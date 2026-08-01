"""The LangGraph RAG workflow.

Graph shape::

    START -> condense_query -> retrieve -> format_context -> generate -> finalize -> END
                                   |                            ^
                                   +--- (no chunks) ------------+  (skipped)

Each node is a small pure-ish function over :class:`RagState`, so extending the
pipeline — a reranker after ``retrieve``, a guardrail after ``generate``, a
web-search fallback branch — means adding a node and an edge, not rewriting the
control flow.

``retrieve`` short-circuits: when nothing is retrieved, ``generate`` is skipped
entirely. That guarantees the exact "not found" sentence (rather than hoping the
model complies) and avoids paying for a generation that cannot be grounded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import LLMError, build_llm, friendly_llm_error, invoke_llm, stream_llm
from .prompts import (
    CONDENSE_QUESTION_SYSTEM_PROMPT,
    NO_ANSWER_TEXT,
    RAG_SYSTEM_PROMPT,
    build_answer_prompt,
    build_condense_prompt,
    build_context_block,
)
from .retriever import RagRetriever, RetrievalError, RetrievedChunk
from .utils import truncate

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# State & results
# --------------------------------------------------------------------------- #
class RagState(TypedDict, total=False):
    """State threaded through the graph."""

    question: str
    history: list[tuple[str, str]]
    video_ids: list[str]
    top_k: int
    search_type: str

    search_query: str
    chunks: list[RetrievedChunk]
    context: str
    answer: str
    error: str | None


@dataclass
class SourceCitation:
    """A citation rendered under an assistant answer."""

    index: int
    video_id: str
    title: str
    author: str
    url: str
    timestamp: str
    timestamp_url: str
    score: float
    snippet: str
    thumbnail: str = ""


@dataclass
class RagResponse:
    """The result of one full pipeline run."""

    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    search_query: str = ""
    chunks_retrieved: int = 0
    error: str | None = None
    #: True when the answer text was already emitted as ``token`` events, so the
    #: UI should keep what it rendered instead of re-writing the placeholder.
    streamed: bool = False

    @property
    def grounded(self) -> bool:
        """True when the answer is backed by at least one retrieved chunk."""
        return bool(self.sources) and NO_ANSWER_TEXT not in self.answer


def _to_citations(chunks: Sequence[RetrievedChunk]) -> list[SourceCitation]:
    """Convert retrieved chunks into UI-ready citations."""
    citations: list[SourceCitation] = []
    for position, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        citations.append(
            SourceCitation(
                index=position,
                video_id=chunk.video_id,
                title=chunk.title,
                author=str(meta.get("author", "")),
                url=str(meta.get("url", "")),
                timestamp=str(meta.get("timestamp", "")),
                timestamp_url=str(meta.get("timestamp_url", meta.get("url", ""))),
                score=round(float(chunk.score), 4),
                snippet=truncate(chunk.document.page_content.strip(), 320),
                thumbnail=str(meta.get("thumbnail", "")),
            )
        )
    return citations


def _history_messages(history: Sequence[tuple[str, str]], turns: int) -> list[BaseMessage]:
    """Convert the last ``turns`` exchanges into LangChain messages."""
    if not history or turns <= 0:
        return []
    recent = list(history)[-(turns * 2) :]
    messages: list[BaseMessage] = []
    for role, content in recent:
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _history_text(history: Sequence[tuple[str, str]], turns: int) -> str:
    """Render recent history as plain text for the query-condensing prompt."""
    lines: list[str] = []
    for role, content in list(history)[-(turns * 2) :]:
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {truncate(content, 400)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class RagPipeline:
    """Compiles and runs the LangGraph RAG workflow."""

    def __init__(
        self,
        retriever: RagRetriever,
        settings: Settings,
        *,
        llm: Any | None = None,
        condense_llm: Any | None = None,
    ) -> None:
        """Build the pipeline.

        ``llm``/``condense_llm`` exist for dependency injection in tests; in
        normal use they are constructed from ``settings``.
        """
        self._retriever = retriever
        self._settings = settings
        # Streaming model for answers; a separate cheap, deterministic one for
        # query rewriting so its tokens never leak into the streamed answer.
        self._llm = llm or build_llm(settings, streaming=True)
        self._condense_llm = condense_llm or build_llm(settings, streaming=False, temperature=0.0)
        self._graph = self._build_graph()

    # -- nodes ------------------------------------------------------------ #
    def _node_condense_query(self, state: RagState) -> dict[str, Any]:
        """Rewrite a follow-up question into a standalone search query."""
        question = (state.get("question") or "").strip()
        history = state.get("history") or []

        if not history:
            return {"search_query": question}

        history_text = _history_text(history, self._settings.history_turns)
        if not history_text:
            return {"search_query": question}

        try:
            rewritten = invoke_llm(
                self._condense_llm,
                [
                    SystemMessage(content=CONDENSE_QUESTION_SYSTEM_PROMPT),
                    HumanMessage(content=build_condense_prompt(question, history_text)),
                ],
            ).strip()
        except LLMError as exc:
            # Rewriting is an optimisation, never a hard dependency.
            logger.warning("Query condensation failed (%s); using the raw question", exc)
            return {"search_query": question}

        # Guard against a model that returns an explanation instead of a query.
        if not rewritten or len(rewritten) > 400:
            return {"search_query": question}

        if rewritten.lower() != question.lower():
            logger.info("Condensed query: %r -> %r", question, rewritten)
        return {"search_query": rewritten}

    def _node_retrieve(self, state: RagState) -> dict[str, Any]:
        """Fetch the most relevant chunks for the (condensed) query."""
        query = state.get("search_query") or state.get("question") or ""
        try:
            chunks = self._retriever.retrieve(
                query,
                k=state.get("top_k") or self._settings.retrieval_k,
                search_type=state.get("search_type") or self._settings.search_type,
                video_ids=state.get("video_ids") or None,
            )
        except RetrievalError as exc:
            return {"chunks": [], "error": str(exc)}
        return {"chunks": chunks}

    def _node_format_context(self, state: RagState) -> dict[str, Any]:
        """Render retrieved chunks into the numbered CONTEXT block."""
        chunks = state.get("chunks") or []
        context = build_context_block(
            [chunk.document for chunk in chunks],
            max_total_chars=self._settings.max_context_chars,
        )
        return {"context": context}

    def _node_generate(self, state: RagState) -> dict[str, Any]:
        """Call Gemini with the grounded prompt, streaming tokens as they arrive."""
        question = state.get("question") or ""
        context = state.get("context") or ""

        messages: list[BaseMessage] = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
        messages.extend(_history_messages(state.get("history") or [], self._settings.history_turns))
        messages.append(HumanMessage(content=build_answer_prompt(question, context)))

        try:
            # Streaming here (rather than invoke) is what lets LangGraph's
            # "messages" stream mode surface tokens to the UI in real time.
            parts = list(stream_llm(self._llm, messages))
        except LLMError as exc:
            return {"answer": "", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"answer": "", "error": friendly_llm_error(exc)}

        answer = "".join(parts).strip()
        if not answer:
            return {"answer": NO_ANSWER_TEXT}
        return {"answer": answer}

    def _node_no_context(self, state: RagState) -> dict[str, Any]:
        """Produce the fixed refusal without spending a generation call."""
        return {"answer": NO_ANSWER_TEXT, "context": ""}

    @staticmethod
    def _route_after_retrieve(state: RagState) -> Literal["format_context", "no_context"]:
        """Skip generation entirely when retrieval came back empty."""
        if state.get("error"):
            return "no_context"
        return "format_context" if state.get("chunks") else "no_context"

    # -- graph ------------------------------------------------------------ #
    def _build_graph(self):
        """Wire and compile the state graph."""
        builder = StateGraph(RagState)
        builder.add_node("condense_query", self._node_condense_query)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("format_context", self._node_format_context)
        builder.add_node("generate", self._node_generate)
        builder.add_node("no_context", self._node_no_context)

        builder.add_edge(START, "condense_query")
        builder.add_edge("condense_query", "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"format_context": "format_context", "no_context": "no_context"},
        )
        builder.add_edge("format_context", "generate")
        builder.add_edge("generate", END)
        builder.add_edge("no_context", END)
        return builder.compile()

    @property
    def graph(self):
        """The compiled LangGraph application (exposed for visualisation/tests)."""
        return self._graph

    # -- entry points ----------------------------------------------------- #
    def _initial_state(
        self,
        question: str,
        history: Sequence[tuple[str, str]] | None,
        video_ids: Sequence[str] | None,
        top_k: int | None,
        search_type: str | None,
    ) -> RagState:
        """Build the graph input for one turn."""
        return {
            "question": question.strip(),
            "history": list(history or []),
            "video_ids": list(video_ids or []),
            "top_k": top_k or self._settings.retrieval_k,
            "search_type": search_type or self._settings.search_type,
        }

    @staticmethod
    def _to_response(state: dict[str, Any]) -> RagResponse:
        """Convert terminal graph state into a :class:`RagResponse`."""
        chunks: list[RetrievedChunk] = state.get("chunks") or []
        error = state.get("error")
        answer = (state.get("answer") or "").strip()
        if error and not answer:
            answer = error
        return RagResponse(
            answer=answer or NO_ANSWER_TEXT,
            sources=_to_citations(chunks) if answer and NO_ANSWER_TEXT not in answer else [],
            search_query=state.get("search_query") or "",
            chunks_retrieved=len(chunks),
            error=error,
        )

    def answer(
        self,
        question: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        video_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        search_type: str | None = None,
    ) -> RagResponse:
        """Run the pipeline to completion and return the full response."""
        if not question or not question.strip():
            return RagResponse(answer="Please enter a question.")
        state = self._initial_state(question, history, video_ids, top_k, search_type)
        try:
            final = self._graph.invoke(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG pipeline failed")
            return RagResponse(answer=friendly_llm_error(exc), error=str(exc))
        return self._to_response(final)

    def stream(
        self,
        question: str,
        *,
        history: Sequence[tuple[str, str]] | None = None,
        video_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        search_type: str | None = None,
    ) -> Iterator[tuple[str, Any]]:
        """Stream one turn.

        Yields ``("token", str)`` for each generated delta, then exactly one
        ``("final", RagResponse)`` carrying the answer and its citations.
        """
        if not question or not question.strip():
            yield "final", RagResponse(answer="Please enter a question.")
            return

        state = self._initial_state(question, history, video_ids, top_k, search_type)
        final_state: dict[str, Any] = {}
        streamed_any = False

        try:
            for mode, payload in self._graph.stream(state, stream_mode=["messages", "values"]):
                if mode == "messages":
                    message_chunk, metadata = payload
                    if metadata.get("langgraph_node") != "generate":
                        continue  # ignore the query-rewriting model's tokens
                    text = _chunk_text(message_chunk)
                    if text:
                        streamed_any = True
                        yield "token", text
                elif mode == "values":
                    final_state = payload
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG streaming failed")
            yield "final", RagResponse(answer=friendly_llm_error(exc), error=str(exc))
            return

        response = self._to_response(final_state)
        # Tells the UI whether it already rendered this text token-by-token.
        response.streamed = streamed_any
        yield "final", response


def _chunk_text(message_chunk: Any) -> str:
    """Extract text from a streamed message chunk, skipping reasoning blocks."""
    content = getattr(message_chunk, "content", message_chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in {"thinking", "reasoning"}:
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def build_pipeline(retriever: RagRetriever, settings: Settings) -> RagPipeline:
    """Factory used by the Streamlit layer."""
    return RagPipeline(retriever, settings)
