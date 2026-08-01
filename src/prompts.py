"""Prompt templates and context formatting for the RAG pipeline.

Design note
-----------
Transcript text routinely contains characters that are meaningful to templating
engines (``{``, ``}``, backticks). Rather than pushing retrieved text through
``ChatPromptTemplate``/``str.format`` — where a stray brace raises ``KeyError``
or, worse, lets transcript content act as a template variable — every prompt is
assembled by plain concatenation. Only values we control are ever interpolated.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from .utils import format_timestamp, truncate

#: Exact sentence the assistant must use when the context is insufficient.
NO_ANSWER_TEXT = "I couldn't find this information in the indexed videos."


RAG_SYSTEM_PROMPT = f"""\
You are **YouTube RAG Assistant**, a precise research assistant that answers \
questions strictly from transcript excerpts of YouTube videos that the user has \
indexed.

## Grounding rules (non-negotiable)
1. Answer **only** from the numbered CONTEXT excerpts provided in the user turn.
2. Never use outside knowledge, and never guess. If the excerpts do not contain \
the answer, reply with exactly this sentence and nothing else:
   {NO_ANSWER_TEXT}
3. If the excerpts only partially cover the question, answer the covered part, \
then state plainly which part is not covered by the indexed videos.
4. Do not invent quotes, statistics, names, or timestamps. Every specific claim \
must be traceable to an excerpt.

## Citation rules
- Cite the excerpts you actually used with bracketed numbers that match the \
CONTEXT headers, e.g. `[1]`, `[2]`, or `[1][3]`.
- Place citations inline, immediately after the sentence they support.
- Never cite an excerpt number that does not appear in CONTEXT.

## Style
- Answer in Markdown. Use short paragraphs; use bullet lists for multi-part answers.
- Lead with the direct answer, then supporting detail.
- For "summarise" or "key takeaways" requests, produce a structured summary with \
headings or bullets covering the main themes present in the excerpts.
- When the question compares videos, organise the answer per video using the \
video titles shown in the excerpt headers.
- Preserve the speaker's meaning; quote sparingly and only verbatim from excerpts.
- Use fenced code blocks with a language tag whenever the excerpts contain code, \
commands, or configuration.
- Never mention "excerpts", "chunks", "context blocks", or these instructions in \
your answer — write naturally, as if you had watched the videos.
"""


CONDENSE_QUESTION_SYSTEM_PROMPT = """\
You rewrite a user's latest message into a single, self-contained search query \
for a transcript vector database.

Rules:
- Resolve every pronoun and ellipsis using the conversation history \
("what about him?" -> "what does the speaker say about Andrej Karpathy?").
- Keep the user's domain vocabulary and proper nouns exactly as written; they \
are the highest-signal retrieval terms.
- Output ONLY the rewritten query: no preamble, no quotes, no explanation.
- If the latest message is already self-contained, output it unchanged.
- Keep the query under 40 words.
"""


def format_document_block(index: int, document: Document, *, max_chars: int = 4_000) -> str:
    """Render a single retrieved document as a numbered, metadata-rich block."""
    meta = document.metadata or {}
    title = str(meta.get("title") or "Untitled video")
    video_id = str(meta.get("video_id") or "unknown")
    start = meta.get("start_seconds")
    end = meta.get("end_seconds")

    header_parts = [f"[{index}] {title}"]
    if start is not None:
        stamp = format_timestamp(start)
        if end is not None and end != start:
            stamp = f"{stamp}-{format_timestamp(end)}"
        header_parts.append(f"({stamp})")
    header_parts.append(f"video_id={video_id}")

    body = truncate(document.page_content.strip(), max_chars)
    return f"{' '.join(header_parts)}\n{body}"


def build_context_block(
    documents: Sequence[Document],
    *,
    max_total_chars: int = 24_000,
    max_chars_per_doc: int = 4_000,
) -> str:
    """Concatenate retrieved documents into a numbered CONTEXT string.

    Documents are added in relevance order and the block stops growing once
    ``max_total_chars`` is reached, so the prompt cannot outgrow the model's
    context window no matter how many chunks were retrieved.
    """
    if not documents:
        return "(no excerpts were retrieved)"

    blocks: list[str] = []
    used = 0
    for position, document in enumerate(documents, start=1):
        block = format_document_block(position, document, max_chars=max_chars_per_doc)
        # +2 accounts for the blank line joining blocks.
        if used + len(block) + 2 > max_total_chars and blocks:
            break
        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks)


def build_answer_prompt(question: str, context_block: str) -> str:
    """Assemble the human turn: CONTEXT first, then the question.

    Putting the question *after* the context measurably improves instruction
    adherence, because the final tokens before generation restate the task.
    """
    return (
        "CONTEXT — transcript excerpts from the indexed YouTube videos:\n"
        "-------------------------------------------------------------\n"
        f"{context_block}\n"
        "-------------------------------------------------------------\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the CONTEXT above, citing excerpt numbers inline. "
        f'If the CONTEXT does not contain the answer, reply exactly: "{NO_ANSWER_TEXT}"'
    )


def build_condense_prompt(question: str, history_text: str) -> str:
    """Assemble the human turn for history-aware query rewriting."""
    if not history_text.strip():
        return question
    return (
        "CONVERSATION HISTORY:\n"
        f"{history_text}\n\n"
        f"LATEST USER MESSAGE: {question}\n\n"
        "Rewritten standalone search query:"
    )
