"""YouTube RAG — Streamlit application entry point.

Run with::

    streamlit run app.py

This module owns application *flow* only: session state, the sidebar workflow
and the chat loop. Rendering lives in :mod:`src.ui`; all retrieval and
generation logic lives behind :class:`~src.graph.RagPipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import streamlit as st

from src import __version__
from src.chunking import chunk_transcript
from src.config import AVAILABLE_CHAT_MODELS, ConfigurationError, Settings, get_settings
from src.embeddings import EmbeddingError, GeminiEmbeddings, build_embeddings
from src.graph import RagPipeline, RagResponse
from src.llm import LLMError, resolve_chat_model
from src.retriever import RagRetriever
from src.transcript import TranscriptError, VideoTranscript, fetch_transcripts
from src.ui import (
    inject_css,
    render_copy_block,
    render_empty_state,
    render_footer,
    render_header,
    render_sources,
    render_stats,
    render_video_row,
    render_welcome,
    sidebar_section,
    typing_indicator,
)
from src.utils import parse_url_input, setup_logging
from src.vector_store import VectorStoreError, VectorStoreManager

logger = logging.getLogger(__name__)

#: Options offered by the "Retrieved chunks (k)" slider.
TOP_K_CHOICES: tuple[int, ...] = tuple(range(2, 21))

EXAMPLE_QUESTIONS = (
    "Summarise the entire video.",
    "What are the key takeaways?",
    "What is the speaker's opinion on AI?",
    "What were the main topics discussed?",
)


# --------------------------------------------------------------------------- #
# Service wiring
# --------------------------------------------------------------------------- #
@dataclass
class Services:
    """The long-lived objects shared across Streamlit reruns."""

    settings: Settings
    embeddings: GeminiEmbeddings
    manager: VectorStoreManager
    retriever: RagRetriever
    pipeline: RagPipeline
    #: Set when the configured chat model was unavailable and one was substituted.
    model_warning: str | None = None


@st.cache_resource(show_spinner=False)
def build_services(cache_key: str, _settings: Settings) -> Services:
    """Construct the RAG stack once and reuse it across reruns.

    Streamlit re-executes this script top-to-bottom on every interaction, so the
    Chroma handle, the embedding client and the compiled graph must be cached or
    they would be rebuilt on every keystroke.

    Args:
        cache_key: Identity of the configuration; changing it rebuilds the stack.
        _settings: The settings object (underscore-prefixed so Streamlit does not
            try to hash it).
    """
    embeddings = build_embeddings(_settings)
    manager = VectorStoreManager(_settings, embeddings)
    retriever = RagRetriever(manager, embeddings, _settings)

    # Probe once per session: Google retires models but still lists them, so a
    # configured name is no guarantee this key can call it.
    model_name, model_warning = resolve_chat_model(_settings)
    effective = _settings.replace(model_name=model_name)

    pipeline = RagPipeline(retriever, effective)
    logger.info("Services initialised (%s)", cache_key)
    return Services(effective, embeddings, manager, retriever, pipeline, model_warning)


def services_cache_key(settings: Settings) -> str:
    """Identity of every setting that requires rebuilding the service stack."""
    return "|".join(
        [
            settings.model_name,
            settings.embedding_model,
            str(settings.chroma_db_path),
            settings.chroma_collection,
            f"{settings.temperature:.2f}",
            str(settings.max_output_tokens),
            settings.google_api_key[-6:],
        ]
    )


def init_session_state(settings: Settings) -> None:
    """Seed every session-state key the app reads.

    Widget-backed keys are seeded with *real* defaults rather than ``None``.
    Streamlit raises if a widget is given both a ``key`` whose session value
    already exists and an explicit ``value``/``index``, so the widgets below are
    declared key-only and read their defaults from here.
    """
    defaults: dict[str, Any] = {
        "messages": [],
        "last_report": None,
        "confirm_clear": False,
        # Clamped into the slider's option range: Streamlit raises if a
        # session value is not one of the widget's options, and RETRIEVAL_K
        # is free-form in .env.
        "top_k": min(max(settings.retrieval_k, TOP_K_CHOICES[0]), TOP_K_CHOICES[-1]),
        "search_type": settings.search_type,
        "model_name": settings.model_name,
        "video_filter": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Indexing workflow
# --------------------------------------------------------------------------- #
def process_videos(services: Services, raw_input: str, *, force: bool) -> dict[str, Any]:
    """Fetch, chunk and index every valid video in ``raw_input``.

    Returns:
        A report dict consumed by :func:`render_report`.
    """
    settings = services.settings
    video_ids, invalid = parse_url_input(raw_input)
    report: dict[str, Any] = {"indexed": [], "skipped": [], "failed": [], "invalid": invalid}

    if not video_ids:
        return report

    known = set(services.manager.indexed_video_ids())
    pending = video_ids if force else [vid for vid in video_ids if vid not in known]
    for video_id in video_ids:
        if video_id not in pending:
            report["skipped"].append(video_id)

    if not pending:
        return report

    progress = st.progress(0.0, text="Fetching transcripts…")
    status = st.empty()
    total = len(pending)
    completed = 0

    def on_result(
        video_id: str, transcript: VideoTranscript | None, error: TranscriptError | None
    ) -> None:
        nonlocal completed
        completed += 1
        label = transcript.title[:46] if transcript else video_id
        verb = "Fetched" if transcript else "Failed"
        progress.progress(completed / (total * 2), text=f"{verb} {completed}/{total}: {label}")

    transcripts, failures = fetch_transcripts(
        pending,
        languages=settings.transcript_languages,
        max_workers=settings.transcript_max_workers,
        on_result=on_result,
    )
    report["failed"].extend({"video_id": f.video_id, "error": f.message} for f in failures)

    for position, transcript in enumerate(transcripts, start=1):
        fraction = 0.5 + (position / max(1, len(transcripts))) * 0.5
        progress.progress(min(fraction, 1.0), text=f"Indexing: {transcript.title[:46]}")
        status.caption(f"Chunking and embedding **{transcript.title[:70]}**…")
        try:
            chunks = chunk_transcript(transcript, settings, services.embeddings)
            if not chunks:
                raise VectorStoreError("The transcript produced no usable chunks.")
            result = services.manager.index_transcript(transcript, chunks, force=force)
        except (EmbeddingError, VectorStoreError, ConfigurationError) as exc:
            report["failed"].append({"video_id": transcript.video_id, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - never lose the whole batch
            logger.exception("Indexing failed for %s", transcript.video_id)
            report["failed"].append({"video_id": transcript.video_id, "error": str(exc)})
            continue

        if result.skipped:
            report["skipped"].append(transcript.video_id)
        else:
            report["indexed"].append(
                {"title": result.title, "chunks": result.chunk_count, "video_id": result.video_id}
            )

    progress.progress(1.0, text="Done")
    progress.empty()
    status.empty()
    return report


def render_report(report: dict[str, Any]) -> None:
    """Turn an indexing report into toasts and inline messages."""
    if not report:
        return
    for entry in report.get("indexed", []):
        st.toast(f"✅ Indexed “{entry['title'][:40]}” ({entry['chunks']} chunks)", icon="✅")
    if report.get("skipped"):
        st.toast(f"⏭️ Skipped {len(report['skipped'])} already-indexed video(s)", icon="⏭️")
    for entry in report.get("failed", []):
        st.error(f"**{entry['video_id']}** — {entry['error']}", icon="🚫")
    if report.get("invalid"):
        preview = ", ".join(report["invalid"][:3])
        st.warning(f"Ignored {len(report['invalid'])} invalid link(s): {preview}", icon="⚠️")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(services: Services | None, settings: Settings) -> None:
    """Render the whole sidebar: input, status, library, stats and settings."""
    with st.sidebar:
        st.markdown("### ▶ YouTube RAG")
        st.caption(f"v{__version__} · LangGraph + Gemini")

        # -- Add videos -------------------------------------------------- #
        sidebar_section("Add videos")
        raw = st.text_area(
            "YouTube links",
            key="url_input",
            height=110,
            placeholder="https://youtu.be/dQw4w9WgXcQ\nhttps://www.youtube.com/watch?v=…",
            label_visibility="collapsed",
            help="One or more links — newline, comma or space separated. Duplicates are ignored.",
        )
        preview_ids, preview_invalid = parse_url_input(raw or "")
        if preview_ids or preview_invalid:
            bits = []
            if preview_ids:
                bits.append(f"✅ {len(preview_ids)} valid")
            if preview_invalid:
                bits.append(f"⚠️ {len(preview_invalid)} invalid")
            st.caption(" · ".join(bits))

        force = st.checkbox(
            "Re-index if already present",
            value=False,
            help="Refetch and rebuild vectors for videos that are already in the database.",
        )

        disabled = services is None or not preview_ids
        if st.button(
            "⚡ Process videos", type="primary", use_container_width=True, disabled=disabled
        ):
            if services is not None:
                with st.spinner("Processing…"):
                    st.session_state.last_report = process_videos(services, raw or "", force=force)
                st.rerun()

        if st.session_state.last_report:
            render_report(st.session_state.last_report)
            st.session_state.last_report = None

        st.divider()

        if services is None:
            st.info("Add your `GOOGLE_API_KEY` to `.env` to enable indexing.", icon="🔑")
            return

        # -- Library ------------------------------------------------------ #
        records = services.manager.list_videos()
        sidebar_section(f"Indexed videos ({len(records)})")
        if not records:
            st.caption("Nothing indexed yet.")
        else:
            for record in records:
                render_video_row(record)
                if st.button(
                    "Remove",
                    key=f"del_{record.video_id}",
                    use_container_width=True,
                    help=f"Delete “{record.title[:50]}” from the index",
                ):
                    services.manager.delete_video(record.video_id)
                    st.toast(f"🗑️ Removed “{record.title[:40]}”", icon="🗑️")
                    st.rerun()

        st.divider()

        # -- Statistics --------------------------------------------------- #
        sidebar_section("Database")
        render_stats(services.manager.stats())
        st.caption(f"`{services.manager.stats()['path']}`")

        if records:
            if not st.session_state.confirm_clear:
                if st.button("🗑️ Clear database", use_container_width=True):
                    st.session_state.confirm_clear = True
                    st.rerun()
            else:
                st.warning("Delete every indexed video? This cannot be undone.", icon="⚠️")
                left, right = st.columns(2)
                if left.button("Yes, clear", type="primary", use_container_width=True):
                    services.manager.clear()
                    st.session_state.messages = []
                    st.session_state.confirm_clear = False
                    st.toast("Database cleared", icon="🗑️")
                    st.rerun()
                if right.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

        st.divider()

        # -- Settings ------------------------------------------------------ #
        sidebar_section("Settings")
        models = list(AVAILABLE_CHAT_MODELS)
        if settings.model_name not in models:
            models.insert(0, settings.model_name)
        if st.session_state.model_name not in models:
            st.session_state.model_name = models[0]
        st.selectbox(
            "Chat model",
            models,
            key="model_name",
            help="Flash is fastest; Pro reasons more deeply on complex questions.",
        )
        st.select_slider(
            "Retrieved chunks (k)",
            options=list(TOP_K_CHOICES),
            key="top_k",
            help="More chunks give broader context at the cost of latency and tokens.",
        )
        st.radio(
            "Search strategy",
            ["mmr", "similarity"],
            key="search_type",
            horizontal=True,
            help="MMR diversifies results; similarity returns the closest matches.",
        )
        if records:
            options = [record.video_id for record in records]
            # A deleted video would leave a stale id here, which Streamlit
            # rejects as an out-of-range default.
            stale = [vid for vid in st.session_state.video_filter if vid not in options]
            if stale:
                st.session_state.video_filter = [
                    vid for vid in st.session_state.video_filter if vid in options
                ]
            titles = {record.video_id: record.title for record in records}
            st.multiselect(
                "Limit search to",
                options=options,
                format_func=lambda vid: titles.get(vid, vid)[:40],
                key="video_filter",
                help="Leave empty to search every indexed video.",
            )


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def history_pairs() -> list[tuple[str, str]]:
    """Session messages as ``(role, content)`` pairs for the pipeline."""
    return [(message["role"], message["content"]) for message in st.session_state.messages]


def stream_to_ui(pipeline: RagPipeline, question: str, **kwargs: Any) -> RagResponse:
    """Stream one answer into the chat area and return the final response."""
    placeholder = st.empty()
    placeholder.markdown(typing_indicator(), unsafe_allow_html=True)

    collected: list[str] = []
    final: RagResponse | None = None

    try:
        for kind, payload in pipeline.stream(question, **kwargs):
            if kind == "token":
                collected.append(payload)
                # The trailing block cursor makes streaming feel responsive.
                placeholder.markdown("".join(collected) + " ▌")
            else:
                final = payload
    except LLMError as exc:
        placeholder.empty()
        st.error(str(exc), icon="🚫")
        return RagResponse(answer=str(exc), error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat turn failed")
        placeholder.empty()
        st.error(f"Something went wrong: {exc}", icon="🚫")
        return RagResponse(answer=f"Something went wrong: {exc}", error=str(exc))

    if final is None:
        final = RagResponse(answer="".join(collected))

    placeholder.markdown(final.answer)
    return final


def render_chat(services: Services) -> None:
    """Render the message history and handle the next turn."""
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources") or [])
                render_copy_block(message["content"])

    question = st.chat_input("Ask anything about your indexed videos…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        response = stream_to_ui(
            services.pipeline,
            question,
            history=history_pairs()[:-1],
            video_ids=st.session_state.video_filter or None,
            top_k=st.session_state.top_k,
            search_type=st.session_state.search_type,
        )
        render_sources(response.sources, expanded=False)
        render_copy_block(response.answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.answer, "sources": response.sources}
    )
    st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="YouTube RAG · Chat with videos",
        page_icon="▶️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    base_settings = get_settings()
    setup_logging(base_settings.log_level)
    inject_css()
    init_session_state(base_settings)

    # Apply live overrides chosen in the sidebar settings panel.
    settings = base_settings
    if st.session_state.model_name and st.session_state.model_name != base_settings.model_name:
        settings = settings.replace(model_name=st.session_state.model_name)

    services: Services | None = None
    status_label, status_kind = "Ready", "ok"
    startup_error: str | None = None

    if not settings.has_api_key:
        status_label, status_kind = "API key missing", "err"
        startup_error = (
            "**GOOGLE_API_KEY is not set.**\n\n"
            "1. Copy `.env.example` to `.env`\n"
            "2. Add your key: `GOOGLE_API_KEY=…`\n"
            "3. Restart the app\n\n"
            "Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)."
        )
    else:
        try:
            services = build_services(services_cache_key(settings), settings)
        except (EmbeddingError, ConfigurationError, VectorStoreError, LLMError) as exc:
            status_label, status_kind = "Configuration error", "err"
            startup_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Startup failed")
            status_label, status_kind = "Startup failed", "err"
            startup_error = f"Could not start the RAG stack: {exc}"

    video_count = len(services.manager.list_videos()) if services else 0
    if services and video_count == 0:
        status_label, status_kind = "No videos indexed", "warn"

    render_header(status_label=status_label, status_kind=status_kind, video_count=video_count)
    render_sidebar(services, settings)

    if startup_error:
        st.error(startup_error, icon="🔑")
        render_footer(
            model=settings.model_name,
            embedding_model=settings.embedding_model,
            video_count=0,
            vector_count=0,
        )
        return

    assert services is not None  # narrowed by the startup_error guard above

    if services.model_warning:
        st.warning(services.model_warning, icon="🔄")

    if services.manager.embedding_warning:
        st.warning(services.manager.embedding_warning, icon="⚠️")

    if video_count == 0:
        render_empty_state()
    else:
        if not st.session_state.messages:
            render_welcome(EXAMPLE_QUESTIONS)
        else:
            left, right = st.columns([1, 5])
            if left.button("🧹 Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        render_chat(services)

    stats = services.manager.stats()
    render_footer(
        model=services.settings.model_name,
        embedding_model=stats["embedding_model"],
        video_count=stats["videos"],
        vector_count=stats["vectors"],
    )


if __name__ == "__main__":
    main()
