"""Presentation layer: CSS, layout primitives and reusable Streamlit components.

Keeping every piece of markup here means ``app.py`` stays a readable description
of application flow rather than a wall of HTML.

All interpolated values are passed through :func:`html.escape`, because video
titles and channel names come from a third party and land inside raw HTML.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

import streamlit as st

from .graph import SourceCitation
from .utils import format_timestamp, human_int
from .vector_store import VideoRecord

APP_TITLE = "YouTube RAG"
APP_TAGLINE = "Chat with the knowledge inside any YouTube video"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
:root {
  --yt-bg: #0b0f19;
  --yt-surface: #141a29;
  --yt-surface-2: #1b2236;
  --yt-border: #263049;
  --yt-text: #e6e9f0;
  --yt-muted: #93a0bd;
  --yt-primary: #6366f1;
  --yt-primary-soft: rgba(99, 102, 241, 0.14);
  --yt-accent: #22d3ee;
  --yt-success: #34d399;
  --yt-warning: #fbbf24;
  --yt-danger: #f87171;
  --yt-radius: 14px;
}

/* Trim Streamlit's default chrome so the app fills the viewport. */
[data-testid="stAppViewContainer"] > .main .block-container {
  padding-top: 1.6rem;
  padding-bottom: 7rem;
  max-width: 1180px;
}
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

/* ---------------- Header ---------------- */
.yt-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap;
  padding: 1.1rem 1.4rem; margin-bottom: 1.4rem;
  background: linear-gradient(135deg, rgba(99,102,241,.18) 0%, rgba(34,211,238,.08) 55%, rgba(20,26,41,.6) 100%);
  border: 1px solid var(--yt-border); border-radius: var(--yt-radius);
  box-shadow: 0 8px 28px rgba(0,0,0,.28);
}
.yt-header__left { display: flex; align-items: center; gap: .9rem; min-width: 0; }
.yt-header__logo {
  width: 42px; height: 42px; flex: none; border-radius: 12px;
  display: grid; place-items: center; font-size: 1.35rem;
  background: linear-gradient(135deg, var(--yt-primary), var(--yt-accent));
  box-shadow: 0 4px 16px rgba(99,102,241,.4);
}
.yt-header__title { font-size: 1.4rem; font-weight: 750; letter-spacing: -.02em; line-height: 1.15;
  background: linear-gradient(90deg, #fff, #b9c2ff);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.yt-header__sub { font-size: .82rem; color: var(--yt-muted); margin-top: .1rem; }

.yt-badge {
  display: inline-flex; align-items: center; gap: .45rem; white-space: nowrap;
  padding: .38rem .8rem; border-radius: 999px;
  font-size: .76rem; font-weight: 600; letter-spacing: .01em;
  border: 1px solid transparent;
}
.yt-badge--ok      { background: rgba(52,211,153,.13); color: var(--yt-success); border-color: rgba(52,211,153,.3); }
.yt-badge--warn    { background: rgba(251,191,36,.13); color: var(--yt-warning); border-color: rgba(251,191,36,.3); }
.yt-badge--err     { background: rgba(248,113,113,.13); color: var(--yt-danger);  border-color: rgba(248,113,113,.3); }
.yt-badge--neutral { background: rgba(147,160,189,.13); color: var(--yt-muted);   border-color: rgba(147,160,189,.26); }

.yt-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; flex: none; }
.yt-dot--live { animation: yt-pulse 1.8s ease-in-out infinite; }
@keyframes yt-pulse {
  0%,100% { opacity: 1; box-shadow: 0 0 0 0 currentColor; }
  50%     { opacity: .65; box-shadow: 0 0 0 5px rgba(52,211,153,0); }
}

/* ---------------- Cards ---------------- */
.yt-card {
  background: var(--yt-surface); border: 1px solid var(--yt-border);
  border-radius: var(--yt-radius); padding: .95rem 1.05rem; margin-bottom: .7rem;
}
.yt-card--empty { text-align: center; padding: 2.6rem 1.4rem; }
.yt-card--empty h3 { margin: .3rem 0 .5rem; font-size: 1.12rem; color: var(--yt-text); }
.yt-card--empty p  { color: var(--yt-muted); font-size: .9rem; margin: 0 auto; max-width: 46ch; line-height: 1.6; }
.yt-card--empty .yt-emoji { font-size: 2.4rem; }

/* ---------------- Stats ---------------- */
.yt-stats { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: .55rem; }
.yt-stat {
  background: var(--yt-surface-2); border: 1px solid var(--yt-border);
  border-radius: 11px; padding: .6rem .7rem;
}
.yt-stat__value { font-size: 1.12rem; font-weight: 720; color: var(--yt-text); line-height: 1.1; }
.yt-stat__label { font-size: .68rem; color: var(--yt-muted); text-transform: uppercase;
  letter-spacing: .06em; margin-top: .18rem; }

/* ---------------- Indexed video rows ---------------- */
.yt-video { display: flex; gap: .6rem; align-items: flex-start;
  background: var(--yt-surface-2); border: 1px solid var(--yt-border);
  border-radius: 11px; padding: .55rem .65rem; }
.yt-video__thumb { width: 58px; height: 34px; object-fit: cover; border-radius: 6px; flex: none; }
.yt-video__body { min-width: 0; flex: 1; }
.yt-video__title {
  font-size: .8rem; font-weight: 620; color: var(--yt-text); line-height: 1.3;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.yt-video__meta { font-size: .68rem; color: var(--yt-muted); margin-top: .22rem; }
.yt-video__meta a { color: var(--yt-muted); text-decoration: none; }
.yt-video__meta a:hover { color: var(--yt-accent); }

/* ---------------- Source citations ---------------- */
.yt-source {
  border: 1px solid var(--yt-border); border-left: 3px solid var(--yt-primary);
  background: var(--yt-surface-2); border-radius: 10px;
  padding: .65rem .8rem; margin-bottom: .55rem;
}
.yt-source__head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-bottom: .35rem; }
.yt-source__idx {
  background: var(--yt-primary-soft); color: #a5b4fc; border: 1px solid rgba(99,102,241,.35);
  border-radius: 6px; padding: .08rem .42rem; font-size: .7rem; font-weight: 700; flex: none;
}
.yt-source__title { font-size: .82rem; font-weight: 620; color: var(--yt-text); }
.yt-source__time {
  font-size: .7rem; color: var(--yt-accent); text-decoration: none;
  background: rgba(34,211,238,.1); border: 1px solid rgba(34,211,238,.25);
  border-radius: 6px; padding: .08rem .4rem; white-space: nowrap;
}
.yt-source__time:hover { background: rgba(34,211,238,.2); }
.yt-source__score { margin-left: auto; font-size: .68rem; color: var(--yt-muted); white-space: nowrap; }
.yt-source__text { font-size: .78rem; color: var(--yt-muted); line-height: 1.55; margin: 0; }

/* ---------------- Chat ---------------- */
[data-testid="stChatMessage"] {
  background: var(--yt-surface); border: 1px solid var(--yt-border);
  border-radius: var(--yt-radius); padding: .9rem 1.1rem; margin-bottom: .85rem;
  animation: yt-fade .28s ease-out;
}
@keyframes yt-fade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
[data-testid="stChatMessage"] p { line-height: 1.68; }
[data-testid="stChatMessage"] pre { border-radius: 10px; border: 1px solid var(--yt-border); }
[data-testid="stChatMessage"] code { font-size: .84em; }
[data-testid="stChatInput"] { border-radius: 12px; border-color: var(--yt-border); }

/* Typing indicator shown while the graph retrieves. */
.yt-typing { display: inline-flex; gap: .28rem; align-items: center; padding: .2rem 0; }
.yt-typing span {
  width: 7px; height: 7px; border-radius: 50%; background: var(--yt-muted);
  animation: yt-bounce 1.3s infinite ease-in-out;
}
.yt-typing span:nth-child(2) { animation-delay: .18s; }
.yt-typing span:nth-child(3) { animation-delay: .36s; }
@keyframes yt-bounce { 0%,80%,100% { transform: scale(.7); opacity: .5; } 40% { transform: scale(1); opacity: 1; } }

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid var(--yt-border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.yt-sidebar-title {
  font-size: .74rem; font-weight: 700; color: var(--yt-muted);
  text-transform: uppercase; letter-spacing: .09em; margin: .3rem 0 .55rem;
}

/* ---------------- Buttons ---------------- */
.stButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--yt-border);
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.stButton > button:hover { transform: translateY(-1px); border-color: var(--yt-primary); }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--yt-primary), #818cf8); border: none;
  box-shadow: 0 4px 14px rgba(99,102,241,.35);
}

/* ---------------- Footer ---------------- */
.yt-footer {
  margin-top: 2rem; padding: .85rem 1.1rem;
  border: 1px solid var(--yt-border); border-radius: var(--yt-radius);
  background: var(--yt-surface);
  display: flex; gap: 1.2rem; flex-wrap: wrap; justify-content: center;
  font-size: .74rem; color: var(--yt-muted);
}
.yt-footer strong { color: var(--yt-text); font-weight: 620; }

/* Scrollbars */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a3350; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #38446b; }

@media (max-width: 640px) {
  .yt-header { flex-direction: column; align-items: flex-start; }
  .yt-stats { grid-template-columns: 1fr 1fr; }
}
</style>
"""


def inject_css() -> None:
    """Inject the application stylesheet once per rerun."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _esc(value: Any) -> str:
    """HTML-escape any value for safe interpolation into markup."""
    return html.escape(str(value if value is not None else ""), quote=True)


# --------------------------------------------------------------------------- #
# Header / footer
# --------------------------------------------------------------------------- #
def render_header(*, status_label: str, status_kind: str, video_count: int) -> None:
    """Render the sticky-looking title bar with a live status badge.

    Args:
        status_label: Text inside the badge.
        status_kind: One of ``ok``, ``warn``, ``err``, ``neutral``.
        video_count: Number of indexed videos, shown as a second badge.
    """
    kind = status_kind if status_kind in {"ok", "warn", "err", "neutral"} else "neutral"
    live = " yt-dot--live" if kind == "ok" else ""
    st.markdown(
        f"""
        <div class="yt-header">
          <div class="yt-header__left">
            <div class="yt-header__logo">▶</div>
            <div>
              <div class="yt-header__title">{_esc(APP_TITLE)}</div>
              <div class="yt-header__sub">{_esc(APP_TAGLINE)}</div>
            </div>
          </div>
          <div style="display:flex; gap:.5rem; flex-wrap:wrap;">
            <span class="yt-badge yt-badge--neutral">📚 {video_count} video{"s" if video_count != 1 else ""}</span>
            <span class="yt-badge yt-badge--{kind}"><span class="yt-dot{live}"></span>{_esc(status_label)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(*, model: str, embedding_model: str, video_count: int, vector_count: int) -> None:
    """Render the footer strip with model and index information."""
    st.markdown(
        f"""
        <div class="yt-footer">
          <span>🧠 LLM <strong>{_esc(model)}</strong></span>
          <span>🔢 Embeddings <strong>{_esc(embedding_model)}</strong></span>
          <span>📚 Indexed <strong>{human_int(video_count)}</strong></span>
          <span>🧩 Vectors <strong>{human_int(vector_count)}</strong></span>
          <span>⚙️ LangGraph · LangChain · ChromaDB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar components
# --------------------------------------------------------------------------- #
def sidebar_section(title: str) -> None:
    """Render a small uppercase section label in the sidebar."""
    st.markdown(f'<div class="yt-sidebar-title">{_esc(title)}</div>', unsafe_allow_html=True)


def render_stats(stats: dict[str, Any]) -> None:
    """Render the database statistics grid."""
    duration = stats.get("duration_seconds", 0.0)
    disk_mb = stats.get("disk_bytes", 0) / (1024 * 1024)
    cells = [
        (human_int(stats.get("videos", 0)), "Videos"),
        (human_int(stats.get("vectors", 0)), "Vectors"),
        (human_int(stats.get("words", 0)), "Words"),
        (format_timestamp(duration) if duration else "0:00", "Runtime"),
        (f"{disk_mb:.1f} MB", "On disk"),
        (str(stats.get("dimension") or "—"), "Dimensions"),
    ]
    body = "".join(
        f'<div class="yt-stat"><div class="yt-stat__value">{_esc(value)}</div>'
        f'<div class="yt-stat__label">{_esc(label)}</div></div>'
        for value, label in cells
    )
    st.markdown(f'<div class="yt-stats">{body}</div>', unsafe_allow_html=True)


def render_video_row(record: VideoRecord) -> None:
    """Render one indexed-video row in the sidebar."""
    thumb = (
        f'<img class="yt-video__thumb" src="{_esc(record.thumbnail)}" alt="" loading="lazy"/>'
        if record.thumbnail
        else '<div class="yt-video__thumb" style="background:#222b42;"></div>'
    )
    kind = "auto-captions" if record.auto_generated else "captions"
    st.markdown(
        f"""
        <div class="yt-video">
          {thumb}
          <div class="yt-video__body">
            <div class="yt-video__title">{_esc(record.title)}</div>
            <div class="yt-video__meta">
              {_esc(record.author)} · {human_int(record.chunk_count)} chunks · {_esc(kind)}
              · <a href="{_esc(record.url)}" target="_blank" rel="noopener">open ↗</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Chat components
# --------------------------------------------------------------------------- #
def typing_indicator() -> str:
    """HTML for the animated three-dot typing indicator."""
    return '<div class="yt-typing"><span></span><span></span><span></span></div>'


def render_sources(sources: Sequence[SourceCitation], *, expanded: bool = False) -> None:
    """Render citation cards beneath an assistant answer."""
    if not sources:
        return
    label = f"📎 {len(sources)} source{'s' if len(sources) != 1 else ''}"
    with st.expander(label, expanded=expanded):
        for source in sources:
            time_link = (
                f'<a class="yt-source__time" href="{_esc(source.timestamp_url)}" '
                f'target="_blank" rel="noopener">▶ {_esc(source.timestamp)}</a>'
                if source.timestamp_url
                else ""
            )
            st.markdown(
                f"""
                <div class="yt-source">
                  <div class="yt-source__head">
                    <span class="yt-source__idx">{source.index}</span>
                    <span class="yt-source__title">{_esc(source.title)}</span>
                    {time_link}
                    <span class="yt-source__score">relevance {source.score:.0%}</span>
                  </div>
                  <p class="yt-source__text">{_esc(source.snippet)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_copy_block(text: str) -> None:
    """Offer the raw answer for copying.

    ``st.code`` renders a native copy-to-clipboard button, which avoids shipping
    a custom component just for this.
    """
    if not text:
        return
    with st.expander("📋 Copy answer", expanded=False):
        st.code(text, language="markdown")


def render_empty_state() -> None:
    """Render the placeholder shown before anything has been indexed."""
    st.markdown(
        """
        <div class="yt-card yt-card--empty">
          <div class="yt-emoji">🎬</div>
          <h3>No videos indexed yet</h3>
          <p>Paste one or more YouTube links in the sidebar and press
          <strong>Process videos</strong>. Once the transcripts are indexed you can ask
          questions, request summaries, and compare videos — every answer cites the
          exact moment it came from.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome(examples: Iterable[str]) -> None:
    """Render the welcome card listing example questions."""
    items = "".join(f"<li>{_esc(example)}</li>" for example in examples)
    st.markdown(
        f"""
        <div class="yt-card">
          <div style="font-weight:650; margin-bottom:.45rem;">👋 Ready when you are</div>
          <div style="color:var(--yt-muted); font-size:.87rem; line-height:1.7;">
            Your videos are indexed. Try asking:
            <ul style="margin:.5rem 0 0 1.1rem; padding:0;">{items}</ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
