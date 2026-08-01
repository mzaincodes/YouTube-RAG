"""Small, dependency-free helpers shared across the application.

Nothing in this module imports LangChain, Chroma or Streamlit, which keeps it
cheap to import and trivially unit-testable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from typing import Iterable, Iterator, Sequence
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

#: A YouTube video id is always 11 characters of URL-safe base64.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: Hosts we accept as YouTube.
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

#: URL path prefixes that carry the video id as the following path segment.
_PATH_PREFIXES = ("embed", "shorts", "live", "v")

_LOGGING_CONFIGURED = False


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, writing to stderr with a compact format.

    Streamlit re-runs the script constantly, so this guards against installing
    duplicate handlers (which would multiply every log line).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # These libraries are extremely chatty at INFO/DEBUG.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "chromadb.telemetry",
        "google_genai",
        "google.generativeai",
        "grpc",
        "watchdog",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


# --------------------------------------------------------------------------- #
# YouTube URL handling
# --------------------------------------------------------------------------- #
def extract_video_id(url_or_id: str) -> str | None:
    """Extract the 11-character video id from any common YouTube URL form.

    Accepts ``watch?v=``, ``youtu.be/``, ``/embed/``, ``/shorts/``, ``/live/``
    and bare video ids. Returns ``None`` when the input is not a YouTube video.

    >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42")
    'dQw4w9WgXcQ'
    >>> extract_video_id("not a url")
    """
    if not url_or_id:
        return None

    candidate = url_or_id.strip()
    if not candidate:
        return None

    # A bare video id pasted directly.
    if _VIDEO_ID_RE.match(candidate):
        return candidate

    # ``urlparse`` needs a scheme to populate ``netloc``.
    if "://" not in candidate:
        candidate = f"https://{candidate.lstrip('/')}"

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None

    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        return None

    # youtu.be/<id>
    if host.endswith("youtu.be"):
        first = parsed.path.lstrip("/").split("/")[0]
        return first if _VIDEO_ID_RE.match(first) else None

    # youtube.com/watch?v=<id>
    query = parse_qs(parsed.query)
    for key in ("v", "video_id"):
        for value in query.get(key, []):
            if _VIDEO_ID_RE.match(value):
                return value

    # youtube.com/{embed,shorts,live,v}/<id>
    segments = [seg for seg in parsed.path.split("/") if seg]
    for index, segment in enumerate(segments):
        if segment in _PATH_PREFIXES and index + 1 < len(segments):
            nxt = segments[index + 1]
            if _VIDEO_ID_RE.match(nxt):
                return nxt

    return None


def is_valid_youtube_url(url: str) -> bool:
    """True when :func:`extract_video_id` can resolve ``url`` to a video id."""
    return extract_video_id(url) is not None


def canonical_url(video_id: str) -> str:
    """Return the canonical watch URL for ``video_id``."""
    return f"https://www.youtube.com/watch?v={video_id}"


def timestamped_url(video_id: str, seconds: float | int | None) -> str:
    """Return a watch URL that deep-links to ``seconds`` into the video."""
    base = canonical_url(video_id)
    if seconds is None:
        return base
    try:
        offset = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return base
    return f"{base}&t={offset}s"


def parse_url_input(raw: str) -> tuple[list[str], list[str]]:
    """Split free-form user input into ``(video_ids, invalid_entries)``.

    Input may be newline-, comma- or whitespace-separated. Duplicate video ids
    are removed while preserving the order they were first seen in.
    """
    if not raw or not raw.strip():
        return [], []

    tokens = [tok for tok in re.split(r"[\s,;]+", raw.strip()) if tok]

    video_ids: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        video_id = extract_video_id(token)
        if video_id is None:
            invalid.append(token)
            continue
        if video_id in seen:
            continue
        seen.add(video_id)
        video_ids.append(video_id)

    return video_ids, invalid


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float | int | None) -> str:
    """Format ``seconds`` as ``M:SS`` (or ``H:MM:SS`` past an hour)."""
    if seconds is None:
        return "--:--"
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "--:--"

    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def human_int(value: int | float) -> str:
    """Format an integer with thousands separators (``12345`` -> ``12,345``)."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    """Truncate ``text`` to ``limit`` characters on a word boundary when possible."""
    if limit <= 0 or len(text) <= limit:
        return text
    clipped = text[:limit]
    pivot = clipped.rfind(" ")
    if pivot > limit * 0.6:
        clipped = clipped[:pivot]
    return clipped.rstrip() + suffix


# --------------------------------------------------------------------------- #
# Hashing / identity
# --------------------------------------------------------------------------- #
def content_hash(*parts: str) -> str:
    """Return a stable short SHA-256 digest over ``parts``.

    Used to build deterministic vector ids so that re-indexing the same video
    overwrites the previous vectors instead of duplicating them.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x1f")  # unit separator prevents boundary collisions
    return digest.hexdigest()[:32]


def chunk_id(video_id: str, index: int, text: str) -> str:
    """Deterministic vector id for a single chunk of a video."""
    return f"{video_id}:{index:05d}:{content_hash(text)[:16]}"


# --------------------------------------------------------------------------- #
# Iteration helpers
# --------------------------------------------------------------------------- #
def batched(items: Sequence[object], size: int) -> Iterator[list[object]]:
    """Yield consecutive lists of at most ``size`` items from ``items``."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def dedupe(items: Iterable[str]) -> list[str]:
    """Remove duplicates from ``items`` while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
