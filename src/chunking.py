"""Timestamp-preserving, topic-aware chunking of YouTube transcripts.

Why not a plain character splitter?
-----------------------------------
Raw caption cues are 1–5 second fragments with no relationship to meaning.
Splitting them every *N* characters reliably cuts sentences — and topics — in
half, which produces chunks that retrieve poorly and read badly when quoted.

This module therefore works in two passes:

**Pass 1 — build timed units.** Caption cues are merged into the largest natural
unit available. When the transcript is punctuated (human-written captions) the
unit is a *sentence*; when it is not (YouTube ASR output, which has no full
stops at all) the unit is a fixed *time window*. Either way each unit keeps the
start/end timestamp of the cues it came from.

**Pass 2 — group units into chunks.** With an embedding model available, units
are embedded and the cosine distance between *consecutive* units is measured. A
large distance means the speaker changed subject, so breakpoints are placed at
the high percentile of that distance distribution — topics stay whole. Without
an embedding model (or for very long transcripts, where the extra embedding pass
is not worth the money) the module falls back to a recursive character splitter
that still respects the unit boundaries from pass 1.

Every emitted chunk carries the metadata needed for citation: source video,
chunk number, start/end timestamp, a deep link, title and URL.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .transcript import TranscriptSegment, VideoTranscript
from .utils import chunk_id, format_timestamp, timestamped_url

logger = logging.getLogger(__name__)

#: Sentence-final punctuation used to detect punctuated transcripts.
_SENTENCE_END_RE = re.compile(r"[.!?]['\")\]]*\s")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])['\")\]]*\s+")

#: Below this ratio of sentence terminators per word we treat the transcript as
#: unpunctuated ASR output and switch to time-window units.
_PUNCTUATION_RATIO_THRESHOLD = 0.01

#: Target wall-clock length of a unit when the transcript has no punctuation.
_ASR_WINDOW_SECONDS = 24.0
_ASR_WINDOW_MAX_CHARS = 320


@dataclass
class TimedUnit:
    """An atomic, timestamped piece of transcript that is never split further."""

    text: str
    start: float
    end: float

    @property
    def char_count(self) -> int:
        """Length of :attr:`text`."""
        return len(self.text)


# --------------------------------------------------------------------------- #
# Pass 1 — caption cues -> timed units
# --------------------------------------------------------------------------- #
def _is_punctuated(text: str) -> bool:
    """Heuristically decide whether ``text`` contains real sentence punctuation."""
    words = max(1, len(text.split()))
    terminators = len(_SENTENCE_END_RE.findall(text + " "))
    return (terminators / words) >= _PUNCTUATION_RATIO_THRESHOLD


def _units_from_sentences(segments: Sequence[TranscriptSegment]) -> list[TimedUnit]:
    """Merge cues into sentence-level units, carrying timestamps across."""
    units: list[TimedUnit] = []
    buffer: list[str] = []
    start: float | None = None
    end: float = 0.0

    for segment in segments:
        if start is None:
            start = segment.start
        buffer.append(segment.text)
        end = segment.end

        joined = " ".join(buffer)
        # A cue may contain several sentences; emit all complete ones.
        pieces = _SENTENCE_SPLIT_RE.split(joined)
        if len(pieces) > 1 and _SENTENCE_END_RE.search(joined + " "):
            complete, remainder = pieces[:-1], pieces[-1]
            span = max(end - start, 0.001)
            consumed = 0
            total = max(1, sum(len(piece) for piece in pieces))
            for piece in complete:
                piece = piece.strip()
                if not piece:
                    continue
                # Distribute the cue's time span proportionally to text length.
                piece_start = start + span * (consumed / total)
                consumed += len(piece)
                piece_end = start + span * (consumed / total)
                units.append(TimedUnit(text=piece, start=piece_start, end=piece_end))
            if remainder.strip():
                start = start + span * (consumed / total)
                buffer = [remainder.strip()]
            else:
                start, buffer = None, []

    if buffer and start is not None:
        tail = " ".join(buffer).strip()
        if tail:
            units.append(TimedUnit(text=tail, start=start, end=end))
    return units


def _units_from_time_windows(segments: Sequence[TranscriptSegment]) -> list[TimedUnit]:
    """Merge cues into ~24 s windows for transcripts without punctuation."""
    units: list[TimedUnit] = []
    buffer: list[str] = []
    start: float | None = None
    end: float = 0.0

    for segment in segments:
        if start is None:
            start = segment.start
        buffer.append(segment.text)
        end = segment.end

        span = end - start
        chars = sum(len(item) + 1 for item in buffer)
        if span >= _ASR_WINDOW_SECONDS or chars >= _ASR_WINDOW_MAX_CHARS:
            text = " ".join(buffer).strip()
            if text:
                units.append(TimedUnit(text=text, start=start, end=end))
            start, buffer = None, []

    if buffer and start is not None:
        text = " ".join(buffer).strip()
        if text:
            units.append(TimedUnit(text=text, start=start, end=end))
    return units


def build_timed_units(transcript: VideoTranscript) -> list[TimedUnit]:
    """Convert caption cues into sentence or time-window units."""
    segments = transcript.segments
    if not segments:
        return []
    if _is_punctuated(transcript.full_text):
        units = _units_from_sentences(segments)
        if units:
            logger.debug("%s: %d sentence units", transcript.video_id, len(units))
            return units
    units = _units_from_time_windows(segments)
    logger.debug("%s: %d time-window units", transcript.video_id, len(units))
    return units


# --------------------------------------------------------------------------- #
# Pass 2 — timed units -> chunks
# --------------------------------------------------------------------------- #
def _cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Cosine distance between each consecutive pair of row vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on empty/degenerate units
    unit_vectors = vectors / norms
    similarities = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
    return 1.0 - similarities


def _group_indices_semantic(
    units: Sequence[TimedUnit],
    embedder: Embeddings,
    settings: Settings,
) -> list[list[int]]:
    """Return unit-index groups split at semantic breakpoints."""
    texts = [unit.text for unit in units]
    vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(units):
        raise ValueError("embedder returned an unexpected number of vectors")

    distances = _cosine_distances(vectors)
    if distances.size == 0:
        return [list(range(len(units)))]

    percentile = min(max(settings.semantic_breakpoint_percentile, 50.0), 99.0)
    # ``method="lower"`` snaps the threshold to an *observed* distance instead of
    # interpolating between two of them. With linear interpolation the threshold
    # can land in the gap between two genuine topic shifts, so only the larger of
    # the pair becomes a breakpoint and the other topic boundary is missed.
    threshold = float(np.percentile(distances, percentile, method="lower"))

    groups: list[list[int]] = []
    current: list[int] = [0]
    current_chars = units[0].char_count
    hard_max = settings.chunk_size * 2

    for index, distance in enumerate(distances):
        next_index = index + 1
        next_chars = units[next_index].char_count
        topic_shift = distance >= threshold and current_chars >= settings.semantic_min_chunk_chars
        too_large = current_chars + next_chars > hard_max
        if topic_shift or too_large:
            groups.append(current)
            current, current_chars = [next_index], next_chars
        else:
            current.append(next_index)
            current_chars += next_chars

    groups.append(current)
    return groups


def _group_indices_structural(units: Sequence[TimedUnit], settings: Settings) -> list[list[int]]:
    """Greedy size-based grouping that still never splits a unit in half."""
    groups: list[list[int]] = []
    current: list[int] = []
    current_chars = 0

    for index, unit in enumerate(units):
        if current and current_chars + unit.char_count > settings.chunk_size:
            groups.append(current)
            current, current_chars = [], 0
        current.append(index)
        current_chars += unit.char_count + 1

    if current:
        groups.append(current)
    return groups


def _merge_undersized(
    groups: list[list[int]],
    units: Sequence[TimedUnit],
    minimum: int,
    ceiling: int,
) -> list[list[int]]:
    """Fold groups shorter than ``minimum`` characters into their neighbour.

    A lone 80-character chunk retrieves badly, so tiny groups are absorbed by the
    preceding one. The merge is refused once the result would exceed ``ceiling``:
    without that guard a run of short groups cascades into a single oversized
    chunk that spans every topic — exactly what the semantic split just avoided.
    """
    if len(groups) <= 1:
        return groups

    merged: list[list[int]] = []
    sizes: list[int] = []
    for group in groups:
        size = sum(units[i].char_count for i in group)
        if merged and size < minimum and sizes[-1] + size <= ceiling:
            merged[-1].extend(group)
            sizes[-1] += size
        else:
            merged.append(list(group))
            sizes.append(size)
    return merged


def _overlap_prefix(previous_text: str, overlap: int) -> str:
    """Return the trailing ``overlap`` characters of ``previous_text``.

    The cut is moved to the nearest word boundary so the carried-over context
    never starts mid-word.
    """
    if overlap <= 0 or not previous_text:
        return ""
    tail = previous_text[-overlap:]
    pivot = tail.find(" ")
    if 0 <= pivot < len(tail) - 1:
        tail = tail[pivot + 1 :]
    return tail.strip()


def chunk_transcript(
    transcript: VideoTranscript,
    settings: Settings,
    embedder: Embeddings | None = None,
) -> list[Document]:
    """Split ``transcript`` into retrieval-ready :class:`Document` chunks.

    Args:
        transcript: The fetched transcript to split.
        settings: Chunking configuration (size, overlap, semantic toggles).
        embedder: Optional embedding model. When supplied — and the transcript is
            short enough to be worth the extra pass — semantic breakpoint
            detection is used instead of size-based grouping.

    Returns:
        Chunks in transcript order, each carrying full citation metadata.
    """
    units = build_timed_units(transcript)
    if not units:
        return []

    use_semantic = (
        settings.semantic_chunking
        and embedder is not None
        and 1 < len(units) <= settings.semantic_max_sentences_probe
    )

    groups: list[list[int]] | None = None
    if use_semantic:
        try:
            groups = _group_indices_semantic(units, embedder, settings)  # type: ignore[arg-type]
            logger.info(
                "%s: semantic chunking produced %d groups from %d units",
                transcript.video_id,
                len(groups),
                len(units),
            )
        except Exception as exc:  # noqa: BLE001 - never fail indexing over this
            logger.warning(
                "Semantic chunking failed for %s (%s); using structural chunking",
                transcript.video_id,
                exc,
            )
            groups = None

    if groups is None:
        if len(units) > settings.semantic_max_sentences_probe and settings.semantic_chunking:
            logger.info(
                "%s: %d units exceeds the semantic probe limit; using structural chunking",
                transcript.video_id,
                len(units),
            )
        groups = _group_indices_structural(units, settings)

    groups = _merge_undersized(
        groups, units, settings.semantic_min_chunk_chars, settings.chunk_size
    )
    return _documents_from_groups(transcript, units, groups, settings)


def _documents_from_groups(
    transcript: VideoTranscript,
    units: Sequence[TimedUnit],
    groups: Sequence[Sequence[int]],
    settings: Settings,
) -> list[Document]:
    """Materialise unit groups into ``Document`` objects with citation metadata."""
    # Guard against a pathological group whose units alone blow the size budget.
    hard_max = settings.chunk_size * 3
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True,
    )

    documents: list[Document] = []
    previous_text = ""

    for group in groups:
        if not group:
            continue
        body = " ".join(units[i].text for i in group).strip()
        if not body:
            continue
        start = units[group[0]].start
        end = units[group[-1]].end

        pieces = splitter.split_text(body) if len(body) > hard_max else [body]
        span = max(end - start, 0.001)
        consumed = 0
        total = max(1, sum(len(piece) for piece in pieces))

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            piece_start = start + span * (consumed / total)
            consumed += len(piece)
            piece_end = start + span * (consumed / total)

            prefix = _overlap_prefix(previous_text, settings.chunk_overlap)
            text = f"{prefix} {piece}".strip() if prefix else piece
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "start_seconds": round(float(piece_start), 2),
                        "end_seconds": round(float(piece_end), 2),
                    },
                )
            )
            previous_text = piece

    # Attach index-dependent metadata once the total is known.
    total_chunks = len(documents)
    for index, document in enumerate(documents):
        start_seconds = document.metadata["start_seconds"]
        document.metadata.update(
            {
                "video_id": transcript.video_id,
                "title": transcript.title,
                "author": transcript.author,
                "url": transcript.url,
                "thumbnail": transcript.thumbnail,
                "chunk_index": index,
                "total_chunks": total_chunks,
                "timestamp": format_timestamp(start_seconds),
                "timestamp_url": timestamped_url(transcript.video_id, start_seconds),
                "language": transcript.language_code,
                "auto_generated": transcript.is_generated,
                "char_count": len(document.page_content),
                "source": "youtube",
            }
        )
        document.id = chunk_id(transcript.video_id, index, document.page_content)

    logger.info(
        "%s (%s): %d chunks, avg %d chars",
        transcript.video_id,
        transcript.title[:48],
        total_chunks,
        (sum(len(d.page_content) for d in documents) // total_chunks) if total_chunks else 0,
    )
    return documents
