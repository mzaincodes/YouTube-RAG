"""YouTube transcript and metadata extraction.

Two independent data sources are combined:

* ``youtube-transcript-api`` (>=1.0 instance API) for timed caption segments.
* YouTube's public **oEmbed** endpoint for the video title, channel and
  thumbnail — ``youtube-transcript-api`` deliberately does not expose them and
  the official Data API would require a second API key.

Every failure mode is translated into a :class:`TranscriptError` carrying a
message that is safe to show directly in the UI.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import requests
from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
)

from .utils import canonical_url

logger = logging.getLogger(__name__)

OEMBED_ENDPOINT = "https://www.youtube.com/oembed"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class TranscriptError(RuntimeError):
    """A transcript could not be retrieved; ``str(exc)`` is UI-safe."""

    def __init__(self, video_id: str, message: str) -> None:
        super().__init__(message)
        self.video_id = video_id
        self.message = message


@dataclass(frozen=True)
class TranscriptSegment:
    """A single timed caption cue."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        """Wall-clock end of the cue, in seconds."""
        return self.start + self.duration


@dataclass
class VideoTranscript:
    """A fully materialised transcript plus the video's display metadata."""

    video_id: str
    url: str
    title: str
    author: str
    thumbnail: str
    language: str
    language_code: str
    is_generated: bool
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """The whole transcript as one whitespace-normalised string."""
        return " ".join(segment.text for segment in self.segments if segment.text)

    @property
    def char_count(self) -> int:
        """Number of characters in :attr:`full_text`."""
        return len(self.full_text)

    @property
    def word_count(self) -> int:
        """Approximate number of words in the transcript."""
        return len(self.full_text.split())

    @property
    def duration_seconds(self) -> float:
        """Timestamp of the end of the final cue."""
        return self.segments[-1].end if self.segments else 0.0


# --------------------------------------------------------------------------- #
# Metadata (oEmbed)
# --------------------------------------------------------------------------- #
def _new_session() -> requests.Session:
    """Return a ``requests`` session with a browser-like User-Agent.

    ``requests`` ships with ``certifi``, which avoids the ``CERTIFICATE_VERIFY_FAILED``
    error that bare ``urllib`` raises on stock macOS Python installs.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def fetch_video_metadata(
    video_id: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> dict[str, str]:
    """Fetch title/author/thumbnail via oEmbed.

    Metadata is cosmetic, so any failure degrades to sensible placeholders
    rather than aborting indexing.
    """
    owns_session = session is None
    session = session or _new_session()
    fallback = {
        "title": f"YouTube video {video_id}",
        "author": "Unknown channel",
        "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    }
    try:
        response = session.get(
            OEMBED_ENDPOINT,
            params={"url": canonical_url(video_id), "format": "json"},
            timeout=timeout,
        )
        if response.status_code != 200:
            logger.debug("oEmbed for %s returned HTTP %s", video_id, response.status_code)
            return fallback
        payload = response.json()
        return {
            "title": str(payload.get("title") or fallback["title"]).strip(),
            "author": str(payload.get("author_name") or fallback["author"]).strip(),
            "thumbnail": str(payload.get("thumbnail_url") or fallback["thumbnail"]),
        }
    except (requests.RequestException, ValueError) as exc:
        logger.debug("oEmbed lookup failed for %s: %s", video_id, exc)
        return fallback
    finally:
        if owns_session:
            session.close()


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #
def _friendly_error(video_id: str, exc: Exception) -> TranscriptError:
    """Map a library exception onto a message a non-technical user can act on."""
    if isinstance(exc, TranscriptsDisabled):
        msg = "Subtitles are disabled for this video, so there is no transcript to index."
    elif isinstance(exc, NoTranscriptFound):
        msg = "No transcript is available for this video in any supported language."
    elif isinstance(exc, (VideoUnavailable, InvalidVideoId)):
        msg = "This video is unavailable — it may be private, deleted, or the ID is wrong."
    elif isinstance(exc, VideoUnplayable):
        msg = "YouTube refused to play this video (region lock or removal), so no transcript could be read."
    elif isinstance(exc, AgeRestricted):
        msg = "This video is age-restricted and its transcript cannot be fetched without sign-in."
    elif isinstance(exc, (IpBlocked, RequestBlocked)):
        msg = (
            "YouTube is temporarily blocking transcript requests from this IP address. "
            "Wait a few minutes, or configure a proxy, then try again."
        )
    elif isinstance(exc, YouTubeRequestFailed):
        msg = "YouTube returned an unexpected error. Please retry in a moment."
    elif isinstance(exc, CouldNotRetrieveTranscript):
        msg = "The transcript could not be retrieved for this video."
    elif isinstance(exc, requests.RequestException):
        msg = "Network error while contacting YouTube. Check your internet connection and retry."
    else:
        msg = f"Unexpected error while fetching the transcript: {exc}"
    logger.warning("Transcript failure for %s: %s (%s)", video_id, msg, type(exc).__name__)
    return TranscriptError(video_id, msg)


# --------------------------------------------------------------------------- #
# Transcript fetching
# --------------------------------------------------------------------------- #
def _select_transcript(api: YouTubeTranscriptApi, video_id: str, languages: Sequence[str]):
    """Pick the best available transcript for ``video_id``.

    Preference order:

    1. A manually created transcript in one of ``languages``.
    2. An auto-generated transcript in one of ``languages``.
    3. Any manually created transcript, translated into ``languages[0]`` if possible.
    4. Any transcript at all, translated when possible.
    """
    transcript_list = api.list(video_id)

    try:
        return transcript_list.find_manually_created_transcript(list(languages))
    except Exception:  # noqa: BLE001 - fall through to the next strategy
        pass
    try:
        return transcript_list.find_generated_transcript(list(languages))
    except Exception:  # noqa: BLE001
        pass

    available = list(transcript_list)
    if not available:
        raise NoTranscriptFound(video_id, list(languages), transcript_list)

    # Prefer human-written captions over ASR when falling back to another language.
    available.sort(key=lambda item: item.is_generated)
    target = languages[0].split("-")[0] if languages else "en"
    for transcript in available:
        if transcript.is_translatable:
            try:
                return transcript.translate(target)
            except Exception:  # noqa: BLE001 - target language not offered
                continue
    return available[0]


def fetch_transcript(
    video_id: str,
    *,
    languages: Sequence[str] = ("en",),
    session: requests.Session | None = None,
) -> VideoTranscript:
    """Fetch the transcript and metadata for a single video.

    Raises:
        TranscriptError: for every failure mode, with a UI-safe message.
    """
    languages = tuple(languages) or ("en",)
    try:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=list(languages))
        except NoTranscriptFound:
            # Widen the search: any language, translated when possible.
            fetched = _select_transcript(api, video_id, languages).fetch()

        segments = [
            TranscriptSegment(
                text=" ".join(snippet.text.split()),
                start=float(snippet.start),
                duration=float(snippet.duration),
            )
            for snippet in fetched.snippets
            if snippet.text and snippet.text.strip()
        ]
    except TranscriptError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated below
        raise _friendly_error(video_id, exc) from exc

    if not segments:
        raise TranscriptError(video_id, "The transcript for this video is empty.")

    metadata = fetch_video_metadata(video_id, session=session)
    return VideoTranscript(
        video_id=video_id,
        url=canonical_url(video_id),
        title=metadata["title"],
        author=metadata["author"],
        thumbnail=metadata["thumbnail"],
        language=getattr(fetched, "language", "Unknown"),
        language_code=getattr(fetched, "language_code", "und"),
        is_generated=bool(getattr(fetched, "is_generated", False)),
        segments=segments,
    )


def fetch_transcripts(
    video_ids: Iterable[str],
    *,
    languages: Sequence[str] = ("en",),
    max_workers: int = 4,
    on_result: Callable[[str, VideoTranscript | None, TranscriptError | None], None] | None = None,
) -> tuple[list[VideoTranscript], list[TranscriptError]]:
    """Fetch several transcripts concurrently.

    Network latency dominates transcript extraction, so a small thread pool cuts
    wall-clock time for a playlist-sized batch dramatically. ``on_result`` is
    invoked once per video (from the calling thread) to drive progress UI.

    Returns:
        ``(successes, failures)`` — one entry per input video.
    """
    ids = list(dict.fromkeys(video_ids))  # de-duplicate, preserve order
    if not ids:
        return [], []

    successes: list[VideoTranscript] = []
    failures: list[TranscriptError] = []
    workers = max(1, min(max_workers, len(ids)))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="transcript") as pool:
        futures = {
            pool.submit(fetch_transcript, video_id, languages=languages): video_id
            for video_id in ids
        }
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                transcript = future.result()
            except TranscriptError as exc:
                failures.append(exc)
                if on_result:
                    on_result(video_id, None, exc)
            except Exception as exc:  # noqa: BLE001 - defensive catch-all
                error = _friendly_error(video_id, exc)
                failures.append(error)
                if on_result:
                    on_result(video_id, None, error)
            else:
                successes.append(transcript)
                if on_result:
                    on_result(video_id, transcript, None)

    # Restore the user's original ordering, which the thread pool scrambles.
    order = {video_id: index for index, video_id in enumerate(ids)}
    successes.sort(key=lambda item: order.get(item.video_id, 0))
    failures.sort(key=lambda item: order.get(item.video_id, 0))
    return successes, failures
