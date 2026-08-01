"""Persistent ChromaDB vector store management.

Responsibilities
----------------
* Own the single :class:`~langchain_chroma.Chroma` handle and its lifecycle.
* Index transcripts **idempotently** — re-adding a video replaces its vectors
  rather than duplicating them.
* Maintain a lightweight JSON *registry* of indexed videos so the sidebar can
  render titles and counts without scanning every vector in the collection.
* Detect when the embedding model has changed underneath an existing collection,
  which would otherwise silently return nonsense results.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import Settings
from .embeddings import GeminiEmbeddings, embedding_fingerprint
from .transcript import VideoTranscript

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "video_registry.json"

#: Chroma requires 3-512 chars from [a-zA-Z0-9._-], starting/ending alphanumeric.
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


class VectorStoreError(RuntimeError):
    """Vector store operation failed; ``str(exc)`` is safe to show to the user."""


def sanitize_collection_name(name: str) -> str:
    """Coerce ``name`` into something Chroma will accept."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", (name or "").strip())
    cleaned = cleaned.strip("._-")
    if len(cleaned) < 3:
        cleaned = f"{cleaned}_rag" if cleaned else "youtube_rag"
    cleaned = cleaned[:512]
    if not _COLLECTION_NAME_RE.match(cleaned):
        logger.warning("Collection name %r is invalid; using 'youtube_rag'", name)
        return "youtube_rag"
    return cleaned


@dataclass
class VideoRecord:
    """Registry entry describing one indexed video."""

    video_id: str
    title: str
    author: str
    url: str
    thumbnail: str = ""
    chunk_count: int = 0
    char_count: int = 0
    word_count: int = 0
    duration_seconds: float = 0.0
    language: str = "und"
    auto_generated: bool = False
    indexed_at: str = ""

    @classmethod
    def from_transcript(cls, transcript: VideoTranscript, chunk_count: int) -> "VideoRecord":
        """Build a record from a fetched transcript."""
        return cls(
            video_id=transcript.video_id,
            title=transcript.title,
            author=transcript.author,
            url=transcript.url,
            thumbnail=transcript.thumbnail,
            chunk_count=chunk_count,
            char_count=transcript.char_count,
            word_count=transcript.word_count,
            duration_seconds=round(transcript.duration_seconds, 2),
            language=transcript.language_code,
            auto_generated=transcript.is_generated,
            indexed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


@dataclass
class IndexResult:
    """Outcome of indexing a single video."""

    video_id: str
    title: str
    chunk_count: int
    skipped: bool = False
    replaced: bool = False


class VectorStoreManager:
    """Owns the persistent Chroma collection and the indexed-video registry."""

    def __init__(self, settings: Settings, embeddings: GeminiEmbeddings) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._collection_name = sanitize_collection_name(settings.chroma_collection)
        self._persist_dir = Path(settings.persist_directory)
        self._registry_path = self._persist_dir / REGISTRY_FILENAME
        self._lock = threading.RLock()
        self._store: Chroma | None = None
        self._registry: dict[str, VideoRecord] | None = None
        self.embedding_warning: str | None = None

    # -- Chroma handle ---------------------------------------------------- #
    @property
    def store(self) -> Chroma:
        """The lazily-created Chroma handle (reloads existing data on restart)."""
        with self._lock:
            if self._store is None:
                self._store = self._open_store()
            return self._store

    def _open_store(self) -> Chroma:
        """Open or create the persistent collection."""
        try:
            store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self._persist_dir),
                collection_metadata={
                    # Cosine keeps distances bounded in [0, 2], which makes the
                    # relevance scores shown in the UI meaningful.
                    "hnsw:space": "cosine",
                },
                # Normalise Chroma's cosine *distance* into a 0-1 *relevance*.
                relevance_score_fn=lambda distance: max(0.0, min(1.0, 1.0 - distance / 2.0)),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Could not open the vector database at {self._persist_dir}. "
                f"If it is corrupted, delete the folder and re-index. ({exc})"
            ) from exc

        self._check_embedding_space(store)
        return store

    def _check_embedding_space(self, store: Chroma) -> None:
        """Warn when the collection was built with a different embedding model."""
        fingerprint = embedding_fingerprint(self._embeddings)
        try:
            metadata = dict(store._collection.metadata or {})  # noqa: SLF001
            stored = metadata.get("embedding_fingerprint")
            count = store._collection.count()  # noqa: SLF001
        except Exception:  # noqa: BLE001 - never block startup on bookkeeping
            return

        if not stored and count == 0:
            self._write_fingerprint(store, fingerprint)
            return
        if stored and count > 0:
            stored_model = str(stored).split("|")[0]
            if stored_model != self._embeddings.model_name:
                self.embedding_warning = (
                    f"This database was built with '{stored_model}' but the app is "
                    f"configured for '{self._embeddings.model_name}'. Vectors from "
                    "different models are not comparable — clear the database and "
                    "re-index for correct results."
                )
                logger.warning(self.embedding_warning)

    def _write_fingerprint(self, store: Chroma, fingerprint: str) -> None:
        """Persist the embedding fingerprint onto the collection metadata."""
        try:
            metadata = dict(store._collection.metadata or {})  # noqa: SLF001
            metadata["embedding_fingerprint"] = fingerprint
            metadata.setdefault("hnsw:space", "cosine")
            store._collection.modify(metadata=metadata)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001 - cosmetic bookkeeping only
            logger.debug("Could not persist embedding fingerprint: %s", exc)

    # -- Registry --------------------------------------------------------- #
    def _load_registry(self) -> dict[str, VideoRecord]:
        """Read the registry from disk, rebuilding it from Chroma if missing."""
        if self._registry is not None:
            return self._registry

        records: dict[str, VideoRecord] = {}
        if self._registry_path.exists():
            try:
                raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
                for entry in raw.get("videos", []):
                    known = {f for f in VideoRecord.__dataclass_fields__}
                    record = VideoRecord(**{k: v for k, v in entry.items() if k in known})
                    records[record.video_id] = record
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Registry unreadable (%s); rebuilding from Chroma", exc)
                records = {}

        if not records:
            records = self._rebuild_registry_from_store()
            if records:
                self._registry = records
                self._save_registry()

        self._registry = records
        return records

    def _rebuild_registry_from_store(self) -> dict[str, VideoRecord]:
        """Reconstruct registry entries by scanning vector metadata.

        Only used as a recovery path — it reads every metadata row, which is why
        the registry exists in the first place.
        """
        records: dict[str, VideoRecord] = {}
        try:
            payload = self.store.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Registry rebuild skipped: %s", exc)
            return records

        for metadata in payload.get("metadatas") or []:
            if not metadata:
                continue
            video_id = str(metadata.get("video_id") or "")
            if not video_id:
                continue
            record = records.get(video_id)
            if record is None:
                record = VideoRecord(
                    video_id=video_id,
                    title=str(metadata.get("title") or f"YouTube video {video_id}"),
                    author=str(metadata.get("author") or "Unknown channel"),
                    url=str(metadata.get("url") or ""),
                    thumbnail=str(metadata.get("thumbnail") or ""),
                    language=str(metadata.get("language") or "und"),
                    auto_generated=bool(metadata.get("auto_generated", False)),
                )
                records[video_id] = record
            record.chunk_count += 1
            record.char_count += int(metadata.get("char_count") or 0)
            end = float(metadata.get("end_seconds") or 0.0)
            record.duration_seconds = max(record.duration_seconds, end)

        if records:
            logger.info("Rebuilt registry for %d videos from vector metadata", len(records))
        return records

    def _save_registry(self) -> None:
        """Atomically write the registry to disk."""
        records = self._registry or {}
        payload = {
            "collection": self._collection_name,
            "embedding_model": self._embeddings.model_name,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "videos": [asdict(record) for record in records.values()],
        }
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            handle, temp_path = tempfile.mkstemp(
                dir=str(self._persist_dir), prefix=".registry-", suffix=".tmp"
            )
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
            os.replace(temp_path, self._registry_path)
        except OSError as exc:
            logger.warning("Could not persist the video registry: %s", exc)

    # -- Public API ------------------------------------------------------- #
    def is_indexed(self, video_id: str) -> bool:
        """True when ``video_id`` already has vectors in the collection."""
        with self._lock:
            return video_id in self._load_registry()

    def list_videos(self) -> list[VideoRecord]:
        """Return indexed videos, most recently indexed first."""
        with self._lock:
            records = list(self._load_registry().values())
        records.sort(key=lambda record: record.indexed_at, reverse=True)
        return records

    def indexed_video_ids(self) -> list[str]:
        """Return the ids of every indexed video."""
        with self._lock:
            return list(self._load_registry().keys())

    def index_transcript(
        self,
        transcript: VideoTranscript,
        documents: Sequence[Document],
        *,
        force: bool = False,
    ) -> IndexResult:
        """Add ``documents`` for ``transcript``, replacing any previous vectors.

        Args:
            transcript: The source transcript (supplies registry metadata).
            documents: Chunks produced by :func:`~src.chunking.chunk_transcript`.
            force: Re-index even when the video is already present.

        Raises:
            VectorStoreError: when the write fails.
        """
        video_id = transcript.video_id
        with self._lock:
            registry = self._load_registry()
            already = video_id in registry

            if already and not force:
                logger.info("Skipping %s — already indexed", video_id)
                return IndexResult(video_id, transcript.title, registry[video_id].chunk_count, skipped=True)

            if not documents:
                raise VectorStoreError(f"No chunks were produced for '{transcript.title}'.")

            # Drop stale vectors first so a re-index with different chunking
            # settings cannot leave orphans behind.
            if already:
                self._delete_vectors(video_id)

            ids = [doc.id or f"{video_id}:{i}" for i, doc in enumerate(documents)]
            payload = [
                Document(page_content=doc.page_content, metadata=_sanitize_metadata(doc.metadata))
                for doc in documents
            ]

            try:
                self.store.add_documents(payload, ids=ids)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(
                    f"Failed to write '{transcript.title}' to the vector database: {exc}"
                ) from exc

            registry[video_id] = VideoRecord.from_transcript(transcript, len(documents))
            self._registry = registry
            self._save_registry()
            self._write_fingerprint(self.store, embedding_fingerprint(self._embeddings))

        logger.info("Indexed %s (%d chunks)", video_id, len(documents))
        return IndexResult(video_id, transcript.title, len(documents), replaced=already)

    def _delete_vectors(self, video_id: str) -> None:
        """Remove every vector belonging to ``video_id``."""
        try:
            self.store._collection.delete(where={"video_id": video_id})  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete vectors for %s: %s", video_id, exc)

    def delete_video(self, video_id: str) -> bool:
        """Remove a video's vectors and registry entry. Returns True if removed."""
        with self._lock:
            registry = self._load_registry()
            if video_id not in registry:
                return False
            self._delete_vectors(video_id)
            registry.pop(video_id, None)
            self._registry = registry
            self._save_registry()
        logger.info("Deleted video %s from the index", video_id)
        return True

    def clear(self) -> None:
        """Delete the entire collection and registry."""
        with self._lock:
            try:
                store = self.store
                store.delete_collection()
            except Exception as exc:  # noqa: BLE001
                logger.warning("delete_collection failed (%s); removing files instead", exc)
                self._store = None
                shutil.rmtree(self._persist_dir, ignore_errors=True)

            self._store = None
            self._registry = {}
            try:
                self._registry_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.embedding_warning = None
        logger.info("Cleared collection %r", self._collection_name)

    def count_vectors(self) -> int:
        """Total number of vectors currently stored."""
        try:
            return int(self.store._collection.count())  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            logger.debug("Vector count unavailable: %s", exc)
            return 0

    def stats(self) -> dict[str, Any]:
        """Summary statistics for the sidebar and footer."""
        records = self.list_videos()
        return {
            "videos": len(records),
            "vectors": self.count_vectors(),
            "chunks": sum(record.chunk_count for record in records),
            "characters": sum(record.char_count for record in records),
            "words": sum(record.word_count for record in records),
            "duration_seconds": sum(record.duration_seconds for record in records),
            "collection": self._collection_name,
            "path": str(self._persist_dir),
            "disk_bytes": _directory_size(self._persist_dir),
            "embedding_model": self._embeddings.model_name,
            "dimension": self._embeddings.dimension,
        }

    def is_empty(self) -> bool:
        """True when nothing has been indexed yet."""
        return not self.list_videos()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce metadata into the scalar types Chroma guarantees support for.

    Chroma's contract covers ``str``/``int``/``float``/``bool``; anything else is
    stringified (and ``None`` dropped) so a metadata change can never break a write.
    """
    if not metadata:
        return {}
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            clean[str(key)] = value
        else:
            clean[str(key)] = str(value)
    return clean


def _directory_size(path: Path) -> int:
    """Total size in bytes of every file under ``path``."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        return total
    return total


def build_metadata_filter(video_ids: Iterable[str] | None) -> dict[str, Any] | None:
    """Build a Chroma ``where`` clause restricting search to ``video_ids``.

    Chroma requires the ``$in`` operator for multi-value matches and a bare
    equality clause for a single value.
    """
    ids = [vid for vid in (video_ids or []) if vid]
    if not ids:
        return None
    if len(ids) == 1:
        return {"video_id": ids[0]}
    return {"video_id": {"$in": ids}}
