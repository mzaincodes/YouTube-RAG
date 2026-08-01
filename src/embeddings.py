"""Gemini embedding wrapper with batching, retries, caching and model fallback.

``GoogleGenerativeAIEmbeddings`` already speaks the LangChain ``Embeddings``
protocol, but on its own it is not production-safe: a single 429 in the middle
of indexing a long video loses the whole batch, and Google's embedding model
line-up changes often enough that a hard-coded model name is a liability.

:class:`GeminiEmbeddings` wraps it with

* **bounded batches** (Google caps a request at 100 inputs),
* **exponential backoff with jitter** on transient failures only,
* a **per-process query cache**, so repeated questions cost nothing, and
* **automatic model fallback** across :data:`~src.config.FALLBACK_EMBEDDING_MODELS`.

The instance-level ``task_type`` is deliberately left unset: the underlying
class then defaults to ``RETRIEVAL_DOCUMENT`` for documents and
``RETRIEVAL_QUERY`` for queries, which is what asymmetric retrieval needs.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Sequence

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import FALLBACK_EMBEDDING_MODELS, Settings
from .utils import content_hash

logger = logging.getLogger(__name__)

#: Google rejects embedding requests with more than 100 inputs.
_GOOGLE_MAX_BATCH = 100

#: Substrings that identify a *transient* failure worth retrying.
_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "quota",
    "resource has been exhausted",
    "resource_exhausted",
    "deadline",
    "timeout",
    "timed out",
    "unavailable",
    "internal error",
    "connection",
    "temporarily",
    "overloaded",
)

#: Substrings that identify a *permanent* failure — retrying only wastes time.
_FATAL_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "permission denied",
    "permission_denied",
    "unauthenticated",
    "401",
    "403",
)


class EmbeddingError(RuntimeError):
    """Embedding generation failed; ``str(exc)`` is safe to show to the user."""


def _classify(exc: Exception) -> tuple[bool, str]:
    """Return ``(is_retryable, friendly_message)`` for an embedding exception."""
    text = f"{type(exc).__name__}: {exc}".lower()

    if any(marker in text for marker in _FATAL_MARKERS):
        return False, (
            "Your Google API key was rejected. Check GOOGLE_API_KEY in .env — "
            "you can generate a new key at https://aistudio.google.com/apikey"
        )
    if "not found" in text or "404" in text:
        return False, (
            "The configured embedding model was not found. Set EMBEDDING_MODEL in "
            ".env to a supported model such as models/gemini-embedding-001"
        )
    if any(marker in text for marker in _RETRYABLE_MARKERS):
        return True, (
            "Google's embedding API is rate-limiting or temporarily unavailable. "
            "The app retried automatically but did not succeed — please try again shortly."
        )
    return False, f"Embedding request failed: {exc}"


class GeminiEmbeddings(Embeddings):
    """Production-hardened Gemini embeddings implementing the LangChain protocol."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        query_cache_size: int = 512,
    ) -> None:
        self._settings = settings
        self._model = model or settings.embedding_model
        self._api_key = settings.require_api_key()
        self._batch_size = max(1, min(settings.embed_batch_size, _GOOGLE_MAX_BATCH))
        self._max_retries = max(1, settings.max_retries)
        self._base_delay = max(0.1, settings.retry_base_delay)
        self._query_cache: dict[str, list[float]] = {}
        self._query_cache_size = query_cache_size
        self._lock = threading.Lock()
        self._dimension: int | None = None
        self._client = self._build_client(self._model)

    # -- construction ---------------------------------------------------- #
    def _build_client(self, model: str) -> GoogleGenerativeAIEmbeddings:
        """Instantiate the underlying LangChain embeddings client."""
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=self._api_key)

    @property
    def model_name(self) -> str:
        """The embedding model actually in use (may differ after fallback)."""
        return self._model

    @property
    def dimension(self) -> int | None:
        """Vector width, populated after the first successful call."""
        return self._dimension

    # -- core -------------------------------------------------------------#
    def _call_with_retry(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """Invoke the API for one batch, retrying transient failures."""
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                if is_query:
                    return [self._client.embed_query(texts[0])]
                return self._client.embed_documents(texts, batch_size=len(texts))
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                retryable, message = _classify(exc)
                if not retryable or attempt == self._max_retries:
                    # Permanent failure, or the last attempt was exhausted.
                    raise EmbeddingError(message) from exc
                # Exponential backoff with full jitter avoids thundering herds.
                delay = min(self._base_delay * (2 ** (attempt - 1)), 30.0)
                delay = random.uniform(delay * 0.5, delay)
                logger.warning(
                    "Embedding attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        raise EmbeddingError(_classify(last_exc)[1] if last_exc else "Embedding failed")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` as retrieval documents, in bounded batches.

        Empty strings are embedded as zero vectors locally instead of being sent
        to the API, which rejects them.
        """
        if not texts:
            return []

        # Map non-empty inputs to their original positions.
        payload: list[str] = []
        positions: list[int] = []
        for index, text in enumerate(texts):
            if text and text.strip():
                payload.append(text)
                positions.append(index)

        vectors: list[list[float]] = []
        for start in range(0, len(payload), self._batch_size):
            batch = payload[start : start + self._batch_size]
            vectors.extend(self._call_with_retry(batch, is_query=False))
            logger.debug(
                "Embedded %d/%d documents", min(start + len(batch), len(payload)), len(payload)
            )

        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])

        width = self._dimension or (len(vectors[0]) if vectors else 768)
        results: list[list[float]] = [[0.0] * width for _ in texts]
        for position, vector in zip(positions, vectors):
            results[position] = vector
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed ``text`` as a retrieval query, memoised per process."""
        if not text or not text.strip():
            return [0.0] * (self._dimension or 768)

        key = content_hash(self._model, text)
        with self._lock:
            cached = self._query_cache.get(key)
        if cached is not None:
            return cached

        vector = self._call_with_retry([text], is_query=True)[0]
        if self._dimension is None:
            self._dimension = len(vector)

        with self._lock:
            if len(self._query_cache) >= self._query_cache_size:
                # Cheap FIFO eviction; queries are small and short-lived.
                self._query_cache.pop(next(iter(self._query_cache)), None)
            self._query_cache[key] = vector
        return vector

    def health_check(self) -> None:
        """Embed a trivial string to verify credentials and model availability.

        Raises:
            EmbeddingError: when the key or model is unusable.
        """
        vector = self._call_with_retry(["health check"], is_query=True)[0]
        self._dimension = len(vector)


def build_embeddings(settings: Settings, *, verify: bool = False) -> GeminiEmbeddings:
    """Build a :class:`GeminiEmbeddings`, falling back across known models.

    Args:
        settings: Runtime configuration.
        verify: When ``True``, each candidate model is probed with a real request
            so an unavailable model is detected at startup instead of mid-index.

    Raises:
        EmbeddingError: when no candidate model can be used.
    """
    candidates: list[str] = [settings.embedding_model]
    candidates += [m for m in FALLBACK_EMBEDDING_MODELS if m not in candidates]

    last_error: Exception | None = None
    for model in candidates:
        try:
            embeddings = GeminiEmbeddings(settings, model=model)
            if verify:
                embeddings.health_check()
            if model != settings.embedding_model:
                logger.warning(
                    "Embedding model %r unavailable; using %r instead",
                    settings.embedding_model,
                    model,
                )
            return embeddings
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_error = exc
            _, message = _classify(exc)
            logger.warning("Embedding model %r unusable: %s", model, message)
            # An invalid API key will fail identically for every model.
            if isinstance(exc, EmbeddingError) and "API key" in str(exc):
                raise

    raise EmbeddingError(
        "No usable Google embedding model was found. Set EMBEDDING_MODEL in .env to "
        f"one of: {', '.join(FALLBACK_EMBEDDING_MODELS)}."
    ) from last_error


def embedding_fingerprint(embeddings: GeminiEmbeddings) -> str:
    """Short identifier of the embedding space, stored on the Chroma collection.

    Vectors produced by different models are not comparable, so this is used to
    detect a model change and warn the user before their searches silently
    degrade.
    """
    return f"{embeddings.model_name}|{embeddings.dimension or 'unknown'}"


def preview_texts(texts: Sequence[str], limit: int = 3) -> str:
    """Return a short, log-safe preview of ``texts`` (used for debugging)."""
    sample = [t[:60].replace("\n", " ") for t in list(texts)[:limit]]
    return " | ".join(sample)
