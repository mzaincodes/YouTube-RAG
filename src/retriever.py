"""Retrieval pipeline: similarity search, MMR diversification and compression.

``langchain_chroma`` exposes *either* relevance scores (``similarity_search_with_
relevance_scores``) *or* MMR diversification (``max_marginal_relevance_search``),
never both — MMR discards the scores. Citations in the UI need the scores, so
MMR is reimplemented here on top of a scored similarity fetch: candidate vectors
are read back from Chroma with ``get(include=["embeddings"])`` (a local read, no
API call) and the maximal-marginal-relevance selection runs in NumPy.

An optional lexical compression pass then trims low-signal sentences out of the
retrieved chunks when the assembled context would overflow the prompt budget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from langchain_core.documents import Document

from .config import Settings
from .embeddings import GeminiEmbeddings
from .utils import content_hash
from .vector_store import VectorStoreManager, build_metadata_filter

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")

#: Very common words carry no retrieval signal during lexical compression.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have he her him his
    how i if in into is it its me my no not of on or our out she so than that the their them then
    there these they this to was we were what when where which who why will with would you your
    """.split()
)


@dataclass
class RetrievedChunk:
    """A retrieved document together with its relevance score and rank."""

    document: Document
    score: float
    rank: int

    @property
    def metadata(self) -> dict[str, Any]:
        """Shortcut to the underlying document metadata."""
        return self.document.metadata or {}

    @property
    def video_id(self) -> str:
        """Source video id."""
        return str(self.metadata.get("video_id", "unknown"))

    @property
    def title(self) -> str:
        """Source video title."""
        return str(self.metadata.get("title", "Untitled video"))


class RetrievalError(RuntimeError):
    """Retrieval failed; ``str(exc)`` is safe to show to the user."""


def _tokenize(text: str) -> set[str]:
    """Lower-case content words of ``text``."""
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS}


def _mmr_select(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Maximal Marginal Relevance selection over candidate row vectors.

    Balances similarity to the query against novelty relative to what has
    already been picked, which stops the top-k from being five near-identical
    chunks of the same passage.

    Returns:
        Indices into ``candidate_vectors``, in selection order.
    """
    if candidate_vectors.size == 0:
        return []

    norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = candidate_vectors / norms

    query_norm = np.linalg.norm(query_vector) or 1.0
    query_unit = query_vector / query_norm

    similarity_to_query = normalised @ query_unit
    selected: list[int] = [int(np.argmax(similarity_to_query))]

    while len(selected) < min(k, len(candidate_vectors)):
        selected_matrix = normalised[selected]
        # Similarity of every candidate to its closest already-selected item.
        redundancy = (normalised @ selected_matrix.T).max(axis=1)
        scores = lambda_mult * similarity_to_query - (1.0 - lambda_mult) * redundancy
        scores[selected] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        selected.append(best)

    return selected


def compress_document(document: Document, query: str, *, max_chars: int) -> Document:
    """Trim ``document`` to the sentences most lexically relevant to ``query``.

    A cheap, deterministic alternative to an LLM-based compressor: it costs no
    tokens and adds no latency, which matters because compression runs on every
    retrieved chunk of every turn. Sentence order is preserved so the excerpt
    still reads naturally.
    """
    text = document.page_content
    if len(text) <= max_chars:
        return document

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return Document(
            page_content=text[:max_chars].rsplit(" ", 1)[0] + " …",
            metadata=dict(document.metadata or {}),
            id=document.id,
        )

    query_terms = _tokenize(query)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        terms = _tokenize(sentence)
        overlap = len(terms & query_terms) / (len(query_terms) or 1)
        # A mild length prior keeps single-word fragments from winning.
        scored.append((overlap + min(len(terms), 25) / 500.0, index, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[int, str]] = []
    budget = 0
    for score, index, sentence in scored:
        if budget + len(sentence) > max_chars and kept:
            continue
        kept.append((index, sentence))
        budget += len(sentence) + 1

    kept.sort(key=lambda item: item[0])
    compressed = " ".join(sentence for _, sentence in kept)

    metadata = dict(document.metadata or {})
    metadata["compressed"] = True
    return Document(page_content=compressed or text[:max_chars], metadata=metadata, id=document.id)


class RagRetriever:
    """Query-time retrieval over the persistent Chroma collection."""

    def __init__(
        self,
        store_manager: VectorStoreManager,
        embeddings: GeminiEmbeddings,
        settings: Settings,
    ) -> None:
        self._manager = store_manager
        self._embeddings = embeddings
        self._settings = settings

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        search_type: str | None = None,
        video_ids: Sequence[str] | None = None,
        score_threshold: float | None = None,
        compress: bool = True,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for ``query``.

        Args:
            query: The (already history-resolved) search query.
            k: Number of chunks to return. Defaults to ``RETRIEVAL_K``.
            search_type: ``"mmr"`` or ``"similarity"``. Defaults to ``SEARCH_TYPE``.
            video_ids: Restrict the search to these videos.
            score_threshold: Drop chunks scoring below this (0-1).
            compress: Trim long chunks toward the per-document budget.

        Raises:
            RetrievalError: when the vector store cannot be queried.
        """
        query = (query or "").strip()
        if not query:
            return []

        settings = self._settings
        k = k or settings.retrieval_k
        search_type = (search_type or settings.search_type).lower()
        threshold = settings.score_threshold if score_threshold is None else score_threshold
        metadata_filter = build_metadata_filter(video_ids)

        # Over-fetch so MMR (and threshold filtering) have room to work.
        fetch_k = max(k, settings.retrieval_fetch_k) if search_type == "mmr" else k

        try:
            scored = self._manager.store.similarity_search_with_relevance_scores(
                query, k=fetch_k, filter=metadata_filter
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                f"Could not search the vector database: {exc}. "
                "If the problem persists, clear the database and re-index."
            ) from exc

        if not scored:
            return []

        documents = [doc for doc, _ in scored]
        scores = [float(score) for _, score in scored]

        if search_type == "mmr" and len(documents) > k:
            order = self._mmr_order(query, documents, k)
            documents = [documents[i] for i in order]
            scores = [scores[i] for i in order]
        else:
            documents, scores = documents[:k], scores[:k]

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        per_document_budget = max(600, settings.max_context_chars // max(1, k))

        for document, score in zip(documents, scores):
            if threshold > 0 and score < threshold:
                continue
            # Overlapping chunks can surface near-duplicate text; keep the best.
            fingerprint = content_hash(document.page_content[:400].lower())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            if compress:
                document = compress_document(document, query, max_chars=per_document_budget)

            results.append(RetrievedChunk(document=document, score=score, rank=len(results) + 1))

        logger.info(
            "Retrieved %d/%d chunks for %r (%s, k=%d)",
            len(results),
            len(scored),
            query[:60],
            search_type,
            k,
        )
        return results

    def _mmr_order(self, query: str, documents: Sequence[Document], k: int) -> list[int]:
        """Return indices of ``documents`` reordered by MMR.

        Falls back to the original similarity order if the candidate embeddings
        cannot be read back from Chroma.
        """
        ids = [doc.id for doc in documents if doc.id]
        if len(ids) != len(documents):
            return list(range(min(k, len(documents))))

        try:
            payload = self._manager.store.get(ids=list(ids), include=["embeddings"])
            raw_vectors = payload.get("embeddings")
            returned_ids = payload.get("ids") or []
            if raw_vectors is None or len(returned_ids) != len(documents):
                return list(range(min(k, len(documents))))

            # ``get`` does not preserve the query ordering; realign by id.
            position = {doc_id: i for i, doc_id in enumerate(returned_ids)}
            matrix = np.asarray(raw_vectors, dtype=np.float32)
            ordered = np.stack([matrix[position[doc_id]] for doc_id in ids])
            query_vector = np.asarray(self._embeddings.embed_query(query), dtype=np.float32)
            if query_vector.shape[0] != ordered.shape[1]:
                return list(range(min(k, len(documents))))
        except Exception as exc:  # noqa: BLE001 - diversification is best-effort
            logger.debug("MMR reordering unavailable (%s); using similarity order", exc)
            return list(range(min(k, len(documents))))

        return _mmr_select(query_vector, ordered, k, self._settings.mmr_lambda)
