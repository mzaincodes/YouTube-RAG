"""Centralised, environment-driven configuration for the YouTube RAG application.

All tunables live here so that behaviour can be changed through ``.env`` without
touching code. :func:`get_settings` is cached, so the ``.env`` file is parsed
exactly once per process even though Streamlit re-executes ``app.py`` on every
interaction.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

# Project root = the directory that contains ``src/``.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Embedding models known to work with the Google Generative AI API, in the
#: order they are tried when the configured model is unavailable.
FALLBACK_EMBEDDING_MODELS: tuple[str, ...] = (
    "models/gemini-embedding-001",
    "models/text-embedding-004",
    "models/embedding-001",
)

#: Chat models offered in the UI settings panel.
#:
#: ``*-latest`` aliases are listed first because Google retires concrete model
#: names: ``gemini-2.5-flash`` now returns 404 "no longer available to new
#: users" for API keys created after its deprecation, even though it is still
#: listed by the models endpoint. The aliases track whatever is current.
AVAILABLE_CHAT_MODELS: tuple[str, ...] = (
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-pro-latest",
)

#: Order in which chat models are tried when the configured one is unavailable.
#: Ends with the older 2.x names so keys that still have access keep working.
FALLBACK_CHAT_MODELS: tuple[str, ...] = (
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)


class ConfigurationError(RuntimeError):
    """Raised when the application is mis-configured and cannot safely start."""


def _load_env_once() -> None:
    """Load ``.env`` from the project root (or anywhere up the tree)."""
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)
    else:  # pragma: no cover - convenience for unusual layouts
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)


def _str_env(key: str, default: str) -> str:
    """Return a stripped string env var, falling back to ``default`` when blank."""
    raw = os.getenv(key)
    if raw is None:
        return default
    raw = raw.strip().strip('"').strip("'")
    return raw or default


def _int_env(key: str, default: int, *, minimum: int | None = None) -> int:
    """Return an int env var, tolerating malformed values by using ``default``."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(float(raw.strip()))
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", key, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s below minimum %s; clamping", key, value, minimum)
        return minimum
    return value


def _float_env(key: str, default: float) -> float:
    """Return a float env var, tolerating malformed values by using ``default``."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", key, raw, default)
        return default


def _bool_env(key: str, default: bool) -> bool:
    """Return a boolean env var accepting ``1/true/yes/on`` (case-insensitive)."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _list_env(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Return a comma-separated env var as a tuple of stripped strings."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of every runtime knob the application exposes."""

    # --- Credentials -----------------------------------------------------
    google_api_key: str = ""

    # --- Models ----------------------------------------------------------
    model_name: str = "gemini-flash-latest"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.2
    max_output_tokens: int = 4096

    # --- Vector store ----------------------------------------------------
    chroma_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "chroma_db")
    chroma_collection: str = "youtube_rag"

    # --- Chunking --------------------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 200
    semantic_chunking: bool = True
    semantic_breakpoint_percentile: float = 90.0
    semantic_min_chunk_chars: int = 400
    semantic_max_sentences_probe: int = 1500

    # --- Embeddings ------------------------------------------------------
    embed_batch_size: int = 64
    max_retries: int = 5
    retry_base_delay: float = 1.5

    # --- Retrieval -------------------------------------------------------
    retrieval_k: int = 6
    retrieval_fetch_k: int = 24
    search_type: str = "mmr"  # "similarity" | "mmr"
    mmr_lambda: float = 0.55
    score_threshold: float = 0.0

    # --- Conversation ----------------------------------------------------
    history_turns: int = 6
    max_context_chars: int = 24_000

    # --- Transcripts -----------------------------------------------------
    transcript_languages: tuple[str, ...] = ("en", "en-US", "en-GB")
    transcript_max_workers: int = 4

    # --- Misc ------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    @property
    def has_api_key(self) -> bool:
        """True when a non-placeholder Google API key is present."""
        key = self.google_api_key.strip()
        return bool(key) and not key.lower().startswith("your_")

    @property
    def persist_directory(self) -> str:
        """Chroma persistence directory as a string, created on demand."""
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        return str(self.chroma_db_path)

    def require_api_key(self) -> str:
        """Return the API key or raise a user-friendly :class:`ConfigurationError`."""
        if not self.has_api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY is not set. Create a .env file in the project root "
                "with GOOGLE_API_KEY=<your key> (see .env.example), then restart "
                "the app. Get a free key at https://aistudio.google.com/apikey"
            )
        return self.google_api_key.strip()

    def replace(self, **overrides: object) -> "Settings":
        """Return a copy with ``overrides`` applied (used by the UI settings panel)."""
        from dataclasses import replace as _replace

        return _replace(self, **overrides)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the :class:`Settings` instance from the environment."""
    _load_env_once()

    raw_path = _str_env("CHROMA_DB_PATH", "./chroma_db")
    chroma_path = Path(raw_path).expanduser()
    if not chroma_path.is_absolute():
        chroma_path = (PROJECT_ROOT / chroma_path).resolve()

    settings = Settings(
        google_api_key=_str_env("GOOGLE_API_KEY", ""),
        model_name=_str_env("MODEL_NAME", "gemini-flash-latest"),
        embedding_model=_str_env("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        temperature=_float_env("TEMPERATURE", 0.2),
        max_output_tokens=_int_env("MAX_OUTPUT_TOKENS", 4096, minimum=256),
        chroma_db_path=chroma_path,
        chroma_collection=_str_env("CHROMA_COLLECTION", "youtube_rag"),
        chunk_size=_int_env("CHUNK_SIZE", 1200, minimum=200),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 200, minimum=0),
        semantic_chunking=_bool_env("SEMANTIC_CHUNKING", True),
        semantic_breakpoint_percentile=_float_env("SEMANTIC_BREAKPOINT_PERCENTILE", 90.0),
        semantic_min_chunk_chars=_int_env("SEMANTIC_MIN_CHUNK_CHARS", 400, minimum=100),
        semantic_max_sentences_probe=_int_env("SEMANTIC_MAX_SENTENCES_PROBE", 1500, minimum=50),
        embed_batch_size=_int_env("EMBED_BATCH_SIZE", 64, minimum=1),
        max_retries=_int_env("MAX_RETRIES", 5, minimum=1),
        retry_base_delay=_float_env("RETRY_BASE_DELAY", 1.5),
        retrieval_k=_int_env("RETRIEVAL_K", 6, minimum=1),
        retrieval_fetch_k=_int_env("RETRIEVAL_FETCH_K", 24, minimum=1),
        search_type=_str_env("SEARCH_TYPE", "mmr").lower(),
        mmr_lambda=_float_env("MMR_LAMBDA", 0.55),
        score_threshold=_float_env("SCORE_THRESHOLD", 0.0),
        history_turns=_int_env("HISTORY_TURNS", 6, minimum=0),
        max_context_chars=_int_env("MAX_CONTEXT_CHARS", 24_000, minimum=2_000),
        transcript_languages=_list_env("TRANSCRIPT_LANGUAGES", ("en", "en-US", "en-GB")),
        transcript_max_workers=_int_env("TRANSCRIPT_MAX_WORKERS", 4, minimum=1),
        log_level=_str_env("LOG_LEVEL", "INFO").upper(),
    )

    # ``chunk_overlap`` must stay strictly below ``chunk_size`` or the splitters
    # loop forever; guard rather than trusting the operator.
    if settings.chunk_overlap >= settings.chunk_size:
        safe_overlap = max(0, settings.chunk_size // 5)
        logger.warning(
            "CHUNK_OVERLAP (%s) >= CHUNK_SIZE (%s); reducing overlap to %s",
            settings.chunk_overlap,
            settings.chunk_size,
            safe_overlap,
        )
        settings = settings.replace(chunk_overlap=safe_overlap)

    if settings.search_type not in {"similarity", "mmr"}:
        logger.warning("Unknown SEARCH_TYPE=%r; falling back to 'mmr'", settings.search_type)
        settings = settings.replace(search_type="mmr")

    if settings.retrieval_fetch_k < settings.retrieval_k:
        settings = settings.replace(retrieval_fetch_k=settings.retrieval_k * 4)

    # Google's SDK reads GOOGLE_API_KEY from the environment in several code
    # paths; make sure both spellings are populated for consistency.
    if settings.has_api_key:
        os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)

    return settings


def reset_settings_cache() -> None:
    """Clear the cached settings (used after the user edits ``.env`` at runtime)."""
    get_settings.cache_clear()
