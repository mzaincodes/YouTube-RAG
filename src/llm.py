"""Gemini chat model construction and error translation.

Two production concerns are handled here that a bare ``ChatGoogleGenerativeAI(...)``
call does not address:

**Safety thresholds.** Gemini's defaults block at MEDIUM, and ordinary video
transcripts (news, true crime, medicine, politics) trip those filters often
enough to break a RAG answer for no good reason. The categories are raised to
``BLOCK_ONLY_HIGH`` so genuinely harmful output is still refused while ordinary
content is not.

**Error translation.** Raw ``google.genai`` exceptions leak stack traces and
request ids into the chat window. :func:`friendly_llm_error` maps them onto
sentences a user can act on.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from .config import FALLBACK_CHAT_MODELS, Settings

logger = logging.getLogger(__name__)

#: Model suggested in error messages — an alias, so the advice cannot go stale.
_RECOMMENDED_MODEL = "gemini-flash-latest"

#: Substrings marking a model that this API key can never use, however often we
#: retry. Anything matching these is skipped during model resolution.
_MODEL_UNUSABLE_MARKERS = (
    "no longer available",
    "not found",
    "404",
    "permission",
    "403",
    # A model whose free-tier quota is exhausted is unusable *right now*; during
    # resolution we move on rather than making the user wait it out.
    "429",
    "quota",
    "resource_exhausted",
    "rate limit",
)


class LLMError(RuntimeError):
    """LLM invocation failed; ``str(exc)`` is safe to show to the user."""


def _default_safety_settings() -> dict[Any, Any]:
    """Relax Gemini's content filters to ``BLOCK_ONLY_HIGH``.

    Built defensively: if the enum members ever change, the app runs with
    Gemini's defaults instead of failing to start.
    """
    try:
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    except AttributeError as exc:  # pragma: no cover - defensive
        logger.debug("Safety settings unavailable (%s); using API defaults", exc)
        return {}


def build_llm(
    settings: Settings,
    *,
    streaming: bool = True,
    model: str | None = None,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Construct the Gemini chat model.

    Args:
        settings: Runtime configuration.
        streaming: Enable token streaming (the chat UI needs this).
        model: Override ``MODEL_NAME``.
        temperature: Override ``TEMPERATURE``.

    Raises:
        LLMError: when the model cannot be constructed.
    """
    api_key = settings.require_api_key()
    try:
        return ChatGoogleGenerativeAI(
            model=model or settings.model_name,
            google_api_key=api_key,
            temperature=settings.temperature if temperature is None else temperature,
            max_output_tokens=settings.max_output_tokens,
            max_retries=settings.max_retries,
            streaming=streaming,
            safety_settings=_default_safety_settings(),
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMError(friendly_llm_error(exc)) from exc


def friendly_llm_error(exc: Exception) -> str:
    """Translate an LLM exception into an actionable message."""
    text = f"{type(exc).__name__}: {exc}".lower()

    if "api key not valid" in text or "api_key_invalid" in text or "unauthenticated" in text:
        return (
            "Your Google API key was rejected. Check GOOGLE_API_KEY in .env — "
            "generate a new key at https://aistudio.google.com/apikey"
        )
    # Google keeps retired models in the models-list endpoint but rejects them at
    # call time for keys created after the deprecation date, so this is NOT the
    # same as a typo'd model name and needs its own message.
    if "no longer available" in text:
        return (
            f"This Gemini model has been retired for newly created API keys. "
            f"Set MODEL_NAME in .env to '{_RECOMMENDED_MODEL}' (an alias that always "
            "tracks the current Flash model), then restart the app."
        )
    if "permission" in text or "403" in text:
        return (
            f"Your API key does not have access to this model. Try MODEL_NAME={_RECOMMENDED_MODEL} "
            "in .env, or enable the Generative Language API for your project."
        )
    if "not found" in text or "404" in text:
        return (
            "The configured Gemini model was not found. Set MODEL_NAME in .env to a "
            f"supported model such as {_RECOMMENDED_MODEL} or gemini-3.5-flash."
        )
    if "429" in text or "quota" in text or "rate limit" in text or "resource_exhausted" in text:
        return (
            "You have hit Gemini's rate limit or free-tier quota. Pro models have very "
            f"little free quota — switch MODEL_NAME to {_RECOMMENDED_MODEL} in .env, or "
            "wait a minute and try again."
        )
    if "safety" in text or "blocked" in text:
        return (
            "Gemini blocked this response under its safety filters. Try rephrasing your "
            "question."
        )
    if any(marker in text for marker in ("timeout", "timed out", "deadline")):
        return "The request to Gemini timed out. Please try again."
    if any(marker in text for marker in ("connection", "network", "unavailable", "503")):
        return "Could not reach Gemini. Check your internet connection and try again."
    if "recitation" in text:
        return (
            "Gemini stopped generating because the answer too closely reproduced its "
            "training data. Try rephrasing your question."
        )
    return f"The language model returned an error: {exc}"


def invoke_llm(llm: BaseChatModel, messages: list[BaseMessage]) -> str:
    """Invoke ``llm`` and return plain text, translating any failure."""
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        raise LLMError(friendly_llm_error(exc)) from exc
    return _message_text(response)


def stream_llm(llm: BaseChatModel, messages: list[BaseMessage]) -> Iterator[str]:
    """Yield text deltas from ``llm``, translating any failure."""
    try:
        for chunk in llm.stream(messages):
            text = _message_text(chunk)
            if text:
                yield text
    except Exception as exc:  # noqa: BLE001
        raise LLMError(friendly_llm_error(exc)) from exc


def _message_text(message: Any) -> str:
    """Extract plain text from a message whose content may be a content-block list.

    LangChain 1.x models can return ``content`` as a list of typed blocks
    (text, thinking, tool calls); only the text blocks belong in the answer.
    """
    if isinstance(message, (AIMessage, BaseMessage)):
        content = message.content
    else:
        content = message

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Skip reasoning blocks — they are not part of the answer.
                if block.get("type") in {"thinking", "reasoning"}:
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def resolve_chat_model(settings: Settings) -> tuple[str, str | None]:
    """Find a chat model this API key can actually call.

    Google lists retired models in the models endpoint but rejects them at call
    time for keys created after the deprecation date, so availability cannot be
    determined from configuration alone — it has to be probed.

    The configured model is tried first, so the normal case costs exactly one
    tiny request. Only when that fails do the fallbacks get probed.

    Returns:
        ``(model_name, warning)`` — ``warning`` is non-``None`` when the
        configured model had to be replaced, and is safe to show in the UI.
    """
    configured = settings.model_name
    candidates = [configured] + [m for m in FALLBACK_CHAT_MODELS if m != configured]
    first_error: str | None = None

    for model in candidates:
        try:
            probe = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.require_api_key(),
                max_output_tokens=16,
                temperature=0.0,
            )
            probe.invoke("ok")
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            message = f"{type(exc).__name__}: {exc}".lower()
            if first_error is None:
                first_error = friendly_llm_error(exc)
            if any(marker in message for marker in _MODEL_UNUSABLE_MARKERS):
                logger.warning("Chat model %r unusable: %s", model, str(exc)[:160])
                continue
            # A network blip says nothing about the model; keep the choice.
            logger.warning("Probe for %r failed transiently (%s); keeping it", model, type(exc).__name__)
            return model, None

        if model != configured:
            warning = (
                f"'{configured}' is not available to your API key, so the app switched to "
                f"'{model}'. Set MODEL_NAME={model} in .env to silence this notice."
            )
            logger.warning(warning)
            return model, warning
        return model, None

    # Nothing worked — keep the configured model so the real error surfaces.
    logger.error("No usable chat model found; keeping %r", configured)
    return configured, first_error


def health_check(settings: Settings) -> tuple[bool, str]:
    """Verify the API key and model with one tiny request.

    Returns:
        ``(ok, message)`` — ``message`` is UI-safe when ``ok`` is ``False``.
    """
    try:
        llm = build_llm(settings, streaming=False)
        response = llm.invoke("Reply with the single word: ok")
        return True, _message_text(response).strip() or "ok"
    except LLMError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, friendly_llm_error(exc)
