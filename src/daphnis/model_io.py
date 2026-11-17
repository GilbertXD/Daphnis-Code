"""Helpers for context overflow and attachment privacy."""

import re
from typing import Any

from .errors import APIError, ResponseError

_HIDDEN_CONTENT = "[内容已隐藏]"
_IMAGE_DATA_URL = re.compile(r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=]+", re.IGNORECASE)


def hide_attachment_echo(value: Any, private_values: tuple[str, ...]) -> Any:
    """Remove inline images and private attachment values from model output."""
    if isinstance(value, str):
        value = _IMAGE_DATA_URL.sub(_HIDDEN_CONTENT, value)
        for payload in private_values:
            if len(value) >= len(payload):
                value = value.replace(payload, _HIDDEN_CONTENT)
        return value
    if isinstance(value, dict):
        return {hide_attachment_echo(key, private_values): hide_attachment_echo(item, private_values)
                for key, item in value.items()}
    if isinstance(value, list):
        return [hide_attachment_echo(item, private_values) for item in value]
    return value


_CONTEXT_CODES = {
    "context_length_exceeded", "context_window_exceeded", "context_size_exceeded",
    "max_context_length_exceeded", "model_context_window_exceeded",
}
_CONTEXT_MESSAGE = re.compile(
    r"maximum context length is \d|context (?:length|window|size) (?:has been )?exceeded"
    r"|(?:exceeds?|exceeded) (?:the )?(?:maximum |model )?context (?:length|window|size)"
    r"|input tokens? exceed(?:s|ed)? (?:the )?limit",
    re.IGNORECASE,
)


class ContextOverflowError(RuntimeError):
    """The provider explicitly rejected the context size."""

    def __init__(self, *, retryable: bool = True) -> None:
        super().__init__("The provider rejected the request because the context window was exceeded.")
        self.retryable = retryable


def is_context_overflow(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    error = payload.get("error", payload)
    if isinstance(error, str):
        return bool(_CONTEXT_MESSAGE.search(error))
    if not isinstance(error, dict):
        return False
    if any(
        isinstance(value := error.get(field), str) and value.lower() in _CONTEXT_CODES
        for field in ("code", "type")
    ):
        return True
    message = error.get("message")
    return isinstance(message, str) and bool(_CONTEXT_MESSAGE.search(message))


def is_http_context_overflow(exc: APIError | ResponseError) -> bool:
    if exc.response.status_code not in (200, 400, 413, 422, 500):
        return False
    try:
        return is_context_overflow(exc.response.json())
    except (ValueError, RuntimeError):
        return False
