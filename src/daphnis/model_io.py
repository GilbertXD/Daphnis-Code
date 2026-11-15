"""Opt-in observation at Saturnus's public response boundary."""

import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any

from saturnus import APIError, ChatClient, ResponseError

logger = logging.getLogger(__name__)

_PREVIEW_MESSAGE_LIMIT = 1024
_PREVIEW_ROLES = ("system", "developer", "user", "assistant", "tool", "function")
_HIDDEN_CONTENT = "[内容已隐藏]"
_ATTACHMENT_PREVIEW_NOTICE = "部分内容因附件安全策略而隐藏。"
_IMAGE_DATA_URL = re.compile(r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=]+", re.IGNORECASE)


def hide_image_echo(value: Any, payloads: tuple[str, ...]) -> Any:
    """Remove inline images and exact echoes of uploaded data from model output."""
    if isinstance(value, str):
        value = _IMAGE_DATA_URL.sub(_HIDDEN_CONTENT, value)
        for payload in payloads:
            if len(value) >= len(payload):
                value = value.replace(payload, _HIDDEN_CONTENT)
        return value
    if isinstance(value, dict):
        return {hide_image_echo(key, payloads): hide_image_echo(item, payloads) for key, item in value.items()}
    if isinstance(value, list):
        return [hide_image_echo(item, payloads) for item in value]
    return value


def preview_request(body: dict, *, attachment_sensitive: bool) -> dict:
    if not attachment_sensitive:
        return body
    preview: dict[str, Any] = {}
    model = body.get("model")
    if isinstance(model, str) and len(model) <= 256:
        preview["model"] = model
    messages = body.get("messages")
    if isinstance(messages, list):
        preview["message_count"] = len(messages)
        preview["messages"] = []
        for message in messages[-_PREVIEW_MESSAGE_LIMIT:]:
            role = message.get("role") if isinstance(message, dict) else None
            preview["messages"].append({
                "role": role if role in _PREVIEW_ROLES else "unknown",
                "content": _HIDDEN_CONTENT,
            })
    return preview


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


class ObservedChatClient(ChatClient):
    def __init__(self, *, events, provider, **options):
        super().__init__(**options)
        self._events = events
        self._request_event = None

        def identify(event):
            if event.data["provider"] == provider:
                self._request_event = event

        events.on("model.started", identify)

    async def _observe(self, event, response, *, failed=False):
        if event is None:
            return
        fields = {
            "session_id": event.session_id,
            "turn_id": event.turn_id,
            "request_number": event.data["request_number"],
        }
        if "purpose" in event.data:
            fields["purpose"] = event.data["purpose"]
        sensitive = bool(event.data.get("attachment_sensitive"))
        try:
            body = json.loads(response.request.content)
            if sensitive and not isinstance(body, dict):
                return
            if sensitive:
                fields["omitted"] = _ATTACHMENT_PREVIEW_NOTICE
            await self._events.emit(
                "model.request",
                **fields,
                http_status=response.status_code,
                body=preview_request(body, attachment_sensitive=sensitive),
            )
            if failed and not sensitive:
                try:
                    body = response.json()
                except (ValueError, RuntimeError):
                    body = None
                await self._events.emit(
                    "model.response_error",
                    **fields,
                    http_status=response.status_code,
                    body=body,
                )
        except (ValueError, RuntimeError):
            logger.debug("Model HTTP body unavailable for observation")

    async def create_with_response(self, **options):
        event = self._request_event
        try:
            result = await super().create_with_response(**options)
        except (APIError, ResponseError) as exc:
            await self._observe(event, exc.response, failed=True)
            raise
        await self._observe(event, result.response)
        return result

    @asynccontextmanager
    async def stream_with_response(self, **options):
        event = self._request_event
        observed = False
        try:
            async with super().stream_with_response(**options) as result:
                await self._observe(event, result.response)
                observed = True
                yield result
        except (APIError, ResponseError) as exc:
            if not observed:
                await self._observe(event, exc.response, failed=True)
            raise
