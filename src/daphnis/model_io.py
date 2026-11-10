"""Opt-in observation at Saturnus's public response boundary."""

import json
import logging
from contextlib import asynccontextmanager

from saturnus import APIError, ChatClient, ResponseError

logger = logging.getLogger(__name__)


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
        try:
            await self._events.emit(
                "model.request",
                **fields,
                http_status=response.status_code,
                body=json.loads(response.request.content),
            )
            if failed:
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
