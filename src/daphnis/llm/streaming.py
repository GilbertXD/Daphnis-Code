"""Decode server-sent events from a model HTTP response."""

from collections.abc import AsyncGenerator
from typing import Any

import httpx

from .._json import decode_json
from ..errors import ResponseError, TransportError


async def _iter_sse_lines(
    response: httpx.Response,
) -> AsyncGenerator[str, None]:
    """Split lines using only SSE's CR, LF, and CRLF delimiters."""
    line: list[str] = []
    skip_lf = False

    async for text in response.aiter_text():
        for char in text:
            if skip_lf:
                skip_lf = False
                if char == "\n":
                    continue

            if char in "\r\n":
                completed = "".join(line)
                line.clear()
                skip_lf = char == "\r"
                yield completed
            else:
                line.append(char)

    if line:
        yield "".join(line)


async def iter_chunks(
    response: httpx.Response,
) -> AsyncGenerator[dict[str, Any], None]:
    """Read JSON data events from a Chat Completions stream."""
    data_lines: list[str] = []
    first_line = True
    response.encoding = "utf-8"
    lines = _iter_sse_lines(response)

    try:
        async for line in lines:
            if first_line:
                line = line.removeprefix("\ufeff")
                first_line = False

            if line == "":
                if not data_lines:
                    continue

                event_data = "\n".join(data_lines)
                data_lines.clear()

                if event_data == "[DONE]":
                    return

                try:
                    chunk = decode_json(event_data)
                except ValueError as exc:
                    raise ResponseError(
                        "The stream contained invalid JSON.",
                        response=response,
                        event_data=event_data,
                    ) from exc

                if not isinstance(chunk, dict):
                    raise ResponseError(
                        "Expected a JSON object in the stream.",
                        response=response,
                        event_data=event_data,
                    )

                yield chunk
                continue

            if line.startswith(":"):
                continue

            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]

            if field == "data":
                data_lines.append(value)

    except httpx.RequestError as exc:
        raise TransportError("The streaming response could not be read.") from exc
    finally:
        await lines.aclose()

    raise ResponseError(
        "The stream ended before the [DONE] marker.",
        response=response,
        event_data="\n".join(data_lines) if data_lines else None,
    )
