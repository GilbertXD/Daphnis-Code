"""HTTP client for Chat Completions requests."""

import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from math import isfinite
from types import TracebackType
from typing import Any, Self

import httpx

from .._json import decode_json, validate_json
from ..errors import APIError, ResponseError, TransportError, ValidationError
from ..messages import validate_message
from ..models.compatibility import Compatibility
from .streaming import iter_chunks


class ChatClient:
    """An asynchronous client for Chat Completions APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint_path: str = "chat/completions",
        api_key: str | None = None,
        auth_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        compatibility: Compatibility | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        _validate_client_config(
            base_url=base_url,
            endpoint_path=endpoint_path,
            api_key=api_key,
            timeout=timeout,
        )

        if api_key is not None and auth_headers is not None:
            raise ValidationError(
                "api_key and auth_headers cannot be provided together."
            )

        headers = _validate_headers(
            auth_headers,
            allow_authorization=True,
        )
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        if compatibility is not None and not isinstance(compatibility, Compatibility):
            raise ValidationError(
                "compatibility must be a Compatibility instance or None."
            )

        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise ValidationError(
                "transport must be an AsyncBaseTransport instance or None."
            )

        url = httpx.URL(base_url)
        endpoint = httpx.URL(endpoint_path)
        path, separator, query = url.raw_path.partition(b"?")
        self._endpoint_url = url.copy_with(
            raw_path=(path.rstrip(b"/") + b"/" + endpoint.raw_path + separator + query)
        )

        self._compatibility = compatibility
        self._auth_header_names = {name.lower() for name in headers}
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        await self._http.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._http.__aexit__(exc_type, exc, traceback)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def _prepare_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        stream: bool = False,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        _validate_request(model=model, messages=messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if stream:
            body["stream"] = True

        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise ValidationError("extra_body must be an object.")

            conflicts = {"model", "messages", "stream"}.intersection(extra_body)
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValidationError(
                    f"extra_body conflicts with reserved fields: {names}."
                )
            body.update(extra_body)

        if self._compatibility is not None:
            body = self._compatibility.apply(body)

        validate_json(body)
        headers = _validate_headers(
            extra_headers,
            protected_names=self._auth_header_names,
        )
        return body, headers

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send one non-streaming Chat Completions request."""

        body, headers = self._prepare_request(
            model=model,
            messages=messages,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )

        try:
            response = await self._http.post(
                self._endpoint_url,
                json=body,
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise TransportError("The HTTP request could not be completed.") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise APIError(
                f"The server returned HTTP {response.status_code}.",
                response=response,
            ) from exc

        try:
            data = decode_json(response.content)
        except ValueError as exc:
            raise ResponseError(
                "The server returned invalid JSON.",
                response=response,
            ) from exc

        if not isinstance(data, dict):
            raise ResponseError(
                "Expected a JSON object response.",
                response=response,
            )

        return data

    @asynccontextmanager
    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[AsyncIterator[dict[str, Any]], None]:
        """Open a streaming request and yield its JSON chunks."""
        body, headers = self._prepare_request(
            model=model,
            messages=messages,
            extra_body=extra_body,
            extra_headers=extra_headers,
            stream=True,
        )

        request = self._http.build_request(
            "POST",
            self._endpoint_url,
            json=body,
            headers=headers,
        )

        try:
            response = await self._http.send(request, stream=True)
        except httpx.RequestError as exc:
            raise TransportError("The HTTP request could not be completed.") from exc

        try:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                try:
                    await response.aread()
                except httpx.RequestError as read_exc:
                    raise TransportError(
                        "The server error response could not be read."
                    ) from read_exc

                raise APIError(
                    f"The server returned HTTP {response.status_code}.",
                    response=response,
                ) from exc

            content_type = response.headers.get("content-type", "")
            media_type = content_type.partition(";")[0].strip().lower()
            if media_type != "text/event-stream":
                try:
                    await response.aread()
                except httpx.RequestError as exc:
                    raise TransportError("The response could not be read.") from exc

                raise ResponseError(
                    "Expected a text/event-stream response.",
                    response=response,
                )

            chunks = iter_chunks(response)
            try:
                yield chunks
            finally:
                await chunks.aclose()
        finally:
            await response.aclose()


def _validate_client_config(
    *,
    base_url: str,
    endpoint_path: str,
    api_key: str | None,
    timeout: float,
) -> None:
    """Validate connection settings before creating the HTTP client."""
    if not isinstance(base_url, str) or not base_url:
        raise ValidationError("base_url must be a non-empty string.")

    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in base_url):
        raise ValidationError("base_url contains whitespace or control characters.")

    try:
        url = httpx.URL(base_url)
    except (httpx.InvalidURL, UnicodeError) as exc:
        raise ValidationError("base_url must be a valid URL.") from exc

    if url.scheme not in {"http", "https"} or not url.host:
        raise ValidationError("base_url must be an absolute HTTP or HTTPS URL.")

    if "#" in base_url:
        raise ValidationError("base_url must not contain a fragment.")

    if url.userinfo:
        raise ValidationError("base_url must not contain authentication credentials.")

    if not isinstance(endpoint_path, str) or not endpoint_path:
        raise ValidationError("endpoint_path must be a non-empty string.")

    if endpoint_path.startswith("/") or "?" in endpoint_path or "#" in endpoint_path:
        raise ValidationError(
            "endpoint_path must be relative and must not contain a query or fragment."
        )

    if any(
        char.isspace() or ord(char) < 32 or ord(char) == 127 for char in endpoint_path
    ):
        raise ValidationError(
            "endpoint_path contains whitespace or control characters."
        )

    if any(segment in {".", ".."} for segment in endpoint_path.split("/")):
        raise ValidationError("endpoint_path must not contain dot segments.")

    try:
        endpoint = httpx.URL(endpoint_path)
    except (httpx.InvalidURL, UnicodeError) as exc:
        raise ValidationError("endpoint_path must be a valid relative path.") from exc

    if endpoint.scheme or endpoint.host:
        raise ValidationError("endpoint_path must be a relative path.")

    if api_key is not None and (
        not isinstance(api_key, str)
        or not api_key
        or any(ord(char) < 33 or ord(char) > 126 for char in api_key)
    ):
        raise ValidationError(
            "api_key must be a non-empty ASCII string without whitespace."
        )

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValidationError("timeout must be a finite positive number.")

    try:
        valid_timeout = isfinite(timeout) and timeout > 0
    except OverflowError as exc:
        raise ValidationError("timeout must be a finite positive number.") from exc

    if not valid_timeout:
        raise ValidationError("timeout must be a finite positive number.")


def _validate_request(
    *,
    model: str,
    messages: list[dict[str, Any]],
) -> None:
    if not isinstance(model, str) or not model.strip():
        raise ValidationError("model must be a non-empty string.")

    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a non-empty list.")

    for index, message in enumerate(messages):
        validate_message(message, path=f"messages[{index}]")


def _validate_headers(
    headers: dict[str, str] | None,
    *,
    allow_authorization: bool = False,
    protected_names: set[str] | None = None,
) -> dict[str, str]:
    """Validate request headers and return a separate dictionary."""
    if headers is None:
        return {}

    if not isinstance(headers, dict):
        raise ValidationError("Headers must be an object.")

    reserved = {
        "content-type",
        "content-length",
        "transfer-encoding",
        "host",
    }
    if not allow_authorization:
        reserved.add("authorization")
    if protected_names:
        reserved.update(name.lower() for name in protected_names)
    seen = set()

    for name, value in headers.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[-!#$%&'*+.^_`|~0-9A-Za-z]+", name) is None
        ):
            raise ValidationError("Header names must be valid ASCII tokens.")

        if not isinstance(value, str):
            raise ValidationError(f"Header {name!r} must have a string value.")

        normalized = name.lower()
        if normalized in reserved:
            raise ValidationError(f"Header {name!r} is controlled by ChatClient.")

        if normalized in seen:
            raise ValidationError(f"Duplicate header name: {name!r}.")
        seen.add(normalized)

        if value != value.strip(" \t") or any(
            (ord(char) < 32 and char != "\t") or ord(char) > 126 for char in value
        ):
            raise ValidationError(
                f"Header {name!r} contains invalid characters or whitespace."
            )

    return dict(headers)
