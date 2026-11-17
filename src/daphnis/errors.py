"""Errors for local validation, tool execution, and model communication."""

import httpx


class DaphnisError(Exception):
    """Base exception for these Daphnis operations."""


class ValidationError(DaphnisError):
    """The request failed local validation."""


class ToolCallValidationError(ValidationError):
    """A tool call failed validation, with structured context when available."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        argument_path: list[str | int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.argument_path = None if argument_path is None else list(argument_path)


class ToolExecutionError(DaphnisError):
    """A tool handler or its result conversion failed."""

    def __init__(
        self,
        message: str,
        *,
        tool_call_id: str,
        tool_name: str,
    ) -> None:
        super().__init__(message)
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


class CommunicationError(Exception):
    """Base exception for model HTTP and response protocol errors."""


class APIError(CommunicationError):
    """The server returned an unsuccessful HTTP status."""

    def __init__(self, message: str, *, response: httpx.Response) -> None:
        super().__init__(message)
        self.response = response


class ResponseError(CommunicationError):
    """The server response does not satisfy the expected protocol."""

    def __init__(
        self,
        message: str,
        *,
        response: httpx.Response,
        event_data: str | None = None,
    ) -> None:
        super().__init__(message)
        self.response = response
        self.event_data = event_data


class TransportError(CommunicationError):
    """The HTTP request or response read could not be completed."""
