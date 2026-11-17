"""Register, validate, and execute Chat Completions function tools."""

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import isawaitable, iscoroutinefunction
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as SchemaValidationError
from jsonschema.validators import validator_for

from .._json import decode_json, validate_json
from ..errors import ToolCallValidationError, ToolExecutionError, ValidationError


@dataclass(frozen=True)
class _RegisteredTool:
    definition: dict[str, Any]
    handler: Callable[..., Any]


@dataclass(frozen=True)
class ToolResult:
    """The value and message content returned by one tool execution."""

    tool_call_id: str
    value: Any
    content: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Register local handlers and export Chat Completions tool definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Validate and store a tool without executing its handler."""
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Tool name must be a non-empty string.")

        if name in self._tools:
            raise ValidationError(f"Tool {name!r} is already registered.")

        if not isinstance(description, str):
            raise ValidationError("Tool description must be a string.")

        if not callable(handler):
            raise ValidationError("Tool handler must be callable.")

        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValidationError(
                "Tool parameters must be a JSON Schema with type 'object'."
            )

        validate_json(parameters)

        if "$schema" in parameters:
            if not isinstance(parameters["$schema"], str):
                raise ValidationError("$schema must be a string.")

            validator_class = validator_for(parameters, default=None)
            if validator_class is None:
                raise ValidationError("Unsupported JSON Schema dialect.")
        else:
            validator_class = Draft202012Validator

        try:
            validator_class.check_schema(parameters)
        except SchemaError as exc:
            raise ValidationError(
                "Tool parameters must be a valid JSON Schema."
            ) from exc

        definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

        self._tools[name] = _RegisteredTool(
            definition=deepcopy(definition),
            handler=handler,
        )

    def definitions(self) -> list[dict[str, Any]]:
        """Return independent tool definitions without local handlers."""

        return deepcopy([tool.definition for tool in self._tools.values()])

    def select(self, names: list[str]) -> "ToolRegistry":
        """Copy selected tools in registration order, retaining handler references."""
        if not isinstance(names, list):
            raise ValidationError("Selected tool names must be a list.")

        requested: set[str] = set()
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValidationError("Selected tool names must be non-empty strings.")
            if name in requested:
                raise ValidationError(f"Tool {name!r} is selected more than once.")
            if name not in self._tools:
                raise ValidationError(f"Tool {name!r} is not registered.")
            requested.add(name)

        selected = ToolRegistry()
        for name, tool in self._tools.items():
            if name in requested:
                selected._tools[name] = _RegisteredTool(
                    definition=deepcopy(tool.definition),
                    handler=tool.handler,
                )
        return selected

    def validate_call(self, call: dict[str, Any]) -> dict[str, Any]:
        """Validate a function call and return its parsed arguments."""
        if not isinstance(call, dict):
            raise ToolCallValidationError(
                "Tool call must be an object.", code="invalid_call"
            )

        raw_id = call.get("id")
        call_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else None
        function = call.get("function")
        raw_name = function.get("name") if isinstance(function, dict) else None
        name = raw_name if isinstance(raw_name, str) and raw_name.strip() else None

        def call_error(
            message: str,
            code: str = "invalid_call",
            *,
            argument_path: list[str | int] | None = None,
        ) -> ToolCallValidationError:
            return ToolCallValidationError(
                message,
                code=code,
                tool_call_id=call_id,
                tool_name=name,
                argument_path=argument_path,
            )

        try:
            validate_json(call)
        except ValidationError as exc:
            raise call_error("Tool call data must be valid JSON.") from exc

        if call_id is None:
            raise call_error("Tool call id must be a non-empty string.")

        if call.get("type") != "function":
            raise call_error("Tool call type must be 'function'.")

        if not isinstance(function, dict):
            raise call_error("Tool call function must be an object.")

        if name is None:
            raise call_error("Tool name must be a non-empty string.")

        tool = self._tools.get(name)
        if tool is None:
            raise call_error(f"Tool {name!r} is not registered.", "unknown_tool")

        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            raise call_error("Tool arguments must be a JSON string.")

        try:
            arguments = decode_json(raw_arguments)
        except (ValueError, RecursionError) as exc:
            raise call_error(
                "Tool arguments must be valid JSON.", "invalid_arguments_json"
            ) from exc

        try:
            validate_json(arguments)
        except ValidationError as exc:
            raise call_error(
                "Tool arguments must be valid finite JSON.", "invalid_arguments_json"
            ) from exc

        if not isinstance(arguments, dict):
            raise call_error(
                "Tool arguments must decode to an object.", "invalid_arguments_type"
            )

        parameters = tool.definition["function"]["parameters"]
        try:
            validator_class = validator_for(parameters, default=Draft202012Validator)
            validator_class(parameters).validate(arguments)
        except SchemaValidationError as exc:
            raise call_error(
                f"Tool arguments do not match the schema: {exc.message}",
                "arguments_schema_mismatch",
                argument_path=list(exc.absolute_path),
            ) from exc
        except Exception as exc:
            raise call_error(
                "Tool parameter schema could not be evaluated.",
                "schema_validation_error",
            ) from exc

        return arguments

    async def execute(self, call: dict[str, Any]) -> ToolResult:
        """Validate and execute one registered tool."""
        arguments = self.validate_call(call)
        argument_snapshot = deepcopy(arguments)
        call_id = call["id"]
        name = call["function"]["name"]
        tool = self._tools[name]

        try:
            if iscoroutinefunction(tool.handler):
                value = await tool.handler(**arguments)
            else:
                value = await asyncio.to_thread(tool.handler, **arguments)
                if isawaitable(value):
                    value = await value
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool {name!r} failed during execution.",
                tool_call_id=call_id,
                tool_name=name,
            ) from exc

        try:
            validate_json(value)
            value = deepcopy(value)

            if isinstance(value, str):
                content = value
            else:
                content = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                )
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool {name!r} returned an invalid result.",
                tool_call_id=call_id,
                tool_name=name,
            ) from exc

        return ToolResult(
            tool_call_id=call_id,
            value=value,
            content=content,
            arguments=argument_snapshot,
        )
