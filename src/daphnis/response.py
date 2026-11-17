"""Assemble Chat Completions chunks and validate response choices."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ._json import validate_json
from .errors import ValidationError

_RESPONSE_METADATA_FIELDS = ("id", "model", "created")


def _response_choice(response: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(response, dict):
        raise ValidationError("Response must be an object.")

    choices = response.get("choices")
    if not isinstance(choices, list):
        raise ValidationError("response.choices must be a list.")

    indices: set[int] = set()
    selected = None
    for position, choice in enumerate(choices):
        path = f"response.choices[{position}]"
        if not isinstance(choice, dict):
            raise ValidationError(f"{path} must be an object.")
        index = choice.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValidationError(f"{path}.index must be a non-negative integer.")
        if index in indices:
            raise ValidationError(f"Duplicate choice index in response: {index}.")
        indices.add(index)
        if index == 0:
            selected = choice, path

    if selected is None:
        raise ValidationError("Response has no choice with index 0.")
    return selected


def _validate_tool_calls(value: Any, *, path: str) -> None:
    if value is None:
        return

    if not isinstance(value, list):
        raise ValidationError(f"{path} must be a list or null.")

    for position, call in enumerate(value):
        call_path = f"{path}[{position}]"

        if not isinstance(call, dict):
            raise ValidationError(f"{call_path} must be an object.")

        index = call.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValidationError(f"{call_path}.index must be a non-negative integer.")

        call_id = call.get("id")
        if call_id is not None and (
            not isinstance(call_id, str) or not call_id.strip()
        ):
            raise ValidationError(f"{call_path}.id must be a non-empty string or null.")

        tool_type = call.get("type")
        if tool_type is not None and tool_type != "function":
            raise ValidationError(f"{call_path}.type must be 'function' or null.")

        function = call.get("function")
        if function is None:
            continue

        if not isinstance(function, dict):
            raise ValidationError(f"{call_path}.function must be an object or null.")

        name = function.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValidationError(
                f"{call_path}.function.name must be a non-empty string or null."
            )

        arguments = function.get("arguments")
        if arguments is not None and not isinstance(arguments, str):
            raise ValidationError(
                f"{call_path}.function.arguments must be a string or null."
            )


def _validate_chunk(chunk: Any) -> None:
    if not isinstance(chunk, dict):
        raise ValidationError("chunk must be an object.")

    try:
        validate_json(chunk)
    except ValidationError as exc:
        raise ValidationError("Chunk data must be valid JSON.") from exc

    for name in ("id", "model"):
        value = chunk.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValidationError(f"chunk.{name} must be a non-empty string or null.")

    created = chunk.get("created")
    if created is not None and (
        isinstance(created, bool) or not isinstance(created, int) or created < 0
    ):
        raise ValidationError("chunk.created must be a non-negative integer or null.")

    usage = chunk.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise ValidationError("chunk.usage must be an object or null.")

    choices = chunk.get("choices")
    if not isinstance(choices, list):
        raise ValidationError("chunk.choices must be a list.")

    indices: set[int] = set()
    for position, choice in enumerate(choices):
        path = f"chunk.choices[{position}]"

        if not isinstance(choice, dict):
            raise ValidationError(f"{path} must be an object.")

        index = choice.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValidationError(f"{path}.index must be a non-negative integer.")
        if index in indices:
            raise ValidationError(f"Duplicate choice index in chunk: {index}.")
        indices.add(index)

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ValidationError(f"{path}.delta must be an object.")
        _validate_tool_calls(delta.get("tool_calls"), path=f"{path}.delta.tool_calls")

        for name in ("content", "reasoning_content"):
            value = delta.get(name)
            if value is not None and not isinstance(value, str):
                raise ValidationError(f"{path}.delta.{name} must be a string or null.")

        for name, value in (
            ("delta.role", delta.get("role")),
            ("finish_reason", choice.get("finish_reason")),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValidationError(
                    f"{path}.{name} must be a non-empty string or null."
                )


@dataclass
class _ToolCallState:
    id: str | None = None
    type: str | None = None
    name: str | None = None
    argument_parts: list[str] = field(default_factory=list)


@dataclass
class _ChoiceState:
    role: str | None = None
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    tool_calls: dict[int, _ToolCallState] = field(default_factory=dict)


class ResponseAccumulator:
    """Incrementally assemble Chat Completions response chunks."""

    def __init__(self) -> None:
        self._choices: dict[int, _ChoiceState] = {}
        self._metadata: dict[str, Any] = {}
        self._usage: dict[str, Any] | None = None
        self._has_usage = False
        self._is_finalized = False

    def _validate_metadata(self, chunk: dict[str, Any]) -> None:
        for name in ("id", "model"):
            value = chunk.get(name)
            if (
                value is not None
                and name in self._metadata
                and self._metadata[name] != value
            ):
                raise ValidationError(f"Response metadata conflict: {name}.")

    def _validate_choice_states(self, chunk: dict[str, Any]) -> None:
        for choice in chunk["choices"]:
            index = choice["index"]
            state = self._choices.get(index)
            if state is None:
                continue

            delta = choice["delta"]
            role = delta.get("role")
            if role is not None and state.role is not None and role != state.role:
                raise ValidationError(f"Role conflict for choice {index}.")

            if state.finish_reason is None:
                continue

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and finish_reason != state.finish_reason:
                raise ValidationError(f"Finish reason conflict for choice {index}.")

            if role is not None and state.role is None:
                raise ValidationError(f"Finished choice {index} cannot gain a role.")

            if any(
                delta.get(name)
                for name in ("content", "reasoning_content", "tool_calls")
            ):
                raise ValidationError(f"Finished choice {index} cannot receive output.")

    def _validate_tool_identities(self, chunk: dict[str, Any]) -> None:
        identities: dict[tuple[int, int], dict[str, str | None]] = {}

        for choice in chunk["choices"]:
            choice_index = choice["index"]

            for call in choice["delta"].get("tool_calls") or []:
                tool_index = call["index"]
                key = (choice_index, tool_index)

                if key not in identities:
                    state = self._choices.get(choice_index)
                    tool = (
                        state.tool_calls.get(tool_index) if state is not None else None
                    )
                    identities[key] = {
                        "id": tool.id if tool is not None else None,
                        "name": tool.name if tool is not None else None,
                    }

                identity = identities[key]
                function = call.get("function") or {}

                for name, value in (
                    ("id", call.get("id")),
                    ("name", function.get("name")),
                ):
                    if value is None:
                        continue

                    previous = identity[name]
                    if previous is not None and previous != value:
                        raise ValidationError(
                            f"Tool call identity conflict for choice "
                            f"{choice_index}, tool {tool_index}: {name}."
                        )

                    identity[name] = value

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        """Add one response chunk without modifying the input."""
        if self._is_finalized:
            raise ValidationError("Cannot add chunks after finalization.")

        _validate_chunk(chunk)
        self._validate_metadata(chunk)
        self._validate_choice_states(chunk)
        self._validate_tool_identities(chunk)
        stored = deepcopy(chunk)

        for name in _RESPONSE_METADATA_FIELDS:
            value = stored.get(name)
            if value is not None:
                self._metadata.setdefault(name, value)

        if "usage" in stored:
            self._has_usage = True
            if stored["usage"] is not None:
                self._usage = stored["usage"]

        for choice in stored["choices"]:
            index = choice["index"]
            state = self._choices.setdefault(index, _ChoiceState())
            if state.finish_reason is not None:
                continue
            delta = choice["delta"]

            role = delta.get("role")
            if role is not None:
                state.role = role

            content = delta.get("content")
            if content is not None:
                state.content_parts.append(content)

            reasoning = delta.get("reasoning_content")
            if reasoning is not None:
                state.reasoning_parts.append(reasoning)

            for call in delta.get("tool_calls") or []:
                tool = state.tool_calls.setdefault(call["index"], _ToolCallState())

                call_id = call.get("id")
                if call_id is not None:
                    tool.id = call_id

                tool_type = call.get("type")
                if tool_type is not None:
                    tool.type = tool_type

                function = call.get("function") or {}

                name = function.get("name")
                if name is not None:
                    tool.name = name

                arguments = function.get("arguments")
                if arguments is not None:
                    tool.argument_parts.append(arguments)

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                state.finish_reason = finish_reason

    def _snapshot(self) -> dict[str, Any]:
        """Return an independent snapshot of the accumulated result."""
        choices: list[dict[str, Any]] = []

        for index, state in sorted(self._choices.items()):
            message: dict[str, Any] = {}

            if state.role is not None:
                message["role"] = state.role

            if state.content_parts:
                message["content"] = "".join(state.content_parts)

            choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": state.finish_reason,
                }
            )

            if state.reasoning_parts:
                message["reasoning_content"] = "".join(state.reasoning_parts)

            if state.tool_calls:
                tool_calls: list[dict[str, Any]] = []

                for _, tool in sorted(state.tool_calls.items()):
                    function: dict[str, Any] = {}
                    call: dict[str, Any] = {"function": function}

                    if tool.id is not None:
                        call["id"] = tool.id

                    if tool.type is not None:
                        call["type"] = tool.type

                    if tool.name is not None:
                        function["name"] = tool.name

                    if tool.argument_parts:
                        function["arguments"] = "".join(tool.argument_parts)

                    tool_calls.append(call)

                message["tool_calls"] = tool_calls

        result = {**self._metadata, "choices": choices}
        if self._has_usage:
            result["usage"] = deepcopy(self._usage)
        return result

    def finalize(self) -> dict[str, Any]:
        """Validate and seal the result after the caller exhausts the stream.

        Failed validation leaves the accumulator unsealed.
        Repeated calls return independent snapshots of the same sealed result.
        """
        if not self._choices:
            raise ValidationError("Cannot finalize without any choices.")

        for index, state in self._choices.items():
            if state.finish_reason is None:
                raise ValidationError(f"Choice {index} has no finish reason.")
            if state.finish_reason == "tool_calls" and not state.tool_calls:
                raise ValidationError(f"Choice {index} finished with no tool calls.")

            for tool_index, tool in state.tool_calls.items():
                if (
                    tool.id is None
                    or tool.type is None
                    or tool.name is None
                    or not tool.argument_parts
                ):
                    raise ValidationError(
                        f"Incomplete tool call for choice {index}, tool {tool_index}."
                    )

        result = self._snapshot()
        self._is_finalized = True
        return result
