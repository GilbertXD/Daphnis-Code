"""Explicit Chat Completions request and message mappings."""

import re
from collections.abc import Callable
from copy import deepcopy
from inspect import iscoroutine, iscoroutinefunction
from itertools import pairwise
from typing import Any

from .._json import validate_json
from ..errors import ValidationError
from ..messages import validate_message

_PROTOCOL_FIELDS = {"model", "messages", "stream"}
_MESSAGE_PROTOCOL_FIELDS = {"role", "content"}
_CONTENT_FIELD = re.compile(r"[\w-]+")
_FieldPath = tuple[str, ...]
_ContentRule = tuple[str, tuple[tuple[_FieldPath, _FieldPath], ...]]


def _content_type(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.isprintable()
        and not any(char.isspace() for char in value)
    )


def _field_path(value: Any) -> _FieldPath:
    path = tuple(value.split(".")) if isinstance(value, str) else ()
    if not path or any(
        not _CONTENT_FIELD.fullmatch(part) or part.isdecimal() for part in path
    ):
        raise ValidationError("Content field paths must name object fields.")
    if path[0] == "type":
        raise ValidationError("Use the content part type rule to change its type.")
    return path


def _validate_nonoverlapping(paths: tuple[_FieldPath, ...]) -> None:
    ordered = sorted(paths)
    for parent, child in pairwise(ordered):
        if child[: len(parent)] == parent:
            raise ValidationError("Content field paths must not overlap.")


def _content_rules(mappings: dict | None) -> dict[str, _ContentRule]:
    if mappings is None:
        return {}
    if not isinstance(mappings, dict):
        raise ValidationError("content_part_mappings must be an object.")

    result = {}
    for source, rule in mappings.items():
        if not _content_type(source) or not isinstance(rule, dict):
            raise ValidationError("Content part rules must map types to objects.")
        target = rule.get("type")
        if not _content_type(target) or rule.keys() - {"type", "fields"}:
            raise ValidationError("Content part rules require a valid target type.")
        fields = rule.get("fields", {})
        if not isinstance(fields, dict):
            raise ValidationError("Content field mappings must be an object.")
        paths = tuple(
            (_field_path(key), _field_path(value)) for key, value in fields.items()
        )
        _validate_nonoverlapping(tuple(source for source, _ in paths))
        _validate_nonoverlapping(tuple(target for _, target in paths))
        result[source] = target, paths
    return result


def _field_parent(
    part: dict[str, Any], path: _FieldPath, *, location: str
) -> dict[str, Any]:
    parent = part
    for key in path[:-1]:
        if key not in parent:
            return {}
        child = parent[key]
        if not isinstance(child, dict):
            raise ValidationError(
                f"{location}: content field path crosses a non-object value."
            )
        parent = child
    return parent


def _remove_field(part: dict[str, Any], path: _FieldPath) -> None:
    parents = []
    for key in path[:-1]:
        parents.append((part, key))
        part[key] = part[key].copy()
        part = part[key]
    del part[path[-1]]
    for parent, key in reversed(parents):
        if parent[key]:
            break
        del parent[key]


def _insert_field(part: dict[str, Any], path: _FieldPath, value: Any) -> None:
    for key in path[:-1]:
        part[key] = part.get(key, {}).copy()
        part = part[key]
    part[path[-1]] = value


def _map_content_part(
    part: dict[str, Any], rule: _ContentRule, *, location: str
) -> dict[str, Any]:
    target_type, fields = rule
    moves = []
    # Resolve every move against the original part so newly written fields never match.
    for source, target in fields:
        if source == target:
            continue
        source_parent = _field_parent(part, source, location=location)
        if source[-1] not in source_parent:
            continue
        if target[-1] in _field_parent(part, target, location=location):
            raise ValidationError(
                f"{location}: content field mapping target is already present."
            )
        moves.append((source, target, source_parent[source[-1]]))

    mapped = {**part, "type": target_type}
    for source, _, _ in moves:
        _remove_field(mapped, source)
    for _, target, value in moves:
        _insert_field(mapped, target, value)
    return mapped


class Compatibility:
    """Explicit compatibility rules for requests, messages, and content parts."""

    def __init__(
        self,
        *,
        field_mappings: dict[str, str] | None = None,
        value_mappings: dict[str, dict[str, Any]] | None = None,
        message_field_mappings: dict[str, dict[str, str]] | None = None,
        content_part_mappings: dict | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        mappings = {} if field_mappings is None else field_mappings

        if not isinstance(mappings, dict):
            raise ValidationError("field_mappings must be an object.")

        for source, target in mappings.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValidationError("Field mappings must map strings to strings.")

            if not source or not target:
                raise ValidationError("Field mapping names must be non-empty strings.")

            if source in _PROTOCOL_FIELDS or target in _PROTOCOL_FIELDS:
                raise ValidationError("Protocol fields cannot be remapped.")

        values = {} if value_mappings is None else value_mappings

        if not isinstance(values, dict):
            raise ValidationError("value_mappings must be an object.")

        for name, aliases in values.items():
            if not isinstance(name, str) or not isinstance(aliases, dict):
                raise ValidationError("Value mappings must map field names to objects.")

            if not name:
                raise ValidationError(
                    "Value mapping field names must be non-empty strings."
                )

            if name in _PROTOCOL_FIELDS:
                raise ValidationError("Protocol fields cannot be remapped.")

        validate_json(values)

        message_mappings = (
            {} if message_field_mappings is None else message_field_mappings
        )
        if not isinstance(message_mappings, dict):
            raise ValidationError("message_field_mappings must be an object.")

        for role, rules in message_mappings.items():
            if not isinstance(role, str) or not role.strip():
                raise ValidationError(
                    "Message mapping roles must be non-empty strings."
                )
            if not isinstance(rules, dict):
                raise ValidationError("Each message role must map to an object.")

            for source, target in rules.items():
                if (
                    not isinstance(source, str)
                    or not source.strip()
                    or not isinstance(target, str)
                    or not target.strip()
                ):
                    raise ValidationError(
                        "Message mappings must map non-empty strings to non-empty strings."
                    )
                if (
                    source in _MESSAGE_PROTOCOL_FIELDS
                    or target in _MESSAGE_PROTOCOL_FIELDS
                ):
                    raise ValidationError("Message protocol fields cannot be remapped.")

        if validator is not None and (
            not callable(validator)
            or iscoroutinefunction(validator)
            or iscoroutinefunction(type(validator).__call__)
        ):
            raise ValidationError("validator must be a synchronous callable or None.")

        self._field_mappings = dict(mappings)
        self._value_mappings = deepcopy(values)
        self._message_field_mappings = deepcopy(message_mappings)
        self._content_part_mappings = _content_rules(content_part_mappings)
        self._validator = validator

    def _map_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rename message fields in an already isolated request snapshot."""
        if not isinstance(messages, list):
            raise ValidationError("messages must be a list for message mappings.")

        result = []
        for index, message in enumerate(messages):
            path = f"messages[{index}]"
            validate_message(message, path=path)
            rules = self._message_field_mappings.get(message["role"], {})
            mapped: dict[str, Any] = {}
            for source, value in message.items():
                target = rules.get(source, source)
                if target != source and target in message:
                    raise ValidationError(
                        f"{path}: mapping target {target!r} is already present."
                    )
                if target in mapped:
                    raise ValidationError(f"{path}: multiple fields map to {target!r}.")
                mapped[target] = value
            result.append(mapped)
        return result

    def _map_content_parts(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            raise ValidationError("messages must be a list for content part mappings.")

        result = []
        for index, message in enumerate(messages):
            path = f"messages[{index}]"
            validate_message(message, path=path)
            content = message.get("content")
            if not isinstance(content, list):
                result.append(message)
                continue
            parts = []
            for position, part in enumerate(content):
                kind = part.get("type") if isinstance(part, dict) else None
                rule = (
                    self._content_part_mappings.get(kind)
                    if isinstance(kind, str)
                    else None
                )
                parts.append(
                    part
                    if rule is None
                    else _map_content_part(
                        part, rule, location=f"{path}.content[{position}]"
                    )
                )
            result.append({**message, "content": parts})
        return result

    def apply(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply mappings once and return an independent JSON snapshot."""
        if not isinstance(body, dict):
            raise ValidationError("Compatibility input must be an object.")

        validate_json(body)
        result: dict[str, Any] = {}

        for source, value in deepcopy(body).items():
            target = self._field_mappings.get(source, source)

            if target != source and target in body:
                raise ValidationError(f"Mapping target {target!r} is already present.")

            if target in result:
                raise ValidationError(f"Multiple fields map to {target!r}.")

            aliases = self._value_mappings.get(source)
            if aliases is not None and isinstance(value, str) and value in aliases:
                value = deepcopy(aliases[value])

            result[target] = value

        if self._message_field_mappings and "messages" in result:
            result["messages"] = self._map_messages(result["messages"])

        if self._content_part_mappings and "messages" in result:
            result["messages"] = self._map_content_parts(result["messages"])

        if self._validator is not None:
            outcome = self._validator(deepcopy(result))
            if outcome is not None:
                if iscoroutine(outcome):
                    outcome.close()
                raise ValidationError("validator must return None.")

        return result
