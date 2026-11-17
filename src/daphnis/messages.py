"""Store and edit Chat Completions messages."""

from copy import deepcopy
from typing import Any

from ._json import validate_json
from .errors import ValidationError


def validate_message(
    message: dict[str, Any], *, path: str = "message", require_role: bool = True
) -> None:
    """Validate the basic structure without restricting extension fields."""
    if not isinstance(message, dict):
        raise ValidationError(f"{path} must be an object.")

    role = message.get("role")
    if (require_role or "role" in message) and (
        not isinstance(role, str) or not role.strip()
    ):
        raise ValidationError(f"{path}.role must be a non-empty string.")

    content = message.get("content")
    if content is not None and not isinstance(content, (str, list)):
        raise ValidationError(f"{path}.content must be a string, list, or null.")


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    validate_message(message)
    validate_json(message)
    return deepcopy(message)


class Messages:
    """Optional storage and editing for Chat Completions message dictionaries."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        if messages is not None and not isinstance(messages, list):
            raise ValidationError("messages must be a list or None.")

        self._messages = [_copy_message(message) for message in messages or []]

    def __len__(self) -> int:
        return len(self._messages)

    def add(self, message: dict[str, Any]) -> None:
        """Append an independent copy of a valid message."""
        self._messages.append(_copy_message(message))

    def _validate_index(self, index: int, *, allow_end: bool = False) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValidationError("index must be an integer, not a boolean.")
        size = len(self._messages)
        upper = size if allow_end else size - 1
        if index < -size or index > upper:
            raise ValidationError("Message index is out of range.")

    def get(self, index: int) -> dict[str, Any]:
        """Return an independent copy; negative indices count from the end."""
        self._validate_index(index)
        return deepcopy(self._messages[index])

    def insert(self, index: int, message: dict[str, Any]) -> None:
        """Insert before an index, permitting len(self) but rejecting overflow."""
        self._validate_index(index, allow_end=True)
        self._messages.insert(index, _copy_message(message))

    def replace(self, index: int, message: dict[str, Any]) -> None:
        """Replace one whole message after validation, without merging fields."""
        self._validate_index(index)
        self._messages[index] = _copy_message(message)

    def remove(self, index: int) -> None:
        """Remove one message by index without changing any other message."""
        self._validate_index(index)
        del self._messages[index]

    def _add_with_extra(
        self, message: dict[str, Any], extra: dict[str, Any] | None
    ) -> None:
        if extra is not None:
            if not isinstance(extra, dict):
                raise ValidationError("extra must be an object or None.")
            conflicts = message.keys() & extra.keys()
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValidationError(f"extra conflicts with message fields: {names}.")
            message.update(extra)
        self.add(message)

    def add_system(
        self,
        content: str | list[dict[str, Any]] | None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a system message with optional extension fields."""
        self._add_with_extra({"role": "system", "content": content}, extra)

    def add_user(
        self,
        content: str | list[dict[str, Any]] | None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a user message with optional extension fields."""
        self._add_with_extra({"role": "user", "content": content}, extra)

    def add_assistant(
        self,
        content: str | list[dict[str, Any]] | None = None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append an assistant message, defaulting to explicit null content."""
        self._add_with_extra({"role": "assistant", "content": content}, extra)

    def add_tool_result(
        self,
        *,
        tool_call_id: str,
        content: str | list[dict[str, Any]] | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append a tool result without executing or serializing a tool."""
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ValidationError("tool_call_id must be a non-empty string.")
        self._add_with_extra(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content},
            extra,
        )

    def snapshot(self) -> list[dict[str, Any]]:
        """Return independent message dictionaries for inspection or a request."""
        return deepcopy(self._messages)
