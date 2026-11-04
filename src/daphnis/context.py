"""Build request context without changing committed or working messages."""

from typing import Any

from saturnus import Messages


class Context:
    def __init__(self, *, system_prompt: str | None = None) -> None:
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None.")
        self.system_prompt = system_prompt

    def build(self, working_messages: Messages) -> list[dict[str, Any]]:
        """Working messages already include all committed session history."""
        messages = working_messages.snapshot()
        if self.system_prompt is not None:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages
