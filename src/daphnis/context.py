"""Build request context without changing committed or working messages."""

from typing import Any

from saturnus import Messages

from .compaction import SUMMARY_POLICY, CompactionCheckpoint, project


class Context:
    def __init__(self, *, system_prompt: str | None = None) -> None:
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None.")
        self.system_prompt = system_prompt

    def build(
        self, working_messages: Messages, checkpoint: CompactionCheckpoint | None = None,
    ) -> list[dict[str, Any]]:
        """Working messages include all committed and pending raw history."""
        messages = working_messages.snapshot()
        prompt = self.system_prompt
        if checkpoint is not None:
            messages = project(messages, checkpoint)
            prompt = f"{prompt}\n\n{SUMMARY_POLICY}" if prompt is not None else SUMMARY_POLICY
        if prompt is not None:
            messages.insert(0, {"role": "system", "content": prompt})
        return messages
