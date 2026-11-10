"""Build request context without changing committed or working messages."""

from typing import Any

from saturnus import Messages

from .compaction import SUMMARY_POLICY, CompactionCheckpoint, project


def history_range_hint(through: int) -> str:
    return (
        "Earlier session messages have been compacted. Original visible records may be "
        "retrieved using history_search and history_read. "
        f"Compacted message range: [0, {through})."
    )


class Context:
    def __init__(self, *, system_prompt: str | None = None) -> None:
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None.")
        self.system_prompt = system_prompt

    def build(
        self, working_messages: Messages, checkpoint: CompactionCheckpoint | None = None,
        *, summary_reasoning: bool = False, history_lookup: bool = False,
    ) -> list[dict[str, Any]]:
        """Working messages include all committed and pending raw history."""
        messages = working_messages.snapshot()
        prompt = self.system_prompt
        if checkpoint is not None:
            messages = project(messages, checkpoint, summary_reasoning=summary_reasoning)
            prompt = f"{prompt}\n\n{SUMMARY_POLICY}" if prompt is not None else SUMMARY_POLICY
            if history_lookup:
                prompt += "\n\n" + history_range_hint(checkpoint.through)
        if prompt is not None:
            messages.insert(0, {"role": "system", "content": prompt})
        return messages
