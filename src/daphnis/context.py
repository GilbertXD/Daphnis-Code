"""Build request context without changing committed or working messages."""

from typing import Any

from saturnus import Messages

from .compaction import SUMMARY_POLICY, CompactionCheckpoint, project, safe_boundaries
from .turn_state import project_terminated_turns, project_turn_messages


def history_range_hint(through: int) -> str:
    return (
        f"Earlier session messages have been compacted. Compacted message range: [0, {through}). "
        "If the task needs original evidence, use retrieval to find visible records in this Session. "
        "Use history_read with a cited message_id or a range returned by retrieval."
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
        """Build the protocol-ready Working view from committed messages."""
        messages = working_messages.snapshot()
        if messages and len(messages) not in safe_boundaries(messages):
            raise ValueError("Cannot request a model with an incomplete tool exchange.")
        prompt = self.system_prompt
        if checkpoint is not None:
            messages = project(messages, checkpoint, summary_reasoning=summary_reasoning)
            prompt = f"{prompt}\n\n{SUMMARY_POLICY}" if prompt is not None else SUMMARY_POLICY
            if history_lookup:
                prompt += "\n\n" + history_range_hint(checkpoint.through)
        if prompt is not None:
            messages.insert(0, {"role": "system", "content": prompt})
        return project_turn_messages(project_terminated_turns(messages))
