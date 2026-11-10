"""Committed conversation state and execution preferences."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from saturnus import Messages

from .compaction import CompactionCheckpoint
from .usage import UsageTotals


@dataclass
class Session:
    workspace: Path | str
    provider: str
    model_id: str
    id: str = field(default_factory=lambda: uuid4().hex)
    messages: Messages = field(default_factory=Messages)
    usage: UsageTotals = field(default_factory=UsageTotals)
    thinking: str | None = None
    enabled_tools: tuple[str, ...] = ()
    _saved_revision: tuple[str, str] | None = field(default=None, repr=False, compare=False)
    title: str | None = None
    compaction: CompactionCheckpoint | None = None

    def __post_init__(self) -> None:
        for name in ("id", "provider", "model_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if (
            not isinstance(self.workspace, (str, Path))
            or not str(self.workspace).strip()
        ):
            raise ValueError("workspace must be a non-empty path.")
        self.workspace = Path(self.workspace).expanduser().resolve()
        if not isinstance(self.messages, Messages):
            raise TypeError("messages must be a Saturnus Messages instance.")
        if not isinstance(self.usage, UsageTotals):
            raise TypeError("usage must be a UsageTotals instance.")
        if self.thinking is not None and (
            not isinstance(self.thinking, str) or not self.thinking.strip()
        ):
            raise ValueError("thinking must be a non-empty string or None.")
        if not isinstance(self.enabled_tools, tuple):
            raise TypeError("enabled_tools must be a tuple.")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.enabled_tools
        ):
            raise ValueError("enabled_tools must contain non-empty tool names.")
        if len(set(self.enabled_tools)) != len(self.enabled_tools):
            raise ValueError("enabled_tools must not contain duplicates.")
        if self.title is not None and (
            not isinstance(self.title, str) or not self.title.strip()
        ):
            raise ValueError("title must be a non-empty string or None.")
        if self.compaction is not None:
            if not isinstance(self.compaction, CompactionCheckpoint):
                raise TypeError("compaction must be a CompactionCheckpoint or None.")
            self.compaction.validate(self.messages.snapshot())
