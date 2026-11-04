"""Committed conversation state, independent of execution and storage."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from saturnus import Messages


@dataclass
class Session:
    workspace: Path | str
    provider: str
    model_id: str
    id: str = field(default_factory=lambda: uuid4().hex)
    messages: Messages = field(default_factory=Messages)

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
