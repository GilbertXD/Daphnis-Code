"""Explicit, versioned, atomic local Session persistence."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from saturnus import Messages

from .session import Session
from .usage import UsageTotals


@dataclass(frozen=True)
class SessionInfo:
    id: str
    workspace: Path
    provider: str
    model_id: str
    message_count: int
    updated_at: float


class History:
    def __init__(self, directory: str | Path | None = None) -> None:
        location = (
            Path.home() / ".daphnis" / "sessions"
            if directory is None
            else Path(directory)
        )
        self.directory = location.expanduser().resolve()

    def list_sessions(self) -> tuple[SessionInfo, ...]:
        """List saved facts; invalid files raise rather than silently disappearing."""
        if not self.directory.exists():
            return ()
        items = []
        for path in self.directory.glob("*.json"):
            try:
                session = self.load(path.stem)
                items.append(
                    SessionInfo(
                        session.id,
                        Path(session.workspace),
                        session.provider,
                        session.model_id,
                        len(session.messages),
                        path.stat().st_mtime,
                    )
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot list saved session {path.name}: {exc}"
                ) from exc
        return tuple(sorted(items, key=lambda item: item.updated_at, reverse=True))

    def _path(self, session_id: str) -> Path:
        if (
            not isinstance(session_id, str)
            or not session_id.strip()
            or session_id in (".", "..")
            or any(character in session_id for character in ("/", "\\", "\0"))
        ):
            raise ValueError("session_id must be a non-empty filename component.")
        return self.directory / f"{session_id}.json"

    def save(self, session: Session) -> None:
        destination = self._path(session.id)
        validated = Session(
            id=session.id,
            workspace=session.workspace,
            provider=session.provider,
            model_id=session.model_id,
            messages=Messages(session.messages.snapshot()),
            usage=session.usage,
            thinking=session.thinking,
            enabled_tools=session.enabled_tools,
        )
        data = {
            "version": 2,
            "id": validated.id,
            "workspace": str(validated.workspace),
            "provider": validated.provider,
            "model_id": validated.model_id,
            "thinking": validated.thinking,
            "enabled_tools": list(validated.enabled_tools),
            "messages": validated.messages.snapshot(),
            "usage": asdict(validated.usage),
        }
        serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{session.id}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(serialized + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        # Missing files remain FileNotFoundError; invalid data never creates a session.
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict):
            raise ValueError("Session data must be a JSON object.")
        version = data.get("version")
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported session file version.")
        required = {"id", "workspace", "provider", "model_id", "messages"}
        if version == 2:
            required.update(("thinking", "enabled_tools"))
        if not required.issubset(data) or data["id"] != session_id:
            raise ValueError("Session data is missing fields or has a mismatched id.")
        if not isinstance(data["messages"], list):
            raise ValueError("Session messages must be a JSON array.")
        enabled_tools = data["enabled_tools"] if version == 2 else []
        if not isinstance(enabled_tools, list):
            raise TypeError("Session enabled_tools must be a JSON array.")
        return Session(
            id=data["id"],
            workspace=data["workspace"],
            provider=data["provider"],
            model_id=data["model_id"],
            thinking=data["thinking"] if version == 2 else None,
            enabled_tools=tuple(enabled_tools),
            messages=Messages(data["messages"]),
            usage=UsageTotals.from_dict(data["usage"])
            if "usage" in data
            else UsageTotals(unrecorded=bool(data["messages"])),
        )
