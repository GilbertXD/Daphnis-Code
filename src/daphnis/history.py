"""Explicit, versioned, atomic local Session persistence."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from saturnus import Messages

from .session import Session


class History:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()

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
        )
        data = {
            "version": 1,
            "id": validated.id,
            "workspace": str(validated.workspace),
            "provider": validated.provider,
            "model_id": validated.model_id,
            "messages": validated.messages.snapshot(),
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
        if type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("Unsupported session file version.")
        required = {"id", "workspace", "provider", "model_id", "messages"}
        if not required.issubset(data) or data["id"] != session_id:
            raise ValueError("Session data is missing fields or has a mismatched id.")
        if not isinstance(data["messages"], list):
            raise ValueError("Session messages must be a JSON array.")
        return Session(
            id=data["id"],
            workspace=data["workspace"],
            provider=data["provider"],
            model_id=data["model_id"],
            messages=Messages(data["messages"]),
        )
