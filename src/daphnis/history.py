"""Explicit, versioned, atomic local Session persistence."""

import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

from saturnus import Messages

from .session import Session
from .usage import UsageTotals

if os.name == "posix":
    import fcntl


class SessionConflictError(RuntimeError):
    """A different writer changed the persisted Session."""


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
        serialized = (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
        if os.name != "posix":
            raise OSError("Session writes require POSIX file locking.")
        self.directory.mkdir(parents=True, exist_ok=True)
        # Lock the stable sidecar, not the inode replaced by the commit.
        lock_path = destination.with_suffix(".lock")
        with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDONLY | os.O_NOFOLLOW, 0o600)) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                revision = (session.id, sha256(destination.read_bytes()).hexdigest())
            except FileNotFoundError:
                revision = None
            expected = session._saved_revision
            if expected is not None and expected[0] != session.id:
                expected = None
            if revision != expected:
                raise SessionConflictError(
                    f"Session write conflict for {session.id}; reload before saving."
                )
            self._replace(destination, serialized)
            session._saved_revision = (session.id, sha256(serialized).hexdigest())

    def _replace(self, destination: Path, serialized: bytes) -> None:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=self.directory,
                prefix=f".{destination.stem}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        # Missing files remain FileNotFoundError; invalid data never creates a session.
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Session data must be a JSON object.")  # noqa: TRY004 - Malformed history file.
        version = data.get("version")
        if type(version) is not int or version != 2:
            raise ValueError("Unsupported session file version.")
        required = {
            "id",
            "workspace",
            "provider",
            "model_id",
            "thinking",
            "enabled_tools",
            "messages",
            "usage",
        }
        if not required.issubset(data) or data["id"] != session_id:
            raise ValueError("Session data is missing fields or has a mismatched id.")
        if not isinstance(data["messages"], list):
            raise ValueError("Session messages must be a JSON array.")  # noqa: TRY004 - Malformed history file.
        enabled_tools = data["enabled_tools"]
        if not isinstance(enabled_tools, list):
            raise TypeError("Session enabled_tools must be a JSON array.")
        return Session(
            id=data["id"],
            workspace=data["workspace"],
            provider=data["provider"],
            model_id=data["model_id"],
            thinking=data["thinking"],
            enabled_tools=tuple(enabled_tools),
            messages=Messages(data["messages"]),
            usage=UsageTotals.from_dict(data["usage"]),
            _saved_revision=(session_id, sha256(raw).hexdigest()),
        )
