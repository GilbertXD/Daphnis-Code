"""Explicit, versioned, atomic local Session persistence."""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self, TypeVar

from saturnus import Messages, ValidationError

from .compaction import CompactionCheckpoint
from .session import Session
from .tool_state import interrupted_tool_group
from .usage import UsageTotals

if os.name == "posix":
    import fcntl


class SessionConflictError(RuntimeError):
    """A different writer changed the persisted Session."""


class SessionBusyError(RuntimeError):
    """Another operation holds the Session lease."""


class UnsupportedSessionVersionError(ValueError):
    """The saved Session uses an unsupported file version."""


class SessionLease:
    """Own the stable sidecar lock until the operation and its I/O finish."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.fd: int | None = None

    def __enter__(self) -> Self:
        if self.fd is not None:
            raise RuntimeError("The Session lease is already held.")
        if os.name != "posix":
            raise OSError("Session writes require POSIX file locking.")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        path = self.destination.with_suffix(".lock")
        fd = os.open(path, os.O_CREAT | os.O_RDONLY | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise SessionBusyError(f"Session is busy: {self.destination.stem}") from exc
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


_T = TypeVar("_T")


async def finish_io(operation: Awaitable[_T]) -> _T:
    """Finish started I/O before propagating cancellation to the lease owner."""
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.wait((task,))
            except asyncio.CancelledError:
                continue
        if not task.cancelled():
            task.exception()
        raise


@dataclass(frozen=True)
class SessionInfo:
    id: str
    workspace: Path
    provider: str
    model_id: str
    message_count: int
    updated_at: float
    title: str | None = None


@dataclass(frozen=True)
class SessionIssue:
    filename: str
    error: str


@dataclass(frozen=True)
class SessionScan:
    sessions: tuple[SessionInfo, ...]
    issues: tuple[SessionIssue, ...]


class History:
    def __init__(self, directory: str | Path | None = None) -> None:
        location = (
            Path.home() / ".daphnis" / "sessions"
            if directory is None
            else Path(directory)
        )
        self.directory = location.expanduser().resolve()

    def list_sessions(self) -> tuple[SessionInfo, ...]:
        """List valid saved Sessions, newest first."""
        return self.scan_sessions().sessions

    def scan_sessions(self) -> SessionScan:
        """Scan saved Sessions and report per-file errors without exposing file contents."""
        try:
            paths = list(self.directory.iterdir())
        except FileNotFoundError:
            return SessionScan((), ())
        items = []
        issues = []
        for path in paths:
            if not path.name.endswith(".json"):
                continue
            try:
                session = self.load(path.name.removesuffix(".json"))
                items.append(
                    SessionInfo(
                        session.id,
                        Path(session.workspace),
                        session.provider,
                        session.model_id,
                        len(session.messages),
                        path.stat().st_mtime,
                        session.title,
                    )
                )
            except (OSError, RuntimeError, TypeError, ValueError, ValidationError) as exc:
                issues.append(SessionIssue(path.name, type(exc).__name__))
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return SessionScan(tuple(items), tuple(issues))

    def _path(self, session_id: str) -> Path:
        if (
            not isinstance(session_id, str)
            or not session_id.strip()
            or session_id in (".", "..")
            or any(character in session_id for character in ("/", "\\", "\0"))
        ):
            raise ValueError("session_id must be a non-empty filename component.")
        return self.directory / f"{session_id}.json"

    @staticmethod
    def _check_revision(session: Session, destination: Path) -> None:
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

    @asynccontextmanager
    async def lease(self, session: Session) -> AsyncIterator[SessionLease]:
        lease = SessionLease(self._path(session.id))

        def acquire() -> None:
            lease.__enter__()
            self._check_revision(session, lease.destination)

        try:
            await finish_io(asyncio.to_thread(acquire))
            yield lease
        finally:
            lease.__exit__()

    async def save_async(self, session: Session, *, lease: SessionLease | None = None) -> None:
        await finish_io(asyncio.to_thread(self.save, session, lease=lease))

    async def rename(self, session_id: str, title: str) -> None:
        def update() -> None:
            self.save(replace(self.load(session_id), title=title))

        await finish_io(asyncio.to_thread(update))

    async def delete(self, session: Session) -> None:
        async with self.lease(session) as lease:
            # Keep the stable lock file so concurrent processes lock the same inode.
            await finish_io(asyncio.to_thread(lease.destination.unlink, missing_ok=True))

    def save(self, session: Session, *, lease: SessionLease | None = None) -> None:
        destination = self._path(session.id)
        if lease is None:
            with SessionLease(destination) as acquired:
                self.save(session, lease=acquired)
            return
        if lease.destination != destination or lease.fd is None:
            raise ValueError("An active lease for the same Session is required.")
        self._check_revision(session, destination)
        validated = Session(
            id=session.id,
            workspace=session.workspace,
            provider=session.provider,
            model_id=session.model_id,
            messages=Messages(session.messages.snapshot()),
            usage=session.usage,
            thinking=session.thinking,
            enabled_tools=session.enabled_tools,
            title=session.title,
            compaction=session.compaction,
            system_prompt=session.system_prompt,
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
        interrupted_tool_group(data["messages"])
        if validated.title is not None:
            data["title"] = validated.title
        if validated.compaction is not None:
            data["compaction"] = asdict(validated.compaction)
        if validated.system_prompt is not None:
            data["system_prompt"] = validated.system_prompt
        serialized = (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
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
            raise UnsupportedSessionVersionError("Unsupported session file version.")
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
        checkpoint = data.get("compaction")
        if checkpoint is not None:
            if not isinstance(checkpoint, dict) or checkpoint.keys() != {"through", "summary"}:
                raise ValueError("Invalid session compaction checkpoint.")
            checkpoint = CompactionCheckpoint(**checkpoint)
        messages = Messages(data["messages"])
        interrupted_tool_group(data["messages"])
        return Session(
            id=data["id"],
            workspace=data["workspace"],
            provider=data["provider"],
            model_id=data["model_id"],
            thinking=data["thinking"],
            enabled_tools=tuple(enabled_tools),
            messages=messages,
            usage=UsageTotals.from_dict(data["usage"]),
            _saved_revision=(session_id, sha256(raw).hexdigest()),
            title=data.get("title"),
            compaction=checkpoint,
            system_prompt=data.get("system_prompt"),
        )
