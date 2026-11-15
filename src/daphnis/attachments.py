"""Validate uploaded text and store immutable, Session-scoped snapshots."""

import os
import re
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 256 * 1024
MAX_FILES = 10
ATTACHMENT_ONLY_PROMPT = "已附上文本文件。"
ATTACHMENT_TOOLS = frozenset({"read_attachment", "search_attachment"})
_ID = re.compile(r"[0-9a-f]{32}")
_BINARY = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".go", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".xml",
    ".csv", ".tsv", ".log", ".sql", ".sh", ".bash", ".zsh", ".rs", ".java", ".kt", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".cs", ".rb", ".php", ".swift", ".vue", ".svelte", ".tex", ".diff", ".patch",
}
_PRIVATE_NAMES = {"credentials.json", "application_default_credentials.json", "secrets.json"}


def valid_id(value: str) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _name(name: str) -> None:
    if (
        not isinstance(name, str) or not 0 < len(name) <= 255 or name != name.strip()
        or any(char in name for char in "/\\") or any(ord(char) < 32 for char in name)
        or _BINARY.search(name) or Path(name).suffix.lower() not in _EXTENSIONS
        or name.lower().startswith((".env", "id_rsa.", "id_ed25519.", "id_ecdsa.", "id_dsa."))
        or name.lower() in _PRIVATE_NAMES
    ):
        raise ValueError("附件须为受支持的文本文件，不能包含路径或使用已知的凭据文件名。")
    try:
        name.encode("utf-8")
    except UnicodeError:
        raise ValueError("附件文件名不是有效的 UTF-8。") from None


@dataclass(frozen=True)
class TextAttachmentInput:
    name: str
    text: str


@dataclass(frozen=True)
class TextAttachment:
    id: str
    name: str
    size_bytes: int
    sha256: str
    lines: int
    mime_type: str = "text/plain"

    def __post_init__(self) -> None:
        _name(self.name)
        if (
            not valid_id(self.id) or self.mime_type != "text/plain"
            or type(self.size_bytes) is not int or not 0 <= self.size_bytes <= MAX_FILE_BYTES
            or not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or type(self.lines) is not int or not 1 <= self.lines <= self.size_bytes + 1
        ):
            raise ValueError("附件元数据无效。")


@dataclass(frozen=True)
class AttachmentBatch:
    files: tuple[TextAttachment, ...]
    submission_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple) or not 0 < len(self.files) <= MAX_FILES or not all(
            isinstance(item, TextAttachment) for item in self.files
        ):
            raise TypeError("附件引用无效。")
        if sum(item.size_bytes for item in self.files) > MAX_TOTAL_BYTES:
            raise ValueError("附件总大小超过 256 KiB。")
        if self.submission_id is not None and not valid_id(self.submission_id):
            raise ValueError("附件发送标识无效。")

    @classmethod
    def from_json(cls, data: dict) -> "AttachmentBatch":
        if not isinstance(data, dict) or not isinstance(data.get("files"), list):
            raise TypeError("附件引用无效。")
        return cls(tuple(TextAttachment(**item) for item in data["files"]), data.get("submission_id"))


def prepare_attachments(uploads: Sequence[TextAttachmentInput], submission_id: str | None) -> AttachmentBatch:
    if not 0 < len(uploads) <= MAX_FILES:
        raise ValueError("每次最多可发送 10 个附件。")
    files = []
    for upload in uploads:
        if not isinstance(upload, TextAttachmentInput) or not isinstance(upload.text, str):
            raise TypeError("附件内容必须是 UTF-8 文本。")
        _name(upload.name)
        if len(upload.text) > MAX_FILE_BYTES:
            raise ValueError("每个附件最多 128 KiB。")
        try:
            data = upload.text.encode("utf-8")
        except UnicodeError:
            raise ValueError("附件内容必须是有效的 UTF-8。") from None
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("每个附件最多 128 KiB。")
        if _BINARY.search(upload.text) or re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", upload.text):
            raise ValueError("附件不能包含二进制内容或私钥。")
        files.append(TextAttachment(
            id=uuid4().hex, name=upload.name, size_bytes=len(data),
            sha256=sha256(data).hexdigest(), lines=upload.text.count("\n") + 1,
        ))
    return AttachmentBatch(tuple(files), submission_id)


def attachment_metadata(batch: AttachmentBatch) -> list[dict]:
    return [asdict(item) for item in batch.files]


class AttachmentStore:
    def __init__(self, history_directory: Path):
        name = "attachments" if history_directory.name == "sessions" else f"{history_directory.name}.attachments"
        self.directory = history_directory.with_name(name)

    @contextmanager
    def _session_directory(self, session_id: str, *, create: bool = False) -> Iterator[tuple[int, int]]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("附件会话标识无效。")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent = os.open(self.directory.parent, flags)
        try:
            if create:
                try:
                    os.mkdir(self.directory.name, 0o700, dir_fd=parent)
                except FileExistsError:
                    pass
                os.fsync(parent)
            root = os.open(self.directory.name, flags, dir_fd=parent)
            try:
                self._private(root)
                if create:
                    try:
                        os.mkdir(session_id, 0o700, dir_fd=root)
                    except FileExistsError:
                        pass
                    os.fsync(root)
                directory = os.open(session_id, flags, dir_fd=root)
                try:
                    self._private(directory)
                    yield root, directory
                finally:
                    os.close(directory)
            finally:
                os.close(root)
        finally:
            os.close(parent)

    @staticmethod
    def _private(fd: int) -> None:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError("附件存储必须由当前用户独占。")

    def _read(self, directory: int, item: TextAttachment) -> str:
        fd = os.open(item.id + ".txt", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            self._private(stream.fileno())
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != item.size_bytes:
                raise OSError("附件快照不可用。")
            data = stream.read(item.size_bytes + 1)
        if len(data) != item.size_bytes or sha256(data).hexdigest() != item.sha256:
            raise OSError("附件快照校验失败。")
        return data.decode("utf-8")

    def read(self, session_id: str, item: TextAttachment) -> str:
        with self._session_directory(session_id) as (_, directory):
            return self._read(directory, item)

    def write(self, session_id: str, batch: AttachmentBatch, uploads: Sequence[TextAttachmentInput]) -> None:
        with self._session_directory(session_id, create=True) as (_, directory):
            for item, upload in zip(batch.files, uploads, strict=True):
                temporary = ".tmp-" + item.id
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(upload.text.encode("utf-8"))
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o400)
                        os.fsync(stream.fileno())
                    # Publish without overwriting an existing immutable snapshot.
                    os.link(temporary, item.id + ".txt", src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                    os.unlink(temporary, dir_fd=directory)
                    self._read(directory, item)
                finally:
                    try:
                        os.unlink(temporary, dir_fd=directory)
                    except FileNotFoundError:
                        pass
            os.fsync(directory)

    def delete(self, session_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            return
        try:
            with self._session_directory(session_id) as (root, directory):
                for name in os.listdir(directory):
                    os.unlink(name, dir_fd=directory)
                os.rmdir(session_id, dir_fd=root)
        except FileNotFoundError:
            pass
