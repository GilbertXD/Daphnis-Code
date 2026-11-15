"""Validate uploads and store immutable, Session-scoped snapshots."""

import os
import re
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from PIL import Image

from .text import decode_text

MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 256 * 1024
MAX_FILES = 10
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 20 * 1024 * 1024
MAX_IMAGE_REQUEST_BYTES = 32 * 1024 * 1024
MAX_IMAGES = 4
MAX_IMAGE_PIXELS = 20_000_000
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOCUMENTS = 1
ATTACHMENT_ONLY_PROMPT = "已附上文本文件。"
IMAGE_ATTACHMENT_ONLY_PROMPT = "已附上图片。"
DOCUMENT_ATTACHMENT_ONLY_PROMPT = "已附上文档。"
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
_IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
_PNG_END = b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
DOCUMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_OFFICE_PARTS = {".docx": "word/document.xml", ".pptx": "ppt/presentation.xml", ".xlsx": "xl/workbook.xml"}


def valid_id(value: str) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _name(name: str, *, image: bool = False, document: bool = False) -> None:
    if (
        not isinstance(name, str) or not 0 < len(name) <= 255 or name != name.strip()
        or any(char in name for char in "/\\") or any(ord(char) < 32 for char in name)
        or _BINARY.search(name) or name in {".", ".."}
        or (not (image or document) and Path(name).suffix.lower() not in _EXTENSIONS)
        or name.lower().startswith((".env", "id_rsa.", "id_ed25519.", "id_ecdsa.", "id_dsa."))
        or name.lower() in _PRIVATE_NAMES
    ):
        if document:
            raise ValueError("文档附件名称无效，不能包含路径或使用已知的凭据文件名。")
        if image:
            raise ValueError("图片附件名称无效，不能包含路径或使用已知的凭据文件名。")
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
class ImageAttachmentInput:
    name: str
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class DocumentAttachmentInput:
    name: str
    data: bytes
    mime_type: str
    reading_mode: str = "text"


AttachmentInput = TextAttachmentInput | ImageAttachmentInput | DocumentAttachmentInput


def validate_resource_uri(uri: str) -> None:
    if (not isinstance(uri, str) or not 0 < len(uri) <= 2048
        or any(char.isspace() or ord(char) < 32 for char in uri)
        or _BINARY.search(uri) or re.search(r'[%](?![0-9a-fA-F]{2})|[<>"{}|\\^`]', uri)
        or not urlsplit(uri).scheme):
        raise ValueError("MCP 资源 URI 无效。")
    uri.encode("utf-8")


def _resource_metadata(name: str, server: str, uri: str, mime_type: str, source: str) -> None:
    validate_resource_uri(uri)
    if (not isinstance(name, str) or not 0 < len(name) <= 255 or _BINARY.search(name)
        or any(ord(char) < 32 for char in name)
        or not isinstance(server, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", server)
        or source not in ("embedded", "standalone")):
        raise ValueError("MCP 资源元数据无效。")
    name.encode("utf-8")
    base = mime_type.partition(";")[0].strip().lower() if isinstance(mime_type, str) else ""
    if (not base or len(mime_type) > 255 or not mime_type.isprintable()
        or not (re.fullmatch(r"text/[a-z0-9.+_-]+", base) or base in {
            "application/json", "application/xml", "application/yaml", "application/toml",
            "application/javascript", "application/x-ndjson",
        } or re.fullmatch(r"application/[a-z0-9.+_-]+\+json", base))):
        raise ValueError("MCP 资源的 MIME 类型不受支持。")


def snapshot_bytes(text: str) -> bytes:
    if not isinstance(text, str):
        raise TypeError("附件内容必须是 UTF-8 文本。")
    if len(text) > MAX_FILE_BYTES:
        raise ValueError("每个附件或资源最多 128 KiB。")
    try:
        data = text.encode("utf-8")
    except UnicodeError:
        raise ValueError("附件内容必须是有效的 UTF-8。") from None
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("每个附件或资源最多 128 KiB。")
    if _BINARY.search(text):
        raise ValueError("附件或资源不能包含二进制内容。")
    return data


@dataclass(frozen=True)
class TextResourceInput(TextAttachmentInput):
    server: str
    uri: str
    mime_type: str
    source: str
    draft_id: str | None = None

    def __post_init__(self) -> None:
        _resource_metadata(self.name, self.server, self.uri, self.mime_type, self.source)
        snapshot_bytes(self.text)
        if self.draft_id is not None and (self.source != "standalone" or not valid_id(self.draft_id)):
            raise ValueError("MCP 资源草稿标识无效。")


def resource_snapshot_id(submission_id: str, draft_id: str) -> str:
    return sha256(f"{submission_id}:{draft_id}".encode("ascii")).hexdigest()[:32]


@dataclass(frozen=True)
class TextAttachment:
    id: str
    name: str
    size_bytes: int
    sha256: str
    lines: int
    mime_type: str = "text/plain"
    source: str | None = None
    server: str | None = None
    uri: str | None = None

    @property
    def model_name(self) -> str:
        return "MCP text resource" if self.source else self.name

    def to_json(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def __post_init__(self) -> None:
        if self.source is None:
            _name(self.name)
            if self.server is not None or self.uri is not None or self.mime_type != "text/plain":
                raise ValueError("附件元数据无效。")
        else:
            _resource_metadata(self.name, self.server, self.uri, self.mime_type, self.source)
        if (
            not valid_id(self.id)
            or type(self.size_bytes) is not int or not 0 <= self.size_bytes <= MAX_FILE_BYTES
            or not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or type(self.lines) is not int or not 1 <= self.lines <= self.size_bytes + 1
        ):
            raise ValueError("附件元数据无效。")


def _valid_dimensions(width: int, height: int) -> bool:
    return (
        type(width) is int and type(height) is int
        and width > 0 and height > 0 and width * height <= MAX_IMAGE_PIXELS
    )


def _image_dimensions(data: bytes, mime_type: str) -> tuple[int, int]:
    if not isinstance(mime_type, str) or mime_type not in _IMAGE_EXTENSIONS:
        raise ValueError("图片附件仅支持 PNG、JPEG 和 WebP。")
    try:
        with Image.open(BytesIO(data), formats=tuple(_IMAGE_FORMATS)) as image:
            if _IMAGE_FORMATS[image.format] != mime_type:
                raise ValueError("图片内容与声明的 MIME 类型不一致。")
            size = image.size
            if not _valid_dimensions(*size):
                raise ValueError("图片最多 2000 万像素。")
            if getattr(image, "n_frames", 1) != 1 or image.get_format_mimetype() == "image/apng":
                raise ValueError("图片附件不支持动图。")
            # Pillow's PNG verifier stops before checking the IEND checksum.
            if image.format == "PNG" and not data.endswith(_PNG_END):
                raise ValueError("PNG 图片结束块缺失或损坏。")
            image.verify()
        with Image.open(BytesIO(data), formats=tuple(_IMAGE_FORMATS)) as image:
            image.load()
    except Image.DecompressionBombError:
        raise ValueError("图片最多 2000 万像素。") from None
    except (OSError, SyntaxError, EOFError, IndexError, OverflowError):
        raise ValueError("图片已损坏或格式不受支持。") from None
    return size


@dataclass(frozen=True)
class ImageAttachment:
    id: str
    name: str
    size_bytes: int
    sha256: str
    mime_type: str
    width: int
    height: int

    def __post_init__(self) -> None:
        _name(self.name, image=True)
        if (
            not valid_id(self.id)
            or type(self.size_bytes) is not int or not 0 < self.size_bytes <= MAX_IMAGE_BYTES
            or not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
            or not isinstance(self.mime_type, str) or self.mime_type not in _IMAGE_EXTENSIONS
            or not _valid_dimensions(self.width, self.height)
        ):
            raise ValueError("图片附件元数据无效。")

    def to_json(self) -> dict:
        return {"type": "image", **asdict(self)}


def _document_format(name: str, mime_type: str, reading_mode: str) -> str:
    _name(name, document=True)
    extension = Path(name).suffix.lower()
    if extension not in DOCUMENT_MIME_TYPES or mime_type != DOCUMENT_MIME_TYPES[extension]:
        raise ValueError("文档须为 PDF、DOCX、PPTX 或 XLSX，文件扩展名与 MIME 类型必须一致。")
    if reading_mode not in ("text", "visual") or (extension == ".xlsx" and reading_mode != "text"):
        raise ValueError("文档阅读方式无效。XLSX 仅支持文字模式。")
    return extension


def _validate_document_signature(data: bytes, extension: str) -> None:
    if extension == ".pdf":
        if data.startswith(b"%PDF-") and b"%%EOF" in data[-1024:]:
            return
    elif data.startswith(b"PK\x03\x04"):
        try:
            with ZipFile(BytesIO(data)) as archive:
                # Inspect the container directory without opening or decompressing any part.
                archive.getinfo("[Content_Types].xml")
                archive.getinfo(_OFFICE_PARTS[extension])
            return
        except (BadZipFile, KeyError, UnicodeError, ValueError):
            pass
    raise ValueError("文档文件签名无效，文件可能已损坏或格式不符。")


@dataclass(frozen=True)
class DocumentAttachment:
    id: str
    name: str
    size_bytes: int
    sha256: str
    mime_type: str
    reading_mode: str

    def __post_init__(self) -> None:
        _document_format(self.name, self.mime_type, self.reading_mode)
        if (
            not valid_id(self.id)
            or type(self.size_bytes) is not int or not 0 < self.size_bytes <= MAX_DOCUMENT_BYTES
            or not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
        ):
            raise ValueError("文档附件元数据无效。")

    def to_json(self) -> dict:
        return {"type": "document", **asdict(self)}


Attachment = TextAttachment | ImageAttachment | DocumentAttachment


def attachment_only_prompt(items: Sequence[Attachment | AttachmentInput]) -> str:
    if any(isinstance(item, (DocumentAttachment, DocumentAttachmentInput)) for item in items):
        return DOCUMENT_ATTACHMENT_ONLY_PROMPT
    if any(isinstance(item, (ImageAttachment, ImageAttachmentInput)) for item in items):
        return IMAGE_ATTACHMENT_ONLY_PROMPT
    return ATTACHMENT_ONLY_PROMPT


@dataclass(frozen=True)
class AttachmentBatch:
    files: tuple[Attachment, ...]
    submission_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple) or not self.files or not all(
            isinstance(item, (TextAttachment, ImageAttachment, DocumentAttachment)) for item in self.files
        ):
            raise TypeError("附件引用无效。")
        if len(self.texts) > MAX_FILES:
            raise ValueError("每次最多可发送 10 个文本附件。")
        if sum(item.size_bytes for item in self.texts) > MAX_TOTAL_BYTES:
            raise ValueError("附件总大小超过 256 KiB。")
        if len(self.images) > MAX_IMAGES:
            raise ValueError("每次最多可发送 4 张图片。")
        if sum(item.size_bytes for item in self.images) > MAX_IMAGE_TOTAL_BYTES:
            raise ValueError("图片附件总大小超过 20 MiB。")
        if len(self.documents) > MAX_DOCUMENTS:
            raise ValueError("每轮最多可发送一份文档。")
        if self.submission_id is not None and not valid_id(self.submission_id):
            raise ValueError("附件发送标识无效。")

    def to_json(self) -> dict:
        return {"files": [item.to_json() for item in self.files], "submission_id": self.submission_id}

    @classmethod
    def from_json(cls, data: dict) -> "AttachmentBatch":
        if not isinstance(data, dict) or not isinstance(data.get("files"), list):
            raise TypeError("附件引用无效。")
        files = []
        for entry in data["files"]:
            if not isinstance(entry, dict):
                raise TypeError("附件引用无效。")
            item = entry.copy()
            match item.pop("type", "text"):
                case "text":
                    files.append(TextAttachment(**item))
                case "image":
                    files.append(ImageAttachment(**item))
                case "document":
                    files.append(DocumentAttachment(**item))
                case _:
                    raise ValueError("附件类型无效。")
        return cls(tuple(files), data.get("submission_id"))

    @property
    def texts(self) -> tuple[TextAttachment, ...]:
        return tuple(item for item in self.files if isinstance(item, TextAttachment))

    @property
    def images(self) -> tuple[ImageAttachment, ...]:
        return tuple(item for item in self.files if isinstance(item, ImageAttachment))

    @property
    def documents(self) -> tuple[DocumentAttachment, ...]:
        return tuple(item for item in self.files if isinstance(item, DocumentAttachment))


def prepare_attachments(uploads: Sequence[AttachmentInput], submission_id: str | None) -> AttachmentBatch:
    if not 0 < len(uploads) <= MAX_FILES + MAX_IMAGES + MAX_DOCUMENTS:
        raise ValueError("每次最多可发送 10 个文本附件、4 张图片和一份文档。")
    image_count = sum(isinstance(upload, ImageAttachmentInput) for upload in uploads)
    document_count = sum(isinstance(upload, DocumentAttachmentInput) for upload in uploads)
    if image_count > MAX_IMAGES:
        raise ValueError("每次最多可发送 4 张图片。")
    if document_count > MAX_DOCUMENTS:
        raise ValueError("每轮最多可发送一份文档。")
    if len(uploads) - image_count - document_count > MAX_FILES:
        raise ValueError("每次最多可发送 10 个文本附件。")
    files = []
    for upload in uploads:
        if isinstance(upload, DocumentAttachmentInput):
            extension = _document_format(upload.name, upload.mime_type, upload.reading_mode)
            if not isinstance(upload.data, bytes):
                raise TypeError("文档附件必须提供 bytes。")
            if not 0 < len(upload.data) <= MAX_DOCUMENT_BYTES:
                raise ValueError("文档附件不能为空，且每份最多 10 MiB。")
            _validate_document_signature(upload.data, extension)
            files.append(DocumentAttachment(
                uuid4().hex, upload.name, len(upload.data), sha256(upload.data).hexdigest(),
                upload.mime_type, upload.reading_mode,
            ))
            continue
        if isinstance(upload, ImageAttachmentInput):
            _name(upload.name, image=True)
            if not isinstance(upload.data, bytes):
                raise TypeError("图片附件必须提供 bytes。")
            if not 0 < len(upload.data) <= MAX_IMAGE_BYTES:
                raise ValueError("图片附件不能为空，且每张最多 5 MiB。")
            width, height = _image_dimensions(upload.data, upload.mime_type)
            files.append(ImageAttachment(
                uuid4().hex, upload.name, len(upload.data), sha256(upload.data).hexdigest(),
                upload.mime_type, width, height,
            ))
            continue
        if not isinstance(upload, TextAttachmentInput) or not isinstance(upload.text, str):
            raise TypeError("附件内容必须是 UTF-8 文本。")
        source = {}
        snapshot_id = uuid4().hex
        if isinstance(upload, TextResourceInput):
            source = {"server": upload.server, "uri": upload.uri, "mime_type": upload.mime_type, "source": upload.source}
            if upload.draft_id and submission_id:
                snapshot_id = resource_snapshot_id(submission_id, upload.draft_id)
        else:
            _name(upload.name)
        data = snapshot_bytes(upload.text)
        if re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", upload.text):
            raise ValueError("附件不能包含二进制内容或私钥。")
        files.append(TextAttachment(
            id=snapshot_id, name=upload.name, size_bytes=len(data),
            sha256=sha256(data).hexdigest(), lines=upload.text.count("\n") + 1, **source,
        ))
    return AttachmentBatch(tuple(files), submission_id)


def attachment_metadata(batch: AttachmentBatch) -> list[dict]:
    return [
        {"id": item.id, "name": item.model_name, "size_bytes": item.size_bytes, "lines": item.lines}
        if isinstance(item, TextAttachment) and item.source else item.to_json()
        for item in batch.files
    ]


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

    @staticmethod
    def _filename(item: Attachment) -> str:
        if isinstance(item, DocumentAttachment):
            return item.id + Path(item.name).suffix.lower()
        return item.id + (_IMAGE_EXTENSIONS[item.mime_type] if isinstance(item, ImageAttachment) else ".txt")

    def _read(self, directory: int, item: Attachment) -> bytes:
        fd = os.open(self._filename(item), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            self._private(stream.fileno())
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != item.size_bytes:
                raise OSError("附件快照不可用。")
            data = stream.read(item.size_bytes + 1)
        if len(data) != item.size_bytes or sha256(data).hexdigest() != item.sha256:
            raise OSError("附件快照校验失败。")
        if isinstance(item, ImageAttachment) and _image_dimensions(data, item.mime_type) != (item.width, item.height):
            raise OSError("图片快照尺寸校验失败。")
        return data

    def read(self, session_id: str, item: TextAttachment) -> str:
        with self._session_directory(session_id) as (_, directory):
            return decode_text(self._read(directory, item))

    def read_image(self, session_id: str, item: ImageAttachment) -> bytes:
        with self._session_directory(session_id) as (_, directory):
            return self._read(directory, item)

    def read_document(self, session_id: str, item: DocumentAttachment) -> bytes:
        with self._session_directory(session_id) as (_, directory):
            return self._read(directory, item)

    def write(self, session_id: str, batch: AttachmentBatch, uploads: Sequence[AttachmentInput]) -> None:
        with self._session_directory(session_id, create=True) as (_, directory):
            for item, upload in zip(batch.files, uploads, strict=True):
                temporary = ".tmp-" + item.id
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(upload.text.encode("utf-8") if isinstance(upload, TextAttachmentInput) else upload.data)
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o400)
                        os.fsync(stream.fileno())
                    # Publish without overwriting an existing immutable snapshot.
                    try:
                        os.link(temporary, self._filename(item), src_dir_fd=directory,
                                dst_dir_fd=directory, follow_symlinks=False)
                    except FileExistsError:
                        # A retried submission may already have published this snapshot.
                        pass
                    os.unlink(temporary, dir_fd=directory)
                    verified = self._read(directory, item)
                    if isinstance(item, TextAttachment):
                        decode_text(verified)
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
