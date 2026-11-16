"""DeepSeek Files API and account-scoped cache of parsed document references."""

import asyncio
import json
import os
import re
import sqlite3
import stat
from collections.abc import Callable, Mapping
from contextlib import closing, contextmanager
from hashlib import sha256
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

import httpx

from ..attachments import DocumentAttachment
from .documents import DocumentInputError, PreparedDocuments

_LIFETIME = 30 * 24 * 60 * 60
_CONFIRMATION_WINDOW = 300
_FILE_ID = re.compile(r"file-api-[a-fA-F0-9-]{32,36}")
_MISSING = "the following file_ids do not exist or are not created under your account:"
_UPLOAD_ERRORS = {
    400: "文档无法解析，请检查格式、文件完整性和密码保护。",
    401: "文档服务鉴权失败，请检查模型连接配置。",
    402: "文档解析需要可用的服务额度。",
    403: "当前连接无权上传文档。",
    404: "文档上传接口不可用，请检查模型连接配置。",
    408: "文档上传超时，请稍后重试。",
    413: "文档超过解析服务的大小上限。",
    429: "文档解析请求过多，请稍后重试。",
    500: "文档解析失败，请稍后重试。",
    503: "文档解析服务繁忙，请稍后重试。",
    504: "文档解析超时，请缩小文档后再试。",
}
_UNKNOWN = "远端文档上传结果尚未确认。稍后重试时会先核对远端记录。"


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _payload(response: httpx.Response) -> dict:
    try:
        result = response.json()
    except (ValueError, RuntimeError):
        return {}
    return result if isinstance(result, dict) else {}


class DeepSeekFiles:
    def __init__(self, connection: Mapping[str, Any], cache_directory: Path) -> None:
        self._endpoint = httpx.URL(connection["base_url"]).join("/files")
        headers = dict(connection.get("auth_headers") or {})
        if key := connection.get("api_key"):
            headers["Authorization"] = f"Bearer {key}"
        self._scope = _digest([str(self._endpoint), sorted((key.lower(), value) for key, value in headers.items())])
        self._cache_path = cache_directory / "deepseek-files.sqlite3"
        self._upload_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._http = httpx.AsyncClient(
            headers=headers, transport=connection.get("transport"),
            timeout=httpx.Timeout(150.0, connect=15.0, write=125.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @contextmanager
    def _database(self):
        try:
            self._cache_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self._cache_path, os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_RDWR, 0o600)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077):
                    raise OSError("Document cache must be private.")
            finally:
                os.close(fd)
            with closing(sqlite3.connect(self._cache_path, timeout=1)) as db, db:
                db.row_factory = sqlite3.Row
                db.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        cache_key TEXT PRIMARY KEY, scope TEXT NOT NULL, name TEXT NOT NULL,
                        created_at INTEGER NOT NULL, file_id TEXT, expires_at INTEGER, available INTEGER NOT NULL
                    )
                """)
                yield db
        except (OSError, sqlite3.Error):
            raise DocumentInputError("无法读取或保存文档的远端引用缓存。") from None

    @staticmethod
    def _ready(entry) -> bool:
        return bool(entry and entry["available"] and entry["expires_at"] > time())

    def _lookup(self, key: str):
        with self._database() as db:
            return db.execute("SELECT * FROM documents WHERE cache_key = ?", (key,)).fetchone()

    def _claim(self, key: str, name: str):
        with self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            entry = db.execute("SELECT * FROM documents WHERE cache_key = ?", (key,)).fetchone()
            if self._ready(entry) or (entry and entry["file_id"] is None):
                return entry
            db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, NULL, NULL, 0)",
                (key, self._scope, name, int(time())),
            )
        return None

    def _forget_pending(self, key: str, name: str) -> None:
        with self._database() as db:
            db.execute("DELETE FROM documents WHERE cache_key = ? AND name = ? AND file_id IS NULL", (key, name))

    def _record(self, key: str, name: str, obj: dict, purpose: str, size: int) -> str:
        ref, created, expiry = obj.get("id"), obj.get("created_at"), obj.get("expires_at")
        remote_size = obj.get("bytes")
        if (
            not isinstance(ref, str) or not _FILE_ID.fullmatch(ref)
            or obj.get("filename") != name or obj.get("purpose") != purpose
            or type(remote_size) is not int or remote_size < 0
            # Spreadsheet uploads report the size after conversion to Markdown.
            or (purpose != "user_data" and remote_size != size)
            or type(created) is not int or type(expiry) is not int or not 0 < created < expiry
        ):
            raise DocumentInputError(_UNKNOWN)
        with self._database() as db:
            db.execute(
                "UPDATE documents SET file_id = ?, created_at = ?, expires_at = ?, available = 1 "
                "WHERE cache_key = ? AND name = ? AND file_id IS NULL",
                (ref, created, expiry, key, name),
            )
        if expiry <= time():
            raise DocumentInputError("远端文档已过期，请重试。")
        return ref

    async def _reconcile(self, key: str, entry, purpose: str, size: int) -> str | None:
        try:
            response = await self._http.get(self._endpoint, params={"purpose": purpose, "order": "desc", "limit": 1000})
        except httpx.RequestError:
            raise DocumentInputError(_UNKNOWN) from None
        if response.status_code != 200:
            raise DocumentInputError(_UNKNOWN)
        payload = _payload(response)
        files = payload.get("data")
        if not isinstance(files, list):
            raise DocumentInputError(_UNKNOWN)
        for obj in files:
            if isinstance(obj, dict) and obj.get("filename") == entry["name"]:
                return self._record(key, entry["name"], obj, purpose, size)
        # Upload plus parsing can take four minutes. Re-upload only on an explicit retry
        # after that window, when a complete list confirms that the attempt is absent.
        if payload.get("has_more") is not False or time() - entry["created_at"] < _CONFIRMATION_WINDOW:
            raise DocumentInputError(_UNKNOWN)
        self._forget_pending(key, entry["name"])
        return None

    async def _reference(self, item: DocumentAttachment, read: Callable[[DocumentAttachment], bytes]) -> str:
        extension = Path(item.name).suffix.lower()
        if extension == ".xlsx":
            purpose = "user_data"
        else:
            purpose = "rich_layout_recognition" if item.reading_mode == "visual" else "text_recognition"
        key = _digest([self._scope, item.sha256, item.mime_type, purpose])
        # A second Session can reuse the first upload once the provider has replied.
        lock = self._upload_locks.get(key)
        if lock is None:
            lock = self._upload_locks[key] = asyncio.Lock()
        async with lock:
            return await self._lookup_or_upload(key, item, read, purpose, extension)

    async def _lookup_or_upload(
        self, key: str, item: DocumentAttachment, read: Callable[[DocumentAttachment], bytes],
        purpose: str, extension: str,
    ) -> str:
        entry = self._lookup(key)
        if self._ready(entry):
            return entry["file_id"]
        if entry and entry["file_id"] is None:
            ref = await self._reconcile(key, entry, purpose, item.size_bytes)
            if ref is not None:
                return ref
        try:
            data = await asyncio.to_thread(read, item)
        except (OSError, ValueError):
            raise DocumentInputError("原始文档快照不可用。") from None
        name = f"daphnis-{uuid4().hex}{extension}"
        entry = self._claim(key, name)
        if self._ready(entry):
            return entry["file_id"]
        if entry is not None:
            raise DocumentInputError(_UNKNOWN)
        try:
            response = await self._http.post(
                self._endpoint,
                data={"purpose": purpose, "expires_after[anchor]": "created_at",
                      "expires_after[seconds]": str(_LIFETIME)},
                files={"file": (name, data, item.mime_type)},
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            self._forget_pending(key, name)
            raise DocumentInputError("无法连接文档上传服务，请稍后重试。") from None
        except httpx.RequestError:
            raise DocumentInputError(_UNKNOWN) from None
        if response.status_code in _UPLOAD_ERRORS:
            self._forget_pending(key, name)
            raise DocumentInputError(_UPLOAD_ERRORS[response.status_code])
        if response.status_code != 200:
            raise DocumentInputError(_UNKNOWN)
        return self._record(key, name, _payload(response), purpose, item.size_bytes)

    async def prepare(
        self, messages: list[dict[str, Any]], documents: dict[int, tuple[DocumentAttachment, ...]],
        read: Callable[[DocumentAttachment], bytes],
    ) -> PreparedDocuments:
        if sum(map(len, documents.values())) > 20:
            raise DocumentInputError("有效上下文中的文档超过解析服务上限，请先压缩历史。")
        prepared = messages.copy()
        refs = []
        for index, items in documents.items():
            message = messages[index]
            content = message.get("content")
            parts = content.copy() if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for item in items:
                ref = await self._reference(item, read)
                refs.append(ref)
                parts.append({"type": "file", "file_id": ref})
            prepared[index] = {**message, "content": parts}
        return PreparedDocuments(prepared, tuple(refs))

    async def invalidate_missing(self, response: httpx.Response, prepared: PreparedDocuments) -> bool:
        error = _payload(response).get("error")
        if response.status_code != 400 or not isinstance(error, dict):
            return False
        message = error.get("message")
        if not isinstance(message, str) or _MISSING not in message.lower():
            return False
        missing = {match.lower().replace("-", "") for match in _FILE_ID.findall(message)}
        refs = [ref for ref in prepared.private_values if ref.lower().replace("-", "") in missing]
        if not refs:
            return False
        with self._database() as db:
            db.executemany(
                "UPDATE documents SET available = 0 WHERE scope = ? AND file_id = ?",
                ((self._scope, ref) for ref in refs),
            )
        return True
