"""Read and validate public metadata from the official MCP Registry."""

import asyncio
import json
import re
from urllib.parse import quote

import httpx2

from .mcp.public_http import public_url

_BASE = "https://registry.modelcontextprotocol.io/v0.1"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_MAX_RESPONSE = 2 * 1024 * 1024
_OFFICIAL = "io.modelcontextprotocol.registry/official"


class RegistryError(RuntimeError):
    """A safe message for an unavailable Registry or unusable response."""


def _text(value, maximum=500):
    return value[:maximum] if isinstance(value, str) and not any(ord(char) < 32 for char in value) else ""


def _items(value):
    return value[:30] if isinstance(value, list) else []


def _object(value):
    return value if isinstance(value, dict) else {}


def _remote(value):
    if not isinstance(value, dict) or not isinstance(value.get("url"), str):
        return None
    url = _text(value["url"], 2048)
    transport = _text(value.get("type"), 40)
    if not transport or not url or url != value["url"]:
        return None
    auth = "none"
    if transport != "streamable-http":
        return {"transport": transport, "url": url, "auth": "unsupported", "reason": "仅支持 Streamable HTTP。"}
    try:
        public_url(httpx2.URL(url))
    except (httpx2.InvalidURL, ValueError):
        return {"transport": transport, "url": url, "auth": "unsupported", "reason": "需要可校验的公网 HTTPS 地址。"}
    headers = value.get("headers", [])
    if "{" in url or "}" in url or "variables" in value or not isinstance(headers, list) or len(headers) > 1:
        auth = "unsupported"
    elif headers:
        header = _object(headers[0])
        if str(header.get("name", "")).lower() == "authorization" and (
            isinstance(header.get("value"), str)
            and re.fullmatch(r"Bearer \{[A-Za-z0-9_]+\}", header["value"], flags=re.IGNORECASE)
        ):
            auth = "bearer"
        else:
            auth = "unsupported"
    return {"transport": transport, "url": url, "auth": auth,
            "reason": "该地址需要当前未支持的认证头或模板参数。" if auth == "unsupported" else None}


def _server(record):
    record = _object(record)
    server = _object(record.get("server"))
    name = server.get("name")
    version = server.get("version")
    if not isinstance(name, str) or len(name) > 255 or not _NAME.fullmatch(name):
        return None
    if not isinstance(version, str) or len(version) > 100 or not _VERSION.fullmatch(version):
        return None
    official = _object(_object(record.get("_meta")).get(_OFFICIAL))
    repository = _object(server.get("repository"))
    return {
        "name": name, "version": version, "publisher": name.split("/")[0],
        "title": _text(server.get("title"), 160), "description": _text(server.get("description"), 2000),
        "repository": _text(repository.get("url"), 1024), "source": _text(repository.get("source"), 100),
        "status": _text(official.get("status"), 40),
        "remotes": [remote for item in _items(server.get("remotes")) if (remote := _remote(item)) is not None],
        "packages": [
            {"type": _text(item.get("registryType"), 40), "identifier": _text(item["identifier"], 200),
             "version": _text(item.get("version"), 100)}
            for item in _items(server.get("packages"))
            if isinstance(item, dict) and _text(item.get("identifier"), 200)
        ],
    }


class RegistryDiscovery:
    def __init__(self, *, transport: httpx2.AsyncBaseTransport | None = None):
        self.transport = transport

    async def _get(self, path, params=None):
        try:
            async with (
                asyncio.timeout(10),
                httpx2.AsyncClient(transport=self.transport, trust_env=False, timeout=6) as client,
                client.stream("GET", _BASE + path, params=params, headers={"Accept": "application/json"}) as response,
            ):
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_RESPONSE:
                        raise RegistryError("官方 MCP Registry 返回的数据过大。")
                    body.extend(chunk)
                return _object(json.loads(body))
        except (TimeoutError, httpx2.HTTPError, ValueError, RecursionError):
            raise RegistryError("官方 MCP Registry 不可用或返回了无效数据，请稍后重试。") from None

    async def search(self, query: str, cursor: str | None):
        query = query.strip()
        if not 3 <= len(query) <= 160 or (cursor is not None and not 0 < len(cursor) <= 4096):
            raise ValueError("请输入 3–160 个字符的关键词。")
        params = {"search": query, "version": "latest", "limit": 15}
        if cursor is not None:
            params["cursor"] = cursor
        data = await self._get("/servers", params)
        metadata = data.get("metadata")
        if not isinstance(data.get("servers"), list) or not isinstance(metadata, dict):
            raise RegistryError("官方 MCP Registry 返回了无效的搜索结果。")
        next_cursor = metadata.get("nextCursor")
        if next_cursor is not None and (not isinstance(next_cursor, str) or len(next_cursor) > 4096):
            raise RegistryError("官方 MCP Registry 返回了无效的分页标识。")
        return {
            "servers": [server for record in data["servers"][:15] if (server := _server(record)) is not None],
            "cursor": next_cursor or None,
        }

    async def detail(self, name: str, version: str):
        if len(name) > 255 or len(version) > 100 or not _NAME.fullmatch(name) or not _VERSION.fullmatch(version):
            raise ValueError("无效的 Registry 服务名称或版本。")
        server = _server(await self._get(f"/servers/{quote(name, safe='')}/versions/{quote(version, safe='')}"))
        if not server or server["name"] != name or (version != "latest" and server["version"] != version):
            raise RegistryError("官方 MCP Registry 返回了无效的服务详情。")
        return server
