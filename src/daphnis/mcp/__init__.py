"""MCP tools and prompts over SDK transports."""

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from typing import Any, TextIO

import httpx2
from mcp import Client, StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.client.subscriptions import ToolsListChanged
from mcp.types import (
    EmptyResult,
    GetPromptResult,
    PingRequest,
    Prompt,
    TextContent,
    Tool,
    ToolListChangedNotification,
)

from ..terminal_text import sanitize_terminal
from .public_http import public_transport, public_url

logger = logging.getLogger(__name__)
TOOL_PREFIX = "mcp__"
MAX_TOOL_RESULT_BYTES = 65_536
_CONNECT_TIMEOUT = 15
_CALL_TIMEOUT = 30
_PROMPT_TIMEOUT = 10
_PROMPT_BYTES = 65_536
_PROMPT_PAGES = 32
_PROMPT_FIELDS = 64
_HTTP_HEALTH_INTERVAL = 15
_HTTP_HEALTH_TIMEOUT = 5
_STDERR_LIMIT = 16 * 1024
_STDERR_EOF_TIMEOUT = 0.25


class _DiagnosticCapture(asyncio.Protocol):
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.truncated = False
        self.closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def data_received(self, data: bytes) -> None:
        remaining = _STDERR_LIMIT - len(self.buffer)
        self.buffer.extend(data[:remaining])
        self.truncated |= len(data) > remaining

    def connection_lost(self, exc: Exception | None) -> None:
        self.closed.set_result(None)

    def log(self, name: str) -> None:
        text = sanitize_terminal(self.buffer.decode("utf-8", errors="replace")).rstrip()
        if text or self.truncated:
            note = "\n[MCP stderr truncated]" if self.truncated else ""
            logger.warning("MCP %s stderr:\n%s%s", name, text, note)


@asynccontextmanager
async def _stdio_errlog(name: str) -> AsyncIterator[TextIO]:
    if os.name != "posix":
        # Anonymous pipes on these platforms need different async I/O support.
        # Keep the SDK's process lifecycle and fail closed for diagnostics.
        with os.fdopen(os.open(os.devnull, os.O_WRONLY), "w", encoding="utf-8") as errlog:
            yield errlog
        return

    read_fd, write_fd = os.pipe()
    with (
        os.fdopen(read_fd, "rb", buffering=0) as pipe,
        os.fdopen(write_fd, "w", encoding="utf-8") as errlog,
    ):
        capture = _DiagnosticCapture()
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: capture, pipe
        )
        try:
            yield errlog
        finally:
            errlog.close()
            try:
                # Normally the SDK has already reaped the child. A descendant
                # may still hold stderr open, so EOF must never delay teardown.
                await asyncio.wait((capture.closed,), timeout=_STDERR_EOF_TIMEOUT)
            finally:
                transport.close()
                if capture.buffer:
                    await asyncio.to_thread(capture.log, name)


def _json_size(value: Any, limit: int) -> int | None:
    remaining = limit
    for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(value):
        if len(chunk) > remaining:
            return None
        remaining -= len(chunk.encode("utf-8"))
        if remaining < 0:
            return None
    return limit - remaining


def _result_fits(value: dict[str, Any]) -> bool:
    # Match Saturnus ToolResult serialization, including UTF-8 and JSON escaping.
    return _json_size(value, MAX_TOOL_RESULT_BYTES) is not None


def _bounded_result(value: dict[str, Any]) -> dict[str, Any]:
    if _result_fits(value):
        return value
    omitted = [key for key in ("text", "structured_content") if key in value]
    limited = {key: item for key, item in value.items() if key not in omitted}
    limited.update(truncated=True, limit_bytes=MAX_TOOL_RESULT_BYTES, omitted_fields=omitted)
    for key in tuple(omitted):
        limited[key] = value[key]
        if _result_fits(limited):
            omitted.remove(key)
        else:
            del limited[key]
    return limited


@dataclass(frozen=True)
class MCPStatus:
    name: str
    state: str
    tool_count: int
    error: str | None


class MCPPromptError(RuntimeError):
    """A safe diagnostic for a failed prompt request."""


def _valid_prompt_name(name: str) -> bool:
    return isinstance(name, str) and 0 < len(name) <= 256 and bool(name.strip()) and not re.search(
        r"[\x00-\x1f\x7f-\x9f]", name,
    )


def validate_prompt_request(name: str, arguments: dict[str, str]) -> None:
    if not _valid_prompt_name(name) or not isinstance(arguments, dict) or len(arguments) > _PROMPT_FIELDS:
        raise ValueError("MCP 模板名称或参数无效。")
    if not all(_valid_prompt_name(key) and isinstance(value, str) for key, value in arguments.items()):
        raise ValueError("MCP 模板参数必须是字符串。")
    if _json_size(arguments, _PROMPT_BYTES) is None:
        raise ValueError("MCP 模板参数超出大小上限。")


class _WatchedReadStream:
    """Notify the owner when the SDK's receive stream reaches EOF."""

    def __init__(self, stream, eof: asyncio.Event) -> None:
        self._stream, self._eof = stream, eof

    def __getattr__(self, name):
        return getattr(self._stream, name)

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._stream.__aexit__(*exc)

    async def __aiter__(self):
        async for message in self._stream:
            yield message
        self._eof.set()


def _check_schema_references(value: Any) -> None:
    # Argument validation must not fetch URLs supplied by an external server.
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key in ("$ref", "$dynamicRef", "$recursiveRef")
                and isinstance(item, str)
                and not item.startswith("#")
            ):
                raise ValueError("MCP tool schemas must use local references.")
            _check_schema_references(item)
    elif isinstance(value, list):
        for item in value:
            _check_schema_references(item)


def _http_url(value: str | None) -> httpx2.URL:
    if not isinstance(value, str):
        raise TypeError("MCP URL 必须是字符串。")
    try:
        url = httpx2.URL(value)
    except (httpx2.InvalidURL, ValueError):
        raise ValueError("MCP URL 无效。") from None
    if not url.host or url.userinfo or url.fragment:
        raise ValueError("MCP URL 需要主机名，且不能包含凭据或片段。")
    if url.scheme == "http":
        loopback = url.host == "localhost"
        try:
            loopback = loopback or ip_address(url.host).is_loopback
        except ValueError:
            pass
        if loopback:
            return url
    if url.scheme != "https":
        raise ValueError("MCP 远程地址需要 HTTPS，本机回环地址允许 HTTP。")
    return url


class MCPConnection:
    def __init__(
        self,
        name: str,
        command: str | None = None,
        args: Sequence[str] = (),
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        transport: str = "stdio",
        url: str | None = None,
        auth: Mapping[str, str] | None = None,
        public_only: bool = False,
    ) -> None:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,20}", name):
            raise ValueError("MCP Server 名称须为 1–20 位字母、数字或连字符。")
        if transport not in ("stdio", "streamable_http"):
            raise ValueError("MCP transport 须为 stdio 或 streamable_http。")
        if transport == "stdio" and (
            not isinstance(command, str) or not command.strip() or "\0" in command
        ):
            raise ValueError("MCP command 必须是非空的可执行程序名。")
        if transport == "stdio" and (url is not None or auth is not None):
            raise ValueError("stdio 不使用 URL 或 HTTP 认证。")
        if transport == "streamable_http" and (
            command is not None or args or cwd is not None or env is not None
        ):
            raise ValueError("Streamable HTTP 不使用 command、args、cwd 或 env。")
        if not isinstance(args, (list, tuple)) or any(
            not isinstance(arg, str) or "\0" in arg for arg in args
        ):
            raise ValueError("MCP args 必须是字符串数组。")
        if cwd is not None and not isinstance(cwd, (str, Path)):
            raise TypeError("MCP cwd 必须是目录路径。")
        if env is not None and (
            not isinstance(env, Mapping)
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                for key, value in env.items()
            )
        ):
            raise ValueError("MCP env 必须将子进程变量名映射到本地变量名。")
        self.name = name
        self.prefix = f"{TOOL_PREFIX}{name}__"
        self._command, self._args, self._cwd = command, tuple(args), cwd
        self._env = dict(env or {})
        self._url = _http_url(url) if transport == "streamable_http" else None
        if type(public_only) is not bool or (public_only and self._url is None):
            raise ValueError("在线发现的 MCP 服务须使用 Streamable HTTP。")
        if public_only:
            public_url(self._url)
        self._public_only = public_only
        if auth is not None and (
            not isinstance(auth, Mapping)
            or set(auth) != {"type", "env"}
            or auth["type"] != "bearer"
            or not isinstance(auth["env"], str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", auth["env"])
        ):
            raise ValueError("MCP auth 只支持 bearer，env 须为本地变量名。")
        self._bearer_env = auth["env"] if auth is not None else None
        self._http_error: tuple[str, str] | None = None
        self._client: Client | None = None
        self.tools: dict[str, Tool] = {}
        self.state = "Disconnected"
        self.error: str | None = None
        self._stop = asyncio.Event()
        self._tools_changed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def status(self) -> MCPStatus:
        return MCPStatus(
            self.name,
            self.state,
            len(self.tools) if self.available else 0,
            self.error,
        )

    @property
    def _live(self) -> bool:
        return (
            self._task is not None
            and not self._task.done()
            and self._client is not None
            and not self._stop.is_set()
        )

    @property
    def available(self) -> bool:
        return self.state == "Connected" and self._live

    @property
    def supports_prompts(self) -> bool:
        client = self._client
        return self.available and client is not None and client.server_capabilities.prompts is not None

    def _can_publish(self) -> bool:
        return self.state == "Connecting" and self._live

    async def start(
        self,
        on_error: Callable[[str, str], Awaitable[None]],
        on_tools_changed: Callable[["MCPConnection"], Awaitable[None]],
    ) -> dict[str, Tool]:
        self.state, self.error = "Connecting", None
        self._http_error = None
        self._stop.clear()
        self._tools_changed.clear()
        ready = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._serve(ready, on_error, on_tools_changed), name=f"mcp:{self.name}"
        )
        self._task = task
        started = False
        try:
            async with asyncio.timeout(_CONNECT_TIMEOUT):
                await asyncio.wait((ready, task), return_when=asyncio.FIRST_COMPLETED)
                if task.done():
                    await task
                    raise RuntimeError("MCP 连接在启动时关闭。")
            started = True
            return ready.result()
        except asyncio.CancelledError:
            self.state = "Disconnected"
            raise
        except TimeoutError:
            self.error = "连接或发现工具超时。"
        except Exception as exc:  # noqa: BLE001 - keep SDK and subprocess details out of errors.
            self.error = (
                str(exc)
                if isinstance(exc, _MCPConfigError)
                else "无法连接或读取工具列表。"
            )
        finally:
            if not started:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if self._http_error:
            self.error = self._http_error[1]
        self.state = "Error"
        raise RuntimeError(self.error)

    def _stdio_parameters(self) -> StdioServerParameters:
        if self._cwd is not None and not Path(self._cwd).is_dir():
            raise _MCPConfigError("MCP 工作目录不存在。")
        env = {}
        for target, source in self._env.items():
            value = os.getenv(source)
            if not value:
                raise _MCPConfigError(f"缺少 MCP 环境变量 {source}。")
            env[target] = value
        return StdioServerParameters(
            command=self._command, args=list(self._args), cwd=self._cwd, env=env
        )

    @asynccontextmanager
    async def _transport(self):
        if self._url is None:
            parameters = self._stdio_parameters()
            async with (
                _stdio_errlog(self.name) as errlog,
                stdio_client(parameters, errlog=errlog) as (read, write),
            ):
                errlog.close()  # The child owns its copy after the SDK spawns it.
                yield _WatchedReadStream(read, self._stop), write
            return

        token = None
        if self._bearer_env is not None:
            token = os.getenv(self._bearer_env)
            if not token:
                raise _MCPConfigError(f"缺少 MCP 环境变量 {self._bearer_env}。")
            if not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", token):
                raise _MCPConfigError("MCP Bearer Token 格式无效。")

        origin = (self._url.scheme, self._url.host, self._url.port)

        async def authorize(request):
            if (request.url.scheme, request.url.host, request.url.port) != origin:
                raise httpx2.RequestError("MCP 跨来源请求已拦截。")
            request.headers.pop("Authorization", None)
            if token is not None:
                request.headers["Authorization"] = f"Bearer {token}"

        async def check_response(response):
            status = response.status_code
            if status in (401, 403):
                suffix = "在线发现仅支持 Bearer Token。" if self._public_only else ""
                error = (
                    "mcp_auth_failed",
                    f"MCP 认证失败（HTTP {status}）。{suffix}",
                )
            elif response.is_redirect:
                error = (
                    "mcp_redirect",
                    "MCP 地址发生重定向，请配置最终地址。",
                )
            else:
                return
            if response.request.method == "POST":
                with suppress(ValueError, httpx2.RequestNotRead):
                    method = json.loads(response.request.content).get("method")
                    if method in {"prompts/list", "prompts/get"}:
                        # Let the SDK reject just this request without following a redirect.
                        response.status_code = 403
                        response.headers.pop("Location", None)
                        return
            self._http_error = error
            # Stop before the SDK can follow or log a server-controlled Location.
            raise httpx2.RequestError(error[1])

        try:
            transport = await public_transport(self._url) if self._public_only else None
        except ValueError as exc:
            raise _MCPConfigError(str(exc)) from None

        async with (
            httpx2.AsyncClient(
                timeout=httpx2.Timeout(8, read=300),
                trust_env=False,
                transport=transport,
                event_hooks={"request": [authorize], "response": [check_response]},
            ) as http,
            streamable_http_client(str(self._url), http_client=http) as (read, write),
        ):
            yield _WatchedReadStream(read, self._stop), write

    async def _wait_for_stop(self, client: Client) -> None:
        if self._url is None:
            await self._stop.wait()
            return
        # Stateless HTTP has no persistent channel to report EOF while idle.
        while not self._stop.is_set():
            try:
                async with asyncio.timeout(_HTTP_HEALTH_INTERVAL):
                    await self._stop.wait()
            except TimeoutError:
                async with asyncio.timeout(_HTTP_HEALTH_TIMEOUT):
                    if client.session.discover_result is not None:
                        await client.session.send_discover(client.protocol_version)
                    else:
                        await client.session.send_request(PingRequest(), EmptyResult)

    async def discover(self) -> dict[str, Tool]:
        client = self._client
        if client is None:
            raise RuntimeError("MCP 连接不可用。")
        async with asyncio.timeout(_CONNECT_TIMEOUT):
            return await self._discover(client)

    async def _discover(self, client: Client) -> dict[str, Tool]:
        if client.server_capabilities.tools is None:
            return {}
        tools = {}
        cursor = None
        seen_cursors = set()
        original_names = set()
        while True:
            page = await client.list_tools(cursor=cursor, cache_mode="bypass")
            for tool in page.tools:
                _check_schema_references(tool.input_schema)
                alias = self.prefix + tool.name
                if (
                    len(alias) > 64
                    or tool.name.startswith("__")
                    or not re.fullmatch(r"[A-Za-z0-9_-]+", tool.name)
                ):
                    alias = (
                        self.prefix + "__" + sha256(tool.name.encode()).hexdigest()[:32]
                    )
                if alias in tools or tool.name in original_names:
                    raise ValueError("MCP Server 返回了重复的工具名称。")
                tools[alias] = tool
                original_names.add(tool.name)
            cursor = page.next_cursor
            if cursor is None:
                return tools
            if cursor in seen_cursors:
                raise ValueError("MCP Server 返回了重复的分页游标。")
            seen_cursors.add(cursor)

    @asynccontextmanager
    async def _prompt_request(self) -> AsyncIterator[Client]:
        client = self._client
        if not self.available or client is None:
            raise MCPPromptError("MCP 连接不可用。")
        if client.server_capabilities.prompts is None:
            raise MCPPromptError("此 MCP 服务不支持模板。")
        try:
            async with asyncio.timeout(_PROMPT_TIMEOUT):
                yield client
        except TimeoutError:
            raise MCPPromptError("MCP 模板请求超时。") from None
        except MCPPromptError:
            raise
        except Exception:  # noqa: BLE001 - SDK failures may contain server data or user arguments.
            if not self.available or self._client is not client:
                raise MCPPromptError("MCP 连接已中断。") from None
            raise MCPPromptError("MCP 模板请求失败或响应无效。") from None
        if not self.available or self._client is not client:
            raise MCPPromptError("MCP 连接已中断。")

    async def list_prompts(self) -> tuple[Prompt, ...]:
        async with self._prompt_request() as client:
            prompts: dict[str, Prompt] = {}
            cursor = None
            seen_cursors: set[str] = set()
            remaining = _PROMPT_BYTES
            for _ in range(_PROMPT_PAGES):
                page = await client.list_prompts(cursor=cursor, cache_mode="bypass")
                size = _json_size(page.model_dump(mode="json", by_alias=True), remaining)
                if size is None:
                    raise MCPPromptError("MCP 模板列表超出大小上限。")
                remaining -= size
                if page.result_type != "complete":
                    raise MCPPromptError("MCP 模板列表响应无效。")
                for prompt in page.prompts:
                    arguments = [arg.name for arg in prompt.arguments or ()]
                    if (
                        not _valid_prompt_name(prompt.name) or prompt.name in prompts
                        or len(arguments) > _PROMPT_FIELDS or len(set(arguments)) != len(arguments)
                        or not all(map(_valid_prompt_name, arguments))
                    ):
                        raise MCPPromptError("MCP 模板定义无效或名称重复。")
                    prompts[prompt.name] = prompt
                cursor = page.next_cursor
                if cursor is None:
                    return tuple(prompts.values())
                if cursor in seen_cursors:
                    raise MCPPromptError("MCP 模板列表返回了重复的分页游标。")
                seen_cursors.add(cursor)
            raise MCPPromptError("MCP 模板列表超出分页上限。")

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> GetPromptResult:
        validate_prompt_request(name, arguments)
        async with self._prompt_request() as client:
            # Do not drive server-requested sampling, elicitation, or additional input.
            result = await client.session.get_prompt(name, arguments)
            if result.result_type != "complete" or not result.messages:
                raise MCPPromptError("MCP 模板结果无效。")
            if _json_size(result.model_dump(mode="json", by_alias=True), _PROMPT_BYTES) is None:
                raise MCPPromptError("MCP 模板结果超出大小上限。")
            return result

    async def _serve(
        self,
        ready: asyncio.Future[dict[str, Tool]],
        on_error: Callable[[str, str], Awaitable[None]],
        on_tools_changed: Callable[["MCPConnection"], Awaitable[None]],
    ) -> None:
        # The SDK enters and exits its cancel scopes in this same task.
        try:
            async with Client(
                self._transport(), message_handler=self._on_message
            ) as client:
                self._client = client
                ready.set_result(await self._discover(client))
                watcher = asyncio.create_task(
                    self._watch_tools(client, on_tools_changed)
                )
                try:
                    await self._wait_for_stop(client)
                finally:
                    self._stop.set()
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
                if self.state != "Disconnected":
                    raise ConnectionError("MCP transport closed.")
        except Exception:
            if not ready.done():
                raise
            if self.state != "Disconnected":
                self.state = "Error"
                self.error = (
                    self._http_error[1]
                    if self._http_error
                    else "连接意外中断或无响应。"
                )
                await on_error(self.name, self.error)
        finally:
            self._client = None

    async def _on_message(self, message) -> None:
        if isinstance(message, ToolListChangedNotification):
            self._tools_changed.set()

    async def _watch_tools(
        self, client: Client, on_changed: Callable[["MCPConnection"], Awaitable[None]]
    ) -> None:
        if client.server_capabilities.tools is None:
            return
        if client.session.discover_result is not None:
            # Unsupported or interrupted subscriptions leave manual refresh available.
            try:
                async with asyncio.timeout(_CONNECT_TIMEOUT) as ack:
                    async with client.listen(tools_list_changed=True) as subscription:
                        ack.reschedule(None)
                        if subscription.honored.tools_list_changed:
                            async for event in subscription:
                                if isinstance(event, ToolsListChanged):
                                    await on_changed(self)
            except Exception:  # noqa: BLE001 - subscriptions are optional SDK capabilities.
                return
        elif (tools := client.server_capabilities.tools) and tools.list_changed:
            while True:
                await self._tools_changed.wait()
                self._tools_changed.clear()
                await on_changed(self)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        if not self.available or client is None:
            return {
                "ok": False,
                "error": {"code": "mcp_unavailable", "message": "MCP 连接不可用。"},
            }
        try:
            async with asyncio.timeout(_CALL_TIMEOUT):
                result = await client.session.call_tool(name, arguments)
        except TimeoutError:
            return {
                "ok": False,
                "error": {"code": "mcp_timeout", "message": "MCP 工具调用超时。"},
            }
        except Exception as exc:  # noqa: BLE001 - SDK protocol and transport failures are tool results.
            code, message = self._http_error or (
                "mcp_call_failed",
                f"MCP 工具调用失败（{type(exc).__name__}）。",
            )
            return {
                "ok": False,
                "error": {"code": code, "message": message},
            }
        value: dict[str, Any] = {"ok": not result.is_error}
        text = [
            block.text for block in result.content if isinstance(block, TextContent)
        ]
        if text:
            value["text"] = "\n".join(text)
        if result.structured_content is not None:
            value["structured_content"] = result.structured_content
        if omitted := len(result.content) - len(text):
            value["omitted_content"] = omitted
        if result.is_error:
            value["error"] = {
                "code": "mcp_tool_error",
                "message": "MCP Server 返回工具错误。",
            }
        return _bounded_result(value)

    async def aclose(self) -> None:
        self.state, self.error = "Disconnected", None
        if self._task is not None:
            self._stop.set()
            settled = asyncio.gather(self._task, return_exceptions=True)
            try:
                await asyncio.shield(settled)
            except asyncio.CancelledError:
                await asyncio.shield(settled)
                raise


class _MCPConfigError(ValueError):
    """Safe configuration diagnostics, with variable names but no values."""
