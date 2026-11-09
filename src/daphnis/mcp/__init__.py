"""Local MCP tool discovery and calls over the SDK's stdio transport."""

import asyncio
import re
from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.types import TextContent, Tool

TOOL_PREFIX = "mcp__"
_CONNECT_TIMEOUT = 15
_CALL_TIMEOUT = 30


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


class MCPConnection:
    def __init__(self, name: str, command: str, args: Sequence[str]) -> None:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,20}", name):
            raise ValueError("MCP Server 名称须为 1–20 位字母、数字或连字符。")
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise ValueError("MCP command 必须是非空的可执行程序名。")
        if not isinstance(args, (list, tuple)) or any(
            not isinstance(arg, str) or "\0" in arg for arg in args
        ):
            raise ValueError("MCP args 必须是字符串数组。")
        self.name = name
        # The SDK inherits only its default OS allowlist; no application secrets are passed.
        self._client = Client(StdioServerParameters(command=command, args=list(args)))
        self.tools: dict[str, Tool] = {}
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        ready = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._serve(ready), name=f"mcp:{self.name}")
        self._task = task
        connected = False
        try:
            async with asyncio.timeout(_CONNECT_TIMEOUT):
                await asyncio.wait((ready, task), return_when=asyncio.FIRST_COMPLETED)
                if task.done():
                    await task
                    raise RuntimeError("MCP connection closed during startup.")
            connected = True
        except TimeoutError as exc:
            raise TimeoutError(
                f"MCP Server {self.name!r} 连接或发现工具超时。"
            ) from exc
        except ExceptionGroup as exc:
            raise RuntimeError(
                f"MCP Server {self.name!r} 无法连接或读取工具列表。"
            ) from exc
        finally:
            if not connected:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _serve(self, ready: asyncio.Future[None]) -> None:
        # The SDK enters and exits AnyIO cancel scopes in this same task.
        async with self._client as client:
            cursor = None
            seen_cursors = set()
            original_names = set()
            prefix = f"{TOOL_PREFIX}{self.name}__"
            while True:
                page = await client.list_tools(cursor=cursor)
                for tool in page.tools:
                    _check_schema_references(tool.input_schema)
                    alias = prefix + tool.name
                    if (
                        len(alias) > 64
                        or tool.name.startswith("__")
                        or not re.fullmatch(r"[A-Za-z0-9_-]+", tool.name)
                    ):
                        digest = sha256(tool.name.encode()).hexdigest()[:32]
                        alias = prefix + "__" + digest
                    if alias in self.tools or tool.name in original_names:
                        raise ValueError(
                            f"MCP Server {self.name!r} 返回了重复的工具名称。"
                        )
                    self.tools[alias] = tool
                    original_names.add(tool.name)
                cursor = page.next_cursor
                if cursor is None:
                    break
                if cursor in seen_cursors:
                    raise ValueError(f"MCP Server {self.name!r} 返回了重复的分页游标。")
                seen_cursors.add(cursor)
            ready.set_result(None)
            await self._stop.wait()

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            async with asyncio.timeout(_CALL_TIMEOUT):
                result = await self._client.call_tool(name, arguments)
        except TimeoutError:
            return {
                "ok": False,
                "error": {"code": "mcp_timeout", "message": "MCP 工具调用超时。"},
            }
        except Exception as exc:  # noqa: BLE001 - SDK protocol and transport failures are tool results.
            return {
                "ok": False,
                "error": {
                    "code": "mcp_call_failed",
                    "message": f"MCP 工具调用失败（{type(exc).__name__}）。",
                },
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
        return value

    async def aclose(self) -> None:
        if self._task is not None:
            self._stop.set()
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError:
                await asyncio.gather(asyncio.shield(self._task), return_exceptions=True)
                raise
