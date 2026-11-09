"""Own Daphnis runtime resources and supervise Turn execution."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

from saturnus import ChatClient

from .agent import Agent, TurnResult, TurnStatus
from .context import Context
from .events import EventBus
from .history import History
from .mcp import MCPConnection
from .models import (
    MODELS,
    ModelInfo,
    provider_compatibility,
    resolve_model,
)
from .models.models import validate_catalog
from .session import Session
from .tools import ApprovalRequest, ToolExecutor, create_tools
from .usage import UsageTracker

logger = logging.getLogger(__name__)


class SessionSaveError(RuntimeError):
    """Saving committed state or API-reported token accounting failed."""

    def __init__(self, result: TurnResult, error: Exception) -> None:
        message = (
            "本轮已完成，磁盘保存失败，内存历史保留"
            if result.status == TurnStatus.COMPLETED
            else "本轮未提交，已报告的用量保存失败"
        )
        super().__init__(f"{message}：{error}")
        self.result = result


class Application:
    def __init__(
        self,
        *,
        workspace: str | Path,
        provider: str,
        model_id: str,
        connections: Mapping[str, dict[str, Any]],
        history: History | None = None,
        catalog: Sequence[ModelInfo] = MODELS,
        interactive: bool = False,
    ) -> None:
        self._catalog = tuple(catalog)
        validate_catalog(self._catalog)
        self._connections = {
            name: dict(options) for name, options in connections.items()
        }
        self._clients: dict[str, ChatClient] = {}
        self._closed = False
        self._history = history if history is not None else History()
        self._interactive = interactive
        self._task: asyncio.Task[TurnResult] | None = None
        self._approval: asyncio.Future[bool] | None = None
        self._pending_approval: ApprovalRequest | None = None
        self._mcp: dict[str, MCPConnection] = {}
        self._mcp_lock = asyncio.Lock()
        self._mcp_pending: set[MCPConnection] = set()
        self._tools = create_tools(approval=self._request_approval)
        model = self._model_for(provider, model_id)
        self._session = Session(
            workspace=workspace,
            provider=provider,
            model_id=model_id,
            enabled_tools=self._all_tools if model.supports_tools is True else (),
        )
        if not Path(self._session.workspace).is_dir():
            raise ValueError("Workspace 必须是现有目录。")
        self.events = EventBus()
        self._usage = UsageTracker(self.events)
        self._usage.select_session(self._session.id, self._session.usage)
        self._client(provider)

    def _connection(self, provider: str) -> dict[str, Any]:
        if provider not in self._connections:
            raise ValueError(f"厂商 {provider} 未配置独立连接。")
        return self._connections[provider]

    def _client(self, provider: str) -> ChatClient:
        if provider not in self._clients:
            options = {"timeout": 60.0, **self._connection(provider)}
            options["compatibility"] = provider_compatibility(
                provider, override=options.get("compatibility")
            )
            self._clients[provider] = ChatClient(**options)
        return self._clients[provider]

    def _model_for(self, provider: str, model_id: str) -> ModelInfo:
        self._connection(provider)
        provider_compatibility(provider)
        return resolve_model(provider, model_id, catalog=self._catalog)

    @property
    def _model(self) -> ModelInfo:
        return resolve_model(
            self._session.provider, self._session.model_id, catalog=self._catalog
        )

    @property
    def _running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Application 已关闭。")

    def _ensure_idle(self) -> None:
        self._ensure_open()
        if self._running:
            raise RuntimeError("请先等待或取消当前 Turn，再执行该操作。")
        if self._mcp_lock.locked():
            raise RuntimeError("请等待 MCP 连接操作完成。")

    @property
    def _all_tools(self) -> tuple[str, ...]:
        offline = tuple(
            server.prefix for server in self._mcp.values() if server.state != "Connected"
        )
        return tuple(
            name
            for item in self._tools.definitions()
            if not (name := item["function"]["name"]).startswith(offline)
        )

    async def _mcp_error(self, name: str, error: str) -> None:
        await self.events.emit(
            "mcp.error", session_id=self._session.id, turn_id="", server=name, error=error
        )

    def _mcp_catalog(self, connecting: MCPConnection | None = None) -> ToolExecutor:
        return create_tools(
            approval=self._request_approval,
            connections=(
                server for server in self._mcp.values()
                if server is connecting or server.state == "Connected"
            ),
        )

    async def _connect_mcp(self, connection: MCPConnection) -> None:
        self._ensure_idle()
        async with self._mcp_lock:
            previous = self._mcp.get(connection.name)
            if previous and previous.state == "Connected":
                raise ValueError(f"MCP Server {connection.name!r} 已连接。")
            if previous:
                await previous.aclose()
            self._mcp[connection.name] = connection
            discovered = await connection.start(self._mcp_error, self._mcp_tools_changed)
            previous_tools = connection.tools
            try:
                self._ensure_open()
                connection.tools = discovered
                tools = self._mcp_catalog(connecting=connection)
            except Exception:  # noqa: BLE001 - SDK catalogue errors must not disclose server data.
                error = "无法注册 MCP 工具定义。"
            else:
                if connection._can_publish():
                    connection.state = "Connected"
                    self._tools = tools
                    return
                error = connection.error or "MCP 连接在发布前已中断。"
            connection.tools = previous_tools
            await connection.aclose()
            if self._closed:
                raise RuntimeError("Application 已关闭。")
            connection.state, connection.error = "Error", error
            raise RuntimeError(error)

    async def _disconnect_mcp(self, name: str) -> None:
        self._ensure_idle()
        async with self._mcp_lock:
            connection = self._mcp[name]
            await connection.aclose()
            self._tools = self._mcp_catalog()

    async def _refresh_mcp(self, name: str) -> None:
        self._ensure_idle()
        async with self._mcp_lock:
            connection = self._mcp.get(name)
            if connection is None or connection.state != "Connected":
                raise ValueError(f"MCP Server {name!r} 未连接。")
            await self._refresh_mcp_catalog(connection)

    async def _refresh_mcp_catalog(self, connection: MCPConnection) -> None:
        previous = connection.tools
        try:
            discovered = await connection.discover()
            self._ensure_open()
            if connection.state != "Connected":
                raise RuntimeError("MCP 连接不可用。")
            connection.tools = discovered
            self._tools = self._mcp_catalog()
        except (Exception, asyncio.CancelledError) as exc:
            connection.tools = previous
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RuntimeError("MCP 工具刷新失败，已保留上次目录。") from None
        await self.events.emit(
            "mcp.tools_changed", session_id=self._session.id, turn_id="", server=connection.name
        )

    async def _mcp_tools_changed(self, connection: MCPConnection) -> None:
        self._mcp_pending.add(connection)
        try:
            while not self._closed and connection in self._mcp_pending:
                async with self._mcp_lock:
                    active = self._task if self._running else None
                    if active is None:
                        await self._refresh_pending_mcp()
                        return
                # Wait without adopting cancellation or failure from the Turn.
                await asyncio.wait((active,))
        finally:
            self._mcp_pending.discard(connection)

    async def _refresh_pending_mcp(self) -> None:
        while self._mcp_pending and not self._closed:
            connection = self._mcp_pending.pop()
            if (
                self._mcp.get(connection.name) is connection
                and connection.state == "Connected"
            ):
                try:
                    await self._refresh_mcp_catalog(connection)
                except RuntimeError as exc:
                    await self._mcp_error(connection.name, str(exc))

    async def _request_approval(self, request: ApprovalRequest) -> bool:
        if not self._interactive:
            return False
        if self._approval is not None:
            raise RuntimeError("已有待处理审批。")
        future = asyncio.get_running_loop().create_future()
        self._approval = future
        self._pending_approval = deepcopy(request)
        try:
            return await future
        finally:
            if not future.done():
                future.cancel()
            self._pending_approval = None
            self._approval = None

    def _resolve_approval(self, turn_id: str, tool_call_id: str, approved: bool) -> bool:
        pending = self._pending_approval
        if pending is None or (pending.turn_id, pending.tool_call_id) != (
            turn_id,
            tool_call_id,
        ):
            return False
        if self._approval is None or self._approval.done():
            return False
        self._approval.set_result(approved is True)
        return True

    async def _run_turn(self, prompt: str, *, stream: bool = True) -> TurnResult:
        self._ensure_idle()
        if self._mcp_pending:
            async with self._mcp_lock:
                await self._refresh_pending_mcp()
                self._ensure_open()
        session = self._session
        thinking = session.thinking
        available = set(self._all_tools)
        enabled_tools = [name for name in session.enabled_tools if name in available]
        provider = session.provider
        custom_compatibility = (
            self._connection(provider).get("compatibility") is not None
        )
        agent = Agent(
            client=self._client(provider),
            provider=provider,
            catalog=self._catalog,
            custom_compatibility=custom_compatibility,
            tools=self._tools.select(enabled_tools),
            events=self.events,
            context=Context(
                system_prompt=(
                    "你是 Daphnis，一个终端编程助手。根据实际工具结果完成任务，"
                    "遵守权限拒绝，使用普通 unified diff 修改文件。"
                    "不要主动读取密钥或凭据文件；不要在回答中输出凭据。"
                )
            ),
        )

        async def run():
            try:
                result = await agent.run(
                    session=session,
                    prompt=prompt,
                    stream=stream,
                    thinking_mode=thinking,
                )
            except asyncio.CancelledError:
                # Token use remains a fact even when working messages roll back.
                try:
                    if self._has_usage_to_save():
                        self._save_session(session)
                except Exception:
                    logger.exception("本轮已取消，已报告的用量未能保存。")
                raise
            if result.status == TurnStatus.COMPLETED or self._has_usage_to_save():
                try:
                    self._save_session(session)
                except Exception as exc:
                    raise SessionSaveError(result, exc) from exc
            return result

        task = asyncio.create_task(run())
        self._task = task
        try:
            return await task
        finally:
            if self._task is task:
                self._task = None

    def _save_session(self, session: Session) -> None:
        session.usage = self._usage.session
        self._history.save(session)

    def _has_usage_to_save(self) -> bool:
        return self._usage.session != self._session.usage and (
            bool(len(self._session.messages)) or self._usage.session.reported_requests > 0
        )

    async def _cancel_turn(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with AsyncExitStack() as stack:
            for client in self._clients.values():
                stack.push_async_callback(client.aclose)
            try:
                await self._cancel_turn()
                await self.events.drain()
            finally:
                async with self._mcp_lock:
                    for connection in self._mcp.values():
                        stack.push_async_callback(connection.aclose)
                    if self._mcp:
                        self._mcp.clear()
                        self._tools = create_tools(approval=self._request_approval)
                self._usage.close()
                self.events.clear()
