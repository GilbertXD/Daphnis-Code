"""Own Daphnis runtime resources and supervise Turn execution."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from saturnus import ChatClient

from .agent import Agent, TurnResult, TurnStatus
from .compaction import CompactionCheckpoint
from .context import Context
from .events import Event, EventBus
from .history import (
    History,
    SessionBusyError,
    SessionConflictError,
    SessionLease,
    finish_io,
)
from .mcp import TOOL_PREFIX, MCPConnection
from .model_io import ObservedChatClient
from .models import (
    MODELS,
    ModelInfo,
    provider_compatibility,
    resolve_model,
)
from .models.models import validate_catalog
from .session import Session
from .token_budget import TokenBudget
from .tools import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalRequest,
    ToolExecutor,
    create_tools,
)
from .usage import UsageTracker

logger = logging.getLogger(__name__)
_EVENT_DRAIN_TIMEOUT = 0.5


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
        observe_model_io: bool = False,
    ) -> None:
        self._catalog = tuple(catalog)
        validate_catalog(self._catalog)
        self._connections = {
            name: dict(options) for name, options in connections.items()
        }
        self._clients: dict[str, ChatClient] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._history = history if history is not None else History()
        self._interactive = interactive
        self._observe_model_io = observe_model_io
        self._task: asyncio.Task[TurnResult] | asyncio.Task[CompactionCheckpoint | None] | None = None
        self._approval_mode = ApprovalMode.AUTO
        self._approval: asyncio.Future[ApprovalDecision] | None = None
        self._pending_approval: ApprovalRequest | None = None
        self._trusted_tools: set[tuple[str, str]] = set()
        self._mcp: dict[str, MCPConnection] = {}
        self._state_lock = asyncio.Lock()
        self._mcp_pending: set[MCPConnection] = set()
        self._tools = self._mcp_catalog()
        self._use_saved_tool_preferences = False
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
        self._last_budget: tuple[str, str, str, TokenBudget] | None = None
        self._compacting = False
        for name in ("context.budget", "compaction.started", "compaction.finished"):
            self.events.on(name, self._observe_context)
        self._usage = UsageTracker(self.events)
        self._usage.select_session(self._session.id, self._session.usage)
        self._client(provider)

    def _observe_context(self, event: Event) -> None:
        if event.session_id != self._session.id:
            return
        if event.name == "context.budget":
            data = event.data
            self._last_budget = (
                event.session_id, data["provider"], data["model_id"],
                TokenBudget(
                    data["input_tokens"], data["context_window"], data["output_reserve"],
                    data["compaction_threshold"], data["needs_compaction"],
                ),
            )
        else:
            self._compacting = event.name == "compaction.started"

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
            if self._observe_model_io:
                self._clients[provider] = ObservedChatClient(
                    events=self.events, provider=provider, **options
                )
            else:
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
        if self._state_lock.locked():
            raise RuntimeError("请等待会话或 MCP 更新完成。")

    @property
    def _all_tools(self) -> tuple[str, ...]:
        offline = tuple(
            server.prefix for server in self._mcp.values() if not server.available
        )
        return tuple(
            name
            for item in self._tools.definitions()
            if not (name := item["function"]["name"]).startswith(offline)
        )

    async def _mcp_error(self, name: str, error: str) -> None:
        self._revoke_mcp_trust(name)
        await self.events.emit(
            "mcp.error", session_id=self._session.id, turn_id="", server=name, error=error
        )

    def _mcp_catalog(self, connecting: MCPConnection | None = None) -> ToolExecutor:
        return create_tools(
            approval=self._request_approval,
            trusted=self._trusted_tools,
            connections=(
                server
                for server in self._mcp.values()
                if server is connecting or server.available
            ),
        )

    def _revoke_mcp_trust(self, server: str, changed: set[str] | None = None) -> None:
        prefix = f"{TOOL_PREFIX}{server}__"
        self._trusted_tools.difference_update(
            {
                key
                for key in self._trusted_tools
                if key[1].startswith(prefix) and (changed is None or key[1] in changed)
            }
        )

    async def _connect_mcp(self, connection: MCPConnection) -> None:
        self._ensure_idle()
        async with self._state_lock:
            previous = self._mcp.get(connection.name)
            if previous and previous.state == "Connected":
                raise ValueError(f"MCP Server {connection.name!r} 已连接。")
            self._revoke_mcp_trust(connection.name)
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
                    try:
                        if (
                            previous is None
                            and discovered
                            and not self._use_saved_tool_preferences
                            and self._model.supports_tools is True
                        ):
                            session = replace(
                                self._session,
                                enabled_tools=(*self._session.enabled_tools, *discovered),
                            )
                            await finish_io(self._save_and_publish_session(session))
                    except (SessionBusyError, SessionConflictError, asyncio.CancelledError):
                        connection.tools = previous_tools
                        await finish_io(connection.aclose())
                        raise
                    except (OSError, TypeError, ValueError):
                        error = "无法保存 MCP 工具启用偏好。"
                    else:
                        if not self._closed and connection._can_publish():
                            connection.state = "Connected"
                            self._tools = tools
                            return
                        error = connection.error or "MCP 连接在发布前已中断。"
                else:
                    error = connection.error or "MCP 连接在发布前已中断。"
            connection.tools = previous_tools
            await connection.aclose()
            if self._closed:
                raise RuntimeError("Application 已关闭。")
            connection.state, connection.error = "Error", error
            raise RuntimeError(error)

    async def _disconnect_mcp(self, name: str) -> None:
        self._ensure_idle()
        async with self._state_lock:
            connection = self._mcp[name]
            self._revoke_mcp_trust(name)
            await connection.aclose()
            self._tools = self._mcp_catalog()

    async def _refresh_mcp(self, name: str) -> None:
        self._ensure_idle()
        async with self._state_lock:
            connection = self._mcp.get(name)
            if connection is None or not connection.available:
                raise ValueError(f"MCP Server {name!r} 未连接。")
            await self._refresh_mcp_catalog(connection)

    async def _refresh_mcp_catalog(self, connection: MCPConnection) -> None:
        previous = connection.tools
        try:
            discovered = await connection.discover()
            self._ensure_open()
            if not connection.available:
                raise RuntimeError("MCP 连接不可用。")
            connection.tools = discovered
            self._tools = self._mcp_catalog()
        except (Exception, asyncio.CancelledError) as exc:
            connection.tools = previous
            if isinstance(exc, asyncio.CancelledError):
                raise
            self._revoke_mcp_trust(connection.name)
            raise RuntimeError("MCP 工具刷新失败，已保留上次目录。") from None
        self._revoke_mcp_trust(
            connection.name,
            {name for name in previous if previous[name] != discovered.get(name)},
        )
        await self.events.emit(
            "mcp.tools_changed", session_id=self._session.id, turn_id="", server=connection.name
        )

    async def _mcp_tools_changed(self, connection: MCPConnection) -> None:
        # The active Turn keeps its definitions, but not its automatic authorization.
        self._revoke_mcp_trust(connection.name)
        self._mcp_pending.add(connection)
        try:
            while not self._closed and connection in self._mcp_pending:
                async with self._state_lock:
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
                and connection.available
            ):
                try:
                    await self._refresh_mcp_catalog(connection)
                except RuntimeError as exc:
                    await self._mcp_error(connection.name, str(exc))

    async def _request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        if not self._interactive:
            return ApprovalDecision.DENY
        if self._approval is not None:
            raise RuntimeError("已有待处理审批。")
        future = asyncio.get_running_loop().create_future()
        self._approval = future
        self._pending_approval = deepcopy(request)
        try:
            decision = await future
            if decision is ApprovalDecision.TRUST_SESSION:
                if not request.can_trust:
                    return ApprovalDecision.DENY
                server = request.details.get("server")
                connection = self._mcp.get(server)
                if server is None or (
                    connection is not None
                    and connection.available
                    and connection not in self._mcp_pending
                ):
                    self._trusted_tools.add((request.session_id, request.tool_name))
            return decision
        finally:
            if not future.done():
                future.cancel()
            self._pending_approval = None
            self._approval = None

    def _resolve_approval(
        self, request_id: str, decision: ApprovalDecision
    ) -> bool:
        pending = self._pending_approval
        if self._closed or pending is None or pending.request_id != request_id:
            return False
        if self._approval is None or self._approval.done():
            return False
        self._approval.set_result(decision)
        return True

    async def _run_turn(self, prompt: str, *, stream: bool = True) -> TurnResult:
        self._ensure_idle()
        if self._mcp_pending:
            async with self._state_lock:
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
            tools=self._tools.select(enabled_tools, mode=self._approval_mode),
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
            async with self._history.lease(session) as lease:
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
                            await self._save_session(session, lease=lease)
                    except Exception:
                        logger.exception("本轮已取消，已报告的用量未能保存。")
                    raise
                if result.status == TurnStatus.COMPLETED or self._has_usage_to_save():
                    try:
                        await self._save_session(session, lease=lease)
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

    async def _compact_session(self) -> CompactionCheckpoint | None:
        self._ensure_idle()
        session = self._session
        provider = session.provider
        agent = Agent(
            client=self._client(provider), provider=provider, catalog=self._catalog,
            custom_compatibility=self._connection(provider).get("compatibility") is not None,
            events=self.events,
        )
        fields = {"session_id": session.id, "turn_id": uuid4().hex, "trigger": "manual"}

        async def compact() -> CompactionCheckpoint | None:
            await self.events.emit("compaction.started", **fields)
            try:
                async with self._history.lease(session) as lease:
                    try:
                        checkpoint = await agent.compact(
                            session=session, operation_id=fields["turn_id"],
                        )
                    except (Exception, asyncio.CancelledError):
                        if self._has_usage_to_save():
                            try:
                                await self._save_session(session, lease=lease)
                            except Exception:
                                logger.exception("压缩用量未能保存。")
                        raise
                    if checkpoint is None:
                        if self._has_usage_to_save():
                            await self._save_session(session, lease=lease)
                        await self.events.emit("compaction.finished", **fields, status="skipped")
                        return None
                    updated = replace(session, compaction=checkpoint)

                    async def commit() -> None:
                        await self._save_session(updated, lease=lease)
                        self._session = updated
                        await self.events.emit(
                            "compaction.finished", **fields,
                            status="committed", through=checkpoint.through,
                        )

                    await finish_io(commit())
                    return checkpoint
            except asyncio.CancelledError:
                if self._session is session:
                    await self.events.emit("compaction.finished", **fields, status="cancelled")
                raise
            except Exception:
                await self.events.emit("compaction.finished", **fields, status="failed")
                raise

        task = asyncio.create_task(compact())
        self._task = task
        try:
            return await task
        finally:
            if self._task is task:
                self._task = None

    async def _save_session(self, session: Session, *, lease: SessionLease | None = None) -> None:
        session.usage = self._usage.session
        await self._history.save_async(session, lease=lease)

    async def _save_and_publish_session(self, session: Session) -> None:
        await self._save_session(session)
        self._session = session

    async def _update_session(self, session: Session) -> None:
        async with self._state_lock:
            await finish_io(self._save_and_publish_session(session))

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
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_resources())
        await asyncio.shield(self._close_task)

    async def _close_resources(self) -> None:
        async with AsyncExitStack() as stack:
            for client in self._clients.values():
                stack.push_async_callback(client.aclose)
            try:
                await self._cancel_turn()
                self.events.clear()
                await self.events.drain(timeout=_EVENT_DRAIN_TIMEOUT)
            finally:
                async with self._state_lock:
                    for connection in self._mcp.values():
                        stack.push_async_callback(connection.aclose)
                    if self._mcp:
                        self._mcp.clear()
                        self._tools = self._mcp_catalog()
                self._trusted_tools.clear()
                self._usage.close()
                self.events.clear()
