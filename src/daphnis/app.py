"""UI-independent coordination of the existing Daphnis components."""

import asyncio
import logging
import os
import shlex
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saturnus import ChatClient

from .agent import Agent, TurnResult, TurnStatus
from .context import Context
from .events import EventBus
from .history import History, SessionInfo
from .models import (
    MODELS,
    CapabilityError,
    ModelInfo,
    list_models,
    provider_compatibility,
    require_tools,
    resolve_model,
)
from .models.models import validate_catalog
from .session import Session
from .tools import ApprovalRequest, create_tools
from .usage import UsageTracker

logger = logging.getLogger(__name__)

HELP = """/model [list|序号|模型ID|厂商 模型ID]  查看或切换模型（按厂商分列）
/thinking [default|档位]  查看或设置思考；default 为服务端默认
/tools [all|none|on 工具名...|off 工具名...]  查看或选择工具
/session [会话ID]  列出或恢复会话
/new  新建会话，保留旧会话
/clear  仅清理屏幕，保留历史
/help  显示帮助
Enter 发送；Shift+Enter 换行；Alt+Enter / F2 为换行备用键。
Esc 中断当前轮；Ctrl+C 运行时取消、空闲时退出；Ctrl+Q 取消并退出。"""


def environment_connections() -> tuple[str, str, dict[str, dict[str, Any]]]:
    """Reuse .env/environment configuration; provider profiles never share keys."""
    provider = os.getenv("LLM_PROVIDER", "")
    model_id = os.getenv("LLM_MODEL", "")
    connections = {}
    base_url = os.getenv("LLM_BASE_URL")
    if provider and base_url:
        connections[provider] = {
            "base_url": base_url,
            "api_key": os.getenv("LLM_API_KEY") or None,
        }
    for name in ("deepseek", "mimo", "compatible"):
        prefix = name.upper()
        address = os.getenv(f"{prefix}_BASE_URL")
        if address:
            connections[name] = {
                "base_url": address,
                "api_key": os.getenv(f"{prefix}_API_KEY") or None,
            }
    if not provider or not model_id or provider not in connections:
        raise ValueError("请在 .env 配置 LLM_PROVIDER、LLM_MODEL 和对应的 Base URL。")
    return provider, model_id, connections


@dataclass(frozen=True)
class CommandResult:
    action: str = "message"
    text: str = ""


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
        self.catalog = tuple(catalog)
        validate_catalog(self.catalog)
        self._connections = {
            name: dict(options) for name, options in connections.items()
        }
        self._clients: dict[str, ChatClient] = {}
        self._closed = False
        self.history = history if history is not None else History()
        self.session = Session(
            workspace=workspace, provider=provider, model_id=model_id
        )
        if not Path(self.session.workspace).is_dir():
            raise ValueError("Workspace 必须是现有目录。")
        self.events = EventBus()
        self.usage = UsageTracker(self.events)
        self.usage.select_session(self.session.id, self.session.usage)
        self.interactive = interactive
        self.thinking: str | None = None
        self._task: asyncio.Task[TurnResult] | None = None
        self._approval: asyncio.Future[bool] | None = None
        self.pending_approval: ApprovalRequest | None = None
        self._tools = create_tools(approval=self._request_approval)
        self.all_tools = tuple(
            item["function"]["name"] for item in self._tools.definitions()
        )
        model = self._model_for(provider, model_id)
        self._client(provider)
        self.enabled_tools = self.all_tools if model.supports_tools is True else ()

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
        return resolve_model(provider, model_id, catalog=self.catalog)

    @property
    def model(self) -> ModelInfo:
        return resolve_model(
            self.session.provider, self.session.model_id, catalog=self.catalog
        )

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _idle(self) -> None:
        if self._closed:
            raise RuntimeError("Application 已关闭。")
        if self.running:
            raise RuntimeError("请先等待或取消当前 Turn，再执行该操作。")

    def model_groups(self) -> dict[str, tuple[ModelInfo, ...]]:
        providers = dict.fromkeys(
            (
                "deepseek",
                "mimo",
                *(model.provider for model in self.catalog),
                self.session.provider,
            )
        )
        groups = {
            provider: list_models(provider, catalog=self.catalog)
            for provider in providers
        }
        if self.model not in groups[self.session.provider]:
            groups[self.session.provider] = (*groups[self.session.provider], self.model)
        return groups

    def configured(self, provider: str) -> bool:
        return provider in self._connections

    def set_model(self, provider: str, model_id: str) -> str:
        self._idle()
        model = self._model_for(provider, model_id)
        updated = Session(
            id=self.session.id,
            workspace=self.session.workspace,
            provider=provider,
            model_id=model_id,
            messages=self.session.messages,
            usage=self.usage.session,
        )
        self.history.save(updated)
        self.session = updated
        notices = [f"模型：{provider} / {model_id}"]
        if self.thinking is not None and self.thinking not in (
            model.thinking_modes or ()
        ):
            self.thinking = None
            notices.append("原思考档位不适用，已恢复服务端默认。")
        if self.enabled_tools and model.supports_tools is not True:
            self.enabled_tools = ()
            notices.append("工具能力未确认支持，已停用工具。")
        return "\n".join(notices)

    def set_thinking(self, mode: str | None) -> None:
        self._idle()
        if mode is not None and mode not in (self.model.thinking_modes or ()):
            raise CapabilityError("该模型不支持此思考选项，或能力尚未确认。")
        self.thinking = mode

    def set_tools(self, names: Sequence[str]) -> None:
        self._idle()
        self._tools.select(list(names))  # Saturnus validates names and duplicates.
        if names:
            require_tools(self.model)
        self.enabled_tools = tuple(names)

    def sessions(self) -> tuple[SessionInfo, ...]:
        return self.history.list_sessions()

    def new_session(self) -> Session:
        self._idle()
        self._save_session()
        fresh = Session(
            workspace=self.session.workspace,
            provider=self.session.provider,
            model_id=self.session.model_id,
        )
        self.history.save(fresh)
        self.session = fresh
        self.usage.select_session(fresh.id, fresh.usage)
        return fresh

    def restore_session(self, session_id: str) -> None:
        self._idle()
        restored = self.history.load(session_id)
        if not Path(restored.workspace).is_dir():
            raise ValueError("该会话原工作区已不存在。")
        model = self._model_for(restored.provider, restored.model_id)
        self.session = restored
        self.usage.select_session(restored.id, restored.usage)
        if self.thinking not in (model.thinking_modes or ()):
            self.thinking = None
        if model.supports_tools is not True:
            self.enabled_tools = ()

    async def _request_approval(self, request: ApprovalRequest) -> bool:
        if not self.interactive:
            return False
        if self._approval is not None:
            raise RuntimeError("已有待处理审批。")
        future = asyncio.get_running_loop().create_future()
        self._approval = future
        self.pending_approval = deepcopy(request)
        try:
            return await future
        finally:
            if not future.done():
                future.cancel()
            self.pending_approval = None
            self._approval = None

    def resolve_approval(self, turn_id: str, tool_call_id: str, approved: bool) -> bool:
        pending = self.pending_approval
        if pending is None or (pending.turn_id, pending.tool_call_id) != (
            turn_id,
            tool_call_id,
        ):
            return False
        if self._approval is None or self._approval.done():
            return False
        self._approval.set_result(approved is True)
        return True

    async def send(self, prompt: str, *, stream: bool = True) -> TurnResult:
        self._idle()
        session = self.session
        provider = session.provider
        custom_compatibility = (
            self._connection(provider).get("compatibility") is not None
        )
        agent = Agent(
            client=self._client(provider),
            provider=provider,
            catalog=self.catalog,
            custom_compatibility=custom_compatibility,
            tools=self._tools.select(list(self.enabled_tools)),
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
                    thinking_mode=self.thinking,
                )
            except asyncio.CancelledError:
                # Token use remains a fact even when working messages roll back.
                try:
                    if self._has_usage_to_save():
                        self._save_session()
                except Exception:
                    logger.exception("本轮已取消，已报告的用量未能保存。")
                raise
            if result.status == TurnStatus.COMPLETED or self._has_usage_to_save():
                try:
                    self._save_session()
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

    def _save_session(self) -> None:
        self.session.usage = self.usage.session
        self.history.save(self.session)

    def _has_usage_to_save(self) -> bool:
        return self.usage.session != self.session.usage and (
            bool(len(self.session.messages)) or self.usage.session.reported_requests > 0
        )

    async def cancel(self) -> None:
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
                await self.cancel()
                await self.events.drain()
            finally:
                self.usage.close()

    async def command(self, text: str) -> CommandResult:
        words = shlex.split(text)
        name, args = words[0], words[1:]
        if name == "/help" and not args:
            return CommandResult(text=HELP)
        if name == "/clear" and not args:
            return CommandResult("clear", "已清理屏幕；会话历史保留。")
        self._idle()
        if name == "/new" and not args:
            self.new_session()
            return CommandResult("reload", f"新会话：{self.session.id}")
        if name == "/model":
            groups = self.model_groups()
            choices = tuple(model for models in groups.values() for model in models)
            if not args or args == ["list"]:
                lines = [
                    f"{provider}"
                    + ("" if self.configured(provider) else "（未配置）")
                    + ": "
                    + ", ".join(
                        f"{choices.index(model) + 1}. {model.id}" for model in models
                    )
                    for provider, models in groups.items()
                ]
                return CommandResult("models", "\n".join(lines))
            if len(args) == 1:
                if args[0].isdigit():
                    index = int(args[0]) - 1
                    if index < 0 or index >= len(choices):
                        raise ValueError("模型序号超出范围。")
                    target = choices[index]
                    return CommandResult(
                        text=self.set_model(target.provider, target.id)
                    )
                matches = [model for model in choices if model.id == args[0]]
                if len(matches) == 1:
                    return CommandResult(
                        text=self.set_model(matches[0].provider, args[0])
                    )
                if len(matches) > 1:
                    raise ValueError("模型名属于多个厂商，请使用 /model 厂商 模型ID。")
                return CommandResult(
                    text=self.set_model(self.session.provider, args[0])
                )
            if len(args) == 2:
                return CommandResult(text=self.set_model(args[0], args[1]))
        if name == "/thinking":
            if not args:
                modes = self.model.thinking_modes
                if not modes:
                    return CommandResult(
                        text="思考选项未知。"
                        if modes is None
                        else "该模型没有可配置的思考选项。"
                    )
                return CommandResult(
                    "thinking", "可用：default（服务端默认）、" + ", ".join(modes)
                )
            if len(args) == 1:
                self.set_thinking(None if args[0] == "default" else args[0])
                return CommandResult(text=f"Thinking：{self.thinking or '服务端默认'}")
        if name == "/tools":
            if not args:
                return CommandResult(
                    "tools",
                    "\n".join(
                        ("[启用] " if item in self.enabled_tools else "[停用] ") + item
                        for item in self.all_tools
                    ),
                )
            if args == ["all"]:
                self.set_tools(self.all_tools)
            elif args == ["none"]:
                self.set_tools(())
            elif len(args) > 1 and args[0] in ("on", "off"):
                if any(name not in self.all_tools for name in args[1:]):
                    raise ValueError("包含未知工具。")
                chosen = set(self.enabled_tools)
                if args[0] == "on":
                    chosen.update(args[1:])
                else:
                    chosen.difference_update(args[1:])
                self.set_tools(tuple(name for name in self.all_tools if name in chosen))
            else:
                raise ValueError("用法：/tools all|none|on 工具名...|off 工具名...")
            return CommandResult(
                text=f"启用工具：{len(self.enabled_tools)} 个（逐次审批仍生效）。"
            )
        if name == "/session":
            if not args:
                items = self.sessions()
                return CommandResult(
                    "sessions",
                    "\n".join(
                        f"{item.id}  {item.provider}/{item.model_id}  "
                        f"{item.message_count} 条  {item.workspace}"
                        for item in items
                    )
                    or "尚无已保存会话。",
                )
            if len(args) == 1:
                self.restore_session(args[0])
                return CommandResult("reload", f"恢复会话：{self.session.id}")
        raise ValueError("未知命令或参数，请输入 /help 查看用法。")
