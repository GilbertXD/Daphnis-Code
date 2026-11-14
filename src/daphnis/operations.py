"""Public business operations and read-only views of an Application."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from saturnus import ValidationError

from .compaction import CompactionCheckpoint
from .events import Callback
from .history import finish_io
from .mcp import TOOL_PREFIX, MCPConnection, MCPStatus
from .models import CapabilityError, ModelInfo, list_models, require_tools
from .session import Session
from .tools import ApprovalDecision, ApprovalMode
from .turn_state import RecoveryAction, RecoveryState

if TYPE_CHECKING:
    from .agent import TurnResult
    from .app import Application
    from .history import SessionInfo, SessionScan
    from .token_budget import TokenBudget
    from .tools import ApprovalRequest
    from .usage import UsageTotals


@dataclass(frozen=True)
class SessionSnapshot:
    id: str
    workspace: Path
    provider: str
    model_id: str
    title: str | None = None


class Operations:
    """Use the shared runtime directly; each call reads its current state."""

    def __init__(self, app: "Application") -> None:
        self._app = app

    @property
    def session(self) -> SessionSnapshot:
        session = self._app._session
        return SessionSnapshot(
            session.id, Path(session.workspace), session.provider, session.model_id,
            session.title,
        )

    def message_snapshot(self) -> list[dict[str, Any]]:
        """Return detached copies of committed messages for display."""
        return self._app._session.messages.snapshot()

    def subscribe(self, name: str, callback: Callback) -> Callable[[], None]:
        """Subscribe to detached event snapshots and return an unsubscribe function."""
        return self._app.events.on(name, callback)

    @property
    def model(self) -> ModelInfo:
        return self._app._model

    @property
    def thinking(self) -> str | None:
        return self._app._session.thinking

    @property
    def all_tools(self) -> tuple[str, ...]:
        return self._app._all_tools

    @property
    def enabled_tools(self) -> tuple[str, ...]:
        return self._app._session.enabled_tools

    @property
    def effective_tools(self) -> tuple[str, ...]:
        available = set(self.all_tools)
        return tuple(name for name in self.enabled_tools if name in available)

    @property
    def running(self) -> bool:
        return self._app._running

    @property
    def approval_mode(self) -> ApprovalMode:
        return self._app._approval_mode

    def set_approval_mode(self, mode: ApprovalMode | str) -> None:
        app = self._app
        app._ensure_idle()
        app._approval_mode = ApprovalMode(mode)
        if app._approval_mode is ApprovalMode.ASK:
            app._trusted_tools.clear()

    @property
    def pending_approval(self) -> "ApprovalRequest | None":
        return deepcopy(self._app._pending_approval)

    @property
    def session_usage(self) -> "UsageTotals":
        return self._app._usage.session

    @property
    def turn_usage(self) -> "UsageTotals | None":
        return self._app._usage.turn

    @property
    def context_budget(self) -> "TokenBudget | None":
        observed = self._app._last_budget
        session = self.session
        if observed is not None and observed[:3] == (session.id, session.provider, session.model_id):
            return observed[3]
        return None

    @property
    def compacting(self) -> bool:
        return self._app._compacting

    @property
    def compacted_through(self) -> int:
        checkpoint = self._app._session.compaction
        return checkpoint.through if checkpoint else 0

    def configured(self, provider: str) -> bool:
        return provider in self._app._connections

    def list_models(self) -> dict[str, tuple[ModelInfo, ...]]:
        catalog = self._app._catalog
        model = self.model
        providers = dict.fromkeys(
            ("deepseek", "mimo", *(item.provider for item in catalog), model.provider)
        )
        groups = {
            provider: list_models(provider, catalog=catalog) for provider in providers
        }
        if model not in groups[model.provider]:
            groups[model.provider] = (*groups[model.provider], model)
        return groups

    async def set_model(self, provider: str, model_id: str) -> str:
        app = self._app
        app._ensure_idle()
        model = app._model_for(provider, model_id)
        current = app._session
        thinking, enabled_tools = current.thinking, current.enabled_tools
        notices = [f"模型：{provider} / {model_id}"]
        if thinking is not None and thinking not in (model.thinking_modes or ()):
            thinking = None
            notices.append("原思考档位不适用，已恢复服务端默认。")
        if enabled_tools and model.supports_tools is not True:
            enabled_tools = ()
            notices.append("工具能力未确认支持，已停用工具。")
        updated = replace(
            current,
            provider=provider,
            model_id=model_id,
            thinking=thinking,
            enabled_tools=enabled_tools,
        )
        await app._update_session(updated)
        return "\n".join(notices)

    @staticmethod
    def _validate_thinking(model: ModelInfo, mode: str | None) -> None:
        if mode is not None and mode not in (model.thinking_modes or ()):
            raise CapabilityError(f"该模型不支持思考档位 {mode!r}，或能力尚未确认。")

    def _validate_tools(
        self,
        model: ModelInfo,
        names: tuple[str, ...],
        *,
        retained: tuple[str, ...] = (),
    ) -> None:
        available = set(self.all_tools)
        for name in names:
            if not isinstance(name, str) or (
                name not in available and name not in retained
            ):
                raise ValidationError(
                    f"Tool {name!r} is not registered or currently available."
                )
        self._app._tools.select([name for name in names if name in available])
        if names:
            require_tools(model)

    async def set_thinking(self, mode: str | None) -> None:
        app = self._app
        app._ensure_idle()
        self._validate_thinking(self.model, mode)
        updated = replace(app._session, thinking=mode)
        await app._update_session(updated)

    def get_system_prompt(self) -> str:
        return self._app._session.effective_system_prompt

    async def set_system_prompt(self, prompt: str) -> None:
        app = self._app
        app._ensure_idle()
        if not isinstance(prompt, str):
            raise TypeError("系统提示词必须是字符串。")
        await app._update_session(replace(app._session, system_prompt=prompt))

    async def reset_system_prompt(self) -> None:
        app = self._app
        app._ensure_idle()
        await app._update_session(replace(app._session, system_prompt=None))

    async def set_tools(
        self, names: Sequence[str], *, keep_unavailable: bool = False
    ) -> None:
        app = self._app
        app._ensure_idle()
        if isinstance(names, (str, bytes)):
            raise TypeError("names must be a sequence of tool names.")
        selected = tuple(names)
        if keep_unavailable:
            available = set(self.all_tools)
            selected += tuple(
                name
                for name in self.enabled_tools
                if name not in available and name not in selected
            )
        self._validate_tools(self.model, selected, retained=self.enabled_tools)
        updated = replace(app._session, enabled_tools=selected)
        await app._update_session(updated)

    async def connect_mcp(
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
    ) -> None:
        """Connect an explicitly configured server."""
        await self._app._connect_mcp(
            MCPConnection(
                name,
                command,
                args,
                cwd=cwd,
                env=env,
                transport=transport,
                url=url,
                auth=auth,
            )
        )

    async def reconnect_mcp(self, name: str) -> None:
        connection = self._app._mcp.get(name)
        if connection is None:
            raise ValueError(f"未知 MCP Server {name!r}。")
        await self._app._connect_mcp(connection)

    async def refresh_mcp(self, name: str) -> None:
        await self._app._refresh_mcp(name)

    async def disconnect_mcp(self, name: str) -> None:
        if name not in self._app._mcp:
            raise ValueError(f"MCP Server {name!r} 未连接。")
        await self._app._disconnect_mcp(name)

    def list_mcp_servers(self) -> tuple[str, ...]:
        return tuple(self._app._mcp)

    def mcp_status(self) -> tuple[MCPStatus, ...]:
        return tuple(server.status for server in self._app._mcp.values())

    def list_mcp_tools(self) -> tuple[str, ...]:
        return tuple(
            tool
            for server in self._app._mcp.values()
            if server.available
            for tool in server.tools
        )

    def list_sessions(self) -> "tuple[SessionInfo, ...]":
        return self._app._history.list_sessions()

    def scan_sessions(self) -> "SessionScan":
        return self._app._history.scan_sessions()

    async def new_session(self) -> SessionSnapshot:
        app = self._app
        app._ensure_idle()
        current = app._session
        fresh = Session(
            workspace=current.workspace,
            provider=current.provider,
            model_id=current.model_id,
            thinking=current.thinking,
            enabled_tools=current.enabled_tools,
        )
        async def save_and_switch() -> None:
            await app._save_session(current)
            await app._history.save_async(fresh)
            app._session = fresh
            app._approval_mode = ApprovalMode.AUTO
            app._usage.select_session(fresh.id, fresh.usage)

        async with app._state_lock:
            await finish_io(save_and_switch())
        return self.session

    def restore_session(self, session_id: str) -> None:
        app = self._app
        app._ensure_idle()
        restored = app._history.load(session_id)
        if not Path(restored.workspace).is_dir():
            raise ValueError("该会话原工作区已不存在。")
        if "history_search" in restored.enabled_tools:
            restored = replace(restored, enabled_tools=tuple(dict.fromkeys(
                "retrieval" if name == "history_search" else name for name in restored.enabled_tools
            )))
        model = app._model_for(restored.provider, restored.model_id)
        self._validate_thinking(model, restored.thinking)
        self._validate_tools(
            model,
            restored.enabled_tools,
            retained=tuple(
                name for name in restored.enabled_tools if name.startswith(TOOL_PREFIX)
            ),
        )
        app._session = restored
        app._use_saved_tool_preferences = True
        app._usage.select_session(restored.id, restored.usage)

    async def rename_session(self, session_id: str, title: str) -> None:
        app = self._app
        app._ensure_idle()
        title = title.strip() if isinstance(title, str) else ""
        if not 0 < len(title) <= 120:
            raise ValueError("会话名称必须包含 1–120 个字符。")
        if session_id == app._session.id:
            await app._update_session(replace(app._session, title=title))
        else:
            async with app._state_lock:
                await app._history.rename(session_id, title)

    async def delete_session(self, session_id: str) -> SessionSnapshot:
        """Delete under the history lease; replace an active session with a blank one."""
        app = self._app
        app._ensure_idle()
        current = app._session

        async def delete_and_select() -> None:
            if session_id != current.id:
                await app._history.delete(app._history.load(session_id))
                return
            fresh = Session(
                workspace=current.workspace,
                provider=current.provider,
                model_id=current.model_id,
                thinking=current.thinking,
                enabled_tools=current.enabled_tools,
            )
            await app._history.delete(current)
            app._session = fresh
            app._approval_mode = ApprovalMode.AUTO
            app._usage.select_session(fresh.id, fresh.usage)

        async with app._state_lock:
            await finish_io(delete_and_select())
        return self.session

    async def send(self, prompt: str, *, stream: bool = True) -> "TurnResult":
        return await self._app._run_turn(prompt, stream=stream)

    @property
    def recovery(self) -> RecoveryState | None:
        return self._app._session.recovery

    async def retry_failed_user(self, key: str, *, stream: bool = True) -> "TurnResult":
        return await self._app._run_turn(stream=stream, recovery_action=RecoveryAction.RETRY, recovery_key=key)

    async def continue_interrupted_turn(self, key: str, *, stream: bool = True) -> "TurnResult":
        return await self._app._run_turn(stream=stream, recovery_action=RecoveryAction.CONTINUE, recovery_key=key)

    async def terminate_interrupted_turn(self, key: str) -> "TurnResult":
        return await self._app._run_turn(recovery_action=RecoveryAction.TERMINATE, recovery_key=key)

    async def compact(self) -> CompactionCheckpoint | None:
        return await self._app._compact_session()

    async def cancel(self) -> None:
        self._app._ensure_open()
        await self._app._cancel_turn()

    def resolve_approval(
        self, request_id: str, decision: ApprovalDecision | str
    ) -> bool:
        return self._app._resolve_approval(request_id, ApprovalDecision(decision))

    async def close(self) -> None:
        await self._app.close()
