"""Public business operations and read-only views of an Application."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import CapabilityError, ModelInfo, list_models, require_tools
from .session import Session

if TYPE_CHECKING:
    from .agent import TurnResult
    from .app import Application
    from .history import SessionInfo
    from .tools import ApprovalRequest
    from .usage import UsageTotals


@dataclass(frozen=True)
class SessionSnapshot:
    id: str
    workspace: Path
    provider: str
    model_id: str


class Operations:
    """Use the shared runtime directly; each call reads its current state."""

    def __init__(self, app: "Application") -> None:
        self._app = app

    @property
    def session(self) -> SessionSnapshot:
        session = self._app._session
        return SessionSnapshot(
            session.id, Path(session.workspace), session.provider, session.model_id
        )

    def message_snapshot(self) -> list[dict[str, Any]]:
        """Return detached copies of committed messages for display."""
        return self._app._session.messages.snapshot()

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
    def running(self) -> bool:
        return self._app._running

    @property
    def pending_approval(self) -> "ApprovalRequest | None":
        return deepcopy(self._app._pending_approval)

    @property
    def session_usage(self) -> "UsageTotals":
        return self._app._usage.session

    @property
    def turn_usage(self) -> "UsageTotals | None":
        return self._app._usage.turn

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

    def set_model(self, provider: str, model_id: str) -> str:
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
        app._save_session(updated)
        app._session = updated
        return "\n".join(notices)

    @staticmethod
    def _validate_thinking(model: ModelInfo, mode: str | None) -> None:
        if mode is not None and mode not in (model.thinking_modes or ()):
            raise CapabilityError(f"该模型不支持思考档位 {mode!r}，或能力尚未确认。")

    def _validate_tools(self, model: ModelInfo, names: tuple[str, ...]) -> None:
        self._app._tools.select(list(names))  # Saturnus validates names and duplicates.
        if names:
            require_tools(model)

    def set_thinking(self, mode: str | None) -> None:
        app = self._app
        app._ensure_idle()
        self._validate_thinking(self.model, mode)
        updated = replace(app._session, thinking=mode)
        app._save_session(updated)
        app._session = updated

    def set_tools(self, names: Sequence[str]) -> None:
        app = self._app
        app._ensure_idle()
        if isinstance(names, (str, bytes)):
            raise TypeError("names must be a sequence of tool names.")
        selected = tuple(names)
        self._validate_tools(self.model, selected)
        updated = replace(app._session, enabled_tools=selected)
        app._save_session(updated)
        app._session = updated

    def list_sessions(self) -> "tuple[SessionInfo, ...]":
        return self._app._history.list_sessions()

    def new_session(self) -> SessionSnapshot:
        app = self._app
        app._ensure_idle()
        current = app._session
        app._save_session(current)
        fresh = Session(
            workspace=current.workspace,
            provider=current.provider,
            model_id=current.model_id,
            thinking=current.thinking,
            enabled_tools=current.enabled_tools,
        )
        app._history.save(fresh)
        app._session = fresh
        app._usage.select_session(fresh.id, fresh.usage)
        return self.session

    def restore_session(self, session_id: str) -> None:
        app = self._app
        app._ensure_idle()
        restored = app._history.load(session_id)
        if not Path(restored.workspace).is_dir():
            raise ValueError("该会话原工作区已不存在。")
        model = app._model_for(restored.provider, restored.model_id)
        self._validate_thinking(model, restored.thinking)
        self._validate_tools(model, restored.enabled_tools)
        app._session = restored
        app._usage.select_session(restored.id, restored.usage)

    async def send(self, prompt: str, *, stream: bool = True) -> "TurnResult":
        return await self._app._run_turn(prompt, stream=stream)

    async def cancel(self) -> None:
        self._app._ensure_open()
        await self._app._cancel_turn()

    def resolve_approval(self, turn_id: str, tool_call_id: str, approved: bool) -> bool:
        self._app._ensure_open()
        return self._app._resolve_approval(turn_id, tool_call_id, approved)

    async def close(self) -> None:
        await self._app.close()
