"""Public business operations and read-only views of an Application."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .attachments import (
    Attachment,
    AttachmentInput,
    DocumentAttachment,
    ImageAttachment,
    TextResourceInput,
    attachment_only_prompt,
    prepare_attachments,
    resource_snapshot_id,
    valid_id,
)
from .compaction import CompactionCheckpoint
from .config import load_mcp_config
from .events import Callback
from .history import SessionLease, finish_io
from .mcp import TOOL_PREFIX, MCPConnection, MCPStatus, validate_prompt_request
from .mcp.resources import expand_resource_template
from .mcp_store import MCPServerConfig
from .models import ModelInfo, list_models
from .project import Project
from .prompt_groups import PromptGroup, PromptInput
from .session import Session
from .skills import SkillInfo, SkillScope, SkillState
from .tools import ApprovalDecision, ApprovalMode
from .turn_state import RecoveryAction, RecoveryKind, RecoveryState

if TYPE_CHECKING:
    from mcp.types import GetPromptResult, Prompt, Resource, ResourceTemplate

    from .agent import TurnResult
    from .app import Application
    from .history import SessionInfo, SessionScan
    from .session_runtime import SessionRuntime
    from .token_budget import TokenBudget
    from .tools import ApprovalRequest
    from .tools.questions import Answers, QuestionRequest
    from .usage import UsageTotals


@dataclass(frozen=True)
class SessionSnapshot:
    id: str
    workspace: Path
    provider: str
    model_id: str
    title: str | None = None
    project_id: str | None = None


class Operations:
    """Shared application operations and explicitly targeted Session operations."""

    def __init__(self, app: "Application", *, initial_session_id: str | None = None) -> None:
        self._app = app
        self.initial_session_id = initial_session_id if initial_session_id is not None else app.initial_session_id
        if initial_session_id is not None:
            self.restore_session(initial_session_id)

    def _runtime(self, session_id: str) -> "SessionRuntime":
        return self._app._load_runtime(session_id)

    def session(self, session_id: str) -> SessionSnapshot:
        session = self._runtime(session_id).session
        workspace = Path(session.workspace)
        project_id = self._app._history.resolve_project_id(session.project_id, workspace)
        return SessionSnapshot(
            session.id, workspace, session.provider, session.model_id,
            session.title, project_id,
        )

    def message_snapshot(self, session_id: str) -> list[dict[str, Any]]:
        """Return detached copies of committed messages for display."""
        return self._runtime(session_id).session.messages.snapshot()

    def subscribe(self, name: str, callback: Callback) -> Callable[[], None]:
        """Subscribe to detached event snapshots and return an unsubscribe function."""
        return self._app.events.on(name, callback)

    def model(self, session_id: str) -> ModelInfo:
        return self._app._session_model(self._runtime(session_id).session)

    def thinking(self, session_id: str) -> str | None:
        return self._runtime(session_id).session.thinking

    @property
    def all_tools(self) -> tuple[str, ...]:
        return self.available_tools(self.initial_session_id)

    def available_tools(self, session_id: str) -> tuple[str, ...]:
        return self._app._available_tools(self._runtime(session_id).session.workspace)

    def enabled_tools(self, session_id: str) -> tuple[str, ...]:
        return self._app._selected_tools(self._runtime(session_id), effective=False)

    def effective_tools(self, session_id: str) -> tuple[str, ...]:
        return self._app._selected_tools(self._runtime(session_id), effective=True)

    def running(self, session_id: str) -> bool:
        return self._runtime(session_id).running

    def busy(self, session_id: str) -> bool:
        runtime = self._runtime(session_id)
        return runtime.running or runtime.state_lock.locked()

    def approval_mode(self, session_id: str) -> ApprovalMode:
        return self._runtime(session_id).approval_mode

    def set_approval_mode(self, mode: ApprovalMode | str, *, session_id: str) -> None:
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)
        runtime.approval_mode = ApprovalMode(mode)
        if runtime.approval_mode is ApprovalMode.ASK:
            app._revoke_session_trust(session_id)

    def pending_approval(self, session_id: str) -> "ApprovalRequest | None":
        return deepcopy(self._runtime(session_id).pending_approval)

    def pending_question(self, session_id: str) -> "QuestionRequest | None":
        runtime = self._runtime(session_id)
        if runtime.question_answer is None or runtime.question_answer.done():
            return None
        return deepcopy(runtime.pending_question)

    def answer_question(self, request_id: str, turn_id: str, answers: "Answers", *, session_id: str) -> bool:
        runtime = self._runtime(session_id)
        pending, future = runtime.pending_question, runtime.question_answer
        if pending is None or future is None or future.done() or (
            pending.request_id, pending.turn_id, pending.session_id
        ) != (request_id, turn_id, session_id):
            return False
        future.set_result(pending.validate_answers(answers))
        return True

    def session_usage(self, session_id: str) -> "UsageTotals":
        return self._runtime(session_id).usage.session

    def turn_usage(self, session_id: str) -> "UsageTotals | None":
        return self._runtime(session_id).usage.turn

    def context_budget(self, session_id: str) -> "TokenBudget | None":
        runtime = self._runtime(session_id)
        session, observed = runtime.session, runtime.last_budget
        if observed is not None and observed[:2] == (session.provider, session.model_id):
            return observed[2]
        return None

    def compacting(self, session_id: str) -> bool:
        return self._runtime(session_id).compacting

    def compacted_through(self, session_id: str) -> int:
        checkpoint = self._runtime(session_id).session.compaction
        return checkpoint.through if checkpoint else 0

    def configured(self, provider: str) -> bool:
        return provider in self._app._connections

    def list_models(self, session_id: str) -> dict[str, tuple[ModelInfo, ...]]:
        catalog = self._app._catalog
        model = self.model(session_id)
        providers = dict.fromkeys(
            ("deepseek", "mimo", *(item.provider for item in catalog), model.provider)
        )
        groups = {
            provider: list_models(provider, catalog=catalog) for provider in providers
        }
        if model not in groups[model.provider]:
            groups[model.provider] = (*groups[model.provider], model)
        return groups

    async def set_model(self, provider: str, model_id: str, *, session_id: str) -> str:
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)
        model = app._model_for(provider, model_id)
        current = runtime.session
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
        await app._update_session(runtime, updated)
        return "\n".join(notices)

    async def set_thinking(self, mode: str | None, *, session_id: str) -> None:
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)
        app._validate_thinking(self.model(session_id), mode)
        updated = replace(runtime.session, thinking=mode)
        await app._update_session(runtime, updated)

    def get_system_prompt(self, session_id: str) -> str:
        return self._runtime(session_id).session.effective_system_prompt

    async def set_system_prompt(self, prompt: str, *, session_id: str) -> None:
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)
        if not isinstance(prompt, str):
            raise TypeError("系统提示词必须是字符串。")
        await app._update_session(runtime, replace(runtime.session, system_prompt=prompt))

    async def reset_system_prompt(self, session_id: str) -> None:
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)
        await app._update_session(runtime, replace(runtime.session, system_prompt=None))

    async def set_tools(
        self, names: Sequence[str], *, session_id: str, keep_unavailable: bool = False
    ) -> None:
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)
        if isinstance(names, (str, bytes)):
            raise TypeError("names must be a sequence of tool names.")
        selected = tuple(names)
        available = set(self.available_tools(session_id))
        if keep_unavailable:
            selected += tuple(
                name
                for name in runtime.session.enabled_tools
                if name not in available and name not in selected
            )
        app._validate_tools(self.model(session_id), selected, retained=runtime.session.enabled_tools,
                            workspace=runtime.session.workspace)
        disabled = ((set(runtime.session.disabled_mcp_tools) | {
            name for name in available.union(runtime.session.enabled_tools) if name.startswith(TOOL_PREFIX)
        }) - set(selected))
        updated = replace(runtime.session, enabled_tools=selected, disabled_mcp_tools=tuple(sorted(disabled)))
        await app._update_session(runtime, updated)

    def list_skills(self, session_id: str) -> tuple[SkillInfo, ...]:
        """List global and workspace Skills, including invalid or missing entries."""
        workspace = self._runtime(session_id).session.workspace
        return self._app._skills.list(Path(workspace))

    async def import_skill(self, source: str | Path, scope: SkillScope | str, *, session_id: str) -> SkillInfo:
        """Copy a local package into the selected scope with state disabled."""
        workspace = Path(self._runtime(session_id).session.workspace)
        return await finish_io(asyncio.to_thread(self._app._skills.import_skill, source, SkillScope(scope), workspace))

    async def set_skill_state(self, skill_id: str, state: SkillState | str, *, session_id: str) -> SkillInfo:
        """Set the user's Skill state for future Turns."""
        workspace = Path(self._runtime(session_id).session.workspace)
        return await finish_io(asyncio.to_thread(self._app._skills.set_state, skill_id, SkillState(state), workspace))

    async def remove_skill(self, skill_id: str, *, session_id: str) -> None:
        """Remove the managed copy and its state record in the selected scope."""
        workspace = Path(self._runtime(session_id).session.workspace)
        await finish_io(asyncio.to_thread(self._app._skills.remove, skill_id, workspace))

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
        public_only: bool = False,
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
                public_only=public_only,
            )
        )

    async def reconnect_mcp(self, name: str, *, session_id: str | None = None) -> None:
        name = self._managed_mcp_key(name, session_id)
        if name in self._app._mcp_configs:
            if error := await self.restart_mcp(name, session_id=session_id):
                raise RuntimeError(error)
            return
        connection = self._app._mcp.get(name)
        if connection is None:
            raise ValueError(f"未知 MCP Server {name!r}。")
        await self._app._connect_mcp(connection)

    async def refresh_mcp(self, name: str, *, session_id: str | None = None) -> None:
        await self._app._refresh_mcp(self._managed_mcp_key(name, session_id))

    async def disconnect_mcp(self, name: str, *, session_id: str | None = None) -> None:
        name = self._managed_mcp_key(name, session_id)
        if name not in self._app._mcp:
            raise ValueError(f"MCP Server {name!r} 未连接。")
        await self._app._disconnect_mcp(name)

    def list_mcp_servers(self, session_id: str | None = None) -> tuple[str, ...]:
        return tuple(server.name for server in self._workspace_mcp_connections(session_id))

    def mcp_status(self, session_id: str | None = None) -> tuple[MCPStatus, ...]:
        return tuple(server.status for server in self._workspace_mcp_connections(session_id))

    def managed_mcp(self, session_id: str | None = None) -> tuple[MCPServerConfig, ...]:
        workspace = self._runtime(session_id or self.initial_session_id).session.workspace
        return tuple(config for config in self._app._mcp_store.read().values() if config.applies_to(workspace))

    async def start_mcp(self, *, session_id: str) -> None:
        await self._app._ensure_always_mcp(self._runtime(session_id).session.workspace)

    def _managed_mcp_key(self, server: str, session_id: str | None) -> str:
        workspace = self._runtime(session_id or self.initial_session_id).session.workspace
        for config in self._app._workspace_mcp(workspace):
            if server in (config.id, config.key):
                return config.key
        if ":" not in server and server not in self._app._mcp_configs:
            return server  # Global, explicitly connected Servers also support Resources and Prompts.
        raise ValueError("当前工作区没有此 MCP 服务。")

    def import_mcp(self, path: Path) -> None:
        """Seed managed configuration once, including an intentionally empty list."""
        app = self._app
        app._ensure_mcp_idle()
        store = app._mcp_store
        if store.path.exists():
            return
        with SessionLease(store.path):
            if not store.path.exists():
                configs = {}
                for name, options in load_mcp_config(path).items():
                    if isinstance(options.get("cwd"), Path):
                        options["cwd"] = str(options["cwd"])
                    config = MCPServerConfig(name, "always", options)
                    config.connection(app._history_redactor)
                    configs[name] = config
                store.write(configs)
                app._mcp_configs = configs

    async def save_mcp(
        self, name: str, *, state: str | None = None, enabled: bool | None = None,
        options: dict[str, Any] | None = None, scope: str = "global", description: str | None = None,
        create: bool = False, session_id: str | None = None,
    ) -> str | None:
        if scope not in ("global", "project"):
            raise ValueError("MCP 作用域须为 global 或 project。")
        if enabled is not None:
            if type(enabled) is not bool or state is not None:
                raise ValueError("请指定一种 MCP 加载策略。")
            state = "always" if enabled else "disabled"
        app = self._app
        with app._changing_mcp():
            store = app._mcp_store
            with SessionLease(store.path):
                configs = store.read()
                if create:
                    workspace = self._runtime(session_id or self.initial_session_id).session.workspace if scope == "project" else None
                    key = MCPServerConfig(name, "disabled", {}, workspace).key
                    current = configs.get(key)
                    if current is not None or key in app._mcp:
                        raise ValueError("该作用域已有同名 MCP 服务。")
                else:
                    key = self._managed_mcp_key(name, session_id)
                    current = configs.get(key)
                    if current is None:
                        raise ValueError("MCP 服务不存在。")
                    name, workspace = current.name, current.workspace
                if options is None and current is None:
                    raise ValueError("请提供 MCP 服务配置。")
                options = deepcopy(options if options is not None else current.options)
                if current and current.options.get("public_only"):
                    options["public_only"] = True
                config = MCPServerConfig(
                    name, state if state is not None else current.state if current else "disabled", options,
                    workspace, description if description is not None else current.description if current else "",
                )
                config.connection(app._history_redactor)
                if config == current:
                    return None
                configs[key] = config
                store.write(configs)
            return await finish_io(app._apply_mcp_config(key, config, connect=config.state == "always", changed=True))

    async def restart_mcp(self, name: str, *, session_id: str | None = None) -> str | None:
        app = self._app
        with app._changing_mcp():
            key = self._managed_mcp_key(name, session_id)
            config = app._mcp_configs.get(key)
            if config is None or not config.enabled:
                raise ValueError("请先启用 MCP 服务。")
            return await finish_io(app._apply_mcp_config(key, config, connect=True, changed=False))

    async def delete_mcp(self, name: str, *, session_id: str | None = None) -> None:
        app = self._app
        with app._changing_mcp():
            key = self._managed_mcp_key(name, session_id)
            store = app._mcp_store
            with SessionLease(store.path):
                configs = store.read()
                if configs.pop(key, None) is None:
                    raise ValueError("MCP 服务不存在。")
                store.write(configs)
            await finish_io(app._apply_mcp_config(key, None, connect=False, changed=True))

    def list_mcp_tools(self, session_id: str | None = None) -> tuple[str, ...]:
        return tuple(name for name in self.available_tools(session_id or self.initial_session_id)
                     if name.startswith(TOOL_PREFIX))

    def _workspace_mcp_connections(self, session_id: str | None):
        app = self._app
        workspace = self._runtime(session_id or self.initial_session_id).session.workspace
        return (
            connection for key, connection in app._mcp.items()
            if key not in app._mcp_configs or app._mcp_configs[key].applies_to(workspace)
        )

    def list_mcp_resource_servers(self, session_id: str | None = None) -> tuple[str, ...]:
        return tuple(server.name for server in self._workspace_mcp_connections(session_id) if server.supports_resources)

    async def list_mcp_resources(self, server: str, *, session_id: str | None = None) -> "tuple[Resource, ...]":
        key = self._managed_mcp_key(server, session_id)
        return await self._app._mcp_resource_connection(key).list_resources()

    async def list_mcp_resource_templates(self, server: str, *, session_id: str | None = None) -> "tuple[ResourceTemplate, ...]":
        key = self._managed_mcp_key(server, session_id)
        return await self._app._mcp_resource_connection(key).list_resource_templates()

    async def read_mcp_resource(
        self, server: str, uri: str, arguments: dict[str, str] | None = None, *, session_id: str | None = None,
    ) -> tuple[TextResourceInput, ...]:
        key = self._managed_mcp_key(server, session_id)
        connection = self._app._mcp_resource_connection(key)
        if arguments is None:
            entry = next((item for item in await connection.list_resources() if str(item.uri) == uri), None)
            target = uri
        else:
            entry = next((item for item in await connection.list_resource_templates() if item.uri_template == uri), None)
            target = expand_resource_template(uri, arguments)
        if entry is None:
            raise ValueError("未找到该 MCP 资源，请刷新目录。")
        return await connection.read_resource(target, name=entry.title or entry.name, mime_type=entry.mime_type)

    def list_mcp_prompt_servers(self, session_id: str | None = None) -> tuple[str, ...]:
        return tuple(server.name for server in self._workspace_mcp_connections(session_id) if server.supports_prompts)

    async def list_mcp_prompts(self, server: str, *, session_id: str | None = None) -> "tuple[Prompt, ...]":
        key = self._managed_mcp_key(server, session_id)
        return await self._app._mcp_prompt_connection(key).list_prompts()

    async def get_mcp_prompt(
        self, server: str, name: str, arguments: dict[str, str], *, session_id: str | None = None,
    ) -> "GetPromptResult":
        key = self._managed_mcp_key(server, session_id)
        connection = self._app._mcp_prompt_connection(key)
        validate_prompt_request(name, arguments)
        arguments = dict(arguments)
        prompt = next((item for item in await connection.list_prompts() if item.name == name), None)
        if prompt is None:
            raise ValueError("未找到该 MCP 模板，请刷新模板列表。")
        fields = {field.name: field for field in prompt.arguments or ()}
        if arguments.keys() - fields.keys():
            raise ValueError("MCP 模板包含未定义的参数。")
        if any(field.required and not arguments.get(field.name, "").strip() for field in fields.values()):
            raise ValueError("请填写 MCP 模板的必填参数。")
        return await connection.get_prompt(name, arguments)

    def list_sessions(self) -> "tuple[SessionInfo, ...]":
        return self._app._history.list_sessions()

    def scan_sessions(self) -> "SessionScan":
        return self._app._history.scan_sessions()

    def list_projects(self) -> tuple[Project, ...]:
        return self._app._history.list_projects()

    async def create_project(
        self, name: str, workspace: Path | str | None = None, *, session_id: str,
    ) -> Project:
        root = self.session(session_id).workspace
        if workspace is not None:
            if not str(workspace).strip():
                raise ValueError("项目需要一个工作区目录。")
            root = root / Path(workspace).expanduser()
        return await finish_io(asyncio.to_thread(self._app._history.create_project, name, root))

    async def remove_project(self, project_id: str) -> None:
        self._app._ensure_open()
        await finish_io(asyncio.to_thread(self._app._history.remove_project, project_id))

    def _project_for_workspace(self, project_id: str) -> Project:
        project = self._app._history.load_project(project_id)
        if not project.workspace.is_dir() or project.workspace.resolve() != project.workspace:
            raise ValueError("该项目的工作区已移动或不存在。")
        return project

    async def open_project(self, project_id: str | None, *, session_id: str) -> SessionSnapshot:
        current = self._runtime(session_id).session
        project = self._project_for_workspace(project_id) if project_id is not None else None
        workspace = project.workspace if project is not None else current.workspace
        await self._app._ensure_always_mcp(workspace)
        return self._new_session(current, workspace, project_id)

    async def new_session(self, *, session_id: str) -> SessionSnapshot:
        current = self._runtime(session_id).session
        project_id = self._app._history.resolve_project_id(current.project_id, current.workspace)
        return self._new_session(current, current.workspace, project_id)

    async def remember_session(self, session_id: str) -> None:
        """Keep a blank session restorable while a client holds an unsent draft."""
        app = self._app
        runtime = self._runtime(session_id)
        if runtime.session._saved_revision is not None:
            return
        app._ensure_idle(runtime)
        async with runtime.state_lock:
            await app._save_session(runtime, runtime.session)

    def _new_session(
        self, source: Session, workspace: Path | str, project_id: str | None,
    ) -> SessionSnapshot:
        fresh = Session(
            workspace=workspace,
            provider=source.provider,
            model_id=source.model_id,
            thinking=source.thinking,
            enabled_tools=source.enabled_tools,
            disabled_mcp_tools=source.disabled_mcp_tools,
            project_id=project_id,
        )
        self._app._add_runtime(fresh)
        return self.session(fresh.id)

    def restore_session(self, session_id: str) -> SessionSnapshot:
        return self.session(session_id)

    async def rename_session(self, session_id: str, title: str) -> None:
        runtime = self._runtime(session_id)
        self._app._ensure_idle(runtime)
        title = title.strip() if isinstance(title, str) else ""
        if not 0 < len(title) <= 120:
            raise ValueError("会话名称必须包含 1–120 个字符。")
        await self._app._update_session(runtime, replace(runtime.session, title=title))

    async def move_session(
        self, session_id: str, project_id: str | None, *, expected_workspace: Path | str | None = None,
    ) -> None:
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)
        session = runtime.session
        workspace = session.workspace
        if project_id is not None:
            try:
                project = self._project_for_workspace(project_id)
            except FileNotFoundError as exc:
                raise ValueError("项目不存在。") from exc
            if expected_workspace is not None and Path(expected_workspace) != project.workspace:
                raise ValueError("目标工作目录已变化，请重新确认。")
            workspace = project.workspace
        if session.project_id == project_id and session.workspace == workspace:
            return
        workspace_changed = session.workspace != workspace
        recovery = session.recovery
        if workspace_changed and recovery is not None and recovery.kind is RecoveryKind.INTERRUPTED_TURN:
            raise ValueError("请先继续或终止未完成的工具任务，再更换工作目录。")
        updated = replace(session, project_id=project_id, workspace=workspace)

        async def move() -> None:
            await app._save_and_publish_session(runtime, updated)
            if workspace_changed:
                app._revoke_session_trust(session_id)
                runtime.activated_mcp.intersection_update(config.key for config in app._workspace_mcp(workspace))

        async with runtime.state_lock:
            await finish_io(move())

    async def delete_session(self, session_id: str, *, replacement: bool = False) -> SessionSnapshot | None:
        """Delete under the history lease; optionally return a blank session for the caller."""
        app = self._app
        runtime = self._runtime(session_id)
        app._ensure_idle(runtime)

        async def delete() -> SessionSnapshot | None:
            session = runtime.session
            await app._history.delete(session)
            app._drop_runtime(session_id)
            if replacement and not app._closed:
                project_id = app._history.resolve_project_id(session.project_id, session.workspace)
                return self._new_session(session, session.workspace, project_id)
            return None

        async with runtime.state_lock:
            return await finish_io(delete())

    def attachment_snapshot(self, session_id: str) -> dict[int, list[dict[str, str | int]]]:
        return {
            index: [
                {"id": item.id, "name": item.name, "type": "image", "mime_type": item.mime_type,
                 "width": item.width, "height": item.height}
                if isinstance(item, ImageAttachment)
                else {"id": item.id, "name": item.name, "type": "document", "mime_type": item.mime_type,
                      "size_bytes": item.size_bytes, "reading_mode": item.reading_mode}
                if isinstance(item, DocumentAttachment)
                else {"id": item.id, "name": item.name, "type": "text", "mime_type": item.mime_type}
                for item in batch.files
            ]
            for index, batch in self._runtime(session_id).session.attachment_refs.items()
        }

    async def read_attachment_image(self, session_id: str, message_index: int, attachment_id: str) -> tuple[bytes, str]:
        data, item = await self._read_binary_attachment(session_id, message_index, attachment_id, ImageAttachment)
        return data, item.mime_type

    async def read_attachment_document(
        self, session_id: str, message_index: int, attachment_id: str,
    ) -> tuple[bytes, str, str]:
        data, item = await self._read_binary_attachment(session_id, message_index, attachment_id, DocumentAttachment)
        return data, item.mime_type, item.name

    async def _read_binary_attachment(
        self, session_id: str, message_index: int, attachment_id: str,
        kind: type[ImageAttachment] | type[DocumentAttachment],
    ) -> tuple[bytes, ImageAttachment | DocumentAttachment]:
        app = self._app
        if not valid_id(attachment_id) or type(message_index) is not int:
            raise FileNotFoundError("附件不可用。")
        try:
            runtime = self._runtime(session_id)
        except (OSError, ValueError):
            raise FileNotFoundError("附件不可用。") from None
        session = runtime.session
        batch = session.attachment_refs.get(message_index)
        item = next((entry for entry in batch.files if isinstance(entry, kind) and entry.id == attachment_id), None) if batch else None
        if item is None:
            raise FileNotFoundError("附件不可用。")
        store = app._history.attachments
        try:
            if isinstance(item, ImageAttachment):
                data = await asyncio.to_thread(store.read_image, session.id, item)
            else:
                data = await asyncio.to_thread(store.read_document, session.id, item)
        except (OSError, ValueError):
            raise FileNotFoundError("附件不可用。") from None
        if app._runtimes.get(session_id) is not runtime:
            raise FileNotFoundError("附件不可用。")
        return data, item

    def attachment_submitted(
        self, submission_id: str, prompt: str, attachments: Sequence[AttachmentInput | str], *, session_id: str,
    ) -> bool:
        session = self._runtime(session_id).session

        def identity(item: Attachment):
            if isinstance(item, DocumentAttachment):
                return (item.name, item.sha256, item.mime_type, item.reading_mode)
            if isinstance(item, ImageAttachment):
                return (item.name, item.sha256, item.mime_type)
            return item.id if item.source else (item.name, item.sha256)

        for index, batch in session.attachment_refs.items():
            if batch.submission_id != submission_id:
                continue
            received = []
            for entry in attachments:
                if isinstance(entry, str):
                    received.append(resource_snapshot_id(submission_id, entry))
                else:
                    item, = prepare_attachments((entry,), submission_id).files
                    received.append(identity(item))
            committed = [identity(item) for item in batch.files]
            default = attachment_only_prompt(batch.files)
            if (session.messages.get(index)["content"] != (prompt if prompt.strip() else default)
                or committed != received):
                raise ValueError("该附件发送标识已用于不同内容。")
            return True
        return False

    async def send(
        self, prompt: str, *, session_id: str, stream: bool = True,
        attachments: Sequence[AttachmentInput] = (), submission_id: str | None = None,
    ) -> "TurnResult":
        return await self._app._run_turn(
            self._runtime(session_id), prompt, stream=stream, attachments=attachments, submission_id=submission_id,
        )

    def prompt_groups(self, session_id: str) -> tuple[PromptGroup, ...]:
        return self._runtime(session_id).session.prompt_groups

    def prompt_submitted(self, group_id: str, texts: tuple[str, ...], *, session_id: str) -> bool:
        session = self._runtime(session_id).session
        group = next((group for group in session.prompt_groups if group.id == group_id), None)
        if group is None:
            return False
        if texts != tuple(session.messages.get(i)["content"] for i in range(group.start, group.end)):
            raise ValueError("该 MCP Prompt 运行标识已用于不同内容。")
        return True

    async def run_mcp_prompt(self, prompt_input: PromptInput, *, session_id: str, stream: bool = True) -> "TurnResult":
        return await self._app._run_turn(self._runtime(session_id), stream=stream, prompt_input=prompt_input)

    def recovery(self, session_id: str) -> RecoveryState | None:
        return self._runtime(session_id).session.recovery

    async def retry_failed_user(self, key: str, *, session_id: str, stream: bool = True) -> "TurnResult":
        return await self._app._run_turn(
            self._runtime(session_id), stream=stream, recovery_action=RecoveryAction.RETRY, recovery_key=key,
        )

    async def continue_interrupted_turn(self, key: str, *, session_id: str, stream: bool = True) -> "TurnResult":
        return await self._app._run_turn(
            self._runtime(session_id), stream=stream, recovery_action=RecoveryAction.CONTINUE, recovery_key=key,
        )

    async def terminate_interrupted_turn(self, key: str, *, session_id: str) -> "TurnResult":
        return await self._app._run_turn(
            self._runtime(session_id), recovery_action=RecoveryAction.TERMINATE, recovery_key=key,
        )

    async def compact(self, session_id: str) -> CompactionCheckpoint | None:
        return await self._app._compact_session(self._runtime(session_id))

    async def cancel(self, session_id: str) -> None:
        await self._app._cancel_turn(self._runtime(session_id))

    def resolve_approval(
        self, request_id: str, decision: ApprovalDecision | str, *, session_id: str,
    ) -> bool:
        return self._app._resolve_approval(session_id, request_id, ApprovalDecision(decision))

    async def close(self) -> None:
        await self._app.close()
