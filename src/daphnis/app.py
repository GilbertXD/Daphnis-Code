"""Own Daphnis runtime resources and supervise Turn execution."""

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AsyncExitStack, contextmanager
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import Agent, TurnResult, TurnStatus
from .attachments import (
    AttachmentBatch,
    AttachmentInput,
    ImageAttachmentInput,
    TextAttachmentInput,
    TextResourceInput,
    attachment_only_prompt,
    prepare_attachments,
)
from .compaction import CompactionCheckpoint, CompactionError
from .compaction_settings import CompactionSettings
from .context import Context
from .errors import ValidationError
from .events import Event, EventBus
from .history import (
    History,
    SessionLease,
    finish_io,
)
from .llm.client import ChatClient
from .mcp import TOOL_PREFIX, MCPConnection, MCPPromptError, MCPResourceError
from .mcp_store import MCPStore
from .models import (
    MODELS,
    CapabilityError,
    ModelInfo,
    provider_compatibility,
    require_tools,
    resolve_model,
)
from .models.documents import DocumentAdapter
from .models.models import validate_catalog
from .models.providers import provider_documents
from .models.requests import require_document_input, require_image_input
from .prompt_groups import PromptInput
from .redaction import Redactor
from .session import Session
from .session_runtime import SessionRuntime
from .skills import SkillStore
from .title import build_title_source, fallback_title, normalize_title
from .token_budget import TokenBudget
from .tools import (
    ApprovalDecision,
    ApprovalRequest,
    ToolExecutor,
    create_tools,
)
from .tools.approval import ToolError
from .tools.questions import Answers, QuestionRequest
from .turn_state import RecoveryAction

logger = logging.getLogger(__name__)
_TITLE_TIMEOUT = 5
_EVENT_DRAIN_TIMEOUT = 0.5


class SessionSaveError(RuntimeError):
    """Saving API usage after the last message commit failed."""

    def __init__(self, result: TurnResult, error: Exception) -> None:
        super().__init__(f"已报告的用量保存失败：{error}")
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
        # Accepted for compatibility while HTTP observation is disabled.
        observe_model_io: bool = False,
        max_primary_requests: int = 20,
        max_maintenance_requests: int = 12,
        max_total_requests: int = 32,
        recent_tokens: int | None = None,
        compaction_settings: CompactionSettings | None = None,
        model_compaction_settings: Mapping[tuple[str, str], CompactionSettings] | None = None,
        output_limits: Mapping[tuple[str, str], int] | None = None,
    ) -> None:
        self._catalog = tuple(catalog)
        validate_catalog(self._catalog)
        self._connections = {
            name: dict(options) for name, options in connections.items()
        }
        secrets = []
        for options in self._connections.values():
            if isinstance(key := options.get("api_key"), str):
                secrets.append(key)
            if isinstance(headers := options.get("auth_headers"), Mapping):
                secrets.extend(value for value in headers.values() if isinstance(value, str))
        self._history_redactor = Redactor(secrets)
        self._clients: dict[str, ChatClient] = {}
        self._document_adapters: dict[str, DocumentAdapter | None] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._history = history if history is not None else History()
        self._skills = SkillStore(self._history.directory.parent)
        self._mcp_store = MCPStore(self._history.directory.parent / "mcp.json")
        self._interactive = interactive
        self._request_limits = {
            "max_primary_requests": max_primary_requests,
            "max_maintenance_requests": max_maintenance_requests,
            "max_total_requests": max_total_requests,
        }
        compaction_settings = compaction_settings or CompactionSettings()
        self._compaction_settings = (
            compaction_settings if recent_tokens is None
            else replace(compaction_settings, recent_tokens=recent_tokens)
        )
        self._model_compaction_settings = dict(model_compaction_settings or {})
        self._output_limits = dict(output_limits or {})
        self._trusted_tools: set[tuple[str, str]] = set()
        self._mcp: dict[str, MCPConnection] = {}
        self._mcp_update: asyncio.Future[None] | None = None
        self._mcp_unverified: set[MCPConnection] = set()
        self._tools = self._mcp_catalog()
        model = self._model_for(provider, model_id)
        session = Session(
            workspace=workspace,
            provider=provider,
            model_id=model_id,
            enabled_tools=self._all_tools if model.supports_tools is True else (),
        )
        if not Path(session.workspace).is_dir():
            raise ValueError("Workspace 必须是现有目录。")
        self.events = EventBus()
        self._runtimes: dict[str, SessionRuntime] = {}
        self._add_runtime(session)
        self.initial_session_id = session.id
        for name in ("context.budget", "compaction.started", "compaction.finished"):
            self.events.on(name, self._observe_context)
        for name in ("turn.started", "model.started", "model.usage"):
            self.events.on(name, self._observe_usage)
        self._client(provider)

    def _add_runtime(self, session: Session) -> SessionRuntime:
        if session.id in self._runtimes:
            raise ValueError("会话已加载。")
        self._skills.load_workspace(Path(session.workspace))
        runtime = SessionRuntime(session)
        self._runtimes[session.id] = runtime
        return runtime

    def _load_runtime(self, session_id: str) -> SessionRuntime:
        self._ensure_open()
        # Loading and inserting happen on the event loop without an intervening await.
        if runtime := self._runtimes.get(session_id):
            return runtime
        session = self._history.load(session_id)
        if "history_search" in session.enabled_tools:
            session = replace(session, enabled_tools=tuple(dict.fromkeys(
                "retrieval" if name == "history_search" else name for name in session.enabled_tools
            )))
        self._validate_session(session)
        return self._add_runtime(session)

    def _validate_session(self, session: Session) -> None:
        if not Path(session.workspace).is_dir():
            raise ValueError("该会话原工作区已不存在。")
        model = self._model_for(session.provider, session.model_id)
        self._validate_thinking(model, session.thinking)
        self._validate_tools(
            model,
            session.enabled_tools,
            retained=tuple(name for name in session.enabled_tools if name.startswith(TOOL_PREFIX)),
        )

    def _drop_runtime(self, session_id: str) -> None:
        self._runtimes.pop(session_id, None)
        self._revoke_session_trust(session_id)

    def _revoke_session_trust(self, session_id: str) -> None:
        self._trusted_tools.difference_update({key for key in self._trusted_tools if key[0] == session_id})

    def _observe_context(self, event: Event) -> None:
        runtime = self._runtimes.get(event.session_id)
        if runtime is None:
            return
        if event.name == "context.budget":
            data = event.data
            runtime.last_budget = (
                data["provider"], data["model_id"],
                TokenBudget(**{field.name: data[field.name] for field in fields(TokenBudget)}),
            )
        else:
            runtime.compacting = event.name == "compaction.started"

    def _observe_usage(self, event: Event) -> None:
        if runtime := self._runtimes.get(event.session_id):
            runtime.usage.observe(event)

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
        available = set(self._all_tools)
        for name in names:
            if not isinstance(name, str) or (
                name not in available and name not in retained
            ):
                raise ValidationError(
                    f"Tool {name!r} is not registered or currently available."
                )
        self._tools.select([name for name in names if name in available])
        if names:
            require_tools(model)

    def _document_adapter(self, provider: str) -> DocumentAdapter | None:
        if provider not in self._document_adapters:
            self._document_adapters[provider] = provider_documents(
                provider, self._connection(provider), self._history.directory.parent,
            )
        return self._document_adapters[provider]

    def _session_model(self, session: Session) -> ModelInfo:
        return resolve_model(session.provider, session.model_id, catalog=self._catalog)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Application 已关闭。")

    def _ensure_idle(self, runtime: SessionRuntime) -> None:
        self._ensure_open()
        if self._runtimes.get(runtime.session.id) is not runtime:
            raise FileNotFoundError("会话已关闭。")
        if runtime.running:
            raise RuntimeError("请先等待或取消当前 Turn，再执行该操作。")
        if runtime.state_lock.locked():
            raise RuntimeError("请等待当前会话更新完成。")

    def _active_tasks(self) -> list[asyncio.Task]:
        return [runtime.task for runtime in self._runtimes.values() if runtime.task is not None and runtime.running]

    def _ensure_mcp_idle(self) -> None:
        self._ensure_open()
        if self._mcp_update is not None:
            raise RuntimeError("请等待 MCP 更新完成。")
        if self._active_tasks():
            raise RuntimeError("请先等待或取消当前 Turn，再更新 MCP。")

    @contextmanager
    def _changing_mcp(self) -> Iterator[None]:
        # Reserve the shared catalog without holding a lock during network I/O.
        self._ensure_mcp_idle()
        update = self._mcp_update = asyncio.get_running_loop().create_future()
        try:
            yield
        finally:
            self._mcp_update = None
            update.set_result(None)

    @property
    def _all_tools(self) -> tuple[str, ...]:
        offline = tuple(
            server.prefix for server in self._mcp.values() if not server.available
        )
        return tuple(
            name
            for item in self._tools.definitions()
            if (name := item["function"]["name"]) != "read_skill" and not name.startswith(offline)
        )

    async def _mcp_error(self, name: str, error: str) -> None:
        self._revoke_mcp_trust(name)
        if (connection := self._mcp.get(name)) and not connection.available:
            self._mcp_unverified.discard(connection)
        await self.events.emit(
            "mcp.error", session_id="", turn_id="", server=name, error=error
        )

    def _mcp_prompt_connection(self, server: str) -> MCPConnection:
        self._ensure_open()
        connection = self._mcp.get(server) if isinstance(server, str) else None
        if connection is None or not connection.available:
            raise MCPPromptError("MCP 服务未连接。")
        if not connection.supports_prompts:
            raise MCPPromptError("此 MCP 服务不支持模板。")
        return connection

    def _mcp_resource_connection(self, server: str) -> MCPConnection:
        self._ensure_open()
        connection = self._mcp.get(server) if isinstance(server, str) else None
        if connection is None or not connection.available:
            raise MCPResourceError("MCP 服务未连接。")
        if not connection.supports_resources:
            raise MCPResourceError("此 MCP 服务不支持资源。")
        return connection

    def _mcp_catalog(self, connecting: MCPConnection | None = None) -> ToolExecutor:
        return create_tools(
            approval=self._request_approval,
            question=self._request_question,
            trusted=self._trusted_tools,
            redactor=self._history_redactor,
            search_excluded_directory=self._history.directory,
            attachment_store=self._history.attachments,
            skill_state_paths=(self._skills.state_path, self._skills.lock_path),
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
        with self._changing_mcp():
            await self._start_mcp(connection)

    async def _start_mcp(self, connection: MCPConnection) -> None:
        previous = self._mcp.get(connection.name)
        if previous and previous.state == "Connected":
            raise ValueError(f"MCP Server {connection.name!r} 已连接。")
        self._revoke_mcp_trust(connection.name)
        if previous:
            self._mcp_unverified.discard(previous)
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
        with self._changing_mcp():
            connection = self._mcp[name]
            self._revoke_mcp_trust(name)
            self._mcp_unverified.discard(connection)
            await connection.aclose()
            self._tools = self._mcp_catalog()

    async def _refresh_mcp(self, name: str) -> None:
        with self._changing_mcp():
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
            self._revoke_mcp_trust(connection.name)
            if connection.available:
                self._mcp_unverified.add(connection)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RuntimeError("MCP 工具刷新失败，已保留上次目录。") from None
        self._revoke_mcp_trust(
            connection.name,
            {name for name in previous if previous[name] != discovered.get(name)},
        )
        self._mcp_unverified.discard(connection)
        await self.events.emit(
            "mcp.tools_changed", session_id="", turn_id="", server=connection.name
        )

    async def _mcp_tools_changed(self, connection: MCPConnection) -> None:
        # Active Turns keep their definitions; revoke automatic trust immediately.
        self._revoke_mcp_trust(connection.name)
        self._mcp_unverified.add(connection)
        while not self._closed and self._mcp.get(connection.name) is connection and connection.available:
            if self._mcp_update is not None:
                await asyncio.wait((self._mcp_update,))
                continue
            if active := self._active_tasks():
                await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                continue
            with self._changing_mcp():
                try:
                    await self._refresh_mcp_catalog(connection)
                except RuntimeError as exc:
                    if not self._closed:
                        await self._mcp_error(connection.name, str(exc))
            return

    async def _request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        if not self._interactive:
            return ApprovalDecision.DENY
        runtime = self._runtimes.get(request.session_id)
        if self._closed or runtime is None:
            return ApprovalDecision.DENY
        if runtime.approval is not None:
            raise RuntimeError("已有待处理审批。")
        future = asyncio.get_running_loop().create_future()
        runtime.approval = future
        runtime.pending_approval = deepcopy(request)
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
                    and connection not in self._mcp_unverified
                ):
                    self._trusted_tools.add((request.session_id, request.tool_name))
            return decision
        finally:
            if not future.done():
                future.cancel()
            runtime.pending_approval = None
            runtime.approval = None

    def _resolve_approval(
        self, session_id: str, request_id: str, decision: ApprovalDecision
    ) -> bool:
        runtime = self._runtimes.get(session_id)
        if self._closed or runtime is None:
            return False
        pending = runtime.pending_approval
        if pending is None or (pending.session_id, pending.request_id) != (session_id, request_id):
            return False
        if runtime.approval is None or runtime.approval.done():
            return False
        runtime.approval.set_result(decision)
        return True

    async def _request_question(self, request: QuestionRequest) -> Answers:
        runtime = self._runtimes.get(request.session_id)
        if self._closed or runtime is None or not self.events.has_subscribers("tool.question_requested"):
            raise ToolError("question_unavailable", "No interactive recipient is available for this question.")
        if runtime.question_answer is not None:
            raise RuntimeError("已有待回答的问题。")
        future: asyncio.Future[Answers] = asyncio.get_running_loop().create_future()
        runtime.question_answer = future
        runtime.pending_question = request
        try:
            await self.events.emit(
                "tool.question_requested", session_id=request.session_id, turn_id=request.turn_id,
                request_id=request.request_id,
            )
            return await future
        finally:
            if not future.done():
                future.cancel()
            runtime.pending_question = None
            runtime.question_answer = None

    async def _run_turn(
        self, runtime: SessionRuntime, prompt: str | None = None, *, stream: bool = True,
        recovery_action: RecoveryAction | None = None, recovery_key: str | None = None,
        prompt_input: PromptInput | None = None,
        attachments: Sequence[AttachmentInput] = (), submission_id: str | None = None,
    ) -> TurnResult:
        self._ensure_idle(runtime)
        if self._mcp_update is not None:
            raise RuntimeError("请等待 MCP 更新完成。")
        session = runtime.session
        model = self._model_for(session.provider, session.model_id)
        uploads = tuple(attachments)
        if any(isinstance(item, ImageAttachmentInput) for item in uploads):
            require_image_input(model)
        batch = prepare_attachments(uploads, submission_id) if uploads else None
        if batch and batch.documents:
            require_document_input(model)
        if prompt_input and (uploads or submission_id is not None):
            raise ValueError("MCP Prompt 必须作为独立任务运行。")
        if prompt_input:
            uploads = tuple(prompt_input.resources.values())
        for item in uploads:
            if isinstance(item, TextResourceInput) and (
                self._history_redactor.json(item.text) != item.text or self._history_redactor.text(item.uri) != item.uri
            ):
                raise ValueError("MCP 资源包含未确认的敏感内容。")
        if batch and isinstance(prompt, str) and not prompt.strip():
            prompt = attachment_only_prompt(batch.files)
        thinking = session.thinking
        available = set(self._all_tools)
        enabled_tools = [name for name in session.enabled_tools if name in available]
        skills = None if recovery_action is RecoveryAction.TERMINATE else self._skills.snapshot(Path(session.workspace))
        if skills and skills.enabled and model.supports_tools is True:
            enabled_tools.append("read_skill")
        recovery = session.recovery if recovery_action is not None else None
        skill_result_start = recovery.message_index if recovery else len(session.messages)
        provider = session.provider
        custom_compatibility = (
            self._connection(provider).get("compatibility") is not None
        )
        system_prompt = session.effective_system_prompt
        if session.system_prompt is None:
            system_prompt += "\n\n当前会话工作区：" + json.dumps(str(session.workspace), ensure_ascii=False)
        agent = Agent(
            client=self._client(provider),
            provider=provider,
            catalog=self._catalog,
            custom_compatibility=custom_compatibility,
            tools=self._tools.select(enabled_tools, mode=runtime.approval_mode),
            events=self.events,
            context=Context(system_prompt=system_prompt, skills=skills, skill_result_start=skill_result_start),
            compaction_settings=self._model_compaction_settings.get(
                (provider, session.model_id), self._compaction_settings,
            ),
            **self._request_limits,
        )

        async def run():
            async with self._history.lease(session) as lease:
                if prompt_input and any(group.id == prompt_input.id for group in session.prompt_groups):
                    raise ValueError("该 MCP Prompt 已提交。")
                if submission_id is not None and any(
                    existing.submission_id == submission_id for existing in session.attachment_refs.values()
                ):
                    raise ValueError("该附件发送请求已接受。")
                save_failed = False
                title_source = None
                input_start = len(session.messages)

                async def persist(candidate: Session) -> None:
                    nonlocal save_failed, title_source
                    if runtime.session is not session:
                        raise RuntimeError("Session changed while its Turn was running.")
                    try:
                        added = candidate.attachment_refs.keys() - session.attachment_refs.keys()
                        excerpts = {}
                        if added:
                            snapshots = AttachmentBatch(tuple(
                                item for index in sorted(added) for item in candidate.attachment_refs[index].files
                            ))
                            try:
                                await finish_io(asyncio.to_thread(
                                    self._history.attachments.write, session.id, snapshots, uploads,
                                ))
                            except OSError:
                                raise OSError("无法持久化附件快照。") from None
                            excerpts = {
                                item.id: upload.text for item, upload in zip(snapshots.files, uploads, strict=True)
                                if isinstance(upload, TextResourceInput)
                            }
                        if candidate.title is None and recovery_action is None and len(session.messages) == input_start:
                            source = build_title_source(
                                candidate.messages.snapshot()[input_start:],
                                {index - input_start: candidate.attachment_refs[index] for index in added},
                                excerpts=excerpts, redactor=self._history_redactor,
                            )
                            candidate.title = fallback_title(source)
                            if candidate.title:
                                title_source = source
                        await self._save_session(runtime, candidate, lease=lease)
                    except Exception:
                        save_failed = True
                        raise

                result: TurnResult | None = None
                try:
                    result = await agent.run(
                        session=session,
                        prompt=prompt,
                        prompt_input=prompt_input,
                        recovery_action=recovery_action,
                        recovery_key=recovery_key,
                        stream=stream,
                        thinking_mode=thinking,
                        output_limit=self._output_limits.get((provider, session.model_id)),
                        persist=persist,
                        attachment_batch=batch,
                        attachment_texts=tuple(
                            item.text for item in uploads if isinstance(item, TextAttachmentInput)
                        ) if batch else (),
                        attachment_store=self._history.attachments,
                        document_adapter=(
                            self._document_adapter(provider) if model.supports_document and (
                                (batch and batch.documents) or any(ref.documents for ref in session.attachment_refs.values())
                            ) else None
                        ),
                    )
                    if result.status is TurnStatus.COMPLETED and title_source is not None:
                        await self._name_session(runtime, agent, session, title_source, lease=lease)
                except asyncio.CancelledError:
                    # Reported usage can extend beyond the last saved message boundary.
                    try:
                        if not save_failed and self._has_usage_to_save(runtime):
                            await self._save_session(runtime, session, lease=lease)
                    except Exception:
                        logger.exception("本轮已取消，已报告的用量未能保存。")
                    if result is None:
                        raise
                    # Stopping title generation does not undo the completed reply.
                    return result
                if not save_failed and self._has_usage_to_save(runtime):
                    try:
                        await self._save_session(runtime, session, lease=lease)
                    except Exception as exc:
                        raise SessionSaveError(result, exc) from exc
                return result

        # No await between the idle check and publication of the new task.
        task = asyncio.create_task(run())
        runtime.task = task
        try:
            return await task
        finally:
            if runtime.task is task:
                runtime.task = None

    async def _compact_session(self, runtime: SessionRuntime) -> CompactionCheckpoint | None:
        self._ensure_idle(runtime)
        session = runtime.session
        provider = session.provider
        agent = Agent(
            client=self._client(provider), provider=provider, catalog=self._catalog,
            custom_compatibility=self._connection(provider).get("compatibility") is not None,
            events=self.events,
            context=Context(system_prompt=session.effective_system_prompt),
            compaction_settings=self._model_compaction_settings.get(
                (provider, session.model_id), self._compaction_settings,
            ),
            **self._request_limits,
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
                        if self._has_usage_to_save(runtime):
                            try:
                                await self._save_session(runtime, session, lease=lease)
                            except Exception:
                                logger.exception("压缩用量未能保存。")
                        raise
                    if checkpoint is None:
                        if self._has_usage_to_save(runtime):
                            await self._save_session(runtime, session, lease=lease)
                        await self.events.emit("compaction.finished", **fields, status="skipped")
                        return None
                    updated = replace(session, compaction=checkpoint)

                    async def commit() -> None:
                        await self._save_session(runtime, updated, lease=lease)
                        runtime.session = updated
                        await self.events.emit(
                            "compaction.finished", **fields,
                            status="committed", through=checkpoint.through,
                        )

                    await finish_io(commit())
                    return checkpoint
            except asyncio.CancelledError:
                if runtime.session is session:
                    await self.events.emit("compaction.finished", **fields, status="cancelled")
                raise
            except Exception as exc:
                await self.events.emit(
                    "compaction.finished", **fields, status="failed", error=type(exc).__name__,
                    reason=exc.reason if isinstance(exc, CompactionError) else None,
                )
                raise

        task = asyncio.create_task(compact())
        runtime.task = task
        try:
            return await task
        finally:
            if runtime.task is task:
                runtime.task = None

    async def _save_session(self, runtime: SessionRuntime, session: Session, *, lease: SessionLease | None = None) -> None:
        session.usage = runtime.usage.session
        await self._history.save_async(session, lease=lease)

    async def _name_session(self, runtime: SessionRuntime, agent: Agent, session: Session, source: str, *, lease: SessionLease) -> None:
        title = session.title
        try:
            async with asyncio.timeout(_TITLE_TIMEOUT):
                generated = await agent.generate_title(session=session, source=source)
            if generated:
                title = normalize_title(generated, self._history_redactor) or title
        except Exception as exc:  # noqa: BLE001 - Title generation is optional.
            logger.debug("会话标题生成失败：%s", type(exc).__name__)
        if title != session.title:
            async def save_title() -> None:
                updated = replace(session, title=title)
                await self._save_session(runtime, updated, lease=lease)
                session.title = updated.title
                session.usage = updated.usage
                session._saved_revision = updated._saved_revision

            try:
                await finish_io(save_title())
            except Exception as exc:  # noqa: BLE001 - Keep the persisted fallback title.
                logger.debug("会话标题保存失败：%s", type(exc).__name__)

    async def _save_and_publish_session(self, runtime: SessionRuntime, session: Session) -> None:
        if session._saved_revision is not None or len(session.messages) or self._has_usage_to_save(runtime):
            await self._save_session(runtime, session)
        runtime.session = session

    async def _update_session(self, runtime: SessionRuntime, session: Session) -> None:
        self._ensure_idle(runtime)
        async with runtime.state_lock:
            await finish_io(self._save_and_publish_session(runtime, session))

    def _has_usage_to_save(self, runtime: SessionRuntime) -> bool:
        return runtime.usage.session != runtime.session.usage and (
            bool(len(runtime.session.messages)) or runtime.usage.session.reported_requests > 0
        )

    async def _cancel_turn(self, runtime: SessionRuntime) -> None:
        task = runtime.task
        if task is not None and not task.done():
            if not task.cancelling():
                task.cancel()
            await asyncio.wait((task,))
            if not task.cancelled():
                task.result()

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_resources())
        await asyncio.shield(self._close_task)

    async def _close_resources(self) -> None:
        runtimes = tuple(self._runtimes.values())
        async with AsyncExitStack() as stack:
            try:
                results = await asyncio.gather(
                    *(self._cancel_turn(runtime) for runtime in runtimes), return_exceptions=True,
                )
                if self._mcp_update is not None:
                    await self._mcp_update
                for runtime in runtimes:
                    async with runtime.state_lock:
                        pass  # Wait for any in-flight settings or history write.
                self.events.clear()
                await self.events.drain(timeout=_EVENT_DRAIN_TIMEOUT)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
            finally:
                for client in self._clients.values():
                    stack.push_async_callback(client.aclose)
                for adapter in self._document_adapters.values():
                    if adapter is not None:
                        stack.push_async_callback(adapter.aclose)
                for connection in self._mcp.values():
                    stack.push_async_callback(connection.aclose)
                self._mcp.clear()
                self._mcp_unverified.clear()
                self._tools = self._mcp_catalog()
                self._runtimes.clear()
                self._trusted_tools.clear()
                self.events.clear()
