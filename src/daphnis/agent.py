"""Single-Agent execution with durable conversation and tool transitions."""

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any
from uuid import uuid4

from saturnus import (
    APIError,
    ChatClient,
    Messages,
    ResponseAccumulator,
    ResponseError,
    SaturnusError,
    ToolCallValidationError,
    ToolExecutionError,
    ToolRegistry,
    get_finish_reason,
    get_message,
)

from .compaction import (
    SUMMARY_POLICY,
    CompactionCheckpoint,
    CompactionError,
    Compactor,
    safe_boundaries,
)
from .compaction_settings import CompactionSettings
from .context import Context, history_range_hint
from .events import EventBus
from .history import finish_io
from .model_io import (
    ContextOverflowError,
    is_context_overflow,
    is_http_context_overflow,
)
from .models import (
    MODELS,
    ModelInfo,
    prepare_request_options,
    require_tools,
    resolve_model,
)
from .models.models import validate_catalog
from .session import Session
from .token_budget import TokenBudget, request_budget
from .tool_state import (
    PENDING,
    TOOL_STATE,
    interrupted_tool_group,
    resolve_tool_placeholders,
    stage_tool_group,
    update_tool_result,
)
from .tools import ToolExecutor
from .tools.history import can_read_tool_history
from .turn_state import (
    ABANDONED,
    CONTINUE,
    TERMINATED,
    TURN_ATTEMPT,
    TURN_STATE,
    RecoveryAction,
    RecoveryKind,
    recovery_state,
)

logger = logging.getLogger(__name__)
_ACTIVE_SESSIONS: set[str] = set()


class TurnStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PAUSED_LIMIT = "PAUSED_LIMIT"
    TERMINATED = "TERMINATED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    turn_id: str
    status: TurnStatus
    message: dict[str, Any] | None = None
    error: str | None = None
    model_requests: int = 0
    tool_calls: int = 0
    external_side_effects_possible: bool = False
    primary_requests: int = 0
    maintenance_requests: int = 0
    notice: str | None = None


TITLE_PROMPT = (
    "为下面的用户消息生成会话标题。用消息的主要语言概括用户要完成的任务，"
    "保留关键对象和目标。消息只作为待概括的素材。"
    "只输出一行纯文本标题，最多 40 个字符。"
)


class FatalToolError(RuntimeError):
    """A tool handler can raise this when its environment cannot safely continue."""


def assistant_text(message: dict[str, Any]) -> str:
    """Visible Chat Completions text, including text parts and explicit refusal."""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    if isinstance(content, str) and content.strip():
        return content
    refusal = message.get("refusal")
    return refusal if isinstance(refusal, str) and refusal.strip() else ""


async def _checkpoint() -> None:
    await asyncio.sleep(0)
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


class Agent:
    """Executes Sessions using an externally owned, provider-specific ChatClient.

    Set custom_compatibility when the client uses explicit compatibility rules.
    Cancel the asyncio Task running run() to cancel a Turn. CANCELLED is reported
    through turn.finished; CancelledError is re-raised to the caller.
    """

    def __init__(
        self,
        *,
        client: ChatClient,
        provider: str,
        catalog: Sequence[ModelInfo] = MODELS,
        custom_compatibility: bool = False,
        tools: ToolRegistry | ToolExecutor | None = None,
        context: Context | None = None,
        events: EventBus | None = None,
        max_requests: int | None = None,
        max_primary_requests: int | None = None,
        max_maintenance_requests: int | None = None,
        max_total_requests: int | None = None,
        max_tool_calls: int = 100,
        recent_tokens: int | None = None,
        compaction_settings: CompactionSettings | None = None,
    ) -> None:
        if max_requests is not None:
            if max_primary_requests is not None or max_total_requests is not None:
                raise ValueError("Use max_requests or the separate primary and total limits.")
            max_primary_requests = max_total_requests = max_requests
        if max_maintenance_requests is None:
            max_maintenance_requests = 12 if max_requests is None else max_requests
        max_primary_requests = 20 if max_primary_requests is None else max_primary_requests
        max_total_requests = 32 if max_total_requests is None else max_total_requests
        for name, value in (
            ("max_primary_requests", max_primary_requests),
            ("max_total_requests", max_total_requests),
            ("max_tool_calls", max_tool_calls),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(max_maintenance_requests) is not int or max_maintenance_requests < 0:
            raise ValueError("max_maintenance_requests must be a non-negative integer.")
        self._client = client
        self._provider = provider
        self._catalog = tuple(catalog)
        validate_catalog(self._catalog)
        self._custom_compatibility = custom_compatibility
        self.tools = tools
        self.context = context if context is not None else Context()
        self.events = events if events is not None else EventBus()
        self.max_primary_requests = max_primary_requests
        self.max_maintenance_requests = max_maintenance_requests
        self.max_total_requests = max_total_requests
        self.max_tool_calls = max_tool_calls
        compaction_settings = compaction_settings or CompactionSettings()
        self.compaction_settings = (
            compaction_settings if recent_tokens is None
            else replace(compaction_settings, recent_tokens=recent_tokens)
        )

    async def _request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        event_fields: dict[str, Any],
        extra_body: dict[str, Any],
        extra_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        if not stream:
            response = await self._client.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                extra_headers=extra_headers,
            )
            if "error" in response and is_context_overflow(response):
                raise ContextOverflowError
            if response.get("usage") is not None:
                await self.events.emit(
                    "model.usage", **event_fields, usage=response["usage"]
                )
            return response
        accumulator = ResponseAccumulator()
        refusals: dict[int, list[str]] = {}
        streamed = False
        async with self._client.stream(
            model=model,
            messages=messages,
            extra_body=extra_body,
            extra_headers=extra_headers,
        ) as chunks:
            async for chunk in chunks:
                if "error" in chunk and is_context_overflow(chunk):
                    raise ContextOverflowError(retryable=not streamed)
                accumulator.add_chunk(chunk)
                streamed |= any(
                    any(value for key, value in choice["delta"].items() if key != "role")
                    for choice in chunk["choices"]
                )
                for choice in chunk["choices"]:
                    refusal = choice["delta"].get("refusal")
                    if isinstance(refusal, str) and refusal:
                        refusals.setdefault(choice["index"], []).append(refusal)
                if chunk.get("usage") is not None:
                    await self.events.emit(
                        "model.usage", **event_fields, usage=chunk["usage"]
                    )
                await self.events.emit("model.delta", **event_fields, chunk=chunk)
        response = accumulator.finalize()
        for choice in response["choices"]:
            if choice["index"] in refusals:
                choice["message"]["refusal"] = "".join(refusals[choice["index"]])
        return response

    @staticmethod
    def _parse_response(
        response: dict[str, Any], *, require_visible: bool = True,
    ) -> tuple[dict[str, Any], str | None]:
        message = get_message(response)
        if len(response["choices"]) != 1:
            raise ValueError("Agent requires exactly one response choice.")
        message.setdefault("role", "assistant")
        if message["role"] != "assistant":
            raise ValueError("Expected an assistant response.")
        reason = get_finish_reason(response)
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ValueError("Tool calls must be a list or null.")
        if reason == "stop" and calls:
            raise ValueError("A stop response unexpectedly contains tool calls.")
        if require_visible and reason == "stop" and not assistant_text(message):
            raise ValueError("A stop response has no visible text or refusal.")
        if reason == "tool_calls":
            Agent._validate_tool_calls(calls)
        for field in (TOOL_STATE, TURN_STATE, TURN_ATTEMPT):
            message.pop(field, None)
        return message, reason

    @staticmethod
    def _validate_tool_calls(calls: list[dict[str, Any]] | None) -> None:
        if not calls:
            raise ValueError("Tool-call response has no tool calls.")
        ids: set[str] = set()
        for call in calls:
            if not isinstance(call, dict):
                raise TypeError("Tool call must be an object.")
            call_id = call.get("id")
            function = call.get("function")
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or call_id in ids
                or call.get("type") != "function"
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"].strip()
                or not isinstance(function.get("arguments"), str)
            ):
                raise ValueError("Incomplete or duplicate tool-call identity.")
            ids.add(call_id)

    def _model_context(
        self, history: Messages, checkpoint: CompactionCheckpoint | None,
        *, history_lookup: bool = False,
    ) -> list[dict[str, Any]]:
        if checkpoint is None:
            return self.context.build(history)
        return self.context.build(
            history, checkpoint, summary_reasoning=self._provider == "deepseek",
            history_lookup=history_lookup,
        )

    def _summary_response_format(self) -> dict[str, str] | None:
        if self._provider in {"mimo", "deepseek"} and not self._custom_compatibility:
            return {"type": "json_object"}
        return None

    async def generate_title(self, *, session: Session, source: str) -> str | None:
        model = resolve_model(session.provider, session.model_id, catalog=self._catalog)
        can_disable_thinking = "off" in (model.thinking_modes or ()) and (
            model.provider != "compatible" or self._custom_compatibility
        )
        body = prepare_request_options(
            model=model,
            thinking_mode="off" if can_disable_thinking else None,
            output_limit=min(128, model.max_output_tokens or 128),
            custom_compatibility=self._custom_compatibility,
        )
        fields = {
            "session_id": session.id, "turn_id": uuid4().hex,
            "request_number": 1, "purpose": "title",
        }
        messages = [
            {"role": "system", "content": TITLE_PROMPT},
            {"role": "user", "content": source},
        ]
        await self.events.emit(
            "model.started", **fields, provider=model.provider, model_id=model.id,
        )
        response = await self._request(
            model=model.id, messages=messages, stream=False, event_fields=fields,
            extra_body=body, extra_headers=None,
        )
        message, reason = self._parse_response(response)
        await self.events.emit("model.finished", **fields, finish_reason=reason, response=response)
        if reason != "stop" or message.get("refusal"):
            return None
        return assistant_text(message)

    async def _request_summary(
        self,
        model: ModelInfo,
        messages: list[dict[str, Any]],
        output_limit: int,
        fields: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        can_disable_thinking = "off" in (model.thinking_modes or ()) and (
            model.provider != "compatible" or self._custom_compatibility
        )
        response_format = self._summary_response_format()
        body = prepare_request_options(
            model=model,
            thinking_mode="off" if can_disable_thinking else None,
            output_limit=output_limit,
            extra_body={"response_format": response_format} if response_format is not None else None,
            custom_compatibility=self._custom_compatibility,
        )
        await self.events.emit(
            "model.started", **fields, provider=model.provider, model_id=model.id,
        )
        response = await self._request(
            model=model.id, messages=messages, stream=False, event_fields=fields,
            extra_body=body, extra_headers=extra_headers,
        )
        await _checkpoint()
        try:
            message, reason = self._parse_response(response, require_visible=False)
        except (TypeError, ValueError, SaturnusError) as exc:
            raise CompactionError("Invalid history summary response.", reason="summary_response") from exc
        await self.events.emit(
            "model.finished", **fields, finish_reason=reason, response=response,
        )
        content = message.get("content")
        if reason == "length":
            raise CompactionError(
                "The history summary was truncated. Write a shorter summary.", reason="summary_length",
            )
        if reason != "stop":
            raise CompactionError("The history summary did not finish normally.", reason="summary_finish")
        if not isinstance(content, str) or not content.strip() or message.get("refusal"):
            raise CompactionError("The history summary has no usable text.", reason="summary_content")
        return content

    async def compact(
        self, *, session: Session, operation_id: str,
    ) -> CompactionCheckpoint | None:
        """Propose a manual checkpoint. The caller owns the session transaction."""
        if session.provider != self._provider:
            raise ValueError("Session provider does not match the ChatClient connection.")
        model = resolve_model(session.provider, session.model_id, catalog=self._catalog)
        compactor = Compactor(
            model.context_window, counter=model.token_counter,
            recent_tokens=self.compaction_settings.recent_tokens,
            response_format=self._summary_response_format(),
        )
        raw = session.messages.snapshot()
        fields = {"session_id": session.id, "turn_id": operation_id, "trigger": "manual"}
        if raw and len(raw) not in safe_boundaries(raw):
            raise ValueError("工具交换尚未完成。请先选择继续或终止。")
        previous = session.compaction
        through = compactor.cut(raw, previous)
        if through is None:
            await self.events.emit("compaction.candidate", **fields, reason="no_safe_cut")
            return None
        before = self._request_budget(self._model_context(session.messages, previous), {}, model).input_tokens
        number = 0

        async def summarize(messages: list[dict[str, Any]], output_limit: int) -> str:
            nonlocal number
            if number >= min(self.max_maintenance_requests, self.max_total_requests):
                raise CompactionError(
                    "No maintenance requests available for compaction.", reason="maintenance_limit",
                )
            number += 1
            return await self._request_summary(model, messages, output_limit, {
                "session_id": session.id, "turn_id": operation_id,
                "request_number": number, "purpose": "compaction", "maintenance": True,
            })

        return await self._shorter_checkpoint(
            compactor, raw, previous, through, summarize, before,
            lambda candidate: self._request_budget(self._model_context(session.messages, candidate), {}, model).input_tokens,
            fields, can_retry=lambda: number < min(self.max_maintenance_requests, self.max_total_requests),
        )

    async def _shorter_checkpoint(
        self, compactor: Compactor, raw: list[dict[str, Any]], previous: CompactionCheckpoint | None,
        through: int, summarize: Callable[[list[dict[str, Any]], int], Awaitable[str]], before: int,
        projected_tokens: Callable[[CompactionCheckpoint], int], fields: dict[str, Any],
        *, can_retry: Callable[[], bool], input_target: int | None = None,
    ) -> CompactionCheckpoint | None:
        placeholder = CompactionCheckpoint(through, '{"goals":"x","progress":"","files":"","next_steps":""}')
        fixed = projected_tokens(placeholder)
        if fixed >= before:
            await self.events.emit("compaction.candidate", **fields, reason="no_token_headroom",
                                   through=through, before_tokens=before, after_tokens=fixed)
            return None
        headroom = before - fixed
        if input_target is not None and input_target > fixed:
            headroom = min(headroom, input_target - fixed)
        targets = (max(1, min(compactor.output_limit * 3 // 4, headroom * 3 // 4)), max(1, headroom // 2))
        for target in dict.fromkeys((targets[0], min(targets))):
            candidate = await compactor.summarize(raw, previous, through, summarize, target_tokens=target)
            after = projected_tokens(candidate)
            reason = "reduced" if after < before else "no_token_reduction"
            await self.events.emit("compaction.candidate", **fields, reason=reason, through=through,
                                   before_tokens=before, after_tokens=after)
            if after < before:
                return candidate
            if not can_retry():
                break
        return None

    def _request_budget(
        self, messages: list[dict[str, Any]], body: dict[str, Any], model: ModelInfo,
    ) -> TokenBudget:
        return request_budget(
            messages, body, model.context_window, counter=model.token_counter,
            provider=None if self._custom_compatibility else self._provider,
            default_output_tokens=None if self._custom_compatibility else model.default_output_tokens,
            settings=self.compaction_settings,
        )

    async def _compact_request(
        self,
        *,
        history: Messages,
        checkpoint: CompactionCheckpoint | None,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        model: ModelInfo,
        compactor: Compactor,
        summarize: Callable[[list[dict[str, Any]], int], Awaitable[str]],
        fields: dict[str, Any],
        can_retry: Callable[[], bool],
        attempted_cuts: set[tuple[int, int]],
        trigger: str = "pressure",
        history_lookup: bool = False,
    ) -> tuple[CompactionCheckpoint | None, list[dict[str, Any]]]:
        fields = {**fields, "trigger": trigger}
        overflow = trigger == "overflow"
        before = self._request_budget(messages, body, model).input_tokens
        await self.events.emit("compaction.started", **fields, input_tokens=before)
        status = "skipped"
        error = None
        failure_reason = None
        try:
            raw = history.snapshot() if history_lookup or self.compaction_settings.auto_enabled else []
            history_tool_ids = {}
            if history_lookup:
                counts = Counter(message["tool_call_id"] for message in raw if message["role"] == "tool")
                history_tool_ids = {
                    message["tool_call_id"]: index
                    for index, message in enumerate(raw)
                    if message["role"] == "tool" and counts[message["tool_call_id"]] == 1
                    and can_read_tool_history(history, index, message)
                }
            pruned = compactor.prune(messages, history_tool_ids=history_tool_ids)
            budget = self._request_budget(pruned, body, model)
            messages = pruned
            if self.compaction_settings.auto_enabled and (
                overflow or budget.needs_compaction or budget.needs_safety_compaction
            ):
                available = (
                    budget.compaction_threshold if budget.needs_compaction
                    else budget.hard_input_limit
                )
                input_target = available
                if available is not None:
                    privileged = Messages([
                        message for message in raw if message["role"] in ("system", "developer")
                    ])
                    fixed = self._request_budget(self.context.build(privileged), body, model)
                    available -= fixed.input_tokens + compactor.output_limit
                    available -= compactor.counter.count(SUMMARY_POLICY)
                    if history_lookup:
                        available -= compactor.counter.count(history_range_hint(len(raw)))
                through = compactor.cut(
                    raw, checkpoint, within_turn=True, available_tokens=available,
                )
                cut_key = (checkpoint.through if checkpoint else 0, through) if through is not None else None
                if through is None:
                    failure_reason = "no_safe_cut"
                elif cut_key in attempted_cuts:
                    failure_reason = "already_attempted"
                else:
                    attempted_cuts.add(cut_key)

                    def project_candidate(candidate: CompactionCheckpoint) -> list[dict[str, Any]]:
                        return compactor.prune(
                            self._model_context(history, candidate, history_lookup=history_lookup),
                            history_tool_ids=history_tool_ids,
                        )

                    try:
                        candidate = await self._shorter_checkpoint(
                            compactor, raw, checkpoint, through, summarize, budget.input_tokens,
                            lambda candidate: self._request_budget(project_candidate(candidate), body, model).input_tokens,
                            fields, can_retry=can_retry, input_target=input_target,
                        )
                    except (CompactionError, ContextOverflowError, SaturnusError, ValueError) as exc:
                        error = type(exc).__name__
                        if isinstance(exc, CompactionError):
                            failure_reason = exc.reason
                    else:
                        if candidate is None:
                            failure_reason = "no_token_reduction"
                        else:
                            checkpoint, messages = candidate, project_candidate(candidate)
                            budget = self._request_budget(messages, body, model)
                            status = "staged"
            if overflow or budget.needs_compaction or budget.needs_safety_compaction:
                # The latest exchange can itself be oversized. Keep its IDs and excerpts.
                messages = compactor.prune(messages, include_latest=True, history_tool_ids=history_tool_ids)
            return checkpoint, messages
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            after_tokens = self._request_budget(messages, body, model).input_tokens
            if status == "skipped":
                if error:
                    status = "failed"
                elif after_tokens < before:
                    status = "trimmed"
            await self.events.emit(
                "compaction.finished", **fields, status=status, error=error, reason=failure_reason,
                before_tokens=before, after_tokens=after_tokens,
                through=checkpoint.through if checkpoint else 0,
            )

    async def run(
        self,
        *,
        session: Session,
        prompt: str | None = None,
        recovery_action: RecoveryAction | None = None,
        recovery_key: str | None = None,
        stream: bool = False,
        thinking_mode: str | None = None,
        output_limit: int | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        persist: Callable[[Session], Awaitable[None]] | None = None,
    ) -> TurnResult:
        session_id = session.id
        if session_id in _ACTIVE_SESSIONS:
            raise RuntimeError(f"Session {session_id!r} already has an active Turn.")
        _ACTIVE_SESSIONS.add(session_id)
        turn_id = uuid4().hex
        fields = {"session_id": session_id, "turn_id": turn_id}
        status = TurnStatus.FAILED
        error: str | None = None
        notice: str | None = None
        final_message: dict[str, Any] | None = None
        requests = primary_requests = maintenance_requests = calls_executed = 0
        side_effects = False
        retried_overflow = False
        committed_checkpoint = checkpoint = session.compaction
        try:
            provider, model_id = session.provider, session.model_id
            identity = (
                session_id,
                session.workspace,
                provider,
                model_id,
            )
            committed = session.messages.snapshot()
            await self.events.emit("turn.started", **fields)
            if provider != self._provider:
                raise ValueError(
                    "Session provider does not match the ChatClient connection."
                )
            model = resolve_model(provider, model_id, catalog=self._catalog)
            recovery = recovery_state(committed, session_id)
            if recovery_action is None:
                if not isinstance(prompt, str):
                    raise TypeError("prompt must be a string.")
                if recovery is not None and recovery.kind is RecoveryKind.INTERRUPTED_TURN:
                    raise ValueError("上轮任务未完成，请先选择继续或终止。")
            else:
                kind = RecoveryKind.FAILED_USER if recovery_action is RecoveryAction.RETRY else RecoveryKind.INTERRUPTED_TURN
                if prompt is not None or recovery is None or recovery.kind is not kind or recovery.key != recovery_key:
                    raise ValueError("恢复状态已变化，请刷新后再操作。")
            interrupted = interrupted_tool_group(committed)
            tools = None if recovery_action is RecoveryAction.TERMINATE else self.tools
            if isinstance(tools, ToolExecutor):
                tools = tools.bind(session=session, events=self.events, turn_id=turn_id)
            definitions = tools.definitions() if tools is not None else []
            if definitions:
                require_tools(model)
            history_lookup = {"retrieval", "history_read"}.issubset(
                item["function"]["name"] for item in definitions
            )
            compactor = Compactor(
                model.context_window, counter=model.token_counter,
                recent_tokens=self.compaction_settings.recent_tokens,
                response_format=self._summary_response_format(),
            )

            def check_session() -> None:
                if (
                    (session.id, session.workspace, session.provider, session.model_id) != identity
                    or session.messages.snapshot() != committed
                    or session.compaction != committed_checkpoint
                ):
                    raise RuntimeError("Session changed while its Turn was running.")

            async def commit(
                snapshot: list[dict[str, Any]], *, tool_finished: dict[str, Any] | None = None,
            ) -> None:
                check_session()
                candidate = replace(session, messages=Messages(snapshot), compaction=checkpoint)

                async def save_and_publish() -> None:
                    nonlocal committed, committed_checkpoint
                    if persist is not None:
                        await persist(candidate)
                    session.messages = candidate.messages
                    session.compaction = candidate.compaction
                    session.title = candidate.title
                    session.usage = candidate.usage
                    session._saved_revision = candidate._saved_revision
                    committed = snapshot
                    if committed_checkpoint != checkpoint:
                        committed_checkpoint = checkpoint
                        await self.events.emit("compaction.committed", **fields, through=checkpoint.through)
                    if tool_finished is not None:
                        await self.events.emit("tool.finished", **tool_finished)

                if persist is None:
                    await save_and_publish()
                else:
                    await finish_io(save_and_publish())

            async def process_tool(
                call: dict[str, Any], index: int, request_fields: dict[str, Any],
            ) -> None:
                nonlocal calls_executed, side_effects
                await _checkpoint()
                calls_executed += 1
                call_id = call["id"]
                tool_fields = {
                    **request_fields,
                    "tool_call_id": call_id,
                    "name": call["function"]["name"],
                }
                await self.events.emit(
                    "tool.started", **tool_fields, arguments=call["function"]["arguments"],
                )

                async def before_execute(executing_id: str) -> None:
                    nonlocal side_effects
                    if executing_id != call_id:
                        raise ValueError("Execution barrier received a different tool call ID.")
                    snapshot = update_tool_result(committed, index, executing_id)
                    try:
                        await commit(snapshot)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Could not persist RUNNING for {call_id!r}: {type(exc).__name__}: {exc}"
                        ) from exc
                    side_effects = True

                tool_error = None
                try:
                    if tools is None:
                        raise ToolCallValidationError("Tool is no longer enabled.", code="unknown_tool")
                    if isinstance(tools, ToolExecutor):
                        result = await tools.execute(
                            call, request_number=tool_fields["request_number"], before_execute=before_execute,
                        )
                    else:
                        tools.validate_call(call)
                        await before_execute(call_id)
                        await self.events.emit("tool.executing", **tool_fields)
                        await _checkpoint()
                        result = await tools.execute(call)
                    content = result.content
                    if isinstance(tools, ToolExecutor) and not result.value.get("ok", False):
                        tool_error = result.value.get("error")
                except (ToolCallValidationError, ToolExecutionError) as exc:
                    if isinstance(exc.__cause__, FatalToolError):
                        raise exc.__cause__
                    tool_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "code": getattr(exc, "code", "execution_error"),
                    }
                    content = json.dumps({"error": tool_error}, ensure_ascii=False)
                if not isinstance(content, str):
                    raise TypeError("Tool result content must be a string.")
                await commit(
                    update_tool_result(committed, index, call_id, content=content),
                    tool_finished={
                        **tool_fields, "status": "ERROR" if tool_error else "COMPLETED",
                        "content": content, "error": tool_error,
                    },
                )

            def primary_remaining() -> int:
                return min(self.max_primary_requests - primary_requests, self.max_total_requests - requests)

            maintenance_limited = False

            def maintenance_available() -> bool:
                return maintenance_requests < self.max_maintenance_requests and requests + 1 < self.max_total_requests

            async def summarize(messages: list[dict[str, Any]], limit: int) -> str:
                nonlocal requests, maintenance_requests, maintenance_limited
                if not maintenance_available():
                    maintenance_limited = True
                    raise CompactionError(
                        "No maintenance requests available for compaction.", reason="maintenance_limit",
                    )
                maintenance_requests += 1
                requests += 1
                return await self._request_summary(model, messages, limit, {
                    **fields, "request_number": requests, "purpose": "compaction",
                }, extra_headers)

            async def send_request(
                messages: list[dict[str, Any]], body: dict[str, Any], budget: TokenBudget,
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                nonlocal requests, primary_requests
                primary_requests += 1
                requests += 1
                request_fields = {**fields, "request_number": requests}
                await self.events.emit(
                    "model.started", **request_fields, provider=provider, model_id=model_id,
                )
                await self.events.emit(
                    "context.budget", **request_fields, **asdict(budget),
                    provider=provider, model_id=model_id,
                )
                try:
                    response = await self._request(
                        model=model_id, messages=messages, stream=stream,
                        event_fields=request_fields, extra_body=body, extra_headers=extra_headers,
                    )
                except (APIError, ResponseError) as exc:
                    if is_http_context_overflow(exc):
                        raise ContextOverflowError from exc
                    raise
                return response, request_fields

            await _checkpoint()
            pending = []
            if recovery_action is None:
                if committed and len(committed) not in safe_boundaries(committed):
                    raise ValueError("Cannot append to an incomplete tool exchange.")
                snapshot = [*committed, {"role": "user", "content": prompt}]
                if recovery is not None:
                    index = recovery.message_index
                    snapshot[index] = {**committed[index], TURN_STATE: ABANDONED}
                await commit(snapshot)
            else:
                snapshot = committed.copy()
                terminating = recovery_action is RecoveryAction.TERMINATE
                if interrupted is not None:
                    calls, indices = interrupted
                    self._validate_tool_calls(calls)
                    prefix = committed[:min(indices.values()) - 1]
                    if prefix and len(prefix) not in safe_boundaries(prefix):
                        raise ValueError("Cannot recover a tool group after an incomplete exchange.")
                    snapshot, side_effects = resolve_tool_placeholders(snapshot, indices, terminate=terminating)
                elif len(snapshot) not in safe_boundaries(snapshot):
                    raise ValueError("Cannot recover an incomplete tool exchange.")
                index = recovery.message_index
                snapshot[index] = {
                    **snapshot[index],
                    TURN_STATE: TERMINATED if terminating else CONTINUE,
                    TURN_ATTEMPT: turn_id,
                }
                if terminating:
                    try:
                        await commit(snapshot)
                    except asyncio.CancelledError:
                        if committed != snapshot:
                            raise
                    status = TurnStatus.TERMINATED
                    return TurnResult(session_id, turn_id, status, external_side_effects_possible=side_effects)
                pending = [] if interrupted is None else [
                    (call, indices[call["id"]]) for call in calls
                    if snapshot[indices[call["id"]]].get(TOOL_STATE) == PENDING
                ]
                if len(pending) > self.max_tool_calls:
                    raise RuntimeError("Maximum tool-call count exceeded.")
                await commit(snapshot)
            body = prepare_request_options(
                model=model,
                thinking_mode=thinking_mode,
                tools=definitions or None,
                output_limit=output_limit,
                extra_body=extra_body,
                custom_compatibility=self._custom_compatibility,
            )
            for call, index in pending:
                await process_tool(call, index, {**fields, "request_number": 0})
            attempted_cuts: set[tuple[int, int]] = set()
            warned_about_reserve = False
            while primary_remaining() > 0:
                await _checkpoint()
                maintenance_limited = False
                messages = self._model_context(session.messages, checkpoint, history_lookup=history_lookup)
                budget = self._request_budget(messages, body, model)
                if budget.threshold_blocked_by_reserve and not warned_about_reserve:
                    logger.warning(
                        "压缩阈值（%s Token）不高于输出预留（%s Token）；"
                        "本轮仅按安全上限检查。请提高阈值或降低 output_limit。",
                        budget.trigger_tokens, budget.output_reserve,
                    )
                    warned_about_reserve = True
                if budget.needs_compaction or budget.needs_safety_compaction:
                    checkpoint, messages = await self._compact_request(
                        history=session.messages, checkpoint=checkpoint, messages=messages,
                        body=body, model=model, compactor=compactor, summarize=summarize,
                        fields=fields, history_lookup=history_lookup,
                        can_retry=maintenance_available, attempted_cuts=attempted_cuts,
                    )
                    budget = self._request_budget(messages, body, model)
                if not budget.fits:
                    if maintenance_limited:
                        break
                    raise CompactionError(
                        f"请求超过上下文安全上限（输入 {budget.input_tokens:,} + 输出预留 "
                        f"{budget.output_reserve:,} > {budget.hard_limit:,} Token）。"
                        "请缩短输入或降低 output_limit。",
                        reason="context_safety_limit",
                    )
                while True:
                    try:
                        response, request_fields = await send_request(messages, body, budget)
                        break
                    except ContextOverflowError as exc:
                        failure_fields = {**fields, "request_number": requests}
                        reason = None
                        if retried_overflow:
                            reason = "retry_exhausted"
                        elif not exc.retryable:
                            reason = "stream_started"
                        elif primary_remaining() == 0:
                            reason = "request_limit"
                        if reason is None:
                            before = budget.input_tokens
                            checkpoint, reduced = await self._compact_request(
                                history=session.messages, checkpoint=checkpoint, messages=messages,
                                body=body, model=model, compactor=compactor, summarize=summarize,
                                fields=failure_fields, trigger="overflow", history_lookup=history_lookup,
                                can_retry=maintenance_available, attempted_cuts=attempted_cuts,
                            )
                            budget = self._request_budget(reduced, body, model)
                            if not budget.fits:
                                reason = "window_limit"
                            elif budget.input_tokens >= before:
                                reason = "no_reduction"
                        await self.events.emit(
                            "context.overflow", **failure_fields,
                            retrying=reason is None, reason=reason,
                        )
                        if reason is not None:
                            if reason == "request_limit" or (maintenance_limited and reason in ("window_limit", "no_reduction")):
                                response = None
                                break
                            raise
                        retried_overflow = True
                        messages = reduced
                if response is None:
                    break
                await _checkpoint()
                message, reason = self._parse_response(response)
                await self.events.emit(
                    "model.finished",
                    **request_fields,
                    finish_reason=reason,
                    response=response,
                )
                if reason not in ("stop", "tool_calls"):
                    raise ValueError(f"Model did not finish normally: {reason!r}.")
                if reason == "stop":
                    await _checkpoint()
                    snapshot = [*committed, message]
                    try:
                        await commit(snapshot)
                    except asyncio.CancelledError:
                        if committed != snapshot:
                            raise
                    final_message = deepcopy(message)
                    status = TurnStatus.COMPLETED
                    break
                if tools is None or not definitions:
                    raise ValueError(
                        "Model requested tools but no tools are available."
                    )
                calls = message["tool_calls"]
                if calls_executed + len(calls) > self.max_tool_calls:
                    raise RuntimeError("Maximum tool-call count exceeded.")
                first_result = len(committed) + 1
                await commit(stage_tool_group(session.messages, message).snapshot())
                for index, call in enumerate(calls, start=first_result):
                    await process_tool(call, index, request_fields)
            if status is not TurnStatus.COMPLETED:
                await _checkpoint()
                status = TurnStatus.PAUSED_LIMIT
                if primary_requests == self.max_primary_requests:
                    limit_reason = f"达到模型主请求上限（{primary_requests}/{self.max_primary_requests}）"
                elif requests == self.max_total_requests:
                    limit_reason = f"达到模型总请求上限（{requests}/{self.max_total_requests}）"
                else:
                    limit_reason = "可用于上下文压缩的请求额度不足"
                pending_recovery = recovery_state(committed, session_id)
                action_hint = (
                    "可重试原消息或发送新消息。"
                    if pending_recovery and pending_recovery.kind is RecoveryKind.FAILED_USER
                    else "请选择继续或终止。"
                )
                notice = f"本轮{limit_reason}，进度已保存。{action_hint}"
        except asyncio.CancelledError:
            status = TurnStatus.CANCELLED
            error = "Turn cancelled; executed tool side effects are not rolled back."
            raise
        except Exception as exc:  # noqa: BLE001 - Report unexpected failures as failed Turns.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if checkpoint != committed_checkpoint:
                    await self.events.emit(
                        "compaction.discarded",
                        **fields, through=checkpoint.through,
                    )
                await self.events.emit(
                    "turn.finished",
                    **fields,
                    status=status.value,
                    error=error,
                    notice=notice,
                    model_requests=requests,
                    primary_requests=primary_requests,
                    maintenance_requests=maintenance_requests,
                    tool_calls=calls_executed,
                    external_side_effects_possible=side_effects,
                )
            finally:
                _ACTIVE_SESSIONS.discard(session_id)
        return TurnResult(
            session_id=session_id,
            turn_id=turn_id,
            status=status,
            message=final_message,
            error=error,
            model_requests=requests,
            primary_requests=primary_requests,
            maintenance_requests=maintenance_requests,
            tool_calls=calls_executed,
            external_side_effects_possible=side_effects,
            notice=notice,
        )
