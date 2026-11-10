"""Single-Agent execution with turn-level message commit."""

import asyncio
import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

from saturnus import (
    ChatClient,
    Messages,
    ResponseAccumulator,
    ToolCallValidationError,
    ToolExecutionError,
    ToolRegistry,
    get_finish_reason,
    get_message,
)

from .compaction import CompactionCheckpoint, CompactionError, Compactor
from .context import Context
from .events import EventBus
from .models import (
    MODELS,
    ModelInfo,
    prepare_request_options,
    require_tools,
    resolve_model,
)
from .models.models import validate_catalog
from .session import Session
from .token_budget import request_budget
from .tools import ToolExecutor

_ACTIVE_SESSIONS: set[str] = set()


class TurnStatus(StrEnum):
    COMPLETED = "COMPLETED"
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
        max_requests: int = 20,
        max_tool_calls: int = 100,
    ) -> None:
        for name, value in (
            ("max_requests", max_requests),
            ("max_tool_calls", max_tool_calls),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self._client = client
        self._provider = provider
        self._catalog = tuple(catalog)
        validate_catalog(self._catalog)
        self._custom_compatibility = custom_compatibility
        self.tools = tools
        self.context = context if context is not None else Context()
        self.events = events if events is not None else EventBus()
        self.max_requests = max_requests
        self.max_tool_calls = max_tool_calls

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
            if response.get("usage") is not None:
                await self.events.emit(
                    "model.usage", **event_fields, usage=response["usage"]
                )
            return response
        accumulator = ResponseAccumulator()
        refusals: dict[int, list[str]] = {}
        async with self._client.stream(
            model=model,
            messages=messages,
            extra_body=extra_body,
            extra_headers=extra_headers,
        ) as chunks:
            async for chunk in chunks:
                accumulator.add_chunk(chunk)
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
    def _parse_response(response: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
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
        if reason == "stop" and not assistant_text(message):
            raise ValueError("A stop response has no visible text or refusal.")
        if reason == "tool_calls":
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
        return message, reason

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
        body = prepare_request_options(
            model=model,
            thinking_mode="off" if can_disable_thinking else None,
            output_limit=output_limit,
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
        message, reason = self._parse_response(response)
        content = message.get("content")
        if reason != "stop" or not isinstance(content, str) or message.get("refusal"):
            raise CompactionError("The model did not return a complete history summary.")
        return content

    async def compact(
        self, *, session: Session, operation_id: str,
    ) -> CompactionCheckpoint | None:
        """Propose a manual checkpoint. The caller owns the session transaction."""
        if session.provider != self._provider:
            raise ValueError("Session provider does not match the ChatClient connection.")
        model = resolve_model(session.provider, session.model_id, catalog=self._catalog)
        compactor = Compactor(model.context_window)
        raw = session.messages.snapshot()
        previous = session.compaction
        through = compactor.cut(raw, previous)
        if through is None:
            return None
        number = 0

        async def summarize(messages: list[dict[str, Any]], output_limit: int) -> str:
            nonlocal number
            number += 1
            return await self._request_summary(model, messages, output_limit, {
                "session_id": session.id, "turn_id": operation_id,
                "request_number": number, "purpose": "compaction", "maintenance": True,
            })

        checkpoint = await compactor.summarize(raw, previous, through, summarize)
        before = self.context.build(session.messages, previous)
        after = self.context.build(session.messages, checkpoint)
        if request_budget(after, {}, None).input_tokens >= request_budget(before, {}, None).input_tokens:
            return None
        return checkpoint

    async def run(
        self,
        *,
        session: Session,
        prompt: str,
        stream: bool = False,
        thinking_mode: str | None = None,
        output_limit: int | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> TurnResult:
        session_id = session.id
        if session_id in _ACTIVE_SESSIONS:
            raise RuntimeError(f"Session {session_id!r} already has an active Turn.")
        _ACTIVE_SESSIONS.add(session_id)
        turn_id = uuid4().hex
        fields = {"session_id": session_id, "turn_id": turn_id}
        status = TurnStatus.FAILED
        error: str | None = None
        final_message: dict[str, Any] | None = None
        requests = calls_executed = 0
        side_effects = False
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
            working = Messages(committed)
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a string.")
            working.add_user(prompt)
            tools = (
                self.tools.bind(session=session, events=self.events, turn_id=turn_id)
                if isinstance(self.tools, ToolExecutor)
                else self.tools
            )
            definitions = tools.definitions() if tools is not None else []
            if definitions:
                require_tools(model)
            for _ in range(self.max_requests):
                await _checkpoint()
                messages = self.context.build(working, session.compaction)
                requests += 1
                request_fields = {**fields, "request_number": requests}
                await self.events.emit(
                    "model.started",
                    **request_fields,
                    provider=provider,
                    model_id=model_id,
                )
                body = prepare_request_options(
                    model=model,
                    thinking_mode=thinking_mode,
                    tools=definitions or None,
                    output_limit=output_limit,
                    extra_body=extra_body,
                    custom_compatibility=self._custom_compatibility,
                )
                await self.events.emit(
                    "context.budget",
                    **request_fields,
                    **asdict(request_budget(messages, body, model.context_window)),
                )
                response = await self._request(
                    model=model_id,
                    messages=messages,
                    stream=stream,
                    event_fields=request_fields,
                    extra_body=body,
                    extra_headers=extra_headers,
                )
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
                working.add(message)
                if reason == "stop":
                    await _checkpoint()
                    current_identity = (
                        session.id,
                        session.workspace,
                        session.provider,
                        session.model_id,
                    )
                    if (
                        current_identity != identity
                        or session.messages.snapshot() != committed
                    ):
                        raise RuntimeError(
                            "Session changed while its Turn was running."
                        )
                    final_message = deepcopy(message)
                    # No suspension between commit and terminal notification.
                    session.messages = working
                    status = TurnStatus.COMPLETED
                    break
                if tools is None or not definitions:
                    raise ValueError(
                        "Model requested tools but no tools are available."
                    )
                calls = message["tool_calls"]
                if calls_executed + len(calls) > self.max_tool_calls:
                    raise RuntimeError("Maximum tool-call count exceeded.")
                for call in calls:
                    await _checkpoint()
                    calls_executed += 1
                    tool_fields = {
                        **request_fields,
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                    }
                    await self.events.emit(
                        "tool.started",
                        **tool_fields,
                        arguments=call["function"]["arguments"],
                    )
                    side_effects = True
                    tool_error = None
                    content = None
                    tool_status = "FAILED"
                    try:
                        try:
                            if isinstance(tools, ToolExecutor):
                                result = await tools.execute(call, request_number=requests)
                            else:
                                result = await tools.execute(call)
                            content = result.content
                            if isinstance(tools, ToolExecutor) and not result.value.get(
                                "ok", False
                            ):
                                tool_error = result.value.get("error")
                        except (ToolCallValidationError, ToolExecutionError) as exc:
                            if isinstance(exc.__cause__, FatalToolError):
                                raise exc.__cause__
                            tool_error = {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "code": getattr(exc, "code", "execution_error"),
                            }
                            content = json.dumps(
                                {"error": tool_error}, ensure_ascii=False
                            )
                        await _checkpoint()
                        working.add_tool_result(
                            tool_call_id=call["id"], content=content
                        )
                        tool_status = "ERROR" if tool_error else "COMPLETED"
                    except asyncio.CancelledError as exc:
                        tool_status = "CANCELLED"
                        tool_error = {
                            "type": "CancelledError",
                            "message": str(exc) or "Tool operation cancelled.",
                        }
                        raise
                    except Exception as exc:
                        tool_error = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        raise
                    finally:
                        await self.events.emit(
                            "tool.finished",
                            **tool_fields,
                            status=tool_status,
                            content=content,
                            error=tool_error,
                        )
            else:
                raise RuntimeError("Maximum model-request count exceeded.")
        except asyncio.CancelledError:
            status = TurnStatus.CANCELLED
            error = "Turn cancelled; executed tool side effects are not rolled back."
            raise
        except Exception as exc:  # noqa: BLE001 - Report unexpected failures as failed Turns.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                await self.events.emit(
                    "turn.finished",
                    **fields,
                    status=status.value,
                    error=error,
                    model_requests=requests,
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
            tool_calls=calls_executed,
            external_side_effects_possible=side_effects,
        )
