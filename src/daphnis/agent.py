"""Single-Agent execution with turn-level message commit."""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from saturnus import (
    Messages,
    ResponseAccumulator,
    ToolCallValidationError,
    ToolExecutionError,
    ToolRegistry,
)

from .context import Context
from .events import EventBus
from .models import LLM
from .session import Session

# Shared model selection is protected for the entire Turn, even across Agents.
_MODEL_LOCKS: WeakKeyDictionary[LLM, asyncio.Lock] = WeakKeyDictionary()
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


async def _checkpoint() -> None:
    await asyncio.sleep(0)
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


class Agent:
    """Executes Sessions using an externally owned LLM and optional tool registry.

    Cancel the asyncio Task running run() to cancel a Turn. CANCELLED is reported
    through turn.finished; CancelledError is re-raised to the caller.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        tools: ToolRegistry | None = None,
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
        self.llm = llm
        self.tools = tools
        self.context = context if context is not None else Context()
        self.events = events if events is not None else EventBus()
        self.max_requests = max_requests
        self.max_tool_calls = max_tool_calls
        self._model_lock = _MODEL_LOCKS.setdefault(llm, asyncio.Lock())

    async def _request(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool,
        event_fields: dict[str, Any],
        **options: Any,
    ) -> dict[str, Any]:
        if not stream:
            return await self.llm.create(messages=messages, **options)
        accumulator = ResponseAccumulator()
        async with self.llm.stream(messages=messages, **options) as chunks:
            async for chunk in chunks:
                accumulator.add_chunk(chunk)
                await self.events.emit("model.delta", **event_fields, chunk=chunk)
        return accumulator.finalize()

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Agent requires exactly one response choice.")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise ValueError("Agent requires response choice index 0.")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Response has no complete assistant message.")
        message = deepcopy(message)
        message.setdefault("role", "assistant")
        if message["role"] != "assistant":
            raise ValueError("Expected an assistant response.")
        Messages([message])  # Reuse Saturnus structure/JSON validation.
        reason = choice.get("finish_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("Invalid finish_reason type.")
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ValueError("Tool calls must be a list or null.")
        if reason == "stop" and calls:
            raise ValueError("A stop response unexpectedly contains tool calls.")
        if reason == "tool_calls":
            if not isinstance(calls, list) or not calls:
                raise ValueError("Tool-call response has no tool calls.")
            ids: set[str] = set()
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("Tool call must be an object.")
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
            identity = (
                session.id,
                session.workspace,
                session.provider,
                session.model_id,
            )
            committed = session.messages.snapshot()
            provider, model_id = session.provider, session.model_id
            await self.events.emit("turn.started", **fields)
            async with self._model_lock:
                if provider != self.llm.model.provider:
                    raise ValueError(
                        "Session provider does not match the LLM connection."
                    )
                self.llm.select_model(model_id)
                working = Messages(committed)
                if not isinstance(prompt, str):
                    raise TypeError("prompt must be a string.")
                working.add_user(prompt)
                definitions = self.tools.definitions() if self.tools is not None else []
                if definitions:
                    self.llm.require_tools()
                options = {
                    "thinking_mode": thinking_mode,
                    "output_limit": output_limit,
                    "tools": definitions or None,
                    "extra_body": extra_body,
                    "extra_headers": extra_headers,
                }
                for _ in range(self.max_requests):
                    await _checkpoint()
                    messages = self.context.build(working)
                    requests += 1
                    request_fields = {**fields, "request_number": requests}
                    await self.events.emit(
                        "model.started",
                        **request_fields,
                        provider=provider,
                        model_id=model_id,
                    )
                    response = await self._request(
                        messages=messages,
                        stream=stream,
                        event_fields=request_fields,
                        **options,
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
                    if self.tools is None or not definitions:
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
                        await self.events.emit("tool.started", **tool_fields)
                        side_effects = True
                        tool_error = None
                        content = None
                        tool_status = "FAILED"
                        try:
                            try:
                                result = await self.tools.execute(call)
                                content = result.content
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
                        except asyncio.CancelledError:
                            tool_status = "CANCELLED"
                            tool_error = {
                                "type": "CancelledError",
                                "message": "Tool operation cancelled.",
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
        except Exception as exc:
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
