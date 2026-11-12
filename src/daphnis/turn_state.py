"""Recovery decisions attached to the original user message."""

import json
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

from .tool_state import TOOL_STATE

TURN_STATE = "daphnis_turn_state"
TURN_ATTEMPT = "daphnis_turn_attempt"
ABANDONED = "ABANDONED"
CONTINUE = "CONTINUE"
TERMINATED = "TERMINATED"
_RUNTIME_KEY = "_daphnis_runtime"
_TERMINATED_KEY = "turn_terminated_by_user"
_TERMINATION_NOTE = "[Daphnis runtime: The user terminated this turn after the tool group finished.]"


class RecoveryKind(StrEnum):
    FAILED_USER = "FAILED_USER"
    INTERRUPTED_TURN = "INTERRUPTED_TURN"


class RecoveryAction(StrEnum):
    RETRY = "retry"
    CONTINUE = "continue"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class RecoveryState:
    kind: RecoveryKind
    message_index: int
    key: str


def recovery_state(messages: list[dict[str, Any]], session_id: str) -> RecoveryState | None:
    if not messages or messages[-1]["role"] not in ("user", "tool"):
        return None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message["role"] != "user":
            continue
        if message.get(TURN_STATE) in (ABANDONED, TERMINATED):
            return None
        kind = RecoveryKind.FAILED_USER if index == len(messages) - 1 else RecoveryKind.INTERRUPTED_TURN
        return RecoveryState(kind, index, f"{session_id}:{index}:{message.get(TURN_ATTEMPT, '')}")
    return None


def _result_object(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _reports_interruption(value: dict[str, Any] | None) -> bool:
    return (
        value is not None
        and value.get("source") == "daphnis_recovery"
        and isinstance(error := value.get("error"), dict)
        and error.get("code") in ("execution_skipped", "execution_result_unknown")
    )


def _with_termination(content: str | list, value: dict[str, Any] | None) -> str | list:
    if value is not None:
        runtime = value.get(_RUNTIME_KEY, {})
        if isinstance(runtime, dict):
            if runtime.get(_TERMINATED_KEY) is True:
                return content
            return json.dumps({**value, _RUNTIME_KEY: {**runtime, _TERMINATED_KEY: True}}, ensure_ascii=False)
    if isinstance(content, list):
        note = {"type": "text", "text": _TERMINATION_NOTE}
        return content if content[-1:] == [note] else [*content, note]
    suffix = "\n\n" + _TERMINATION_NOTE
    return content if content.endswith(suffix) else content + suffix


def project_terminated_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate the last result of each visible, completed group at termination."""
    projected = messages.copy()
    users = [index for index, message in enumerate(messages) if message["role"] == "user"]
    for start, end in pairwise([*users, len(messages)]):
        if messages[start].get(TURN_STATE) != TERMINATED:
            continue
        assistant = end - 1
        while assistant > start and messages[assistant]["role"] == "tool":
            assistant -= 1
        results = messages[assistant + 1:end]
        calls = messages[assistant].get("tool_calls")
        if messages[assistant]["role"] != "assistant" or not results or not isinstance(calls, list):
            continue
        call_ids = {
            call["id"] for call in calls
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]
        }
        result_ids = {
            result["tool_call_id"] for result in results
            if TOOL_STATE not in result and isinstance(result.get("content"), (str, list))
            and isinstance(result.get("tool_call_id"), str)
        }
        if len(call_ids) != len(calls) or len(result_ids) != len(results) or call_ids != result_ids:
            continue
        values = [_result_object(result["content"]) for result in results]
        if any(_reports_interruption(value) for value in values):
            continue
        last = results[-1]
        content = _with_termination(last["content"], values[-1])
        if content != last["content"]:
            projected[end - 1] = {**last, "content": content}
    return projected


def project_turn_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in message.items() if key not in (TURN_STATE, TURN_ATTEMPT)}
        for message in messages if message.get(TURN_STATE) != ABANDONED
    ]
