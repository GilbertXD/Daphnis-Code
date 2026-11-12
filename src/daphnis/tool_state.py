"""Internal tool placeholders stored alongside their assistant calls."""

import json
from typing import Any

from saturnus import Messages

TOOL_STATE = "daphnis_tool_state"
PENDING = "PENDING"
RUNNING = "RUNNING"
TERMINATED_SKIP_MESSAGE = "任务已终止，工具未执行。"
_UNKNOWN = "execution_result_unknown"


def _recovery_result(code: str, message: str) -> str:
    return json.dumps({
        "ok": False,
        "source": "daphnis_recovery",
        "error": {"code": code, "message": message},
    }, ensure_ascii=False)


def _is_unknown(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    try:
        value = json.loads(content)
    except (ValueError, RecursionError):
        return False
    return (
        isinstance(value, dict)
        and value.get("source") == "daphnis_recovery"
        and isinstance(value.get("error"), dict)
        and value["error"].get("code") == _UNKNOWN
    )


def resolve_tool_placeholders(
    messages: list[dict[str, Any]], indices: dict[str, int], *, terminate: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Record unknown results and skip pending calls when execution is unsafe."""
    unknown = any(
        messages[index].get(TOOL_STATE) == RUNNING or _is_unknown(messages[index].get("content"))
        for index in indices.values()
    )
    for call_id, index in indices.items():
        state = messages[index].get(TOOL_STATE)
        if state == RUNNING:
            content = _recovery_result(_UNKNOWN, "执行中断，工具可能已经产生副作用，请检查实际环境。")
        elif state == PENDING and (terminate or unknown):
            content = _recovery_result(
                "execution_skipped",
                TERMINATED_SKIP_MESSAGE if terminate else "前序工具结果未知，此工具未执行，需要重新规划。",
            )
        else:
            continue
        messages = update_tool_result(messages, index, call_id, content=content)
    return messages, unknown


def stage_tool_group(history: Messages, assistant: dict[str, Any]) -> Messages:
    group = Messages(history.snapshot())
    group.add(assistant)
    for call in assistant["tool_calls"]:
        group.add_tool_result(tool_call_id=call["id"], content=None, extra={TOOL_STATE: PENDING})
    return group


def update_tool_result(
    messages: list[dict[str, Any]], index: int, call_id: str, *, content: str | None = None,
) -> list[dict[str, Any]]:
    """Copy the snapshot and advance exactly one unfinished result."""
    current = messages[index]
    allowed = (PENDING,) if content is None else (PENDING, RUNNING)
    if (
        current["role"] != "tool"
        or current.get("tool_call_id") != call_id
        or current.get(TOOL_STATE) not in allowed
    ):
        raise ValueError(f"Invalid state transition for tool call {call_id!r}.")
    result = {"role": "tool", "tool_call_id": call_id, "content": content}
    if content is None:
        result[TOOL_STATE] = RUNNING
    snapshot = messages.copy()
    snapshot[index] = result
    return snapshot


def interrupted_tool_group(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    """Locate and validate the trailing group containing internal placeholders."""
    unfinished = {index for index, message in enumerate(messages) if TOOL_STATE in message}
    if not unfinished:
        return None
    start = len(messages) - 1
    while start >= 0 and messages[start]["role"] == "tool":
        start -= 1
    if start < 0 or messages[start]["role"] != "assistant" or min(unfinished) <= start:
        raise ValueError("Tool placeholders must belong to the last assistant tool group.")
    calls = messages[start].get("tool_calls")
    if not isinstance(calls, list) or len(calls) != len(messages) - start - 1:
        raise ValueError("Interrupted tool group has mismatched calls and results.")
    indices = {}
    for index, call in enumerate(calls, start=start + 1):
        result = messages[index]
        call_id = call.get("id") if isinstance(call, dict) else None
        state = result.get(TOOL_STATE)
        if (
            not isinstance(call_id, str) or not call_id.strip() or call_id in indices
            or result.get("tool_call_id") != call_id
        ):
            raise ValueError("Invalid tool result in interrupted tool group.")
        content = result.get("content")
        if index in unfinished:
            if state not in (PENDING, RUNNING) or content is not None:
                raise ValueError("Invalid unfinished tool result.")
        elif not isinstance(content, (str, list)):
            raise ValueError("Invalid finished tool result.")
        indices[call_id] = index
    return calls, indices
