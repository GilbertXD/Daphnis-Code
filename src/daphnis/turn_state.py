"""Recovery decisions attached to the original user message."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

TURN_STATE = "daphnis_turn_state"
TURN_ATTEMPT = "daphnis_turn_attempt"
ABANDONED = "ABANDONED"
CONTINUE = "CONTINUE"
TERMINATED = "TERMINATED"


class RecoveryKind(StrEnum):
    FAILED_USER = "FAILED_USER"
    INTERRUPTED_TURN = "INTERRUPTED_TURN"


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


def project_turn_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in message.items() if key not in (TURN_STATE, TURN_ATTEMPT)}
        for message in messages if message.get(TURN_STATE) != ABANDONED
    ]
