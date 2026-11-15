"""MCP Prompt input validation and provenance stored alongside raw messages."""

import json
import re
from dataclasses import dataclass

MAX_PROMPT_BYTES = 65_536
MAX_PROMPT_MESSAGES = 64
PROMPT_ROLES = ("user", "assistant")


def _identifier(value: str, *, limit: int = 256) -> None:
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
        or re.search(r"[\x00-\x1f\x7f-\x9f]", value)):
        raise ValueError("无效的 MCP Prompt 标识。")
    value.encode("utf-8")


def validate_messages(messages: tuple[tuple[str, str], ...]) -> None:
    if not isinstance(messages, tuple) or not 0 < len(messages) <= MAX_PROMPT_MESSAGES:
        raise ValueError("MCP Prompt 消息数量超出限制。")
    size = 0
    for message in messages:
        if not isinstance(message, tuple) or len(message) != 2:
            raise ValueError("无效的 MCP Prompt 消息。")
        role, content = message
        if role not in PROMPT_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError("MCP Prompt 仅支持非空的 User 和 Assistant 文本。")
        size += len(json.dumps({"role": role, "content": content}, ensure_ascii=False).encode())
        if size > MAX_PROMPT_BYTES:
            raise ValueError("MCP Prompt 文本超出大小限制。")


@dataclass(frozen=True)
class PromptGroup:
    id: str
    server: str
    prompt: str
    start: int
    end: int
    edited: bool

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.id):
            raise ValueError("无效的 MCP Prompt 运行标识。")
        _identifier(self.server)
        _identifier(self.prompt)
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start < self.end:
            raise ValueError("无效的 MCP Prompt 消息范围。")
        if type(self.edited) is not bool:
            raise ValueError("无效的 MCP Prompt 编辑状态。")


@dataclass(frozen=True)
class PromptInput:
    id: str
    server: str
    prompt: str
    messages: tuple[tuple[str, str], ...]
    edited: bool = False

    def __post_init__(self) -> None:
        validate_messages(self.messages)
        self.group(0)

    def group(self, start: int) -> PromptGroup:
        return PromptGroup(self.id, self.server, self.prompt, start, start + len(self.messages), self.edited)

    def snapshot(self) -> list[dict[str, str]]:
        return [{"role": role, "content": text} for role, text in self.messages]
