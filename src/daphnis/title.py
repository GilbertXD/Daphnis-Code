"""Build bounded title material from the committed input of one Turn."""

import re
from collections.abc import Mapping
from typing import Any

from .attachments import ATTACHMENT_ONLY_PROMPT, AttachmentBatch
from .redaction import Redactor
from .terminal_text import sanitize_terminal

_TITLE_LENGTH = 40
_SMALL_TEXT_BYTES = 4096
_ATTACHMENT_LABEL = "附件: "
_PREFIX = re.compile(
    r"^(?:(?:(?:本|此|当前)?(?:会话|对话)的?)?标题\s*[:：]\s*"
    r"|(?:(?:本|此|当前)?(?:会话|对话)的?)?标题\s*(?:是|为)\s*[:：]?\s*"
    r"|(?:这段|本次)(?:对话|会话)关于\s*[:：]?\s*"
    r"|(?:(?:session|conversation)\s+)?title\s*[:：]\s*"
    r"|the title of (?:this|the) (?:session|conversation) is\s*[:：]?\s*)",
    re.IGNORECASE,
)
_MARKDOWN_PREFIX = re.compile(r"^(?:#{1,6}|[-*+]|\d+[.)、])\s+")
_WRAPPERS = (("**", "**"), ("__", "__"), ("~~", "~~"), ("*", "*"), ("_", "_"), ("`", "`"),
             ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))


def _clip(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    return text[:length - 1].rstrip() + "…" if length > 1 else ""


def _fit_lines(lines: dict[int, str], budget: int) -> dict[int, str]:
    fitted = {}
    for remaining, (index, line) in enumerate(lines.items()):
        limit = budget // (len(lines) - remaining) - 1
        if limit <= len("Assistant: "):
            continue
        fitted[index] = _clip(line, limit)
        budget -= len(fitted[index]) + 1
    return fitted


def build_title_source(
    messages: list[dict[str, Any]],
    attachment_refs: Mapping[int, AttachmentBatch],
    *,
    max_chars: int = 3000,
    excerpts: Mapping[str, str] | None = None,
    redactor: Redactor | None = None,
) -> str:
    redact = redactor or Redactor(())

    def visible(text: str) -> str:
        return " ".join(redact.text(sanitize_terminal(text.replace("\r", "\n"))).split())

    texts, refs = {}, {}
    order = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ("user", "assistant") or message.get("tool_calls"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        batch = attachment_refs.get(index)
        role_label = role.capitalize() + ": "
        if isinstance(content, str) and (text := visible(content)) and not (batch and text == ATTACHMENT_ONLY_PROMPT):
            texts[order] = role_label + text
            order += 1
        if batch:
            for item in batch.files:
                name = _clip(visible(item.name), 96)
                mime_type = _clip(visible(item.mime_type), 48)
                line = f"{role_label}{_ATTACHMENT_LABEL}{name} ({mime_type}, {item.size_bytes} 字节)"
                if excerpts and item.size_bytes <= _SMALL_TEXT_BYTES and (excerpt := excerpts.get(item.id)):
                    line += " · " + _clip(visible(excerpt), 200)
                refs[order] = line
                order += 1

    ref_budget = min(sum(len(line) + 1 for line in refs.values()), max_chars // 3) if texts else max_chars
    fitted = {**_fit_lines(texts, max_chars - ref_budget), **_fit_lines(refs, ref_budget)}
    return "\n".join(fitted[index] for index in sorted(fitted))


def fallback_title(source: str) -> str | None:
    first_attachment = first_text = None
    for line in source.splitlines():
        role, _, content = line.partition(": ")
        if content.startswith(_ATTACHMENT_LABEL):
            first_attachment = first_attachment or content[len(_ATTACHMENT_LABEL):].split(" (", 1)[0]
        elif role == "User":
            return _clip(content, _TITLE_LENGTH)
        elif role == "Assistant":
            first_text = first_text or content
    return _clip(first_text or first_attachment or "", _TITLE_LENGTH) or None


def normalize_title(text: str, redactor: Redactor) -> str | None:
    value = redactor.text(sanitize_terminal(text.replace("\r", "\n"))).strip()
    if len(value.splitlines()) != 1:
        return None
    value = " ".join(value.split())
    if "```" in value:
        return None
    while value:
        previous = value
        value = _PREFIX.sub("", _MARKDOWN_PREFIX.sub("", value)).rstrip(" 。！？!?：:；;，,．.")
        for opening, closing in _WRAPPERS:
            if value.startswith(opening) and value.endswith(closing) and len(value) >= len(opening) + len(closing):
                value = value[len(opening):-len(closing)].strip()
                break
        if value == previous:
            break
    if not 0 < len(value) <= _TITLE_LENGTH or re.search(r"[`]|\[[^\]]+\]\(", value):
        return None
    if not any(char.isalnum() for char in value):
        return None
    return value
