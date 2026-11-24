"""Built-in tools for reading, writing, updating, and deleting long-term memories."""

from pathlib import Path
from typing import Any

from ..memory import (
    MAX_CONTENT_CHARS,
    MAX_PREVIEW_CHARS,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_TITLE_CHARS,
    MEMORY_NOTICE,
    MemoryLimitError,
    MemoryNotFoundError,
    MemoryStore,
)
from ..redaction import Redactor
from .approval import Decision, Operation, ToolError

MEMORY_TOOLS = frozenset({"memory_read", "memory_write", "memory_delete"})


def _scope_label(scope: str) -> str:
    return "项目" if scope == "project" else "全局"


def _sanitize_write_fields(
    arguments: dict[str, Any], redactor: Redactor,
) -> tuple[str, str, tuple[str, ...]]:
    title_raw = arguments["title"]
    content_raw = arguments["content"]
    tags_raw = arguments.get("tags") or ()
    if "\0" in title_raw or "\0" in content_raw or any("\0" in tag for tag in tags_raw):
        raise ToolError("invalid_arguments", "Memory fields must not contain null bytes.")
    title = redactor.json(title_raw.strip())
    content = redactor.json(content_raw.strip())
    if not isinstance(title, str) or not title.strip():
        raise ToolError("invalid_arguments", "Memory title must not be empty.")
    if not isinstance(content, str) or not content.strip():
        raise ToolError("invalid_arguments", "Memory content must not be empty.")
    title = title.strip()
    content = content.strip()
    if len(title) > MAX_TITLE_CHARS:
        raise ToolError("invalid_arguments", f"Memory title must not exceed {MAX_TITLE_CHARS} characters.")
    if len(content) > MAX_CONTENT_CHARS:
        raise ToolError("invalid_arguments", f"Memory content must not exceed {MAX_CONTENT_CHARS} characters.")
    if len(tags_raw) > MAX_TAGS:
        raise ToolError("invalid_arguments", f"Memory tags must not exceed {MAX_TAGS} items.")
    tags: list[str] = []
    for item in tags_raw:
        cleaned = redactor.json(item.strip())
        if not isinstance(cleaned, str) or not cleaned.strip():
            raise ToolError("invalid_arguments", "Memory tags must not be empty.")
        cleaned = cleaned.strip()
        if len(cleaned) > MAX_TAG_CHARS:
            raise ToolError("invalid_arguments", f"Each memory tag must not exceed {MAX_TAG_CHARS} characters.")
        if cleaned not in tags:
            tags.append(cleaned)
    return title, content, tuple(tags)


def prepare_memory(
    name: str,
    arguments: dict[str, Any],
    workspace: Path,
    store: MemoryStore | None,
    redactor: Redactor,
) -> Operation:
    if store is None:
        raise ToolError("memory_unavailable", "Long-term memory storage is unavailable.")

    if name == "memory_read":
        async def run_read() -> dict[str, Any]:
            scope = arguments.get("scope", "all")
            entry_id = arguments.get("id")
            try:
                if entry_id is not None:
                    entry = await store.read_entry(workspace, entry_id, scope=scope, redactor=redactor)
                    return {
                        "ok": True,
                        "notice": MEMORY_NOTICE,
                        "entry": {
                            "id": entry.id,
                            "scope": entry.scope,
                            "title": entry.title,
                            "content": entry.content,
                            "tags": list(entry.tags),
                            "updated_at": entry.updated_at,
                        },
                    }
                items, total = await store.list_entries(
                    workspace, scope=scope, limit=arguments.get("limit", 20), redactor=redactor,
                )
                return {
                    "ok": True,
                    "notice": MEMORY_NOTICE,
                    "scope": scope,
                    "total": total,
                    "entries": [
                        {
                            "id": item.id,
                            "scope": item.scope,
                            "title": item.title,
                            "tags": list(item.tags),
                            "preview": item.content[:MAX_PREVIEW_CHARS]
                            + ("…" if len(item.content) > MAX_PREVIEW_CHARS else ""),
                            "updated_at": item.updated_at,
                        }
                        for item in items
                    ],
                }
            except MemoryNotFoundError as exc:
                raise ToolError("memory_not_found", str(exc)) from None
            except ValueError as exc:
                raise ToolError("invalid_arguments", str(exc)) from None

        return Operation(
            decision=Decision.ALLOW,
            reason="Read saved long-term memory entries.",
            targets=(),
            details=arguments,
            run=run_read,
        )

    if name == "memory_write":
        scope = arguments["scope"]
        entry_id = arguments.get("id")
        title, content, tags = _sanitize_write_fields(arguments, redactor)
        label = _scope_label(scope)
        reason = (
            f"更新{label}长期记忆（{entry_id}）需要确认。"
            if entry_id
            else f"写入{label}长期记忆需要确认。"
        )
        details: dict[str, Any] = {
            "action": "update" if entry_id else "create",
            "scope": scope,
            "title": title,
            "tags": list(tags),
            "content": content,
        }
        if entry_id is not None:
            details["id"] = entry_id

        async def run_write() -> dict[str, Any]:
            try:
                saved, created = await store.write_entry(
                    workspace,
                    scope=scope,
                    title=title,
                    content=content,
                    tags=tags,
                    entry_id=entry_id,
                    redactor=redactor,
                )
            except MemoryNotFoundError as exc:
                raise ToolError("memory_not_found", str(exc)) from None
            except MemoryLimitError as exc:
                raise ToolError("memory_limit_exceeded", str(exc)) from None
            except ValueError as exc:
                raise ToolError("invalid_arguments", str(exc)) from None
            return {
                "ok": True,
                "action": "created" if created else "updated",
                "entry": {
                    "id": saved.id,
                    "scope": saved.scope,
                    "title": saved.title,
                    "tags": list(saved.tags),
                    "updated_at": saved.updated_at,
                },
            }

        return Operation(
            decision=Decision.ASK,
            reason=reason,
            targets=(),
            details=details,
            run=run_write,
            allow_session_trust=True,
        )

    if name == "memory_delete":
        scope = arguments["scope"]
        entry_id = arguments["id"]
        label = _scope_label(scope)

        async def run_delete() -> dict[str, Any]:
            try:
                removed = await store.delete_entry(workspace, scope=scope, entry_id=entry_id)
            except MemoryNotFoundError as exc:
                raise ToolError("memory_not_found", str(exc)) from None
            except ValueError as exc:
                raise ToolError("invalid_arguments", str(exc)) from None
            return {
                "ok": True,
                "deleted": {
                    "id": removed.id,
                    "scope": removed.scope,
                    "title": redactor.text(removed.title),
                },
            }

        return Operation(
            decision=Decision.ASK,
            reason=f"删除{label}长期记忆（{entry_id}）需要确认。",
            targets=(),
            details={"scope": scope, "id": entry_id},
            run=run_delete,
            allow_session_trust=True,
        )

    raise ToolError("invalid_operation", f"Unsupported memory tool: {name!r}")
