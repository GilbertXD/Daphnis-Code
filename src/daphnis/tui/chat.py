"""Presentation-only message rows, updated in place as deltas arrive."""

import json
import re
from typing import Any

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Static

from ..agent import assistant_text
from ..events import Event
from ..mcp import TOOL_PREFIX
from .theme import ACCENT, BORDER, ERROR, MUTED, SUCCESS, SURFACE, TEXT, WARNING


class ChatRow(Static):
    def __init__(self, title: str, text: str = "", *, classes: str = "") -> None:
        super().__init__(classes=classes, markup=False)
        self.title = title
        self.text = text
        self.reasoning = ""
        self.outcome = ""
        self.output = ""
        self.edits: tuple[int, int] | None = None
        self.tool_group = None
        self.dirty = True

    def redraw(self) -> None:
        if not self.dirty:
            return
        if self.has_class("tool"):
            state, _, detail = self.outcome.partition(" · ")
            label, icon, color = {
                "COMPLETED": ("", "✓", SUCCESS),
                "ERROR": ("失败", "×", ERROR),
                "FAILED": ("失败", "×", ERROR),
                "CANCELLED": ("已取消", "·", WARNING),
                "运行中": ("", "◌", ACCENT),
                "等待审批": ("等待审批", "?", WARNING),
            }.get(state, (state, "·", MUTED))
            line = Table.grid(expand=True, padding=(0, 1))
            line.add_column(width=1, no_wrap=True)
            line.add_column(width=7, no_wrap=True, overflow="ellipsis")
            line.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
            name = self.title.removeprefix("工具 · ")
            name = {
                "read_file": "Read",
                "list_directory": "List",
                "apply_patch": "Edit",
                "run_command": "Run",
                "search": "Search",
            }.get(name, name)
            target = Text(self.text.splitlines()[0] if self.text else "", style=TEXT)
            if label:
                target.append("  " + label, style=color)
            if state == "COMPLETED" and self.edits is not None:
                target.append(f"  +{self.edits[0]}", style=SUCCESS)
                target.append(f"  −{self.edits[1]}", style=ERROR)
            line.add_row(
                Text(icon, style=color),
                Text(name, style=MUTED),
                target,
            )
            pieces = [line]
            if detail:
                pieces.append(Text("  " + detail, style=WARNING))
            if self.output:
                pieces.append(
                    Panel(
                        Text(self.output, style=TEXT),
                        border_style=BORDER,
                        padding=(0, 1),
                        box=box.ROUNDED,
                        style=f"on {SURFACE}",
                    )
                )
        else:
            body = Markdown(self.text, code_theme="monokai")
            if self.has_class("user"):
                line = Table.grid(expand=True)
                line.add_column(width=3)
                line.add_column(ratio=1)
                line.add_row(Text("❯", style=f"bold {ACCENT}"), body)
                pieces = [line]
            elif self.title == "助手":
                pieces = [body] if self.text else []
            else:
                pieces = [Text("✧ " + self.title, style=ACCENT), body]
        if self.reasoning:
            pieces.append(Text("思考片段：" + self.reasoning[-400:], style="dim"))
        if self.outcome and not self.has_class("tool"):
            pieces.append(Text(self.outcome, style="dim"))
        self.update(Group(*pieces))
        self.dirty = False


def tool_summary(data: dict[str, Any]) -> str:
    name = data.get("name", "")
    if name.startswith(TOOL_PREFIX):
        return f"{name} {data.get('arguments', '{}')}"[:300]
    try:
        arguments = json.loads(data.get("arguments", "{}"))
        if "path" in arguments:
            return str(arguments["path"])[:300]
        if "argv" in arguments:
            return " ".join(arguments["argv"])[:300]
        if "patch" in arguments:
            patch = arguments["patch"]
            if not isinstance(patch, str):
                return "补丁"
            for line in patch.splitlines():
                if line.startswith("+++ ") and line[4:] != "/dev/null":
                    path = line[4:].split("\t")[0]
                    return path.removeprefix("b/")
            return "补丁"
        if "query" in arguments:
            return str(arguments["query"])[:300]
    except (TypeError, ValueError):
        pass
    return ""


def patch_counts(arguments: str) -> tuple[int, int] | None:
    try:
        patch = json.loads(arguments).get("patch")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(patch, str):
        return None
    added = removed = old_left = new_left = 0
    for line in patch.splitlines():
        header = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
        if header:
            old_left = int(header[1]) if header[1] is not None else 1
            new_left = int(header[2]) if header[2] is not None else 1
        elif old_left or new_left:
            if line.startswith("+"):
                added += 1
                new_left -= 1
            elif line.startswith("-"):
                removed += 1
                old_left -= 1
            elif line.startswith(" "):
                old_left -= 1
                new_left -= 1
    return added, removed


def command_output(content: str | None) -> str:
    try:
        data = json.loads(content or "{}")
    except (TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    text = data.get("stdout") or data.get("stderr")
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = Text.from_ansi(text).plain.strip().splitlines()
    return ("…\n" if len(lines) > 3 else "") + "\n".join(lines[-3:])[-360:]


class ToolGroup(Vertical):
    DEFAULT_CSS = """
    ToolGroup { height: auto; margin-left: 3; }
    ToolGroup Collapsible { height: auto; border: none; padding: 0;
        background: $background; }
    ToolGroup Collapsible:focus-within { background-tint: transparent; }
    ToolGroup CollapsibleTitle { color: $text-muted; padding: 0; }
    ToolGroup Collapsible > Contents { height: auto; padding: 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.more: Collapsible | None = None

    async def add_row(self, row: ChatRow) -> None:
        row.tool_group = self
        self.count += 1
        if self.count <= 3:
            await self.mount(row)
        elif self.more is None:
            self.more = Collapsible(
                row,
                title="1 more operation",
                collapsed=True,
                collapsed_symbol="▸",
                expanded_symbol="▾",
            )
            await self.mount(self.more)
        else:
            await self.more.query_one(Collapsible.Contents).mount(row)
            self.more.title = f"{self.count - 3} more operations"

    def reveal(self) -> None:
        if self.more is not None:
            self.more.collapsed = False


class ChatView(VerticalScroll):
    DEFAULT_CSS = """
    ChatView { width: 1fr; height: 1fr; padding: 0; scrollbar-gutter: stable;
        scrollbar-size: 1 1; scrollbar-background: $background;
        scrollbar-color: $panel; }
    ChatRow { height: auto; margin: 1 0 1 3; padding: 0; border: none; }
    ChatRow.user { margin-left: 0; background: $background; padding: 0; }
    ChatRow.command { border-top: solid #48566b; border-bottom: solid #48566b; }
    ChatRow.tool { margin: 0; padding: 0; color: $text-muted; }
    ChatRow.notice { color: $text-muted; }
    ChatRow.welcome { margin: 2 0; padding: 1 2; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.assistants: dict[tuple[str, int], ChatRow] = {}
        self.tools: dict[tuple[str, int, str], ChatRow] = {}
        self.turn_rows: dict[str, list[ChatRow]] = {}
        self.pending_user: ChatRow | None = None
        self._tool_group: ToolGroup | None = None

    def on_mount(self) -> None:
        self.anchor()
        self.set_interval(0.05, self.refresh_rows)

    def refresh_rows(self) -> None:
        for row in self.query(ChatRow):
            if row.dirty:
                row.redraw()

    async def add(self, title: str, text: str = "", *, classes: str = "") -> ChatRow:
        row = ChatRow(title, text, classes=classes)
        if "tool" in classes.split():
            if self._tool_group is None:
                self._tool_group = ToolGroup()
                await self.mount(self._tool_group)
            await self._tool_group.add_row(row)
        else:
            self._tool_group = None
            await self.mount(row)
        row.redraw()
        self.scroll_end(animate=False)
        return row

    async def clear_display(self) -> None:
        await self.remove_children()
        self.assistants.clear()
        self.tools.clear()
        self.turn_rows.clear()
        self.pending_user = None
        self._tool_group = None

    async def restore(self, messages: list[dict[str, Any]]) -> None:
        await self.clear_display()
        calls = {}
        for message in messages:
            role = message["role"]
            if role == "assistant":
                calls = {call["id"]: call["function"] for call in message.get("tool_calls") or ()}
            if role == "tool":
                function = calls.get(message["tool_call_id"], {})
                row = await self.add(
                    "工具 · " + function.get("name", "工具结果"),
                    tool_summary(function),
                    classes="tool",
                )
                row.outcome = "已保存"
                row.edits = patch_counts(function.get("arguments", "{}"))
                if function.get("name") == "run_command":
                    row.output = command_output(message.get("content"))
                elif function.get("name", "").startswith(TOOL_PREFIX):
                    row.output = (message.get("content") or "")[:2000]
                try:
                    result = json.loads(message.get("content") or "{}")
                    if isinstance(result, dict) and "ok" in result:
                        row.outcome = "COMPLETED" if result["ok"] else "ERROR"
                        if isinstance(result.get("error"), dict):
                            row.outcome += (
                                " · " + str(result["error"].get("message", ""))[:300]
                            )
                except (TypeError, ValueError):
                    pass
                row.dirty = True
                if row.output or row.outcome.startswith("ERROR"):
                    row.tool_group.reveal()
            else:
                content = assistant_text(message) if role == "assistant" else message.get("content")
                if (
                    role == "assistant"
                    and not content
                    and not message.get("reasoning_content")
                    and message.get("tool_calls")
                ):
                    continue
                row = await self.add(
                    "你" if role == "user" else "助手" if role == "assistant" else role,
                    content if isinstance(content, str) else "",
                    classes="user" if role == "user" else "",
                )
                row.reasoning = message.get("reasoning_content") or ""
                row.dirty = True
        self.refresh_rows()

    async def event(self, event: Event) -> None:
        data = event.data
        if event.name == "turn.started":
            if self.pending_user is not None:
                self.turn_rows.setdefault(event.turn_id, []).append(self.pending_user)
                self.pending_user = None
        elif event.name in ("model.delta", "model.finished"):
            key = (event.turn_id, data["request_number"])
            if event.name == "model.delta":
                choices = data["chunk"].get("choices", [])
                message = next(
                    (
                        choice.get("delta", {})
                        for choice in choices
                        if choice["index"] == 0
                    ),
                    {},
                )
            else:
                choices = data["response"].get("choices", [])
                message = choices[0].get("message", {}) if choices else {}
            content, reasoning = (
                (message.get("content") or message.get("refusal"))
                if event.name == "model.delta" else assistant_text(message),
                message.get("reasoning_content"),
            )
            if not content and not reasoning:
                return
            row = self.assistants.get(key)
            if row is None:
                row = await self.add("助手")
                self.assistants[key] = row
                self.turn_rows.setdefault(event.turn_id, []).append(row)
            if event.name == "model.delta":
                if isinstance(content, str):
                    row.text += content
                if isinstance(reasoning, str):
                    row.reasoning = (row.reasoning + reasoning)[-400:]
            else:
                row.text = content if isinstance(content, str) else row.text
                row.reasoning = (
                    reasoning if isinstance(reasoning, str) else row.reasoning
                )
            row.dirty = True
        elif event.name.startswith("tool."):
            key = (event.turn_id, data.get("request_number", 0), data["tool_call_id"])
            row = self.tools.get(key)
            if row is None:
                row = await self.add(
                    f"工具 · {data['name']}", tool_summary(data), classes="tool"
                )
                self.tools[key] = row
                self.turn_rows.setdefault(event.turn_id, []).append(row)
            if event.name == "tool.started":
                row.outcome = "运行中"
                row.edits = patch_counts(data.get("arguments", "{}"))
            elif event.name == "tool.approval_requested":
                row.outcome = "等待审批"
            elif event.name == "tool.blocked":
                row.outcome = "已拦截：" + data["reason"]
            elif event.name == "tool.finished":
                row.outcome = data["status"]
                if data.get("error"):
                    row.outcome += " · " + str(data["error"].get("message", ""))[:300]
                try:
                    value = json.loads(data.get("content") or "{}")
                    if isinstance(value, dict):
                        paths = value.get("applied_files") or (
                            [value["path"]] if "path" in value else []
                        )
                        if paths:
                            row.text = "\n".join(str(path) for path in paths)[:800]
                except ValueError:
                    pass
                if data["name"] == "run_command":
                    row.output = command_output(data.get("content"))
                elif data["name"].startswith(TOOL_PREFIX):
                    row.output = (data.get("content") or "")[:2000]
            row.dirty = True
            if row.output or row.outcome.startswith(("ERROR", "FAILED", "等待审批")):
                row.tool_group.reveal()
        elif event.name == "turn.finished":
            if data["status"] != "COMPLETED":
                for row in self.turn_rows.get(event.turn_id, []):
                    row.outcome = f"{data['status']} · 本轮未提交；工具副作用不自动回滚"
                    row.dirty = True
                    if row.tool_group is not None:
                        row.tool_group.reveal()
            self.refresh_rows()
