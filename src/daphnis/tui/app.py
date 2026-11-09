"""Textual terminal presentation over Operations and EventBus."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, SelectionList, Static
from textual.widgets.option_list import Option

from ..app import SessionSaveError
from ..events import Event, EventBus
from ..operations import Operations
from ..terminal_commands import CommandResult, safe_error, terminal_command
from ..tools import ApprovalDecision, ApprovalMode
from .approval import ApprovalScreen
from .chat import ChatView
from .input import PromptInput
from .model_picker import ModelPicker
from .status import UsageView
from .theme import ACCENT, MUTED, SUCCESS, THEME, WARNING


class ChoiceScreen(ModalScreen[Any]):
    BINDINGS = [Binding("escape", "close", "返回")]
    DEFAULT_CSS = """
    ChoiceScreen { align: center middle; background: $background 75%; }
    #choices-box { width: 90%; max-width: 96; height: auto; max-height: 90%;
        border: round $primary 45%; background: $surface; padding: 1 2; }
    #choices-box > Static { height: auto; margin-bottom: 1; }
    #choice-title { color: $primary; text-style: bold; }
    #choice-hint { color: $text-muted; margin-top: 1; margin-bottom: 0; }
    #choice-columns { height: 10; }
    .choice-column { width: 1fr; margin-right: 1; }
    .choice-column Static { height: 2; color: $secondary; text-style: bold; }
    .choice-column OptionList { height: 1fr; border: none; padding: 0;
        background: $surface; }
    .choice-column OptionList:focus { border: none; background-tint: transparent; }
    .choice-column OptionList > .option-list--option-highlighted {
        background: transparent; color: $foreground; }
    .choice-column OptionList:focus > .option-list--option-highlighted {
        background: $primary 18%; color: $primary; text-style: bold; }
    """

    def __init__(
        self, title: str, groups: dict[str, list[tuple[str, Any, bool]]]
    ) -> None:
        super().__init__()
        self.heading = title
        self.groups = groups
        self.values: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="choices-box"):
            yield Static(self.heading, markup=False, id="choice-title")
            with Horizontal(id="choice-columns") as columns:
                rows = max(
                    (
                        sum(label.count("\n") + 1 for label, _, _ in items)
                        for items in self.groups.values()
                    ),
                    default=0,
                )
                columns.styles.height = min(12, max(5, rows + 2))
                for index, (group, options) in enumerate(self.groups.items()):
                    with Vertical(classes="choice-column"):
                        yield Static(group, markup=False)
                        entries = []
                        for number, (label, value, disabled) in enumerate(options):
                            key = f"{index}-{number}"
                            self.values[key] = value
                            entries.append(Option(label, id=key, disabled=disabled))
                        yield OptionList(*entries, id=f"choices-{index}", markup=False)
            yield Static(
                "↑↓ 选择   Tab 切换分列   Enter 确认   Esc 返回",
                markup=False,
                id="choice-hint",
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.values[event.option.id])

    def action_close(self) -> None:
        self.dismiss(None)


class ToolsScreen(ModalScreen[tuple[str, ...] | None]):
    BINDINGS = [Binding("escape", "close", "返回")]
    DEFAULT_CSS = """
    ToolsScreen { align: center middle; background: $background 75%; }
    #tools-box { width: 65; max-width: 95%; height: auto;
        border: round $primary 45%; background: $surface; padding: 1 2; }
    #tools-title { height: 2; color: $primary; text-style: bold; }
    #tools-note { height: auto; color: $text-muted; margin-bottom: 1; }
    #tool-selections { height: 6; border: none; padding: 0; }
    #tool-selections:focus { border: none; background-tint: transparent; }
    #tools-buttons { height: 3; align-horizontal: right; }
    #tools-buttons Button { margin-left: 1; }
    """

    def __init__(self, ops: Operations) -> None:
        super().__init__()
        self.ops = ops

    def compose(self) -> ComposeResult:
        with Vertical(id="tools-box"):
            yield Static("可用工具", markup=False, id="tools-title")
            offline = len(self.ops.enabled_tools) - len(self.ops.effective_tools)
            note = f"Space 勾选 · 审批模式：{self.ops.approval_mode}"
            if offline:
                note += f" · 保留 {offline} 个离线工具的启用偏好"
            yield Static(
                note,
                markup=False,
                id="tools-note",
            )
            yield SelectionList(
                *[
                    (name, name, name in self.ops.enabled_tools)
                    for name in self.ops.all_tools
                ],
                id="tool-selections",
            )
            with Horizontal(id="tools-buttons"):
                yield Button("返回", id="tools-cancel")
                yield Button("应用", id="tools-apply", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(
            tuple(self.query_one(SelectionList).selected)
            if event.button.id == "tools-apply"
            else None
        )

    def action_close(self) -> None:
        self.dismiss(None)


class CoreNotification(Message):
    def __init__(self, event: Event) -> None:
        super().__init__()
        self.event = event


class DaphnisApp(App[None]):
    TITLE = "Daphnis"
    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", "取消 / 退出", priority=True),
        Binding("ctrl+q", "quit_safely", "退出", priority=True),
        Binding("escape", "interrupt", "中断", show=False),
    ]
    CSS = """
    Screen { background: $background; align-horizontal: center; }
    #shell { width: 1fr; height: 1fr; margin: 0 2; }
    #heading { height: 3; padding: 1 0 0 0;
        border-bottom: solid #48566b; text-overflow: ellipsis; }
    #prompt-box { height: 4; border-top: solid #48566b;
        border-bottom: solid #48566b; padding: 0; }
    #prompt-symbol { width: 3; height: 1; color: $primary; text-style: bold; }
    #composer { width: 1fr; height: 1fr; border: none;
        padding: 0; background: $background; }
    #composer:focus { border: none; }
    #composer:disabled { opacity: 70%; }
    #composer .text-area--cursor-line { background: transparent; }
    #composer .text-area--placeholder { color: $foreground 40%; }
    Button { border: none; background: $panel; color: $foreground; min-width: 12; }
    Button:focus { border: none; text-style: bold; }
    #tools-apply { background: $primary; color: $background; }
    #approval-yes { background: $warning; color: $background; }
    """

    def __init__(self, ops: Operations, events: EventBus) -> None:
        super().__init__()
        self.register_theme(THEME)
        self.theme = "daphnis"
        self.ops = ops
        self.events = events
        self._subscriptions = []
        self._sending = False
        self._shutting_down = False
        self._approval_screen: ApprovalScreen | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            yield Static(id="heading", markup=False)
            yield ChatView(id="chat")
            with Horizontal(id="prompt-box"):
                yield Static("❯", id="prompt-symbol", markup=False)
                yield PromptInput(
                    id="composer",
                    placeholder="Ask Daphnis to do something…",
                    show_line_numbers=False,
                )
            yield UsageView(id="usage")

    async def on_mount(self) -> None:
        for name in (
            "turn.started",
            "turn.finished",
            "model.started",
            "model.delta",
            "model.finished",
            "model.usage",
            "tool.started",
            "tool.finished",
            "tool.approval_requested",
            "tool.blocked",
            "mcp.error",
            "mcp.tools_changed",
        ):
            self._subscriptions.append(self.events.on(name, self._forward))
        messages = self.ops.message_snapshot()
        await self.query_one(ChatView).restore(messages)
        if not messages:
            await self.query_one(ChatView).add(
                "从一个任务开始",
                "描述你的目标，一起把它完成。\n\n"
                "`/model` 切换模型　`/session` 恢复会话　`/help` 查看命令",
                classes="welcome",
            )
        self.update_status()
        self.query_one(PromptInput).focus()

    def _forward(self, event: Event) -> None:
        if not self._shutting_down:
            self.post_message(CoreNotification(event))

    def update_status(self) -> None:
        ops = self.ops
        session = ops.session
        busy = self._sending or ops.running
        state = (
            "approval"
            if ops.pending_approval
            else "running"
            if busy
            else "ready"
        )
        workspace = str(session.workspace)
        home = str(Path.home())
        if workspace.startswith(home + "/"):
            workspace = "~" + workspace[len(home) :]
        heading = Table.grid(expand=True, padding=(0, 1))
        heading.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        heading.add_column(justify="right", no_wrap=True)
        label = Text("♄  daphnis", style=f"bold {ACCENT}")
        label.append("  /  " + workspace, style=MUTED)
        color = WARNING if busy else SUCCESS
        mode = ops.approval_mode
        badge = Text(
            f"{mode.upper()}  ·  ",
            style=WARNING if mode is ApprovalMode.FULL else MUTED,
        )
        badge.append("● " + state, style=color)
        heading.add_row(label, badge)
        self.query_one("#heading", Static).update(heading)
        self.query_one("#heading", Static).tooltip = f"会话 {session.id}"
        self.query_one(UsageView).update_usage(
            ops.turn_usage,
            ops.session_usage,
            model_id=session.model_id,
            thinking=ops.thinking,
            tools=len(ops.effective_tools),
        )
        for picker in self.query(ModelPicker):
            picker.disabled = busy

    async def on_core_notification(self, message: CoreNotification) -> None:
        if self._shutting_down:
            return
        event = message.event
        if event.name == "mcp.error":
            self.notify(f"MCP {event.data['server']} · {event.data['error']}", severity="error")
            self.update_status()
            return
        if event.name == "mcp.tools_changed":
            self.update_status()
            return
        await self.query_one(ChatView).event(event)
        if event.name == "tool.approval_requested":
            request = self.ops.pending_approval
            if request is not None and (request.turn_id, request.tool_call_id) == (
                event.turn_id,
                event.data["tool_call_id"],
            ):
                screen = ApprovalScreen(request)
                self._approval_screen = screen

                def answered(value):
                    if self._approval_screen is screen:
                        self._approval_screen = None
                    if not self._shutting_down:
                        self.ops.resolve_approval(
                            request.turn_id,
                            request.tool_call_id,
                            value or ApprovalDecision.DENY,
                        )

                self.push_screen(screen, answered)
        elif event.name == "turn.finished" and self._approval_screen is not None:
            self._approval_screen.dismiss(ApprovalDecision.DENY)
            self._approval_screen = None
        self.update_status()

    async def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        if self._sending or self.ops.running:
            return
        prompt = self.query_one(PromptInput)
        text = message.text
        prompt.load_text("")
        await self.query_one(ChatView).query(".welcome").remove()
        if text.startswith("/"):
            await self.query_one(ChatView).add("你", text, classes="user command")
            self.run_worker(
                self._command(text),
                group="commands",
                exclusive=True,
                exit_on_error=False,
            )
            return
        self._sending = True
        prompt.disabled = True
        chat = self.query_one(ChatView)
        chat.pending_user = await chat.add("你", text, classes="user")
        self.update_status()
        self.run_worker(self._send(text), group="turn", exit_on_error=False)

    async def _send(self, text: str) -> None:
        try:
            result = await self.ops.send(text, stream=True)
            if result.error and not self._shutting_down:
                await self.query_one(ChatView).add(
                    "运行结果", safe_error(result.error), classes="notice"
                )
        except SessionSaveError as exc:
            if not self._shutting_down:
                await self.query_one(ChatView).add(
                    "保存失败", safe_error(exc), classes="notice"
                )
        except (Exception, asyncio.CancelledError) as exc:
            if not self._shutting_down:
                chat = self.query_one(ChatView)
                if chat.pending_user is not None:
                    chat.pending_user.outcome = "已取消/失败 · 未提交"
                    chat.pending_user.dirty = True
                    chat.pending_user = None
                await chat.add(
                    "运行结果",
                    "已取消，本轮未提交。"
                    if isinstance(exc, asyncio.CancelledError)
                    else safe_error(exc),
                    classes="notice",
                )
        finally:
            self._sending = False
            if not self._shutting_down:
                self.query_one(PromptInput).disabled = False
                self.query_one(PromptInput).focus()
                self.update_status()

    async def _command(self, text: str) -> None:
        await self._perform_command(lambda: terminal_command(self.ops, text))

    async def _perform_command(
        self, action: Callable[[], CommandResult | Awaitable[CommandResult]]
    ) -> None:
        try:
            result = action()
            if not isinstance(result, CommandResult):
                result = await result
            chat = self.query_one(ChatView)
            show_text = True
            if result.action == "clear":
                await chat.clear_display()
            elif result.action == "reload":
                await chat.restore(self.ops.message_snapshot())
            elif result.action == "models":
                await chat.query(ModelPicker).remove()
                await chat.mount(ModelPicker(self.ops))
                chat.scroll_end(animate=False)
                show_text = False
            elif result.action == "thinking":
                choices = [
                    ("服务端默认", "default", False),
                    *[
                        (mode, mode, False)
                        for mode in self.ops.model.thinking_modes or ()
                    ],
                ]
                self.push_screen(
                    ChoiceScreen("Thinking", {"思考选项": choices}),
                    self._thinking_selected,
                )
                show_text = False
            elif result.action == "tools":
                self.push_screen(ToolsScreen(self.ops), self._tools_selected)
                show_text = False
            elif result.action == "sessions":
                items = self.ops.list_sessions()
                if items:
                    choices = [
                        (
                            f"{item.id[:12]} · {item.provider}/{item.model_id}\n"
                            f"{item.workspace} · {item.message_count} 条",
                            item.id,
                            not self.ops.configured(item.provider),
                        )
                        for item in items
                    ]
                    self.push_screen(
                        ChoiceScreen("恢复会话", {"已保存会话": choices}),
                        self._session_selected,
                    )
                    show_text = False
            if result.text and show_text:
                await chat.add("提示", result.text, classes="notice")
            for picker in self.query(ModelPicker):
                picker.refresh_models()
            self.update_status()
        except Exception as exc:
            await self.query_one(ChatView).add(
                "提示", safe_error(exc), classes="notice"
            )

    async def on_model_picker_selected(self, message: ModelPicker.Selected) -> None:
        await self._perform_command(
            lambda: CommandResult(
                text=self.ops.set_model(message.provider, message.model_id)
            )
        )
        self.query_one(PromptInput).focus()

    async def _thinking_selected(self, value: str | None) -> None:
        if value is None:
            return

        def select() -> CommandResult:
            self.ops.set_thinking(None if value == "default" else value)
            return CommandResult(text=f"Thinking：{self.ops.thinking or '服务端默认'}")

        await self._perform_command(select)

    def _tools_selected(self, value) -> None:
        if value is not None:
            try:
                self.ops.set_tools(value, keep_unavailable=True)
                self.update_status()
            except Exception as exc:
                self.notify(safe_error(exc), severity="error")

    async def _session_selected(self, value: str | None) -> None:
        if value is None:
            return

        def select() -> CommandResult:
            self.ops.restore_session(value)
            return CommandResult("reload", f"恢复会话：{self.ops.session.id}")

        await self._perform_command(select)

    async def action_cancel_or_quit(self) -> None:
        if self._sending or self.ops.running:
            await self.ops.cancel()
            if self._approval_screen is not None:
                self._approval_screen.dismiss(ApprovalDecision.DENY)
                self._approval_screen = None
        else:
            await self.action_quit_safely()

    async def action_interrupt(self) -> None:
        if self._sending or self.ops.running:
            await self.action_cancel_or_quit()
        else:
            self.query_one(PromptInput).focus()

    async def action_quit_safely(self) -> None:
        self._shutting_down = True
        await self.ops.close()
        self.exit()

    async def on_unmount(self) -> None:
        self._shutting_down = True
        for unsubscribe in self._subscriptions:
            unsubscribe()
        self._subscriptions.clear()
        await self.ops.close()
