"""Explicit approval display; the Application resolves the actual Future."""

import json
from typing import ClassVar

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from ..tools import ApprovalDecision, ApprovalRequest
from .theme import BACKGROUND


class ApprovalScreen(ModalScreen[ApprovalDecision]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("y", "approve", "允许"),
        Binding("t", "toggle_trust", "切换会话信任"),
        Binding("n,escape", "deny", "拒绝"),
    ]
    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; background: $background 75%; }
    #approval-box { width: 90%; max-width: 96; height: 85%; max-height: 32;
        border: round $warning 65%; background: $surface; padding: 1 2; }
    #approval-title { height: 1; margin-bottom: 1;
        color: $warning; text-style: bold; }
    #approval-reason { height: auto; max-height: 3; margin-bottom: 1; }
    #approval-body { height: 1fr; margin-bottom: 1; }
    #approval-body Static { height: auto; margin-bottom: 1; }
    #approval-details { padding: 0 1; color: $text-muted; }
    #approval-patch { background: $background; padding: 0 1; }
    #approval-trust { height: auto; }
    #approval-trust-note { height: auto; color: $text-muted; }
    #approval-hint { height: 1; color: $text-muted; text-overflow: ellipsis; }
    #approval-buttons { height: 3; align: right middle; }
    #approval-buttons Button { margin-left: 1; min-width: 10; }
    """

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        details = dict(self.request.details)
        server = details.pop("server", None)
        tool = self.request.tool_name
        if server is not None:
            tool = f"{server} / {details.pop('tool')!r}"
        with Vertical(id="approval-box"):
            yield Static(
                f"工具执行确认  /  {tool}",
                markup=False,
                id="approval-title",
            )
            yield Static(self.request.reason, markup=False, id="approval-reason")
            patch = details.pop("patch", None)
            lines = []
            if server is not None:
                arguments = details.pop("arguments")
                lines.append(
                    "参数：\n" + json.dumps(arguments, ensure_ascii=False, indent=2)
                )
                if details.get("targets") == []:
                    del details["targets"]
            if isinstance(details.get("changes"), list):
                actions = {"create": "新建", "modify": "修改", "delete": "删除"}
                for change in details.pop("changes"):
                    action = change["action"]
                    lines.append(f"{actions.get(action, action)}  {change['path']}")
            if details:
                lines.append(json.dumps(details, ensure_ascii=False))
            with VerticalScroll(id="approval-body"):
                yield Static(
                    "\n".join(lines),
                    markup=False,
                    id="approval-details",
                )
                if isinstance(patch, str):
                    yield Static(
                        Syntax(
                            patch,
                            "diff",
                            theme="monokai",
                            word_wrap=True,
                            background_color=BACKGROUND,
                        ),
                        id="approval-patch",
                    )
            if self.request.can_trust:
                yield Checkbox("本会话不再询问此工具（T）", id="approval-trust")
                if server is not None:
                    yield Static(
                        "授权适用于此工具，后续调用可能使用不同的参数。",
                        markup=False,
                        id="approval-trust-note",
                    )
            yield Static(
                "Y 允许   N / Esc 拒绝   Ctrl+C 取消本轮",
                markup=False,
                id="approval-hint",
            )
            with Horizontal(id="approval-buttons"):
                yield Button("拒绝 N", id="approval-no", variant="default")
                yield Button("允许 Y", id="approval-yes", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#approval-no", Button).focus()

    def action_approve(self) -> None:
        trust = (
            self.request.can_trust and self.query_one("#approval-trust", Checkbox).value
        )
        self.dismiss(
            ApprovalDecision.TRUST_SESSION if trust else ApprovalDecision.ALLOW_ONCE
        )

    def action_toggle_trust(self) -> None:
        if self.request.can_trust:
            self.query_one("#approval-trust", Checkbox).toggle()

    def action_deny(self) -> None:
        self.dismiss(ApprovalDecision.DENY)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-yes":
            self.action_approve()
        else:
            self.action_deny()
