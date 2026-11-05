"""Multiline prompt editing and local-command submission."""

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class PromptInput(TextArea):
    BINDINGS = [
        Binding("enter", "submit", show=False, priority=True),
        Binding("shift+enter,alt+enter,f2", "newline", show=False, priority=True),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def action_submit(self) -> None:
        if not self.disabled and self.text.strip():
            self.post_message(self.Submitted(self.text.strip()))

    def action_newline(self) -> None:
        if not self.disabled:
            self.insert("\n")
