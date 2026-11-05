"""Inline model selection, using the application's actual catalog."""

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from ..app import Application
from .theme import ACCENT, MUTED, SUCCESS, TEXT


def context_label(value: int | None) -> str:
    if value is None:
        return "未知"
    if value % 1_048_576 == 0:
        return f"{value // 1_048_576}M"
    if value % 1_024 == 0:
        return f"{value // 1_024}K"
    return f"{value:,}"


class ModelPicker(Vertical):
    DEFAULT_CSS = """
    ModelPicker { height: auto; margin: 0 0 1 3; }
    .model-current { height: auto; margin-bottom: 1; }
    .model-columns { height: auto; }
    .model-column { width: 1fr; margin-right: 1; }
    .model-provider { height: 2; color: $text-muted; text-style: bold; }
    .model-options { height: 1fr; border: none; padding: 0; background: $background; }
    .model-options:focus { border: none; background-tint: transparent; }
    .model-options > .option-list--option-highlighted {
        background: transparent; color: $foreground; }
    .model-options:focus > .option-list--option-highlighted {
        background: $primary 15%; color: $primary; text-style: bold; }
    .model-help { height: auto; margin-top: 1; color: $text-muted; }
    """

    class Selected(Message):
        def __init__(self, provider: str, model_id: str) -> None:
            super().__init__()
            self.provider, self.model_id = provider, model_id

    def __init__(self, application: Application) -> None:
        super().__init__()
        self.application = application
        self.groups = {
            provider: models
            for provider, models in application.model_groups().items()
            if models
        }
        self.values: dict[str, tuple[str, str]] = {}

    def compose(self) -> ComposeResult:
        yield Static(classes="model-current", markup=False)
        with Horizontal(classes="model-columns") as columns:
            columns.styles.height = max(map(len, self.groups.values()), default=0) + 2
            for index, (provider, models) in enumerate(self.groups.items()):
                with Vertical(classes="model-column"):
                    name = {"deepseek": "DeepSeek", "mimo": "MiMo"}.get(
                        provider, provider
                    )
                    if not self.application.configured(provider):
                        name += " · 未配置"
                    yield Static(
                        name + "  /  上下文", classes="model-provider", markup=False
                    )
                    yield OptionList(id=f"models-{index}", classes="model-options")
        yield Static(
            "↑↓ / Tab 选择 · Enter 确认 · Esc 返回输入\n"
            "使用 /model <序号或模型名> 切换；/model list 查看列表。",
            classes="model-help",
            markup=False,
        )

    def on_mount(self) -> None:
        self.refresh_models()

    def refresh_models(self) -> None:
        app = self.application
        model = app.model
        summary = Text("✓ ", style=SUCCESS)
        summary.append(model.id, style=f"bold {ACCENT}")
        summary.append(
            f"  ({model.provider})  │  上下文 {context_label(model.context_window)}"
            "  │  输出上限 未记录",
            style=MUTED,
        )
        self.query_one(".model-current", Static).update(
            Panel(
                summary,
                border_style=ACCENT,
                padding=(0, 1),
                box=box.ROUNDED,
                style="on #191e32",
            )
        )
        self.values.clear()
        number = 0
        for index, (provider, models) in enumerate(self.groups.items()):
            options = []
            for item in models:
                number += 1
                key = str(number)
                self.values[key] = provider, item.id
                active = (provider, item.id) == (model.provider, model.id)
                row = Table.grid(expand=True, padding=(0, 1))
                row.add_column(width=4, no_wrap=True)
                row.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
                row.add_column(width=4, no_wrap=True)
                row.add_row(
                    Text(
                        f"{'●' if active else '○'} {number}",
                        style=ACCENT if active else MUTED,
                    ),
                    Text(item.id, style=f"bold {ACCENT}" if active else TEXT),
                    Text(context_label(item.context_window), style=MUTED),
                    style="on #1e2340" if active else None,
                )
                options.append(
                    Option(row, id=key, disabled=not app.configured(provider))
                )
            self.query_one(f"#models-{index}", OptionList).clear_options().add_options(
                options
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.post_message(self.Selected(*self.values[event.option.id]))
