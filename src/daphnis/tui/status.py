"""A responsive footer with model settings and both token scopes."""

from rich.text import Text
from textual.events import Resize
from textual.widgets import Static

from ..terminal_commands import sanitize_terminal
from ..token_budget import TokenBudget
from ..usage import UsageTotals
from .theme import ACCENT, BLUE, MUTED, TEXT, WARNING


def compact_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


class UsageView(Static):
    DEFAULT_CSS = """
    UsageView { height: 2; color: $text-muted; margin-top: 1; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._turn: UsageTotals | None = None
        self._session = UsageTotals()
        self._model_id = ""
        self._thinking: str | None = None
        self._tools = 0
        self._width = 0
        self._context_line = Text()

    @staticmethod
    def _line(label: str, totals: UsageTotals | None, *, exact: bool = False) -> Text:
        line = Text(label + " ", style=MUTED)
        if totals is None:
            line.append("—")
        elif not totals.reported_requests and (totals.requests or totals.unrecorded):
            line.append("未返回完整用量" if totals.requests else "历史未记录")
        else:
            number = (lambda value: f"{value:,}") if exact else compact_tokens
            line.append(
                f"in {number(totals.prompt_tokens)} "
                f"out {number(totals.completion_tokens)} "
                f"total {number(totals.total_tokens)}",
                style=TEXT,
            )
            if totals.unrecorded or totals.reported_requests < totals.requests:
                line.append(" *", style=WARNING)
        return line

    def update_usage(
        self,
        turn: UsageTotals | None,
        session: UsageTotals,
        *,
        model_id: str = "",
        thinking: str | None = None,
        tools: int = 0,
        context_budget: TokenBudget | None = None,
        context_window: int | None = None,
        compacted_through: int = 0,
    ) -> None:
        self._turn, self._session = turn, session
        self._model_id, self._thinking, self._tools = model_id, thinking, tools
        estimated = compact_tokens(context_budget.input_tokens) if context_budget else "—"
        window = compact_tokens(context_window) if context_window is not None else "未知"
        self._context_line = Text(f"上下文 ≈{estimated}/{window}", style=MUTED)
        if context_budget and context_budget.needs_compaction:
            self._context_line.stylize(WARNING)
        if compacted_through:
            self._context_line.append(f" · 摘要至 #{compacted_through}")
        budget_detail = ""
        if context_budget:
            trigger = context_budget.trigger_tokens
            threshold = f"{trigger:,} Token" if trigger is not None else "未知"
            if not context_budget.auto_enabled:
                threshold = "已关闭"
            elif context_budget.threshold_blocked_by_reserve:
                threshold = "不高于输出预留，未生效"
            source = "本地分词" if context_budget.count_source == "local_tokenizer" else "长度估算"
            hard_limit = context_budget.hard_limit
            safety = f"{hard_limit:,} Token" if hard_limit is not None else "未知"
            budget_detail = (
                f"\n计数来源：{source} · 输出预留 {context_budget.output_reserve:,} Token"
                f"\n自动压缩阈值（输入 + 预留）：{threshold}"
                f"\n安全上限（输入 + 预留）：{safety}"
            )
        self.tooltip = (
            str(self._line("本轮", turn, exact=True))
            + "\n"
            + str(self._line("会话", session, exact=True))
            + "\n"
            + self._context_line.plain
            + "（最近主请求的输入估算）"
            + budget_detail
            + "\nin / out = 输入 / 输出 Token；* 表示仅含已报告部分。"
        )
        self._redraw()

    def on_resize(self, event: Resize) -> None:
        if event.size.width != self._width:
            self._width = event.size.width
            self._redraw()

    def _redraw(self) -> None:
        separator = Text("  │  ", style=MUTED)
        meta = Text(sanitize_terminal(self._model_id), style=BLUE)
        meta.append(separator)
        meta.append("思考 ", style=MUTED)
        meta.append(sanitize_terminal(self._thinking or "default"), style=ACCENT)
        meta.append(separator)
        meta.append(f"{self._tools} tools", style=MUTED)
        hint = Text("Enter 发送 · Esc 中断", style=MUTED)
        counts = Text.assemble(
            self._line("本轮", self._turn),
            separator,
            self._line("会话", self._session),
            separator,
            self._context_line,
        )
        full = Text.assemble(meta, separator, counts, separator, hint)
        width = self.size.width or 80
        if full.cell_len <= width:
            lines = [full]
        else:
            top = Text.assemble(meta, "  ·  ", hint)
            if counts.cell_len <= width:
                lines = [top, counts]
            else:
                lines = [
                    top,
                    self._line("本轮", self._turn),
                    self._line("会话", self._session),
                    self._context_line,
                ]
        self.styles.height = len(lines)
        output = Text("\n").join(lines)
        output.no_wrap = True
        output.overflow = "ellipsis"
        self.update(output)
