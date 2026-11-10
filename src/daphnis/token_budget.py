"""Estimate Chat Completions input size and reserve room for output."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

_DEFAULT_RESERVE = 16_384


def _estimate_text(text: str) -> int:
    if text.isascii():
        return (len(text) + 2) // 3
    ascii_count = sum(character.isascii() for character in text)
    non_ascii_bytes = len(text.encode("utf-8", errors="surrogatepass")) - ascii_count
    return (ascii_count + 2) // 3 + (non_ascii_bytes + 1) // 2


@dataclass(frozen=True)
class TokenCounter:
    """Count text locally. Chat framing and server accounting remain estimates."""

    count_text: Callable[[str], int]
    source: Literal["estimate", "local_tokenizer"] = "estimate"

    def count(self, value: Any) -> int:
        return self.count_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


DEFAULT_COUNTER = TokenCounter(_estimate_text)


def estimate_tokens(value: Any) -> int:
    return DEFAULT_COUNTER.count(value)


@dataclass(frozen=True)
class TokenBudget:
    input_tokens: int
    context_window: int | None
    output_reserve: int
    compaction_threshold: int | None
    needs_compaction: bool
    count_source: Literal["estimate", "local_tokenizer"] = "estimate"

    @property
    def fits(self) -> bool:
        return (
            self.context_window is None
            or self.input_tokens + self.output_reserve <= self.context_window
        )


def request_budget(
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    context_window: int | None,
    *,
    provider: str | None = None,
    counter: TokenCounter = DEFAULT_COUNTER,
) -> TokenBudget:
    # Only prompt-bearing options are counted as input. Allow for chat framing.
    input_tokens = counter.count(messages) + 4 * len(messages) + 3
    for field in ("tools", "response_format"):
        if field in body:
            input_tokens += counter.count(body[field])
    limits = [
        limit
        for field in ("output_limit", "max_tokens", "max_completion_tokens")
        if type(limit := body.get(field)) is int and limit > 0
    ]
    if limits:
        reserve = max(limits)
    elif provider == "deepseek":
        # DeepSeek Chat Completions defaults for off, default thinking, and max effort.
        reserve = {"off": 8192, "max": 131_072}.get(body.get("thinking_mode"), 65_536)
    else:
        reserve = (
            _DEFAULT_RESERVE if context_window is None
            else min(_DEFAULT_RESERVE, max(1, context_window // 8))
        )
    threshold = (
        None if context_window is None
        else max(0, context_window - reserve - context_window // 20)
    )
    return TokenBudget(
        input_tokens, context_window, reserve, threshold,
        threshold is not None and input_tokens >= threshold, counter.source,
    )
