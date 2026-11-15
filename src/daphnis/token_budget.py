"""Estimate Chat Completions input size and reserve room for output."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, floor, sqrt
from typing import Any, Literal

from .compaction_settings import CompactionSettings

_DEFAULT_RESERVE = 16_384
_DEFAULT_COMPACTION = CompactionSettings()
_UNKNOWN_DOCUMENT_RESERVE = 48_000


def mimo_image_tokens(width: int, height: int) -> int:
    """Estimate MiMo's image tokens on its 32 px grid before safety margin."""
    patch = 32
    min_pixels, max_pixels = 8_192, 8_388_608
    pixels = width * height
    cols, rows = max(1, round(width / patch)), max(1, round(height / patch))
    aligned_pixels = cols * rows * patch**2
    if aligned_pixels > max_pixels:
        scale = sqrt(max_pixels / pixels)
        cols = max(1, floor(width * scale / patch))
        rows = max(1, floor(height * scale / patch))
    elif aligned_pixels < min_pixels:
        scale = sqrt(min_pixels / pixels)
        cols, rows = ceil(width * scale / patch), ceil(height * scale / patch)
    # A very narrow image can round to one tile on the short side and still exceed the area cap.
    return min(cols * rows, max_pixels // patch**2)


def image_token_reserve(provider: str, width: int, height: int) -> int:
    if provider == "mimo":
        base = mimo_image_tokens(width, height)
        return base + (base + 9) // 10
    if provider == "deepseek":
        return 1024
    # Custom compatible endpoints lack a verified image tokenizer. Retain the P1 reserve.
    tiles = ((width + 31) // 32) * ((height + 31) // 32)
    return 2 * max(1024, tiles)


def _hard_limit(context_window: int | None) -> int | None:
    return None if context_window is None else context_window * 95 // 100


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
    trigger_tokens: int | None = None
    auto_enabled: bool = True
    unknown_documents: int = 0

    @property
    def hard_limit(self) -> int | None:
        return _hard_limit(self.context_window)

    @property
    def hard_input_limit(self) -> int | None:
        limit = self.hard_limit
        return None if limit is None else max(0, limit - self.output_reserve)

    @property
    def threshold_blocked_by_reserve(self) -> bool:
        return (
            self.auto_enabled and self.trigger_tokens is not None
            and self.trigger_tokens <= self.output_reserve
        )

    @property
    def needs_safety_compaction(self) -> bool:
        return not self.fits

    @property
    def fits(self) -> bool:
        limit = self.hard_limit
        return limit is None or self.input_tokens + self.output_reserve <= limit


def request_budget(
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    context_window: int | None,
    *,
    provider: str | None = None,
    counter: TokenCounter = DEFAULT_COUNTER,
    settings: CompactionSettings | None = None,
    default_output_tokens: int | None = None,
    image_reserve: int = 0,
    unknown_documents: int = 0,
) -> TokenBudget:
    # Only prompt-bearing options are counted as input. Allow for chat framing.
    input_tokens = counter.count(messages) + 4 * len(messages) + 3 + image_reserve
    input_tokens += unknown_documents * _UNKNOWN_DOCUMENT_RESERVE
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
    elif default_output_tokens is not None:
        reserve = default_output_tokens
    elif provider == "deepseek":
        # DeepSeek Chat Completions defaults for off, default thinking, and max effort.
        reserve = {"off": 8192, "max": 131_072}.get(body.get("thinking_mode"), 65_536)
    else:
        reserve = (
            _DEFAULT_RESERVE if context_window is None
            else min(_DEFAULT_RESERVE, max(1, context_window // 8))
        )
    settings = settings or _DEFAULT_COMPACTION
    trigger = settings.trigger_for(context_window)
    hard_limit = _hard_limit(context_window)
    if trigger is not None and hard_limit is not None:
        trigger = min(trigger, hard_limit)
    threshold = None if trigger is None else max(0, trigger - reserve)
    return TokenBudget(
        input_tokens=input_tokens, context_window=context_window, output_reserve=reserve,
        compaction_threshold=threshold,
        needs_compaction=settings.auto_enabled and threshold is not None
        and 0 < threshold <= input_tokens,
        count_source=counter.source, trigger_tokens=trigger, auto_enabled=settings.auto_enabled,
        unknown_documents=unknown_documents,
    )
