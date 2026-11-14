"""User preferences for proactive context compaction."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class CompactionSettings:
    auto_enabled: bool = True
    threshold_mode: Literal["percent", "tokens"] = "percent"
    threshold_value: int | float = 0.95
    recent_tokens: int = 4096

    def __post_init__(self) -> None:
        if type(self.auto_enabled) is not bool:
            raise ValueError("auto_enabled must be a boolean.")
        if type(self.recent_tokens) is not int or self.recent_tokens < 0:
            raise ValueError("recent_tokens must be a non-negative integer.")
        if self.threshold_mode == "percent":
            if type(self.threshold_value) not in (int, float) or not 0 < self.threshold_value <= 1:
                raise ValueError("A percent compaction threshold must be greater than 0 and at most 1.")
        elif self.threshold_mode == "tokens":
            if type(self.threshold_value) is not int or self.threshold_value <= 0:
                raise ValueError("A token compaction threshold must be a positive integer.")
        else:
            raise ValueError("threshold_mode must be 'percent' or 'tokens'.")

    def trigger_for(self, context_window: int | None) -> int | None:
        """Return the requested threshold in total tokens, before the safety cap."""
        if self.threshold_mode == "tokens":
            return int(self.threshold_value)
        if context_window is None:
            return None
        return int(Decimal(str(self.threshold_value)) * context_window)
