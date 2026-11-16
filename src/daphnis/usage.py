"""API-reported token accounting, observed independently of message commits."""

from dataclasses import dataclass
from typing import Any

from .events import Event


@dataclass(frozen=True)
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    reported_requests: int = 0
    unrecorded: bool = False

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "requests",
            "reported_requests",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"usage.{name} must be a non-negative integer.")
        if self.reported_requests > self.requests:
            raise ValueError("Reported requests cannot exceed total requests.")
        if type(self.unrecorded) is not bool:
            raise ValueError("usage.unrecorded must be a boolean.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UsageTotals":
        if not isinstance(value, dict):
            raise ValueError("Session usage must be an object.")  # noqa: TRY004 - Malformed history data.
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("Invalid session usage fields.") from exc

    def plus(self, other: "UsageTotals") -> "UsageTotals":
        return UsageTotals(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            requests=self.requests + other.requests,
            reported_requests=self.reported_requests + other.reported_requests,
            unrecorded=self.unrecorded or other.unrecorded,
        )


def _reported_usage(value: Any) -> UsageTotals | None:
    if not isinstance(value, dict):
        return None
    prompt, completion = value.get("prompt_tokens"), value.get("completion_tokens")
    if any(type(count) is not int or count < 0 for count in (prompt, completion)):
        return None
    total = value.get("total_tokens", prompt + completion)
    if type(total) is not int or total < 0:
        return None
    return UsageTotals(prompt, completion, total, requests=1, reported_requests=1)


class UsageTracker:
    """Usage for one Session; each request replaces its latest snapshot."""

    def __init__(self, session_id: str, totals: UsageTotals) -> None:
        self.session_id = session_id
        self.turn_id: str | None = None
        self.turn: UsageTotals | None = None
        self.session = self._base = totals
        self._requests: dict[tuple[bool, int], UsageTotals | None] = {}

    def observe(self, event: Event) -> None:
        if event.session_id != self.session_id:
            return
        if event.name == "turn.started":
            self._base = self.session
            self.turn_id = event.turn_id
            self.turn = UsageTotals()
            self._requests.clear()
            return
        is_title = event.data.get("purpose") == "title"
        maintenance = event.data.get("maintenance") is True
        if maintenance and event.name == "model.started" and event.turn_id != self.turn_id:
            self._base = self.session
            self.turn_id = event.turn_id
            self._requests.clear()
        if not is_title and event.turn_id != self.turn_id:
            return
        key = (is_title, event.data["request_number"])
        self._requests.setdefault(key, None)
        # Null/missing usage is not a new snapshot; an empty object is.
        if event.name == "model.usage" and event.data.get("usage") is not None:
            self._requests[key] = _reported_usage(event.data["usage"])
        totals = title_totals = UsageTotals()
        for (is_title, _), usage in self._requests.items():
            usage = usage or UsageTotals(requests=1)
            if is_title:
                title_totals = title_totals.plus(usage)
            else:
                totals = totals.plus(usage)
        if not maintenance:
            self.turn = totals
        self.session = self._base.plus(totals).plus(title_totals)
