"""In-memory execution state for a Session."""

import asyncio
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

from .tools import ApprovalDecision, ApprovalMode, ApprovalRequest
from .usage import UsageTracker

if TYPE_CHECKING:
    from .agent import TurnResult
    from .compaction import CompactionCheckpoint
    from .events import EventBus
    from .session import Session
    from .token_budget import TokenBudget


@dataclass
class SessionRuntime:
    session: "Session"
    events: InitVar["EventBus"]
    task: asyncio.Task["TurnResult"] | asyncio.Task["CompactionCheckpoint | None"] | None = None
    approval: asyncio.Future[ApprovalDecision] | None = None
    pending_approval: ApprovalRequest | None = None
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    usage: UsageTracker = field(init=False)
    last_budget: tuple[str, str, "TokenBudget"] | None = None
    compacting: bool = False
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self, events: "EventBus") -> None:
        self.usage = UsageTracker(events)
        self.usage.select_session(self.session.id, self.session.usage)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()
