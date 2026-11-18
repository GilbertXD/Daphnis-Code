"""Isolated notifications, with no event-name registry or control hooks."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    name: str
    session_id: str
    turn_id: str
    data: dict[str, Any]


Callback = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Sync listeners run inline; async listeners run as isolated tasks.

    emit() dispatches without suspending. It does not wait for async listeners;
    await drain() when their completion matters, including application shutdown.
    Listeners receive snapshots and must not mutate execution via external refs.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[tuple[object, Callback]]] = {}
        self._tasks: set[asyncio.Future[Any]] = set()

    def on(self, name: str, callback: Callback) -> Callable[[], None]:
        if not isinstance(name, str) or not name.strip() or not callable(callback):
            raise ValueError("A non-empty event name and callable are required.")
        entry = (object(), callback)
        self._listeners.setdefault(name, []).append(entry)

        def unsubscribe() -> None:
            listeners = self._listeners.get(name, [])
            if entry in listeners:
                listeners.remove(entry)
            if not listeners:
                self._listeners.pop(name, None)

        return unsubscribe

    def _completed(self, name: str, task: asyncio.Future[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            logger.warning("Async event listener cancelled for %s", name)
        elif (error := task.exception()) is not None:
            logger.error(
                "Async event listener failed for %s",
                name,
                exc_info=(type(error), error, error.__traceback__),
            )

    def has_subscribers(self, name: str) -> bool:
        return bool(self._listeners.get(name))

    async def emit(
        self, event_name: str, *, session_id: str, turn_id: str, **data: Any
    ) -> None:
        event = Event(
            name=event_name, session_id=session_id, turn_id=turn_id, data=data
        )
        for _, callback in tuple(self._listeners.get(event_name, ())):
            try:
                pending = callback(deepcopy(event))
                if isawaitable(pending):
                    task = asyncio.ensure_future(pending)
                    self._tasks.add(task)
                    task.add_done_callback(
                        lambda done, name=event_name: self._completed(name, done)
                    )
            except asyncio.CancelledError:
                # A listener's own exception is not cancellation of the Turn.
                logger.warning(
                    "Sync event listener raised cancellation for %s", event_name
                )
            except Exception:
                logger.exception("Sync event listener failed for %s", event_name)

    async def drain(self, *, timeout: float | None = None) -> None:
        """With a timeout, cancel listeners still running at the deadline."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while self._tasks:
            await asyncio.wait(
                tuple(self._tasks),
                timeout=None if deadline is None else max(0, deadline - loop.time()),
            )
            if deadline is not None and loop.time() >= deadline:
                for task in self._tasks:
                    task.cancel()
                return

    def clear(self) -> None:
        """Remove subscriptions; already dispatched notifications can still drain."""
        self._listeners.clear()
