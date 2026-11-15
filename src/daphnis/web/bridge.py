"""Translate Operations and public notifications into browser transport frames."""

import asyncio
import json
import time
from collections import OrderedDict, deque
from dataclasses import asdict, fields
from uuid import uuid4

from ..history import SessionScan
from ..operations import Operations
from .redaction import Redactor

_EVENT_NAMES = (
    "turn.started", "turn.finished", "model.started", "model.request",
    "model.delta", "model.usage", "model.finished", "model.response_error",
    "tool.started", "tool.approval_requested", "tool.blocked", "tool.executing", "tool.finished",
    "mcp.error", "mcp.tools_changed",
    "context.budget", "context.overflow",
    "compaction.started", "compaction.finished", "compaction.committed", "compaction.discarded",
)
_MAX_EVENT_BYTES = 1024 * 1024
_MAX_JOURNAL_BYTES = 4 * _MAX_EVENT_BYTES
_MAX_SESSIONS = 16
_MAX_LIVE_CHARS = 256_000


def _model_view(model):
    return {
        field.name: getattr(model, field.name)
        for field in fields(model) if field.name != "token_counter"
    }


class Journal:
    def __init__(self):
        self.entries = deque()
        self.bytes = 0
        self.floor = 0

    def add(self, record):
        size = len(json.dumps(record, ensure_ascii=False).encode())
        self.entries.append((size, record))
        self.bytes += size
        while len(self.entries) > 600 or self.bytes > _MAX_JOURNAL_BYTES:
            removed, old = self.entries.popleft()
            self.bytes -= removed
            self.floor = old["id"]

    def after(self, cursor):
        return [record for _, record in self.entries if record["id"] > cursor]


class Bridge:
    def __init__(self, ops: Operations, redactor: Redactor):
        self.ops = ops
        self.redactor = redactor
        self.changed = asyncio.Event()
        self.revision = 0
        self.cursor = 0
        self.journals = OrderedDict()
        self.task = None
        self.run = None
        self.live = {}
        self.lock = asyncio.Lock()
        self._subscriptions = [ops.subscribe(name, self._forward) for name in _EVENT_NAMES]

    def _notify(self, *, state=False):
        if state:
            self.revision += 1
        self.changed.set()
        self.changed = asyncio.Event()

    def _record(self, name, session_id, turn_id, data):
        journal = self.journals.get(session_id)
        if journal is None:
            journal = Journal()
            self.journals[session_id] = journal
        self.journals.move_to_end(session_id)
        if len(self.journals) > _MAX_SESSIONS:
            self.journals.popitem(last=False)
        self.cursor += 1
        data = self.redactor.json(data)
        if len(json.dumps(data, ensure_ascii=False).encode()) > _MAX_EVENT_BYTES:
            fields = ("name", "tool_call_id", "request_number", "status", "http_status")
            data = {key: data[key] for key in fields if key in data}
            data["omitted"] = "此事件的详细内容超过 1 MiB。"
        journal.add({
            "id": self.cursor, "name": name, "turn_id": turn_id,
            "time": time.time(), "data": data,
        })

    def _forward(self, event):
        data = event.data
        if event.name == "model.delta":
            if self.run is None or event.turn_id != self.run["turn_id"]:
                return
            number = data["request_number"]
            item = self.live.setdefault(number, {"content": "", "reasoning": "", "count": 0})
            if not item["count"]:
                self._record(
                    event.name, event.session_id, event.turn_id,
                    {"request_number": number, "coalesced": True},
                )
            item["count"] += 1
            for choice in data["chunk"]["choices"]:
                if choice["index"] != 0:
                    continue
                delta = choice.get("delta", {})
                for key, value in (
                    ("content", delta.get("content") or delta.get("refusal")),
                    ("reasoning", delta.get("reasoning_content")),
                ):
                    if isinstance(value, str) and len(item[key]) < _MAX_LIVE_CHARS:
                        item[key] += value[:_MAX_LIVE_CHARS - len(item[key])]
        else:
            if event.name == "turn.started" and self.run and self.run["session_id"] == event.session_id:
                self.run["turn_id"] = event.turn_id
                data = {**data, "base_count": self.run["base_count"]}
            if event.name == "model.finished" and self.run and event.turn_id == self.run["turn_id"]:
                item = self.live.get(data["request_number"])
                if item is not None:
                    data = {**data, "chunks": item["count"]}
            if event.name == "turn.finished":
                data = {**data, "committed_count": len(self.ops.message_snapshot())}
            self._record(event.name, event.session_id, event.turn_id, data)
        first_request = (
            event.name == "model.started" and data["request_number"] == 1
            and data.get("purpose") != "title"
            and self.run is not None
        )
        self._notify(state=first_request or event.name in {"mcp.error", "mcp.tools_changed"})

    def state(self):
        ops = self.ops
        history_error = None
        try:
            history = ops.scan_sessions()
        except OSError as exc:
            history = SessionScan((), ())
            history_error = f"无法读取历史会话目录（{type(exc).__name__}）。"
        projects_error = None
        try:
            projects = ops.list_projects()
        except OSError as exc:
            projects = ()
            projects_error = f"无法读取项目目录（{type(exc).__name__}）。"
        mcp_error = None
        try:
            servers = ops.managed_mcp()
        except (OSError, TypeError, ValueError):
            servers = ()
            mcp_error = "无法读取已保存的 MCP 配置，请检查 mcp.json。"
        statuses = {status.name: status for status in ops.mcp_status()}
        state = self.redactor.json({
            "session": asdict(ops.session),
            "sessions": [asdict(session) for session in history.sessions],
            "projects": [asdict(project) for project in projects],
            "projects_error": projects_error,
            "history_issues": [asdict(issue) for issue in history.issues],
            "history_error": history_error,
            "messages": ops.message_snapshot(),
            "recovery": asdict(recovery) if (recovery := ops.recovery) else None,
            "model": _model_view(ops.model),
            "models": [
                {**_model_view(model), "configured": ops.configured(provider)}
                for provider, models in ops.list_models().items() for model in models
            ],
            "thinking": ops.thinking,
            "system_prompt": ops.get_system_prompt(),
            "all_tools": ops.all_tools,
            "enabled_tools": ops.enabled_tools,
            "effective_tools": ops.effective_tools,
            "approval_mode": ops.approval_mode,
        })
        state["mcp_error"] = mcp_error
        state["mcp_prompt_servers"] = ops.list_mcp_prompt_servers()
        state["mcp_servers"] = []
        for server in servers:
            status = statuses.get(server.name)
            options = self.redactor.json(server.options)
            if "env" in options:
                # These are variable names, not the resolved process environment.
                options["env"] = server.options["env"]
            state["mcp_servers"].append({
                "name": server.name, "enabled": server.enabled, "options": options,
                "state": status.state if status and server.enabled else "Disconnected",
                "tool_count": status.tool_count if status and server.enabled else 0,
                "error": self.redactor.text(status.error) if status and server.enabled and status.error else None,
            })
        return state

    def frame(self, previous_session="", revision=-1, cursor=0):
        ops = self.ops
        session = ops.session.id
        journal = self.journals.get(session)
        reset = previous_session != session or (journal is not None and cursor < journal.floor)
        approval = ops.pending_approval
        usage = ops.turn_usage
        budget = ops.context_budget
        frame = {
            "session_id": session,
            "revision": self.revision,
            "cursor": self.cursor,
            "reset": reset,
            "events": journal.after(0 if reset else cursor) if journal else [],
            "status": self.redactor.json({
                "running": ops.running or self.run is not None,
                "approval": asdict(approval) if approval else None,
                "session_usage": asdict(ops.session_usage),
                "turn_usage": asdict(usage) if usage else None,
                "context_budget": {
                    **asdict(budget),
                    "hard_limit": budget.hard_limit,
                    "threshold_blocked_by_reserve": budget.threshold_blocked_by_reserve,
                } if budget else None,
                "compacting": ops.compacting,
                "compacted_through": ops.compacted_through,
            }),
            "run": self.redactor.json(self.run),
            "live": {
                number: {
                    "content": self.redactor.text(item["content"], partial=True),
                    "reasoning": self.redactor.text(item["reasoning"], partial=True),
                    "count": item["count"],
                }
                for number, item in self.live.items()
            },
        }
        if reset or revision != self.revision:
            frame["state"] = self.state()
        return frame

    def start(self, prompt, *, operation=None):
        session_id = self.ops.session.id
        self.run = {"id": uuid4().hex, "session_id": session_id, "turn_id": "", "prompt": prompt,
                    "base_count": len(self.ops.message_snapshot())}
        self.live.clear()
        self.task = asyncio.create_task(self._send(prompt, session_id, operation))
        self._notify()

    async def _send(self, prompt, session_id, operation):
        result = {"status": "FAILED", "error": "本轮未完成。"}
        try:
            turn = await (operation() if operation is not None else self.ops.send(prompt, stream=True))
            result = {
                "status": turn.status,
                "error": turn.error,
                "notice": turn.notice,
                "external_side_effects_possible": turn.external_side_effects_possible,
            }
        except asyncio.CancelledError:
            result = {"status": "CANCELLED", "error": "本轮已停止。已执行的工具操作不会自动撤销。"}
        except Exception as exc:  # noqa: BLE001 - Return failures across the detached send boundary.
            result = {"status": "FAILED", "error": str(exc)}
        finally:
            self._record("web.send_finished", session_id, self.run["turn_id"], result)
            self.run = None
            self.live.clear()
            self._notify(state=True)

    async def cancel(self):
        # A send can still be entering Operations (e.g. applying pending MCP updates).
        task = self.task
        while not task.done() and not self.ops.running:
            await asyncio.wait((task,), timeout=0.02)
        if not task.done():
            await self.ops.cancel()
        await task

    async def close(self):
        for unsubscribe in self._subscriptions:
            unsubscribe()
        try:
            await self.ops.close()
        finally:
            if self.task is not None:
                await self.task
