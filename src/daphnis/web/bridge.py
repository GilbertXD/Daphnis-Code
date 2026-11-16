"""Translate Operations and public notifications into browser transport frames."""

import asyncio
import json
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from uuid import uuid4

from ..attachments import (
    ATTACHMENT_TOOLS,
    attachment_only_prompt,
)
from ..history import SessionScan
from ..operations import Operations
from ..prompt_groups import PromptInput
from .redaction import Redactor

_EVENT_NAMES = (
    "turn.started", "turn.finished", "attachments.committed", "prompt.committed", "model.started", "model.request",
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


def _browser_messages(messages):
    private = set()
    for index, message in enumerate(messages):
        if message["role"] != "tool":
            private = {call["id"] for call in message.get("tool_calls") or ()
                       if call.get("function", {}).get("name") in ATTACHMENT_TOOLS}
        elif message.get("tool_call_id") in private and "content" in message:
            messages[index] = {**message, "content": "[附件工具输出已隐藏]"}
    return messages


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


@dataclass
class WebRun:
    data: dict
    task: asyncio.Task[dict] = field(init=False, repr=False)
    live: dict = field(default_factory=dict)
    accepted: asyncio.Future | None = None
    upload: tuple | None = None
    prompt_input: PromptInput | None = None


class Bridge:
    def __init__(self, ops: Operations, redactor: Redactor):
        self.ops = ops
        self.default_session_id = ops.initial_session_id
        self.redactor = redactor
        self.changed = asyncio.Event()
        self.revision = 0
        self.shared_revision = 0
        self.updates: dict[str, int] = {}
        self.state_revisions: dict[str, int] = {}
        self.cursor = 0
        self.journals = OrderedDict()
        self.runs: dict[str, WebRun] = {}
        self.prompt_previews = OrderedDict()
        self.resource_drafts = OrderedDict()
        self._subscriptions = [ops.subscribe(name, self._forward) for name in _EVENT_NAMES]

    def notify(self, session_id=None, *, state=False, shared=False):
        self.revision += 1
        if session_id is not None:
            self.updates[session_id] = self.revision
            if state:
                self.state_revisions[session_id] = self.revision
        if shared:
            self.shared_revision = self.revision
        self.changed.set()
        self.changed = asyncio.Event()

    def version(self, session_id):
        return max(self.updates.get(session_id, 0), self.shared_revision)

    def busy(self, session_id):
        return session_id in self.runs or self.ops.busy(session_id)

    def forget(self, session_id):
        self.journals.pop(session_id, None)
        self.updates.pop(session_id, None)
        self.state_revisions.pop(session_id, None)

    def _record(self, name, session_id, turn_id, data):
        journal = self.journals.get(session_id)
        if journal is None:
            journal = Journal()
            self.journals[session_id] = journal
        self.journals.move_to_end(session_id)
        if len(self.journals) > _MAX_SESSIONS:
            for old in self.journals:
                if old != session_id and old not in self.runs:
                    del self.journals[old]
                    break
        self.cursor += 1
        data = self.redactor.json(data)
        if len(json.dumps(data, ensure_ascii=False).encode()) > _MAX_EVENT_BYTES:
            fields = ("name", "tool_call_id", "request_number", "purpose", "status", "http_status")
            data = {key: data[key] for key in fields if key in data}
            data["omitted"] = "此事件的详细内容超过 1 MiB。"
        journal.add({
            "id": self.cursor, "name": name, "turn_id": turn_id,
            "time": time.time(), "data": data,
        })

    def _forward(self, event):
        data, session = event.data, event.session_id
        if not session:
            self.notify(shared=True)
            return
        run = self.runs.get(session)
        if event.name in {"attachments.committed", "prompt.committed"}:
            if (
                run and run.accepted and not run.accepted.done()
                and (
                    run.upload is not None and run.upload[0] == data.get("submission_id")
                    or run.prompt_input is not None and run.prompt_input.id == data.get("group_id")
                )
            ):
                run.accepted.set_result(None)
            self.notify(session, state=True, shared=True)
            return
        if event.name == "model.finished" and data.get("attachment_sensitive"):
            data = {key: value for key, value in data.items() if key != "response"}
        if event.name == "model.delta":
            if run is None or event.turn_id != run.data["turn_id"]:
                return
            number = data["request_number"]
            item = run.live.setdefault(number, {"content": "", "reasoning": "", "count": 0})
            if not item["count"]:
                self._record(
                    event.name, session, event.turn_id,
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
            if event.name == "turn.started" and run and run.data.get("kind") != "compaction":
                run.data["turn_id"] = event.turn_id
                data = {**data, "base_count": run.data["base_count"]}
            if (
                event.name == "compaction.started" and data.get("trigger") == "manual"
                and run and run.data.get("kind") == "compaction"
            ):
                run.data["turn_id"] = event.turn_id
            if event.name == "model.finished" and run and event.turn_id == run.data["turn_id"]:
                item = run.live.get(data["request_number"])
                if item is not None:
                    data = {**data, "chunks": item["count"]}
            if event.name == "turn.finished":
                data = {**data, "committed_count": len(self.ops.message_snapshot(session))}
            self._record(event.name, session, event.turn_id, data)
        committed_input = (
            event.name == "model.started" and data["request_number"] == 1
            and data.get("purpose") != "title"
        )
        self.notify(session, state=committed_input, shared=committed_input)

    def _shared_state(self, session):
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
        sessions = [asdict(item) for item in history.sessions]
        saved = {item.id for item in history.sessions}
        sessions.extend(asdict(ops.session(item)) for item in self.runs if item not in saved)
        state = self.redactor.json({
            "session": asdict(ops.session(session)),
            "sessions": sessions,
            "projects": [asdict(project) for project in projects],
            "projects_error": projects_error,
            "history_issues": [asdict(issue) for issue in history.issues],
            "history_error": history_error,
            "all_tools": ops.all_tools,
            "effective_tools": ops.effective_tools(session),
        })
        state["mcp_error"] = mcp_error
        state["mcp_prompt_servers"] = ops.list_mcp_prompt_servers()
        state["mcp_resource_servers"] = ops.list_mcp_resource_servers()
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

    def state(self, session):
        ops = self.ops
        return {
            **self._shared_state(session),
            **self.redactor.json({
                "messages": _browser_messages(ops.message_snapshot(session)),
                "attachment_refs": ops.attachment_snapshot(session),
                "prompt_groups": [asdict(group) for group in ops.prompt_groups(session)],
                "recovery": asdict(recovery) if (recovery := ops.recovery(session)) else None,
                "model": _model_view(ops.model(session)),
                "models": [
                    {**_model_view(model), "configured": ops.configured(provider)}
                    for provider, models in ops.list_models(session).items() for model in models
                ],
                "thinking": ops.thinking(session),
                "system_prompt": ops.get_system_prompt(session),
                "enabled_tools": ops.enabled_tools(session),
                "approval_mode": ops.approval_mode(session),
            }),
        }

    def frame(self, session, previous_session="", revision=-1, cursor=0):
        ops = self.ops
        run = self.runs.get(session)
        journal = self.journals.get(session)
        reset = previous_session != session or (journal is not None and cursor < journal.floor)
        approval = ops.pending_approval(session)
        usage = ops.turn_usage(session)
        budget = ops.context_budget(session)
        frame = {
            "session_id": session,
            "revision": self.version(session),
            "cursor": self.cursor,
            "reset": reset,
            "events": journal.after(0 if reset else cursor) if journal else [],
            "status": self.redactor.json({
                "running": ops.running(session) or run is not None,
                "approval": asdict(approval) if approval else None,
                "session_usage": asdict(ops.session_usage(session)),
                "turn_usage": asdict(usage) if usage else None,
                "context_budget": {
                    **asdict(budget),
                    "hard_limit": budget.hard_limit,
                    "threshold_blocked_by_reserve": budget.threshold_blocked_by_reserve,
                } if budget else None,
                "compacting": ops.compacting(session) or (run is not None and run.data.get("kind") == "compaction"),
                "compacted_through": ops.compacted_through(session),
            }),
            "run": self.redactor.json(run.data) if run else None,
            "live": {
                number: {
                    "content": self.redactor.text(item["content"], partial=True),
                    "reasoning": self.redactor.text(item["reasoning"], partial=True),
                    "count": item["count"],
                }
                for number, item in (run.live.items() if run else ())
            },
        }
        if reset or self.state_revisions.get(session, 0) > revision:
            frame["state"] = self.state(session)
        elif self.shared_revision > revision:
            frame["shared"] = self._shared_state(session)
        return frame

    def cache_resource_draft(self, session, resource):
        draft_id = uuid4().hex
        self.resource_drafts[draft_id] = (session, replace(resource, draft_id=draft_id))
        if len(self.resource_drafts) > 64:
            self.resource_drafts.popitem(last=False)
        return draft_id

    def resource_draft(self, session, draft_id):
        draft = self.resource_drafts.get(draft_id)
        if draft is None or draft[0] != session:
            raise ValueError("MCP 资源草稿已失效，请重新读取并添加。")
        return draft[1]

    def cache_prompt_preview(self, session, server, name, messages):
        preview_id = uuid4().hex
        self.prompt_previews[preview_id] = (session, server, name, messages)
        if len(self.prompt_previews) > 16:
            self.prompt_previews.popitem(last=False)
        return preview_id

    def start(self, session_id, prompt, *, operation=None, attachments=(), submission_id=None,
              prompt_input: PromptInput | None = None):
        run = WebRun({
            "id": uuid4().hex, "session_id": session_id, "turn_id": "", "prompt": prompt,
            "base_count": len(self.ops.message_snapshot(session_id)),
        })
        if attachments:
            run.data["prompt"] = prompt if prompt.strip() else attachment_only_prompt(attachments)
            run.data["attachments"] = [item.name for item in attachments]
            run.upload = (submission_id, prompt, attachments)
        elif prompt_input is not None:
            run.prompt_input = prompt_input
            operation = partial(self.ops.run_mcp_prompt, prompt_input, session_id=session_id)
        if run.upload is not None or run.prompt_input is not None:
            run.accepted = asyncio.get_running_loop().create_future()
        if operation is None:
            operation = partial(
                self.ops.send, prompt, session_id=session_id, stream=True,
                attachments=attachments, submission_id=submission_id,
            )
        self._start_run(session_id, run, self._send(operation))
        return run.accepted

    def _start_run(self, session_id, run, operation):
        self.runs[session_id] = run
        run.task = asyncio.create_task(operation)
        run.task.add_done_callback(partial(self._finished, session_id, run))
        self.notify(session_id, shared=True)

    async def _send(self, operation):
        try:
            turn = await operation()
            return {
                "status": turn.status,
                "error": turn.error,
                "notice": turn.notice,
                "external_side_effects_possible": turn.external_side_effects_possible,
            }
        except Exception as exc:  # noqa: BLE001 - Return failures across the detached send boundary.
            return {"status": "FAILED", "error": str(exc)}

    def start_compaction(self, session_id):
        run = WebRun({"id": uuid4().hex, "session_id": session_id, "turn_id": "", "kind": "compaction"})
        self._start_run(session_id, run, self._compact(session_id))

    async def _compact(self, session_id):
        try:
            checkpoint = await self.ops.compact(session_id)
            return {"status": "skipped"} if checkpoint is None else {"status": "committed", "through": checkpoint.through}
        except Exception as exc:  # noqa: BLE001 - Report failures across the detached operation boundary.
            return {"status": "failed", "error": type(exc).__name__}

    def _finished(self, session_id, run, task):
        turn_id = run.data["turn_id"]
        if run.data.get("kind") == "compaction":
            # A started compaction already emits the outcome of its transaction.
            if not turn_id:
                result = {"status": "cancelled"} if task.cancelled() else task.result()
                self._record("compaction.finished", session_id, run.data["id"], {"trigger": "manual", **result})
        else:
            result = (
                {"status": "CANCELLED", "error": "本轮已停止。已执行的工具操作不会自动撤销。"}
                if task.cancelled() else task.result()
            )
            if run.accepted is not None and not run.accepted.done():
                run.accepted.set_result(self.redactor.text(result["error"] or "任务未能保存。"))
            self._record("web.send_finished", session_id, turn_id, result)
        del self.runs[session_id]
        self.notify(session_id, state=True, shared=True)

    async def cancel(self, session_id, run_id):
        run = self.runs.get(session_id)
        if run is None or run.data["id"] != run_id:
            return False
        if not run.task.done() and not run.task.cancelling():
            run.task.cancel()
        await asyncio.wait((run.task,))
        return True

    async def close(self):
        for unsubscribe in self._subscriptions:
            unsubscribe()
        try:
            await self.ops.close()
        finally:
            await asyncio.gather(*(run.task for run in self.runs.values()), return_exceptions=True)
