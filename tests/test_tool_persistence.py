import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from daphnis import History, Operations, TurnStatus
from daphnis.app import Application
from daphnis.tool_state import PENDING, RUNNING, TOOL_STATE


def write_call(call_id, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "write_file", "arguments": json.dumps(arguments)},
    }


def tool_results(messages):
    return [message for message in messages if message["role"] == "tool"]


class ToolPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.history = History(self.root / "sessions")
        self.calls = [write_call(name, path=f"{name}.txt", content=name) for name in "ab"]
        self.requests = []

    def application(self, *, interactive=False, restore=None):
        def model(request):
            messages = json.loads(request.content)["messages"]
            self.requests.append(messages)
            if len(self.requests) == 1:
                message = {"role": "assistant", "tool_calls": self.calls}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "done"}
                finish = "stop"
            return httpx.Response(200, json={
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            })

        app = Application(
            workspace=self.root, provider="mimo", model_id="mimo-v2.6-flash",
            connections={"mimo": {
                "base_url": "https://model.test", "transport": httpx.MockTransport(model),
            }},
            history=self.history, interactive=interactive,
        )
        ops = Operations(app)
        self.addAsyncCleanup(ops.close)
        if restore is not None:
            ops.restore_session(restore)
        return ops

    def saved(self, ops):
        return self.history.load(ops.session.id).messages.snapshot()

    def approve_writes(self, ops):
        async def approve(event):
            ops.resolve_approval(event.data["request_id"], "allow_once")

        ops.subscribe("tool.approval_requested", approve)

    async def test_group_and_each_transition_are_saved_before_notifications(self):
        self.calls.append(write_call("invalid", content="missing path"))
        ops = self.application(interactive=True)
        self.approve_writes(ops)
        observed = []

        def record(event):
            saved = self.saved(ops)
            observed.append((event.name, event.data, saved, ops.message_snapshot()))

        for name in ("tool.started", "tool.executing", "tool.finished"):
            ops.subscribe(name, record)
        result = await asyncio.wait_for(ops.send("write files", stream=False), 3)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual([
            (name, data["tool_call_id"], [msg.get(TOOL_STATE) for msg in tool_results(saved)])
            for name, data, saved, _ in observed
        ], [
            ("tool.started", "a", [PENDING, PENDING, PENDING]),
            ("tool.executing", "a", [RUNNING, PENDING, PENDING]),
            ("tool.finished", "a", [None, PENDING, PENDING]),
            ("tool.started", "b", [None, PENDING, PENDING]),
            ("tool.executing", "b", [None, RUNNING, PENDING]),
            ("tool.finished", "b", [None, None, PENDING]),
            ("tool.started", "invalid", [None, None, PENDING]),
            ("tool.finished", "invalid", [None, None, None]),
        ])
        for _, _, saved, published in observed:
            self.assertEqual(saved, published)
        self.assertEqual(observed[0][2][1]["tool_calls"], self.calls)
        self.assertEqual([data["status"] for name, data, _, _ in observed if name == "tool.finished"],
                         ["COMPLETED", "COMPLETED", "ERROR"])
        self.assertEqual(self.requests[1][1:], observed[-1][2])
        self.assertTrue(all(TOOL_STATE not in message for message in self.requests[1]))

    async def test_save_failures_stop_all_later_execution_and_notifications(self):
        replace_file = self.history._replace
        for boundary, saved_states, effect, expected_events in (
            (PENDING, [], False, []),
            (RUNNING, [PENDING, PENDING], False, ["tool.started"]),
            ("FINISHED", [RUNNING, PENDING], True, ["tool.started", "tool.executing"]),
        ):
            with self.subTest(boundary=boundary):
                self.requests.clear()
                ops = self.application(interactive=True)
                self.approve_writes(ops)
                observed = []
                for name in ("tool.started", "tool.executing", "tool.finished"):
                    ops.subscribe(name, lambda event, observed=observed: observed.append(event.name))

                def save(path, data, boundary=boundary):
                    results = tool_results(json.loads(data)["messages"])
                    if results and results[0].get(TOOL_STATE, "FINISHED") == boundary:
                        raise OSError("injected save failure")
                    replace_file(path, data)

                with patch.object(self.history, "_replace", side_effect=save):
                    result = await asyncio.wait_for(ops.send("write files", stream=False), 3)
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertIn("injected save failure", result.error)
                self.assertEqual(result.external_side_effects_possible, effect)
                self.assertEqual(observed, expected_events)
                self.assertEqual(len(self.requests), 1)
                saved = self.saved(ops)
                self.assertEqual(ops.message_snapshot(), saved)
                self.assertEqual([msg.get(TOOL_STATE) for msg in tool_results(saved)], saved_states)
                self.assertEqual((self.root / "a.txt").exists(), effect)
                self.assertFalse((self.root / "b.txt").exists())

        # The real write succeeded before its result save failed. Recovery must
        # preserve that uncertainty even though this test knows the file exists.
        resumed = self.application(interactive=True, restore=ops.session.id)
        self.approve_writes(resumed)
        result = await asyncio.wait_for(resumed.send("继续", stream=False), 3)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        first, second = tool_results(self.saved(resumed))
        self.assertEqual(json.loads(first["content"])["error"]["code"], "execution_result_unknown")
        self.assertTrue(json.loads(second["content"])["ok"])

    async def test_reload_requires_explicit_resume_and_rechecks_approval_and_targets(self):
        self.calls.extend(write_call(name, path=f"{name}.txt", content=name) for name in "cd")
        ops = self.application(interactive=True)
        self.approve_writes(ops)

        def interrupt(event):
            if event.data["tool_call_id"] == "a":
                asyncio.current_task().cancel()

        ops.subscribe("tool.executing", interrupt)
        task = asyncio.create_task(ops.send("write files", stream=False))
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        await ops.close()
        interrupted = self.saved(ops)
        self.assertEqual([msg[TOOL_STATE] for msg in tool_results(interrupted)],
                         [RUNNING, PENDING, PENDING, PENDING])
        self.assertFalse((self.root / "a.txt").exists())

        resumed = self.application(interactive=True, restore=ops.session.id)
        resumed.restore_session(ops.session.id)
        self.assertEqual(resumed.message_snapshot(), interrupted)
        self.assertEqual(len(self.requests), 1)
        with self.assertRaisesRegex(ValueError, "继续"):
            await resumed.compact()
        blocked = await resumed.send("new task", stream=False)
        self.assertEqual(blocked.status, TurnStatus.FAILED)
        self.assertIn("继续", blocked.error)
        self.assertEqual(len(self.requests), 1)
        unknown = json.loads(tool_results(self.saved(resumed))[0]["content"])
        self.assertEqual(unknown["source"], "daphnis_recovery")
        self.assertEqual(unknown["error"]["code"], "execution_result_unknown")

        approvals, executed = asyncio.Queue(), []
        resumed.subscribe("tool.approval_requested", approvals.put_nowait)
        resumed.subscribe("tool.executing", lambda event: executed.append(event.data["tool_call_id"]))
        task = asyncio.create_task(resumed.send("继续", stream=False))
        try:
            for name, decision in (("b", "deny"), ("c", "allow_once"), ("d", "allow_once")):
                event = await asyncio.wait_for(approvals.get(), 2)
                self.assertEqual(event.data["tool_call_id"], name)
                if name == "c":
                    (self.root / "c.txt").write_text("changed during approval")
                self.assertTrue(resumed.resolve_approval(event.data["request_id"], decision))
            result = await asyncio.wait_for(task, 2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(executed, ["d"])
        self.assertFalse((self.root / "a.txt").exists())
        self.assertFalse((self.root / "b.txt").exists())
        self.assertEqual((self.root / "c.txt").read_text(), "changed during approval")
        self.assertEqual((self.root / "d.txt").read_text(), "d")
        saved = self.saved(resumed)
        results = [json.loads(msg["content"]) for msg in tool_results(saved)]
        self.assertEqual(results[0], unknown)
        self.assertEqual([value["error"]["code"] for value in results[1:3]],
                         ["permission_denied", "target_changed"])
        self.assertTrue(results[3]["ok"])
        self.assertEqual(self.requests[1][1:], saved[:-1])
        resumed.restore_session(ops.session.id)
        self.assertEqual((await resumed.send("继续", stream=False)).status, TurnStatus.COMPLETED)
        self.assertEqual(executed, ["d"])
        self.assertEqual(tool_results(self.saved(resumed)), tool_results(saved))
