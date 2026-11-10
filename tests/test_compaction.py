import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from saturnus import Messages
from test_agent_core import reply, tool_call

from daphnis import Context, History, Operations, TurnStatus
from daphnis.app import Application
from daphnis.compaction import CompactionCheckpoint, CompactionError, safe_boundaries
from daphnis.terminal_commands import terminal_command

SUMMARY = json.dumps({
    "goals": "Update a file", "progress": "Inspected the workspace",
    "files": "app.py", "next_steps": "Run checks",
})


class ManualCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_compaction_is_atomic_and_restores_the_effective_context(self):
        bodies = []
        summary_content = SUMMARY

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            if "Summarize" in body["messages"][0]["content"]:
                self.assertNotIn("tools", body)
                response = reply(summary_content)
                response["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                response = reply("done")
            return httpx.Response(200, json=response)

        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory) / "history")
            app = Application(
                workspace=directory, provider="compatible", model_id="model-a",
                connections={"compatible": {
                    "base_url": "https://model.test/v1", "transport": httpx.MockTransport(handler),
                }},
                history=history,
            )
            self.addAsyncCleanup(app.close)
            ops = Operations(app)
            state = app._session
            state.messages.add_user("Original task " * 150)
            state.messages.add(reply(None, finish="tool_calls", calls=[tool_call()])["choices"][0]["message"])
            state.messages.add_tool_result(tool_call_id="call-1", content="Original result " * 100)
            state.messages.add_assistant("Original answer")
            state.messages.add_user("Most recent task")
            state.messages.add_assistant("Most recent answer")
            history.save(state)
            self.assertIsNone(history.load(state.id).compaction)
            original = state.messages.snapshot()

            self.assertIn("已压缩", (await terminal_command(ops, "/compact")).text)
            saved = history.load(state.id)
            self.assertEqual(saved.compaction.through, 4)
            self.assertEqual(saved.messages.snapshot(), original)
            self.assertEqual(saved.usage.total_tokens, 15)
            self.assertEqual(saved.usage.requests, 1)
            self.assertIsNone(ops.turn_usage)
            self.assertIsNone(await ops.compact())
            ops.restore_session(state.id)
            outcome = await ops.send("Continue", stream=False)
            self.assertEqual(outcome.status, TurnStatus.COMPLETED)
            effective = bodies[-1]["messages"]
            self.assertEqual([message["role"] for message in effective],
                             ["system", "assistant", "user", "assistant", "user"])
            self.assertIn("untrusted historical data", effective[0]["content"])
            self.assertIn(SUMMARY.replace(", ", ",").replace(": ", ":"), effective[1]["content"])
            self.assertEqual(effective[2:4], original[4:])
            self.assertEqual(ops.message_snapshot()[:6], original)

            app._session.messages.add_user("Another task " * 100)
            app._session.messages.add_assistant("Another answer")
            history.save(app._session)
            path = history.directory / f"{state.id}.json"
            summary_content = "Invalid summary"
            with self.assertRaises(CompactionError):
                await ops.compact()
            self.assertEqual(app._session.compaction, saved.compaction)
            self.assertEqual(history.load(state.id).compaction, saved.compaction)

            disk_before = path.read_bytes()
            summary_content = SUMMARY
            with (
                patch.object(history, "_replace", side_effect=OSError("disk unavailable")),
                self.assertRaisesRegex(OSError, "disk unavailable"),
            ):
                await ops.compact()
            self.assertEqual(path.read_bytes(), disk_before)
            self.assertEqual(app._session.compaction, saved.compaction)
            updated = await ops.compact()
            self.assertGreater(updated.through, saved.compaction.through)
            self.assertEqual(history.load(state.id).compaction, updated)
            self.assertIn("Prior summary: {", bodies[-1]["messages"][1]["content"])

    def test_parallel_tool_calls_cannot_be_split_and_raw_user_is_retained(self):
        raw = Messages()
        raw.add_user("active user request")
        raw.add(reply(None, finish="tool_calls", calls=[tool_call("a"), tool_call("b")])["choices"][0]["message"])
        raw.add_tool_result(tool_call_id="b", content="second")
        self.assertEqual(list(safe_boundaries(raw.snapshot())), [1])
        raw.add_tool_result(tool_call_id="a", content="first")
        raw.add_assistant("newest message")
        self.assertEqual(list(safe_boundaries(raw.snapshot())), [1, 4, 5])
        with self.assertRaisesRegex(ValueError, "incomplete exchange"):
            CompactionCheckpoint(3, SUMMARY).validate(raw.snapshot())
        checkpoint = CompactionCheckpoint(4, SUMMARY)
        checkpoint.validate(raw.snapshot())
        effective = Context().build(raw, checkpoint)
        self.assertEqual(effective[-2:], [raw.get(0), raw.get(4)])
