import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from saturnus import Messages
from test_agent_core import (
    CATALOG,
    make_client,
    registry,
    reply,
    session,
    stream_reply,
    tool_call,
)

from daphnis import Agent, Context, History, Operations, TurnStatus
from daphnis.app import Application, SessionSaveError
from daphnis.compaction import CompactionCheckpoint, CompactionError, safe_boundaries
from daphnis.models import ModelInfo
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


SMALL_MODEL = ModelInfo("model-a", "compatible", context_window=1200, supports_tools=True)


class AutomaticCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_tool_loop_trims_then_summarizes_and_commits_raw_history(self):
        state = session()
        tool_output = "tool-data " * 150
        calls, requests, events = [], [], []

        def execute(text):
            calls.append(text)
            return tool_output

        def handler(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if "Summarize" in messages[0]["content"]:
                self.assertNotIn("tools", body)
                return httpx.Response(200, json=reply(SUMMARY))
            requests.append(messages)
            if len(requests) <= 7:
                return httpx.Response(200, json=reply(
                    None, finish="tool_calls", calls=[tool_call(str(len(requests)))],
                ))
            return httpx.Response(200, json=reply("finished"))

        async with make_client(handler) as client:
            agent = Agent(
                client=client, provider="compatible", catalog=(SMALL_MODEL,),
                tools=registry(execute),
            )
            for event in ("compaction.finished", "compaction.committed", "context.budget"):
                agent.events.on(event, events.append)
            result = await agent.run(session=state, prompt="active user request")
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(len(calls), 7)
        raw = state.messages.snapshot()
        self.assertEqual([item["content"] for item in raw if item["role"] == "tool"], [tool_output] * 7)
        self.assertIsNotNone(state.compaction)
        state.compaction.validate(raw)
        self.assertTrue(any(event.data.get("status") == "trimmed" for event in events))
        self.assertTrue(any(event.data.get("status") == "staged" for event in events))
        self.assertTrue(any(event.name == "compaction.committed" for event in events))
        for sent in requests:
            self.assertIn(raw[0], sent)
            self.assertIn(len(sent), safe_boundaries(sent))
            self.assertNotIn("Tool output excerpt", sent[-1].get("content", "") or "")
        budgets = [event.data for event in events if event.name == "context.budget"]
        self.assertFalse(any(budget["needs_compaction"] for budget in budgets))

    async def test_cancel_or_failure_after_staging_keeps_committed_checkpoint(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                state = session()
                state.messages.add_user("old request")
                state.messages.add_assistant("old answer")
                state.messages.add_user("older raw data " * 350)
                state.messages.add_assistant("older response")
                prior = CompactionCheckpoint(2, SUMMARY)
                state.compaction = prior
                original = state.messages.snapshot()
                entered, discarded = asyncio.Event(), []

                async def handler(request, entered=entered, cancel=cancel):
                    body = json.loads(request.content)
                    if "Summarize" in body["messages"][0]["content"]:
                        return httpx.Response(200, json=reply(SUMMARY))
                    entered.set()
                    if cancel:
                        await asyncio.Future()
                    return httpx.Response(500, json={"error": "server failure"})

                async with make_client(handler) as client:
                    agent = Agent(client=client, provider="compatible", catalog=(SMALL_MODEL,))
                    agent.events.on("compaction.discarded", discarded.append)
                    task = asyncio.create_task(agent.run(session=state, prompt="continue"))
                    await asyncio.wait_for(entered.wait(), 1)
                    self.assertEqual(state.compaction, prior)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        self.assertEqual((await task).status, TurnStatus.FAILED)
                self.assertEqual(state.messages.snapshot(), original)
                self.assertEqual(state.compaction, prior)
                self.assertEqual(discarded[0].data["through"], 4)

    async def test_disk_failure_keeps_each_history_and_checkpoint_pair_consistent(self):
        def handler(request):
            body = json.loads(request.content)
            summary = "Summarize" in body["messages"][0]["content"]
            return httpx.Response(200, json=reply(SUMMARY if summary else "done"))

        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory) / "history")
            app = Application(
                workspace=directory, provider="compatible", model_id="model-a",
                connections={"compatible": {
                    "base_url": "https://model.test/v1", "transport": httpx.MockTransport(handler),
                }}, history=history, catalog=(SMALL_MODEL,),
            )
            self.addAsyncCleanup(app.close)
            ops = Operations(app)
            await ops.set_tools(())
            app._session.messages.add_user("original " * 600)
            app._session.messages.add_assistant("answer")
            history.save(app._session)
            before = history.load(ops.session.id)
            with (
                patch.object(history, "_replace", side_effect=OSError("disk unavailable")),
                self.assertRaises(SessionSaveError),
            ):
                await ops.send("continue", stream=False)
            disk = history.load(ops.session.id)
            self.assertEqual(disk.messages.snapshot(), before.messages.snapshot())
            self.assertIsNone(disk.compaction)
            self.assertEqual(len(ops.message_snapshot()), 4)
            app._session.compaction.validate(ops.message_snapshot())
            self.assertEqual((await ops.send("continue again", stream=False)).status, TurnStatus.COMPLETED)
            restored = history.load(ops.session.id)
            self.assertEqual(restored.messages.snapshot(), ops.message_snapshot())
            self.assertEqual(restored.compaction, app._session.compaction)


class OverflowRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_window_recovers_http_overflow_without_reexecuting_tools(self):
        state = session()
        requested, executed, recoveries = [], [], []

        def execute(text):
            executed.append(text)
            return "large result " * 160

        def handler(request):
            body = json.loads(request.content)
            requested.append(body["messages"])
            if len(requested) == 1:
                return httpx.Response(200, json=reply(
                    None, finish="tool_calls", calls=[tool_call()],
                ))
            if len(requested) == 2:
                return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
            self.assertEqual(requested[-1][-1]["tool_call_id"], "call-1")
            self.assertIn("Tool output excerpt", requested[-1][-1]["content"])
            return httpx.Response(200, json=reply())

        async with make_client(handler) as client:
            agent = Agent(client=client, provider="compatible", catalog=CATALOG, tools=registry(execute))
            agent.events.on("context.overflow", recoveries.append)
            result = await agent.run(session=state, prompt="use a tool")
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(result.model_requests, 3)
        self.assertEqual(len(executed), 1)
        self.assertTrue(recoveries[0].data["retrying"])
        self.assertEqual(state.messages.get(2)["content"], "large result " * 160)

    async def test_streaming_overflow_retries_once_and_discards_checkpoint_on_failure(self):
        state = session()
        state.messages.add_user("large prior history " * 150)
        state.messages.add_assistant("prior answer")
        original = state.messages.snapshot()
        events, requested = [], []

        def handler(request):
            body = json.loads(request.content)
            requested.append(body)
            if not body.get("stream"):
                return httpx.Response(200, json=reply(SUMMARY))
            return stream_reply({"error": {"message": "This model's maximum context length is 1000 tokens."}})

        async with make_client(handler) as client:
            agent = Agent(client=client, provider="compatible")
            agent.events.on("context.overflow", events.append)
            result = await agent.run(session=state, prompt="continue", stream=True)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("context window was exceeded", result.error)
        self.assertEqual([bool(body.get("stream")) for body in requested], [True, False, True])
        self.assertEqual([event.data["retrying"] for event in events], [True, False])
        self.assertEqual(events[-1].data["reason"], "retry_exhausted")
        self.assertIsNone(state.compaction)
        self.assertEqual(state.messages.snapshot(), original)

    async def test_only_explicit_overflow_retries_and_requires_reduction(self):
        cases = (
            (429, {"error": {"code": "rate_limit", "message": "Rate limit exceeded"}}, None),
            (400, {"error": {"message": "Invalid tool input"}}, None),
            (400, {"error": {"message": "Unable to determine maximum context length"}}, None),
            (200, reply("truncated", finish="length"), None),
            (400, {"message": "Input token exceed the limit"}, "no_reduction"),
        )
        for status, payload, reason in cases:
            with self.subTest(status=status, reason=reason):
                events, requests = [], []

                def handler(request, requests=requests, status=status, payload=payload):
                    requests.append(request)
                    return httpx.Response(status, json=payload)

                async with make_client(handler) as client:
                    agent = Agent(client=client, provider="compatible")
                    agent.events.on("context.overflow", events.append)
                    result = await agent.run(session=session(), prompt="cannot reduce this")
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertEqual(len(requests), 1)
                self.assertEqual([event.data["reason"] for event in events], [reason] if reason else [])

    async def test_stream_with_visible_output_is_not_retried(self):
        state = session()
        state.messages.add_user("older data " * 150)
        state.messages.add_assistant("older response")
        requested, events = [], []

        def handler(request):
            requested.append(request)
            return stream_reply(
                {"choices": [{"index": 0, "delta": {"content": "visible output"}, "finish_reason": None}]},
                {"error": {"type": "context_window_exceeded"}},
            )

        async with make_client(handler) as client:
            agent = Agent(client=client, provider="compatible")
            agent.events.on("context.overflow", events.append)
            result = await agent.run(session=state, prompt="continue", stream=True)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertEqual(len(requested), 1)
        self.assertEqual(events[0].data["reason"], "stream_started")
