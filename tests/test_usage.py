import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from daphnis import EventBus, History, TurnStatus
from daphnis.app import Application
from daphnis.models import ModelInfo
from daphnis.usage import UsageTotals, UsageTracker

CATALOG = (
    ModelInfo(id="a", provider="deepseek", supports_tools=True, thinking_modes=()),
    ModelInfo(id="b", provider="mimo", supports_tools=True, thinking_modes=()),
)


def counts(prompt, completion):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def application(root, handler):
    return Application(
        workspace=root,
        provider="deepseek",
        model_id="a",
        catalog=CATALOG,
        connections={
            name: {
                "base_url": f"https://{name}.test",
                "transport": httpx.MockTransport(handler),
            }
            for name in ("deepseek", "mimo")
        },
        history=History(root / "sessions"),
    )


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshots_replace_and_missing_counts_remain_unknown(self):
        events = EventBus()
        tracker = UsageTracker(events)
        tracker.select_session("one", UsageTotals())
        fields = {"session_id": "one", "turn_id": "turn"}
        await events.emit("turn.started", **fields)
        await events.emit("model.started", **fields, request_number=1)
        for usage in (counts(10, 1), counts(10, 5), None):
            await events.emit("model.usage", **fields, request_number=1, usage=usage)
        await events.emit("model.started", **fields, request_number=2)
        self.assertEqual(tracker.turn, UsageTotals(10, 5, 15, 2, 1))
        # Invalid/bool counters never become fabricated zero counts.
        await events.emit(
            "model.usage", **fields, request_number=2, usage=counts(True, 5)
        )
        self.assertEqual(tracker.session, UsageTotals(10, 5, 15, 2, 1))
        tracker.select_session("two", UsageTotals(unrecorded=True))
        await events.emit(
            "model.usage", **fields, request_number=1, usage=counts(999, 999)
        )
        self.assertIsNone(tracker.turn)
        self.assertEqual(tracker.session, UsageTotals(unrecorded=True))
        tracker.close()

    async def test_tool_loop_stream_and_nonstream_totals_survive_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.txt").write_text("hello")
            request_number = 0

            def handler(request):
                nonlocal request_number
                request_number += 1
                body = json.loads(request.content)
                if not body.get("stream", False):
                    data = {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "again"},
                            }
                        ],
                    }
                    if request_number == 3:
                        data["usage"] = counts(50, 5)
                    return httpx.Response(200, json=data)
                usage = counts(request_number * 100, request_number * 10)
                # Cached input and reasoning are already in the parent counts.
                usage["prompt_tokens_details"] = {"cached_tokens": 10}
                usage["completion_tokens_details"] = {"reasoning_tokens": 5}
                delta = {"role": "assistant", "content": "done"}
                finish = "stop"
                if request_number == 1:
                    delta = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "read",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "input.txt"}),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                chunks = [
                    {
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ]
                    },
                    {"choices": [], "usage": usage},
                ]
                data = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks)
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=data + "data: [DONE]\n\n",
                )

            app = application(root, handler)
            seen = []
            off = app.events.on("model.usage", seen.append)
            try:
                result = await app.send("read", stream=True)
                self.assertEqual(result.status, TurnStatus.COMPLETED)
                self.assertEqual(result.model_requests, 2)
                self.assertEqual(len(seen), 2)
                self.assertEqual(app.usage.turn, UsageTotals(300, 30, 330, 2, 2))
                await app.send("again", stream=False)
                self.assertEqual(app.usage.turn, UsageTotals(50, 5, 55, 1, 1))
                self.assertEqual(app.usage.session, UsageTotals(350, 35, 385, 3, 3))
                await app.send("missing usage", stream=False)
                self.assertEqual(app.usage.turn, UsageTotals(requests=1))
                totals = UsageTotals(350, 35, 385, 4, 3)
                old = app.session.id
                await app.command("/clear")
                app.set_model("mimo", "b")
                self.assertEqual(app.usage.session, totals)
                app.new_session()
                self.assertIsNone(app.usage.turn)
                self.assertEqual(app.usage.session, UsageTotals())
                app.restore_session(old)
                self.assertIsNone(app.usage.turn)
                self.assertEqual(app.usage.session, totals)
                self.assertEqual(app.history.load(old).usage, totals)
                path = app.history.directory / f"{old}.json"
                legacy = json.loads(path.read_text())
                del legacy["usage"]
                path.write_text(json.dumps(legacy))
                self.assertTrue(app.history.load(old).usage.unrecorded)
            finally:
                off()
                await app.close()

    async def test_failed_and_cancelled_usage_is_saved_without_working_messages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entered = asyncio.Event()

            class PendingStream(httpx.AsyncByteStream):
                closed = False

                async def __aiter__(self):
                    chunk = {
                        "choices": [{"index": 0, "delta": {"content": "part"}}],
                        "usage": counts(20, 5),
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                    entered.set()
                    await asyncio.Event().wait()

                async def aclose(self):
                    self.closed = True

            pending = PendingStream()

            def handler(request):
                if not json.loads(request.content).get("stream", False):
                    # Usage exists even though the assistant response is malformed.
                    return httpx.Response(
                        200, json={"choices": [], "usage": counts(10, 3)}
                    )
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=pending
                )

            app = application(root, handler)
            try:
                failed = await app.send("invalid response", stream=False)
                self.assertEqual(failed.status, TurnStatus.FAILED)
                self.assertEqual(app.usage.turn, UsageTotals(10, 3, 13, 1, 1))
                task = asyncio.create_task(app.send("cancel stream", stream=True))
                await asyncio.wait_for(entered.wait(), 2)
                await app.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(pending.closed)
                self.assertEqual(app.usage.session, UsageTotals(30, 8, 38, 2, 2))
                self.assertEqual(len(app.session.messages), 0)
                restored = app.history.load(app.session.id)
                self.assertEqual(len(restored.messages), 0)
                self.assertEqual(restored.usage, app.usage.session)
            finally:
                await app.close()
