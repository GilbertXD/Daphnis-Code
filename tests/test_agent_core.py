import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from saturnus import Messages, ToolRegistry

from daphnis import (
    Agent,
    Context,
    EventBus,
    FatalToolError,
    History,
    Session,
    TurnStatus,
)
from daphnis.models import LLM, ModelInfo

CATALOG = tuple(
    ModelInfo(id=name, provider="compatible", supports_tools=True, thinking_modes=())
    for name in ("model-a", "model-b")
)


def make_llm(handler):
    return LLM(
        provider="compatible",
        model_id="model-a",
        base_url="https://model.test/v1",
        catalog=CATALOG,
        transport=httpx.MockTransport(handler),
    )


def session(model_id="model-a"):
    return Session(
        workspace="/tmp/daphnis-project", provider="compatible", model_id=model_id
    )


def reply(content="done", *, finish="stop", calls=None):
    message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": "reasoning",
    }
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }


def tool_call(call_id="call-1", name="echo", arguments='{"text": "hello"}'):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def registry(handler=None):
    tools = ToolRegistry()
    tools.register(
        name="echo",
        description="Echo a value",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=handler if handler is not None else lambda text: text,
    )
    return tools


class AgentCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_commits_once_and_context_is_independent(self):
        requests = []
        state = session()

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(
                body["messages"][0], {"role": "system", "content": "Be concise."}
            )
            # No pending user or assistant messages are committed during a request.
            self.assertEqual(len(state.messages), 2 * (len(requests) - 1))
            return httpx.Response(200, json=reply(f"answer-{len(requests)}"))

        async with make_llm(handler) as llm:
            agent = Agent(
                llm=llm, context=Context(system_prompt="Be concise."), tools=registry()
            )
            for text in ("first", "second"):
                result = await agent.run(session=state, prompt=text)
                self.assertEqual(result.status, TurnStatus.COMPLETED)
                self.assertEqual(result.model_requests, 1)
                self.assertEqual(result.tool_calls, 0)
            self.assertEqual(len(state.messages), 4)
            self.assertEqual(requests[1]["messages"][1:], state.messages.snapshot()[:3])
            self.assertEqual(state.messages.get(0)["role"], "user")
            self.assertEqual(state.messages.get(1)["reasoning_content"], "reasoning")
            result.message["content"] = "mutated result"
            self.assertEqual(state.messages.get(-1)["content"], "answer-2")

    async def test_tools_produce_one_result_each_including_recoverable_errors(self):
        tools = registry()

        def fail():
            raise RuntimeError("tool failed")

        tools.register(
            name="fail", description="Fail", parameters={"type": "object"}, handler=fail
        )
        calls = [
            tool_call(),
            tool_call("call-2", "missing", "{}"),
            tool_call("call-3", arguments="{invalid"),
            tool_call("call-4", "fail", "{}"),
        ]
        requests = []
        events = EventBus()
        tool_events = []
        events.on("tool.finished", tool_events.append)
        state = session()

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(len(state.messages), 0)
            if len(requests) == 1:
                return httpx.Response(
                    200, json=reply(None, finish="tool_calls", calls=calls)
                )
            chain = body["messages"]
            self.assertEqual(chain[1]["tool_calls"], calls)
            self.assertEqual(chain[1]["reasoning_content"], "reasoning")
            self.assertEqual(
                [item["tool_call_id"] for item in chain[2:]],
                [item["id"] for item in calls],
            )
            self.assertEqual(chain[2]["content"], "hello")
            self.assertEqual(
                json.loads(chain[3]["content"])["error"]["code"], "unknown_tool"
            )
            self.assertEqual(
                json.loads(chain[4]["content"])["error"]["code"],
                "invalid_arguments_json",
            )
            self.assertEqual(
                json.loads(chain[5]["content"])["error"]["type"], "ToolExecutionError"
            )
            return httpx.Response(200, json=reply())

        async with make_llm(handler) as llm:
            result = await Agent(llm=llm, tools=tools, events=events).run(
                session=state, prompt="use tools"
            )
        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual((result.model_requests, result.tool_calls), (2, 4))
        self.assertTrue(result.external_side_effects_possible)
        self.assertEqual(len(state.messages), 7)
        self.assertEqual(len(tool_events), 4)

    async def test_stream_and_event_isolation_subscription_and_unsubscription(self):
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": "think"},
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {"content": "hello"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        def handler(request):
            data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=data + "data: [DONE]\n\n",
            )

        events = EventBus()
        seen = []

        def mutate(event):
            event.data["chunk"]["choices"].clear()

        def sync_failure(event):
            raise RuntimeError("observer failure")

        async def async_failure(event):
            raise RuntimeError("async observer failure")

        events.on("model.delta", mutate)
        events.on("model.delta", sync_failure)
        events.on("model.delta", async_failure)
        unsubscribe = events.on("model.delta", seen.append)
        state = session()
        async with make_llm(handler) as llm:
            with self.assertLogs("daphnis.events", level="ERROR"):
                result = await Agent(llm=llm, events=events).run(
                    session=state, prompt="stream", stream=True
                )
                await events.drain()
        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.message["content"], "hello")
        self.assertEqual(result.message["reasoning_content"], "think")
        self.assertEqual(len(seen), 3)
        self.assertTrue(seen[0].data["chunk"]["choices"])
        self.assertTrue(
            all(
                event.session_id == state.id and event.turn_id == result.turn_id
                for event in seen
            )
        )
        unsubscribe()
        unsubscribe()
        # Custom event names require no central registration.
        custom = []
        off = events.on("custom.notification", custom.append)
        await events.emit(
            "custom.notification", session_id=state.id, turn_id=result.turn_id, value=1
        )
        off()
        await events.emit(
            "custom.notification", session_id=state.id, turn_id=result.turn_id, value=2
        )
        self.assertEqual([event.data["value"] for event in custom], [1])

    async def test_failure_limits_and_unsafe_tools_do_not_commit(self):
        cases = (
            ("length", {}, registry(), reply("partial", finish="length")),
            ("content_filter", {}, registry(), reply(None, finish="content_filter")),
            (
                "request_limit",
                {"max_requests": 1},
                registry(),
                reply(None, finish="tool_calls", calls=[tool_call()]),
            ),
            (
                "tool_limit",
                {"max_tool_calls": 1},
                registry(),
                reply(
                    None, finish="tool_calls", calls=[tool_call(), tool_call("call-2")]
                ),
            ),
            (
                "duplicate_ids",
                {},
                registry(),
                reply(None, finish="tool_calls", calls=[tool_call(), tool_call()]),
            ),
        )
        for name, options, tools, response in cases:
            with self.subTest(name=name):
                state = session()
                state.messages.add_user("old question")
                state.messages.add_assistant("old answer")
                original = state.messages.snapshot()
                async with make_llm(
                    lambda request: httpx.Response(200, json=response)
                ) as llm:
                    result = await Agent(llm=llm, tools=tools, **options).run(
                        session=state, prompt="new question"
                    )
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertIsNotNone(result.error)
                self.assertEqual(state.messages.snapshot(), original)

        async def unsafe(text):
            raise FatalToolError("environment unavailable")

        state = session()
        async with make_llm(
            lambda request: httpx.Response(
                200, json=reply(None, finish="tool_calls", calls=[tool_call()])
            )
        ) as llm:
            result = await Agent(llm=llm, tools=registry(unsafe)).run(
                session=state, prompt="use tool"
            )
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("environment unavailable", result.error)
        self.assertTrue(result.external_side_effects_possible)
        self.assertEqual(len(state.messages), 0)

    async def test_cancel_stream_closes_resources_and_preserves_history(self):
        received = asyncio.Event()
        never = asyncio.Event()

        class PendingStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                chunk = {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "partial"},
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                await never.wait()

            async def aclose(self):
                self.closed = True

        stream = PendingStream()
        events = EventBus()
        finished = []
        events.on("model.delta", lambda event: received.set())
        events.on("turn.finished", finished.append)
        state = session()
        async with make_llm(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        ) as llm:
            task = asyncio.create_task(
                Agent(llm=llm, events=events).run(
                    session=state, prompt="stream", stream=True
                )
            )
            await asyncio.wait_for(received.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stream.closed)
        self.assertEqual(len(state.messages), 0)
        self.assertEqual(finished[-1].data["status"], "CANCELLED")

    async def test_cancel_tool_reports_possible_side_effects_and_releases_session(self):
        entered = asyncio.Event()
        never = asyncio.Event()

        async def handler_tool(text):
            entered.set()
            await never.wait()

        requests = []

        def handler(request):
            requests.append(request)
            response = (
                reply(None, finish="tool_calls", calls=[tool_call()])
                if len(requests) == 1
                else reply()
            )
            return httpx.Response(200, json=response)

        events = EventBus()
        finished = []
        events.on("turn.finished", finished.append)
        tool_finished = []
        events.on("tool.finished", tool_finished.append)
        state = session()
        async with make_llm(handler) as llm:
            agent = Agent(llm=llm, tools=registry(handler_tool), events=events)
            task = asyncio.create_task(agent.run(session=state, prompt="tool"))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(len(state.messages), 0)
            self.assertEqual(finished[-1].data["status"], "CANCELLED")
            self.assertTrue(finished[-1].data["external_side_effects_possible"])
            self.assertEqual(tool_finished[-1].data["status"], "CANCELLED")
            result = await agent.run(session=state, prompt="next")
            self.assertEqual(result.status, TurnStatus.COMPLETED)

    async def test_session_exclusion_and_shared_model_selection_are_serialized(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        model_ids = []

        async def wait_tool(text):
            entered.set()
            await release.wait()
            return text

        def handler(request):
            body = json.loads(request.content)
            model_ids.append(body["model"])
            response = (
                reply(None, finish="tool_calls", calls=[tool_call()])
                if len(model_ids) == 1
                else reply()
            )
            return httpx.Response(200, json=response)

        first_session, second_session = session(), session("model-b")
        async with make_llm(handler) as llm:
            first = Agent(llm=llm, tools=registry(wait_tool))
            second = Agent(llm=llm)
            one = asyncio.create_task(first.run(session=first_session, prompt="one"))
            await asyncio.wait_for(entered.wait(), 2)
            two = asyncio.create_task(second.run(session=second_session, prompt="two"))
            await asyncio.sleep(0)
            with self.assertRaisesRegex(RuntimeError, "active Turn"):
                await second.run(session=first_session, prompt="overlap")
            self.assertEqual(model_ids, ["model-a"])
            release.set()
            results = await asyncio.wait_for(asyncio.gather(one, two), 2)
        self.assertEqual(model_ids, ["model-a", "model-a", "model-b"])
        self.assertTrue(
            all(result.status == TurnStatus.COMPLETED for result in results)
        )
        self.assertEqual(first_session.messages.get(0)["content"], "one")
        self.assertEqual(second_session.messages.get(0)["content"], "two")

    async def test_history_round_trip_atomic_failure_and_invalid_files(self):
        state = session()
        state.messages = Messages(
            [
                {"role": "user", "content": "hello"},
                reply(None, finish="tool_calls", calls=[tool_call()])["choices"][0][
                    "message"
                ],
                {"role": "tool", "tool_call_id": "call-1", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "done",
                    "vendor_metadata": {"kept": True},
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            history = History(temporary)
            history.save(state)
            restored = history.load(state.id)
            self.assertEqual(
                (restored.id, restored.provider, restored.model_id, restored.workspace),
                (state.id, state.provider, state.model_id, state.workspace),
            )
            self.assertEqual(restored.messages.snapshot(), state.messages.snapshot())
            path = Path(temporary) / f"{state.id}.json"
            original = path.read_bytes()
            with patch(
                "daphnis.history.os.replace", side_effect=OSError("disk unavailable")
            ):
                with self.assertRaises(OSError):
                    history.save(state)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])
            with self.assertRaises(FileNotFoundError):
                history.load("missing")
            with self.assertRaises(ValueError):
                history.load("../outside")
            path.write_text("{invalid", encoding="utf-8")
            with self.assertRaises(ValueError):
                history.load(state.id)
            data = json.loads(original)
            data["version"] = 2
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                history.load(state.id)
            data["version"] = 1
            data["messages"] = None
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "messages"):
                history.load(state.id)


if __name__ == "__main__":
    unittest.main()
