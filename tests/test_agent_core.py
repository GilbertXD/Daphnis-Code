import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from saturnus import ChatClient, Messages, ToolRegistry

from daphnis import (
    Agent,
    Context,
    EventBus,
    FatalToolError,
    History,
    Session,
    TurnStatus,
)
from daphnis.models import ModelInfo, provider_compatibility
from daphnis.tool_state import RUNNING, TOOL_STATE
from daphnis.turn_state import (
    ABANDONED,
    CONTINUE,
    TURN_ATTEMPT,
    TURN_STATE,
    RecoveryAction,
    RecoveryKind,
)
from daphnis.usage import UsageTotals

CATALOG = tuple(
    ModelInfo(id=name, provider="compatible", supports_tools=True, thinking_modes=())
    for name in ("model-a", "model-b")
)


def make_client(handler):
    return ChatClient(
        base_url="https://model.test/v1",
        compatibility=provider_compatibility("compatible"),
        transport=httpx.MockTransport(handler),
    )


def make_agent(client, **options):
    return Agent(client=client, provider="compatible", catalog=CATALOG, **options)


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


def stream_reply(*chunks):
    data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=data + "data: [DONE]\n\n",
    )


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
    async def test_retry_reuses_user_and_new_prompt_abandons_failed_user(self):
        requests = []
        state = session()

        def handler(request):
            requests.append(json.loads(request.content)["messages"])
            return httpx.Response(500, json={"error": "unavailable"}) if len(requests) < 3 else httpx.Response(200, json=reply())

        async with make_client(handler) as client:
            agent = make_agent(client)
            first = await agent.run(session=state, prompt="original")
            self.assertEqual(first.status, TurnStatus.FAILED)
            key = state.recovery.key
            retry = await agent.run(session=state, recovery_action=RecoveryAction.RETRY, recovery_key=key)
            self.assertEqual(retry.status, TurnStatus.FAILED)
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(len(state.messages), 1)
            self.assertNotEqual(state.recovery.key, key)
            stale = await agent.run(session=state, recovery_action=RecoveryAction.RETRY, recovery_key=key)
            self.assertEqual(stale.status, TurnStatus.FAILED)
            self.assertEqual(len(requests), 2)
            new = await agent.run(session=state, prompt="replacement")
        self.assertEqual(new.status, TurnStatus.COMPLETED)
        self.assertEqual(requests[2], [{"role": "user", "content": "replacement"}])
        self.assertEqual(state.messages.get(0)["content"], "original")
        self.assertEqual(state.messages.get(0)[TURN_STATE], ABANDONED)
        self.assertIsNone(state.recovery)

    async def test_cancellation_during_final_save_uses_the_committed_reply(self):
        state = session()
        entered, release = asyncio.Event(), asyncio.Event()
        with tempfile.TemporaryDirectory() as directory:
            history = History(directory)

            async def persist(candidate):
                if candidate.messages.get(-1)["role"] == "assistant":
                    entered.set()
                    await release.wait()
                await history.save_async(candidate)

            async with make_client(lambda _: httpx.Response(200, json=reply())) as client:
                task = asyncio.create_task(make_agent(client).run(session=state, prompt="request", persist=persist))
                await asyncio.wait_for(entered.wait(), 2)
                task.cancel()
                release.set()
                result = await asyncio.wait_for(task, 2)
            self.assertEqual(result.status, TurnStatus.COMPLETED)
            self.assertIsNone(history.load(state.id).recovery)

    def test_v2_recovery_metadata_keeps_history_indices_and_filters_context(self):
        with tempfile.TemporaryDirectory() as directory:
            history = History(directory)
            state = session()
            state.messages.add_user("failed")
            history.save(state)
            restored = history.load(state.id)
            self.assertEqual(restored.recovery.kind, RecoveryKind.FAILED_USER)
            restored.messages.replace(0, {**restored.messages.get(0), TURN_STATE: ABANDONED})
            restored.messages.add_user("retry", extra={TURN_STATE: CONTINUE, TURN_ATTEMPT: "attempt"})
            history.save(restored)
            restored = history.load(state.id)
            self.assertEqual(restored.messages.get(0)["content"], "failed")
            self.assertEqual(restored.recovery.message_index, 1)
            self.assertEqual(Context().build(restored.messages), [{"role": "user", "content": "retry"}])

    async def test_conversation_commits_user_before_request_and_context_is_independent(self):
        requests = []
        state = session()

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(
                body["messages"][0], {"role": "system", "content": "Be concise."}
            )
            self.assertEqual(state.messages.snapshot(), body["messages"][1:])
            return httpx.Response(200, json=reply(f"answer-{len(requests)}"))

        async with make_client(handler) as client:
            agent = make_agent(
                client, context=Context(system_prompt="Be concise."), tools=registry()
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
            self.assertEqual(state.messages.snapshot(), body["messages"])
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

        async with make_client(handler) as client:
            result = await make_agent(client, tools=tools, events=events).run(
                session=state, prompt="use tools"
            )
        self.assertEqual(result.status, TurnStatus.COMPLETED)
        self.assertEqual((result.model_requests, result.tool_calls), (2, 4))
        self.assertTrue(result.external_side_effects_possible)
        self.assertEqual(len(state.messages), 7)
        self.assertEqual(len(tool_events), 4)

    async def test_tool_turn_matches_across_nonstreaming_and_streaming(self):
        call = tool_call()
        responses = [reply("", finish="tool_calls", calls=[call]), reply()]
        del responses[1]["choices"][0]["message"]["role"]
        for response in responses:
            response["usage"] = {"total_tokens": 12}
        event_names = (
            "turn.started",
            "model.started",
            "model.usage",
            "model.delta",
            "model.finished",
            "tool.started",
            "tool.executing",
            "tool.finished",
            "turn.finished",
        )
        observed = []
        for stream in (False, True):
            with self.subTest(stream=stream):
                events = EventBus()
                seen, requests = [], []
                for name in event_names:
                    events.on(name, seen.append)

                def handler(request, requests=requests):
                    body = json.loads(request.content)
                    response = responses[len(requests)]
                    requests.append(body["messages"])
                    if not body.get("stream"):
                        return httpx.Response(200, json=response)
                    choice = response["choices"][0]
                    delta = dict(choice["message"])
                    if "tool_calls" in delta:
                        delta["tool_calls"] = [{"index": 0, **call}]
                    chunk = {
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": choice["finish_reason"],
                            }
                        ],
                        "usage": response["usage"],
                    }
                    return stream_reply(chunk)

                state = session()
                async with make_client(handler) as client:
                    result = await make_agent(
                        client, tools=registry(), events=events
                    ).run(session=state, prompt="use tool", stream=stream)

                self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                self.assertEqual((result.model_requests, result.tool_calls), (2, 1))
                self.assertEqual(result.message["role"], "assistant")
                self.assertEqual(result.message["content"], "done")
                committed = state.messages.snapshot()
                self.assertEqual(requests[1], committed[:-1])
                self.assertEqual(
                    requests[1][-1],
                    {"role": "tool", "tool_call_id": "call-1", "content": "hello"},
                )
                self.assertTrue(
                    all(
                        event.session_id == state.id and event.turn_id == result.turn_id
                        for event in seen
                    )
                )
                deltas = [event.data for event in seen if event.name == "model.delta"]
                self.assertEqual(len(deltas), 2 if stream else 0)
                if stream:
                    self.assertEqual(deltas[0]["request_number"], 1)
                    self.assertEqual(
                        deltas[0]["chunk"]["choices"][0]["finish_reason"], "tool_calls"
                    )
                observed.append(
                    (
                        committed,
                        [
                            (event.name, event.data)
                            for event in seen
                            if event.name != "model.delta"
                        ],
                    )
                )

        self.assertEqual(observed[0], observed[1])
        _, shared_events = observed[0]
        self.assertEqual(
            [name for name, _ in shared_events],
            [
                "turn.started",
                "model.started",
                "model.usage",
                "model.finished",
                "tool.started",
                "tool.executing",
                "tool.finished",
                "model.started",
                "model.usage",
                "model.finished",
                "turn.finished",
            ],
        )
        self.assertEqual(
            [data for name, data in shared_events if name == "model.finished"],
            [
                {
                    "request_number": 1,
                    "finish_reason": "tool_calls",
                    "response": responses[0],
                },
                {
                    "request_number": 2,
                    "finish_reason": "stop",
                    "response": responses[1],
                },
            ],
        )
        tool_finished = next(
            data for name, data in shared_events if name == "tool.finished"
        )
        self.assertEqual(tool_finished["tool_call_id"], "call-1")
        self.assertEqual(tool_finished["content"], "hello")
        self.assertEqual(shared_events[-1][1]["status"], "COMPLETED")

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
            return stream_reply(*chunks)

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
        async with make_client(handler) as client:
            with self.assertLogs("daphnis.events", level="ERROR"):
                result = await make_agent(client, events=events).run(
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

    async def test_invalid_response_only_commits_user_and_does_not_start_tools(self):
        valid = reply()["choices"][0]
        cases = (
            ({"choices": [valid, {**valid, "index": 1}]}, "exactly one"),
            ({"choices": [{**valid, "index": 1}]}, "no choice with index 0"),
            ({"choices": [{"message": valid["message"]}]}, ".index must be"),
            ({"choices": [{**valid, "message": None}]}, ".message must be"),
            (
                {"choices": [{**valid, "message": {"role": "user"}}]},
                "Expected an assistant",
            ),
            (reply(content=42), ".message.content must be"),
            (reply(finish=False), ".finish_reason must be"),
            (reply(finish=None), "Model did not finish normally"),
            (
                reply(calls=[tool_call()]),
                "stop response unexpectedly contains tool calls",
            ),
            (reply(finish="tool_calls", calls=[]), "no tool calls"),
            (
                reply(finish="tool_calls", calls=[tool_call(arguments={})]),
                "Incomplete or duplicate tool-call identity",
            ),
        )
        for response, error in cases:
            with self.subTest(error=error):
                state = session()
                state.messages.add_user("previous question")
                state.messages.add_assistant("previous answer")
                original = state.messages.snapshot()
                events = EventBus()
                tool_events = []
                events.on("tool.started", tool_events.append)
                async with make_client(
                    lambda request, response=response: httpx.Response(
                        200, json=response
                    )
                ) as client:
                    result = await make_agent(
                        client, tools=registry(), events=events
                    ).run(session=state, prompt="new question")
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertIn(error, result.error)
                self.assertEqual(state.messages.snapshot(), [*original, {"role": "user", "content": "new question"}])
                self.assertEqual(tool_events, [])

    async def test_stop_requires_visible_text_or_refusal(self):
        for fields, expected in (
            ({}, TurnStatus.FAILED),
            ({"content": None}, TurnStatus.FAILED),
            ({"content": ""}, TurnStatus.FAILED),
            ({"content": "  ", "reasoning_content": "thinking"}, TurnStatus.FAILED),
            ({"content": "answer"}, TurnStatus.COMPLETED),
            ({"content": [{"type": "text", "text": "answer"}]}, TurnStatus.COMPLETED),
            ({"content": None, "refusal": "Cannot comply."}, TurnStatus.COMPLETED),
        ):
            with self.subTest(fields=fields):
                state = session()
                response = reply(None)
                response["choices"][0]["message"] = {"role": "assistant", **fields}
                async with make_client(
                    lambda request, response=response: httpx.Response(200, json=response)
                ) as client:
                    result = await make_agent(client).run(session=state, prompt="answer")
                self.assertEqual(result.status, expected, result.error)
                if expected == TurnStatus.FAILED:
                    self.assertIn("no visible text or refusal", result.error)
                    self.assertEqual(len(state.messages), 1)
                else:
                    self.assertEqual(result.message, state.messages.snapshot()[-1])

        async with make_client(lambda request: stream_reply(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "refusal": "Cannot "}}]},
            {"choices": [{"index": 0, "delta": {"refusal": "comply."}, "finish_reason": "stop"}]},
        )) as client:
            result = await make_agent(client).run(session=session(), prompt="answer", stream=True)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(result.message["refusal"], "Cannot comply.")

    async def test_failure_limits_and_unsafe_tools_keep_complete_boundaries(self):
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
                async with make_client(
                    lambda request, response=response: httpx.Response(200, json=response)
                ) as client:
                    result = await make_agent(client, tools=tools, **options).run(
                        session=state, prompt="new question"
                    )
                if name == "request_limit":
                    self.assertEqual(result.status, TurnStatus.PAUSED_LIMIT)
                    self.assertEqual(result.model_requests, 1)
                    self.assertIsNone(result.error)
                    self.assertIsNone(result.message)
                    self.assertIn("继续", result.notice)
                else:
                    self.assertEqual(result.status, TurnStatus.FAILED)
                    self.assertIsNotNone(result.error)
                self.assertEqual(state.messages.snapshot()[:3], [*original, {"role": "user", "content": "new question"}])
                self.assertEqual(len(state.messages), 5 if name == "request_limit" else 3)

        async def unsafe(text):
            raise FatalToolError("environment unavailable")

        state = session()
        async with make_client(
            lambda request: httpx.Response(
                200, json=reply(None, finish="tool_calls", calls=[tool_call()])
            )
        ) as client:
            result = await make_agent(client, tools=registry(unsafe), max_requests=1).run(
                session=state, prompt="use tool"
            )
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("environment unavailable", result.error)
        self.assertTrue(result.external_side_effects_possible)
        self.assertEqual(state.messages.get(-1)[TOOL_STATE], RUNNING)

    async def test_last_request_keeps_tools_and_commits_its_response(self):
        async def check(final_calls, max_primary, max_total):
            requests = []
            state = session()
            tools = registry()
            system = {"role": "system", "content": "Be concise."}
            user = {"role": "user", "content": "use tools"}

            def handler(request):
                requests.append(json.loads(request.content))
                if len(requests) == 1:
                    return httpx.Response(200, json=reply(None, finish="tool_calls", calls=[tool_call()]))
                self.assertEqual(len(requests), 2)
                response = reply(None, finish="tool_calls", calls=final_calls) if final_calls else reply("done")
                return httpx.Response(200, json=response)

            async with make_client(handler) as client:
                result = await make_agent(
                    client, tools=tools, context=Context(system_prompt=system["content"]),
                    max_primary_requests=max_primary, max_total_requests=max_total,
                ).run(session=state, prompt=user["content"])

            messages = state.messages.snapshot()
            self.assertEqual([body["tools"] for body in requests], [tools.definitions()] * 2)
            self.assertEqual([body["messages"] for body in requests], [[system, user], [system, *messages[:3]]])
            self.assertEqual((result.model_requests, result.primary_requests, result.maintenance_requests), (2, 2, 0))
            self.assertIsNone(result.error)
            if final_calls:
                self.assertEqual((result.status, result.tool_calls), (TurnStatus.PAUSED_LIMIT, 3))
                self.assertIsNone(result.message)
                self.assertIn("总请求上限（2/2）", result.notice)
                self.assertEqual(messages[3]["tool_calls"], final_calls)
                self.assertEqual([item["tool_call_id"] for item in messages[4:]], ["call-2", "call-3"])
                self.assertEqual(messages[4]["content"], "hello")
                self.assertEqual(json.loads(messages[5]["content"])["error"]["code"], "unknown_tool")
            else:
                self.assertEqual((result.status, result.tool_calls), (TurnStatus.COMPLETED, 1))
                self.assertEqual(messages[3:], [result.message])
                self.assertEqual(result.message["content"], "done")
                self.assertIsNone(result.notice)

        await check(None, max_primary=2, max_total=5)
        await check([tool_call("call-2"), tool_call("call-3", "missing", "{}")], max_primary=5, max_total=2)

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
        async with make_client(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        ) as client:
            task = asyncio.create_task(
                make_agent(client, events=events).run(
                    session=state, prompt="stream", stream=True
                )
            )
            await asyncio.wait_for(received.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stream.closed)
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(finished[-1].data["status"], "CANCELLED")

    async def test_cancel_tool_reports_possible_side_effects_and_releases_session(self):
        entered = asyncio.Event()
        never = asyncio.Event()

        async def handler_tool(text):
            if text == "second":
                entered.set()
                await never.wait()
            return text

        requests = []

        def handler(request):
            requests.append(request)
            response = (
                reply(None, finish="tool_calls", calls=[tool_call(), tool_call("call-2", arguments='{"text":"second"}')])
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
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        history = History(temporary.name)
        async with make_client(handler) as client:
            agent = make_agent(client, tools=registry(handler_tool), events=events, max_requests=1)
            task = asyncio.create_task(agent.run(session=state, prompt="tool", persist=history.save_async))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            state = history.load(state.id)
            self.assertEqual(state.messages.get(-2)["content"], "hello")
            self.assertEqual(state.messages.get(-1)[TOOL_STATE], RUNNING)
            self.assertEqual(finished[-1].data["status"], "CANCELLED")
            self.assertTrue(finished[-1].data["external_side_effects_possible"])
            self.assertEqual([event.data["tool_call_id"] for event in tool_finished], ["call-1"])
            result = await agent.run(
                session=state, recovery_action=RecoveryAction.CONTINUE,
                recovery_key=state.recovery.key, persist=history.save_async,
            )
            self.assertEqual(result.status, TurnStatus.COMPLETED)
            self.assertEqual(
                json.loads(state.messages.get(-2)["content"])["error"]["code"],
                "execution_result_unknown",
            )

    async def test_shared_client_allows_parallel_sessions_with_distinct_models(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        requests = []

        async def wait_tool(text):
            entered.set()
            await release.wait()
            return text

        def handler(request):
            body = json.loads(request.content)
            requests.append((body["model"], body["max_tokens"]))
            response = (
                reply(None, finish="tool_calls", calls=[tool_call()])
                if len(requests) == 1
                else reply(body["model"])
            )
            return httpx.Response(200, json=response)

        first_session, second_session = session(), session("model-b")
        async with make_client(handler) as client:
            first = make_agent(client, tools=registry(wait_tool))
            second = make_agent(client)
            one = asyncio.create_task(
                first.run(session=first_session, prompt="one", output_limit=100)
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                with self.assertRaisesRegex(RuntimeError, "active Turn"):
                    await second.run(session=first_session, prompt="overlap")
                two = await asyncio.wait_for(
                    second.run(session=second_session, prompt="two", output_limit=200),
                    2,
                )
                self.assertEqual(two.status, TurnStatus.COMPLETED, two.error)
                self.assertEqual(two.message["content"], "model-b")
                self.assertFalse(one.done())
                self.assertEqual(first_session.messages.get(-1)[TOOL_STATE], RUNNING)
            finally:
                release.set()
                result = await asyncio.wait_for(one, 2)
        self.assertEqual(
            requests, [("model-a", 100), ("model-b", 200), ("model-a", 100)]
        )
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(result.message["content"], "model-a")
        self.assertEqual(first_session.messages.get(0)["content"], "one")
        self.assertEqual(second_session.messages.get(0)["content"], "two")

    async def test_session_identity_change_during_turn_prevents_commit(self):
        state = session()
        state.messages.add_user("previous message")
        original = state.messages.snapshot()
        requests = []

        def change_model(text):
            state.model_id = "model-b"
            return text

        def handler(request):
            requests.append(json.loads(request.content)["model"])
            response = (
                reply(None, finish="tool_calls", calls=[tool_call()])
                if len(requests) == 1
                else reply()
            )
            return httpx.Response(200, json=response)

        async with make_client(handler) as client:
            result = await make_agent(client, tools=registry(change_model)).run(
                session=state, prompt="use tool"
            )
        self.assertEqual(requests, ["model-a"])
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("Session changed", result.error)
        self.assertTrue(result.external_side_effects_possible)
        self.assertEqual(state.messages.snapshot()[:-2], [{**original[0], TURN_STATE: ABANDONED}, {"role": "user", "content": "use tool"}])
        self.assertEqual(state.messages.get(-1)[TOOL_STATE], RUNNING)

    async def test_provider_and_tool_capabilities_are_checked_before_request(self):
        states = (
            (
                Session(workspace="/tmp", provider="mimo", model_id="mimo-v2.6-pro"),
                "provider",
            ),
            (session("plain"), "unsupported"),
            (session("unknown"), "unknown"),
        )
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=reply())

        async with make_client(handler) as client:
            agent = Agent(
                client=client,
                provider="compatible",
                catalog=(
                    ModelInfo(id="plain", provider="compatible", supports_tools=False),
                ),
                tools=registry(),
            )
            for state, error in states:
                with self.subTest(error=error):
                    result = await agent.run(session=state, prompt="use tool")
                    self.assertEqual(result.status, TurnStatus.FAILED)
                    self.assertIn(error, result.error)
                    self.assertEqual(result.model_requests, 0)
                    self.assertEqual(len(state.messages), 0)
        self.assertEqual(requests, [])

    async def test_history_round_trip_atomic_failure_and_invalid_files(self):
        state = replace(
            session(),
            thinking="high",
            enabled_tools=("read_file", "run_command"),
            usage=UsageTotals(10, 2, 12, 1, 1),
        )
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
            self.assertEqual(restored.thinking, state.thinking)
            self.assertEqual(restored.enabled_tools, state.enabled_tools)
            self.assertEqual(restored.usage, state.usage)
            path = Path(temporary) / f"{state.id}.json"
            original = path.read_bytes()
            data = json.loads(original)
            self.assertEqual(data["version"], 2)
            self.assertEqual(data["enabled_tools"], ["read_file", "run_command"])
            with (
                patch(
                    "daphnis.history.os.replace",
                    side_effect=OSError("disk unavailable"),
                ),
                self.assertRaises(OSError),
            ):
                history.save(replace(state, thinking="off"))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])
            with self.assertRaises(FileNotFoundError):
                history.load("missing")
            with self.assertRaises(ValueError):
                history.load("../outside")
            path.write_text("{invalid", encoding="utf-8")
            with self.assertRaises(ValueError):
                history.load(state.id)
            path.write_text(json.dumps(data | {"version": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                history.load(state.id)
            path.write_text(json.dumps(data | {"messages": None}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "messages"):
                history.load(state.id)
            path.write_text(
                json.dumps(data | {"enabled_tools": "read_file"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(TypeError, "enabled_tools"):
                history.load(state.id)
            del data["usage"]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                history.load(state.id)


class SessionTests(unittest.TestCase):
    def test_execution_settings_defaults_and_validation(self):
        fresh = session()
        self.assertIsNone(fresh.thinking)
        self.assertEqual(fresh.enabled_tools, ())
        for settings, error, field in (
            ({"thinking": 1}, ValueError, "thinking"),
            ({"thinking": " "}, ValueError, "thinking"),
            ({"enabled_tools": "read_file"}, TypeError, "enabled_tools"),
            ({"enabled_tools": ("",)}, ValueError, "enabled_tools"),
            ({"enabled_tools": (1,)}, ValueError, "enabled_tools"),
            ({"enabled_tools": ("read_file", "read_file")}, ValueError, "duplicates"),
        ):
            with self.subTest(settings=settings), self.assertRaisesRegex(error, field):
                Session(
                    workspace=fresh.workspace,
                    provider=fresh.provider,
                    model_id=fresh.model_id,
                    **settings,
                )


if __name__ == "__main__":
    unittest.main()
