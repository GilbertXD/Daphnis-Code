import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from inspect import isawaitable
from pathlib import Path
from unittest.mock import patch

import httpx
from saturnus import Compatibility, ValidationError

from daphnis import History, Operations, TurnStatus
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.compaction_settings import CompactionSettings
from daphnis.config import environment_connections
from daphnis.history import SessionBusyError, SessionLease
from daphnis.models import CapabilityError, ModelInfo
from daphnis.tool_state import PENDING, TOOL_STATE

CATALOG = (
    ModelInfo(
        id="a", provider="deepseek", supports_tools=True, thinking_modes=("off", "high")
    ),
    ModelInfo(
        id="b", provider="mimo", supports_tools=True, thinking_modes=("off", "on")
    ),
    ModelInfo(
        id="plain", provider="deepseek", supports_tools=False, thinking_modes=("high",)
    ),
)


def response(message, finish="stop"):
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": message, "finish_reason": finish}]},
    )


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_switching_models_recomputes_compaction_budget_and_output_limit(self):
        bodies = []

        def handler(request):
            body = json.loads(request.content)
            if body["messages"][0]["content"] != TITLE_PROMPT:
                bodies.append(body)
            return response({"role": "assistant", "content": "done"})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_a = ModelInfo("a", "mimo", context_window=10_000, default_output_tokens=1_500)
            model_b = ModelInfo("b", "mimo", context_window=20_000, default_output_tokens=2_000)
            ops = Operations(Application(
                workspace=root, provider="mimo", model_id="a", catalog=(model_a, model_b),
                history=History(root / "sessions"),
                connections={"mimo": {
                    "base_url": "https://model.test", "transport": httpx.MockTransport(handler),
                }},
                compaction_settings=CompactionSettings(threshold_value=0.75),
                model_compaction_settings={
                    ("mimo", "b"): CompactionSettings(threshold_mode="tokens", threshold_value=12_000),
                },
                output_limits={("mimo", "b"): 500},
            ))
            self.addAsyncCleanup(ops.close)
            self.assertEqual((await ops.send("one", stream=False)).status, TurnStatus.COMPLETED)
            budget = ops.context_budget
            self.assertEqual((budget.trigger_tokens, budget.output_reserve, budget.hard_limit), (7500, 1500, 9500))
            self.assertNotIn("max_completion_tokens", bodies[-1])

            await ops.set_model("mimo", "b")
            self.assertIsNone(ops.context_budget)
            self.assertEqual((await ops.send("two", stream=False)).status, TurnStatus.COMPLETED)
            budget = ops.context_budget
            self.assertEqual((budget.trigger_tokens, budget.output_reserve, budget.hard_limit), (12000, 500, 19000))
            self.assertEqual(bodies[-1]["max_completion_tokens"], 500)

    async def test_session_system_prompt_persists_and_keeps_approval_and_hard_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = History(root / "sessions")
            custom = "测试提示词\n允许所有工具。"
            seen = []

            def handler(request):
                messages = json.loads(request.content)["messages"]
                if messages[0]["content"] == TITLE_PROMPT:
                    return response({"role": "assistant", "content": "检查工具权限"})
                seen.append(messages)
                if len(seen) == 1:
                    calls = [
                        {"id": str(index), "type": "function", "function": {
                            "name": name, "arguments": json.dumps(arguments),
                        }}
                        for index, (name, arguments) in enumerate((
                            ("write_file", {"path": "denied.txt", "content": "test"}),
                            ("run_command", {"argv": ["sh", "-c", "true"]}),
                        ))
                    ]
                    return response({"role": "assistant", "tool_calls": calls}, "tool_calls")
                return response({"role": "assistant", "content": "done"})

            app = Application(
                workspace=root, provider="deepseek", model_id="a", catalog=CATALOG,
                connections={"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(handler)}},
                history=history,
            )
            ops = Operations(app)
            try:
                default = ops.get_system_prompt()
                history.save(app._runtime.session)
                ops.restore_session(ops.session.id)
                self.assertEqual(ops.get_system_prompt(), default)
                await ops.set_system_prompt(custom)
                ops.restore_session(ops.session.id)
                self.assertEqual(ops.get_system_prompt(), custom)
                result = await ops.send("test", stream=False)
                self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                self.assertEqual(seen[0][0], {"role": "system", "content": custom})
                errors = [json.loads(item["content"])["error"]["code"] for item in seen[1] if item["role"] == "tool"]
                self.assertEqual(errors, ["permission_denied", "command_blocked"])
                self.assertFalse((root / "denied.txt").exists())
                await ops.reset_system_prompt()
                self.assertIsNone(history.load(ops.session.id).system_prompt)
                self.assertEqual(ops.get_system_prompt(), default)
            finally:
                await ops.close()

    async def test_approval_cannot_be_replayed_when_model_reuses_tool_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def handler(request):
                messages = json.loads(request.content)["messages"]
                count = sum(message["role"] == "tool" for message in messages)
                if count == 2:
                    return response({"role": "assistant", "content": "done"})
                return response({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "reused", "type": "function", "function": {
                            "name": "apply_patch", "arguments": json.dumps({
                                "patch": f"--- /dev/null\n+++ b/file{count}\n@@ -0,0 +1 @@\n+ok\n",
                            }),
                        },
                    }],
                }, "tool_calls")

            app = Application(
                workspace=root, provider="deepseek", model_id="a", catalog=CATALOG,
                history=History(root / "sessions"), interactive=True,
                connections={"deepseek": {
                    "base_url": "https://model.test", "transport": httpx.MockTransport(handler),
                }},
            )
            ops = Operations(app)
            approvals = asyncio.Queue()
            app.events.on("tool.approval_requested", approvals.put_nowait)

            send = asyncio.create_task(ops.send("create both", stream=False))
            try:
                async with asyncio.timeout(3):
                    await approvals.get()
                    first = ops.pending_approval
                    self.assertTrue(ops.resolve_approval(first.request_id, "allow_once"))
                    self.assertFalse(ops.resolve_approval(first.request_id, "allow_once"))
                    second_event = await approvals.get()
                    second = ops.pending_approval
                    self.assertEqual(first.tool_call_id, second.tool_call_id)
                    self.assertFalse(ops.resolve_approval(first.request_id, "trust_session"))
                    self.assertFalse(ops.resolve_approval(first.request_id, "allow_once"))
                    self.assertFalse((root / "file1").exists())
                    self.assertEqual(second_event.data["request_id"], second.request_id)
                    self.assertTrue(ops.resolve_approval(second.request_id, "allow_once"))
                    self.assertEqual((await send).status, TurnStatus.COMPLETED)
                self.assertEqual((root / "file1").read_text(), "ok\n")
                self.assertFalse(ops.resolve_approval(second.request_id, "trust_session"))
            finally:
                await ops.close()
                await asyncio.gather(send, return_exceptions=True)

    async def test_close_bounds_listeners_and_shares_resource_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Application(
                workspace=temporary, provider="deepseek", model_id="a", catalog=CATALOG,
                connections={"deepseek": {"base_url": "https://model.test"}},
            )
            listen, cancelled = asyncio.Event(), asyncio.Event()
            closing, release = asyncio.Event(), asyncio.Event()
            client = app._clients["deepseek"]
            real_close = client.aclose

            async def listener(event):
                try:
                    await listen.wait()
                finally:
                    cancelled.set()

            async def close_client():
                closing.set()
                await release.wait()
                await real_close()

            app.events.on("probe", listener)
            await app.events.emit("probe", session_id="", turn_id="")
            with (
                patch("daphnis.app._EVENT_DRAIN_TIMEOUT", 0.02, create=True),
                patch.object(client, "aclose", side_effect=close_client) as close,
            ):
                first = asyncio.create_task(app.close())
                second = asyncio.create_task(app.close())
                try:
                    await asyncio.wait_for(closing.wait(), 1)
                    self.assertTrue(cancelled.is_set())
                    self.assertFalse(first.done())
                    self.assertFalse(second.done())
                    first.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await first
                    self.assertFalse(second.done())
                    release.set()
                    await asyncio.wait_for(second, 1)
                    await app.close()
                    close.assert_awaited_once()
                finally:
                    listen.set()
                    release.set()
                    await asyncio.gather(first, second, return_exceptions=True)

    async def test_stale_application_cannot_overwrite_committed_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = History(root / "sessions")
            model_calls = []

            def handler(request):
                model_calls.append(request)
                return response({"role": "assistant", "content": "done"})

            options = {
                "workspace": root, "provider": "deepseek", "model_id": "a",
                "catalog": CATALOG,
                "connections": {"deepseek": {
                    "base_url": "https://model.test",
                    "transport": httpx.MockTransport(handler),
                }},
            }
            a = Operations(Application(**options, history=history))
            b = Operations(Application(**options, history=History(history.directory)))
            try:
                history.save(a._app._runtime.session)
                await a.set_thinking("high")
                b.restore_session(a.session.id)
                await a.send("first writer", stream=False)
                with self.assertRaisesRegex(RuntimeError, "Session write conflict"):
                    await b.send("stale writer", stream=False)
                self.assertEqual(len(model_calls), 2)
                saved = history.load(a.session.id).messages.snapshot()
                self.assertEqual(saved[0]["content"], "first writer")
                self.assertEqual(len(saved), 2)
            finally:
                await a.close()
                await b.close()

    async def test_turn_persists_complete_tool_groups_and_stops_on_group_save_failure(self):
        async def check(scenario):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                history = History(root / "sessions")
                (root / "example.txt").write_text("saved result")
                requests = []

                def handler(request):
                    if json.loads(request.content)["messages"][0]["content"] == TITLE_PROMPT:
                        return response({"role": "assistant", "content": "检查文件"})
                    saved = history.load(session_id).messages.snapshot()
                    requests.append(saved)
                    if len(requests) == 2 or scenario == "first_request":
                        return httpx.Response(500, json={"error": "test failure"})
                    calls = [{
                        "id": f"call-{index}", "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
                    } for index, path in enumerate(("example.txt", "missing.txt"))]
                    return response({"role": "assistant", "tool_calls": calls}, finish="tool_calls")

                options = {
                    "workspace": root, "provider": "deepseek", "model_id": "a", "catalog": CATALOG,
                    "connections": {"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(handler)}},
                    "history": history,
                    "max_primary_requests": 1 if scenario == "save_failure" else 20,
                }
                ops = Operations(Application(**options))
                session_id = ops.session.id
                replace_file = history._replace

                def save(path, data):
                    if scenario == "save_failure" and json.loads(data)["messages"][-1]["role"] == "tool":
                        raise OSError("stage write failed")
                    replace_file(path, data)

                try:
                    with patch.object(history, "_replace", side_effect=save):
                        result = await ops.send("check files", stream=False)
                    self.assertEqual(result.status, TurnStatus.FAILED)
                    user = [{"role": "user", "content": "check files"}]
                    self.assertEqual(requests[0], user)
                    snapshot = ops.message_snapshot()
                    if scenario == "after_tools":
                        self.assertEqual(requests[1], snapshot)
                        self.assertEqual([message["role"] for message in snapshot], ["user", "assistant", "tool", "tool"])
                        self.assertEqual(json.loads(snapshot[2]["content"])["content"], "saved result")
                        self.assertFalse(json.loads(snapshot[3]["content"])["ok"])
                    else:
                        self.assertEqual(len(requests), 1)
                        self.assertEqual(snapshot, user)
                    if scenario == "save_failure":
                        self.assertIn("stage write failed", result.error)
                        self.assertIsNone(result.notice)
                        self.assertEqual(result.model_requests, 1)
                        self.assertFalse(result.external_side_effects_possible)
                finally:
                    await ops.close()
                restored = Operations(Application(**options))
                try:
                    restored.restore_session(session_id)
                    self.assertEqual(restored.message_snapshot(), snapshot)
                finally:
                    await restored.close()

        for scenario in ("first_request", "after_tools", "save_failure"):
            with self.subTest(scenario=scenario):
                await check(scenario)

    async def test_invalid_or_timed_out_title_keeps_the_saved_fallback(self):
        async def handler(request):
            nonlocal title_requests
            if json.loads(request.content)["messages"][0]["content"] == TITLE_PROMPT:
                title_requests += 1
                if output is None:
                    await asyncio.Future()
                return response({"role": "assistant", "content": output})
            return response({"role": "assistant", "content": "done"})

        with tempfile.TemporaryDirectory() as temporary:
            history = History(Path(temporary) / "sessions")
            ops = Operations(Application(
                workspace=temporary, provider="deepseek", model_id="a", catalog=CATALOG,
                connections={"deepseek": {
                    "base_url": "https://model.test", "transport": httpx.MockTransport(handler),
                }}, history=history,
            ))
            self.addAsyncCleanup(ops.close)
            title_requests = 0
            with patch("daphnis.app._TITLE_TIMEOUT", 0.05):
                for output in ("", "长" * 41, None):
                    result = await ops.send("first task", stream=False)
                    self.assertEqual(result.status, TurnStatus.COMPLETED)
                    saved = history.load(ops.session.id)
                    self.assertEqual(saved.title, "first task")
                    self.assertEqual(saved.messages.snapshot()[-1]["content"], "done")
                    await ops.new_session()
            self.assertEqual(title_requests, 3)

    async def test_cancel_title_keeps_completed_reply_and_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = History(Path(temporary) / "sessions")
            naming = asyncio.Event()

            async def handler(request):
                if json.loads(request.content)["messages"][0]["content"] == TITLE_PROMPT:
                    naming.set()
                    await asyncio.Future()
                return response({"role": "assistant", "content": "done"})

            ops = Operations(Application(
                workspace=temporary, provider="deepseek", model_id="a", catalog=CATALOG,
                connections={"deepseek": {
                    "base_url": "https://model.test", "transport": httpx.MockTransport(handler),
                }},
                history=history,
            ))
            self.addAsyncCleanup(ops.close)
            session_id = ops.session.id
            path = history.directory / f"{session_id}.json"
            task = asyncio.create_task(ops.send("first task", stream=False))
            await asyncio.wait_for(naming.wait(), 2)
            saved = history.load(session_id)
            self.assertEqual(saved.title, "first task")
            self.assertEqual(saved.messages.snapshot()[-1]["content"], "done")
            with self.assertRaises(SessionBusyError), SessionLease(path):
                self.fail("The title request should still hold the session lease.")

            await ops.cancel()
            self.assertEqual((await task).status, TurnStatus.COMPLETED)
            self.assertEqual(history.load(session_id).title, "first task")
            self.assertIsNone(ops.recovery)

    async def test_cancel_during_background_save_keeps_session_busy(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = History(Path(temporary) / "sessions")
            transport = httpx.MockTransport(
                lambda request: response({"role": "assistant", "content": "done"})
            )
            ops = Operations(Application(
                workspace=temporary, provider="deepseek", model_id="a", catalog=CATALOG,
                connections={"deepseek": {"base_url": "https://model.test", "transport": transport}},
                history=history,
            ))
            entered, release = threading.Event(), threading.Event()
            replace_file = history._replace

            def slow_replace(*args):
                entered.set()
                if not release.wait(1):
                    raise TimeoutError("The event loop did not release the save worker.")
                replace_file(*args)

            with patch.object(history, "_replace", side_effect=slow_replace):
                send = asyncio.create_task(ops.send("save", stream=False))
                try:
                    async with asyncio.timeout(3):
                        while not entered.is_set():
                            await asyncio.sleep(0.005)
                    send.cancel()
                    # A pending save may block its worker, but not the event loop.
                    for _ in range(5):
                        await asyncio.sleep(0.025)
                        self.assertFalse(send.done())
                    self.assertFalse(send.done())
                    other = History(history.directory)
                    peer = replace(ops._app._runtime.session)
                    with self.assertRaisesRegex(RuntimeError, "Session is busy"):
                        other.save(peer)
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await send
                    self.assertEqual(other.load(ops.session.id).messages.snapshot(),
                                     [{"role": "user", "content": "save"}])
                    self.assertEqual(len(ops.message_snapshot()), 1)
                finally:
                    release.set()
                    await asyncio.gather(send, return_exceptions=True)
                    await ops.close()

    async def test_approval_holds_cross_process_lease_until_cancelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = History(root / "sessions")
            tool = {"id": "edit", "type": "function", "function": {
                "name": "apply_patch", "arguments": json.dumps({
                    "patch": "--- /dev/null\n+++ b/effect\n@@ -0,0 +1 @@\n+effect\n",
                }),
            }}

            def model(request):
                if json.loads(request.content)["messages"][0]["content"] == TITLE_PROMPT:
                    return response({"role": "assistant", "content": "等待工具审批"})
                body = response({"role": "assistant", "tool_calls": [tool]}, "tool_calls").json()
                body["usage"] = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
                return httpx.Response(200, json=body)

            app = Application(
                workspace=root, provider="mimo", model_id="mimo-v2.6-flash",
                connections={"mimo": {
                    "base_url": "https://model.test",
                    "transport": httpx.MockTransport(model),
                }},
                history=history, interactive=True,
            )
            ops = Operations(app)
            peer = Operations(Application(
                workspace=root, provider="mimo", model_id="mimo-v2.6-flash",
                connections={"mimo": {
                    "base_url": "https://model.test",
                    "transport": httpx.MockTransport(lambda request: response({
                        "role": "assistant", "content": "done",
                    })),
                }},
                history=History(history.directory),
            ))
            seed = replace(app._runtime.session)
            history.save(seed)
            ops.restore_session(seed.id)
            pending = asyncio.Event()
            app.events.on("tool.approval_requested", lambda event: pending.set())
            send = asyncio.create_task(ops.send("pause", stream=False))
            child = """
import asyncio, sys
from daphnis import History, Operations
from daphnis.app import Application
import httpx

async def main():
    workspace, directory, session_id = sys.argv[1:]
    def response(request):
        raise AssertionError("busy session reached the model")
    ops = Operations(Application(
        workspace=workspace, provider="mimo", model_id="mimo-v2.6-flash",
        connections={"mimo": {"base_url": "https://model.test", "transport": httpx.MockTransport(response)}},
        history=History(directory),
    ))
    try:
        ops.restore_session(session_id)
        for attempt in (lambda: ops.send("conflict", stream=False), lambda: ops.set_thinking(None)):
            try:
                await asyncio.wait_for(attempt(), .5)
            except RuntimeError as exc:
                if "Session is busy" in str(exc):
                    continue
                raise
            raise AssertionError("expected busy session")
    finally:
        await ops.close()

asyncio.run(main())
"""
            process = None
            try:
                await asyncio.wait_for(pending.wait(), 2)
                initial = history._path(seed.id).read_bytes()
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", child, str(root), str(history.directory), seed.id,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(process.communicate(), 5)
                self.assertEqual(process.returncode, 0, stderr.decode())
                self.assertEqual(history._path(seed.id).read_bytes(), initial)
                self.assertFalse((root / "effect").exists())
                self.assertEqual((await peer.send("independent", stream=False)).status, TurnStatus.COMPLETED)
                request = ops.pending_approval
                await ops.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await send
                self.assertIsNone(ops.pending_approval)
                self.assertFalse(ops.resolve_approval(request.request_id, "allow_once"))
                saved = history.load(seed.id)
                self.assertEqual(saved.messages.get(-1)[TOOL_STATE], PENDING)
                self.assertEqual(saved.usage.total_tokens, 5)
                peer.restore_session(seed.id)
                self.assertEqual((await peer.send("retry", stream=False)).status, TurnStatus.FAILED)
                self.assertEqual((await peer.continue_interrupted_turn(peer.recovery.key, stream=False)).status, TurnStatus.COMPLETED)
                self.assertFalse((root / "effect").exists())
            finally:
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
                await ops.close()
                await peer.close()
                await asyncio.gather(send, return_exceptions=True)

    async def test_history_models_and_thinking_revalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = History(root / "sessions")
            received = []

            def handler(request):
                body = json.loads(request.content)
                if body["messages"][0]["content"] == TITLE_PROMPT:
                    return response({"role": "assistant", "content": "测试模型"})
                received.append(
                    (
                        request.url.host,
                        request.headers["authorization"],
                        body,
                    )
                )
                return response({"role": "assistant", "content": "hello"})

            class TrackedTransport(httpx.MockTransport):
                closed = False

                async def aclose(self):
                    self.closed = True
                    await super().aclose()

            transports = {
                name: TrackedTransport(handler) for name in ("deepseek", "mimo")
            }
            app = Application(
                workspace=root,
                provider="deepseek",
                model_id="a",
                catalog=CATALOG,
                connections={
                    name: {
                        "base_url": f"https://{name}.test",
                        "api_key": f"{name}-test-key",
                        "transport": transport,
                    }
                    for name, transport in transports.items()
                },
                history=history,
            )
            ops = Operations(app)
            try:
                await ops.set_thinking("high")
                with self.assertRaises(CapabilityError):
                    await ops.set_thinking("invalid")
                self.assertEqual(ops.thinking, "high")
                self.assertEqual(ops.enabled_tools, ops.all_tools)
                with self.assertRaises(ValidationError):
                    await ops.set_tools(("missing_tool",))
                self.assertEqual(ops.enabled_tools, ops.all_tools)
                await ops.set_tools(("read_file",))
                result = await ops.send("hello", stream=False)
                self.assertEqual(result.status, TurnStatus.COMPLETED)
                old = ops.session.id
                self.assertEqual(
                    history.load(old).messages.snapshot(),
                    ops.message_snapshot(),
                )
                original = ops.message_snapshot()
                detached = ops.message_snapshot()
                detached[0]["content"] = "external edit"
                self.assertEqual(ops.message_snapshot(), original)
                peer = Operations(app)
                before = (ops.session, ops.thinking, ops.enabled_tools)
                path = history.directory / f"{old}.json"
                old_file = path.read_bytes()
                for change, args in (
                    (ops.set_model, ("deepseek", "plain")),
                    (ops.set_thinking, ("off",)),
                    (ops.set_tools, ((),)),
                ):
                    with (
                        self.subTest(change=change.__name__),
                        patch(
                            "daphnis.history.os.replace",
                            side_effect=OSError("disk unavailable"),
                        ),
                    ):
                        with self.assertRaises(OSError):
                            await change(*args)
                        self.assertEqual(
                            (ops.session, ops.thinking, ops.enabled_tools), before
                        )
                        self.assertEqual(path.read_bytes(), old_file)
                fresh = await ops.new_session()
                self.assertNotEqual(fresh.id, old)
                self.assertEqual(peer.session, fresh)
                self.assertEqual(len(ops.message_snapshot()), 0)
                client = app._clients["deepseek"]
                self.assertIn("已停用工具", await ops.set_model("deepseek", "plain"))
                self.assertEqual(peer.model.id, "plain")
                self.assertEqual(ops.thinking, "high")
                self.assertEqual(ops.enabled_tools, ())
                with self.assertRaises(FileNotFoundError):
                    history.load(fresh.id)
                with self.assertRaisesRegex(CapabilityError, "unsupported"):
                    await ops.set_tools(("read_file",))
                self.assertEqual(
                    (await ops.send("same provider", stream=False)).status,
                    TurnStatus.COMPLETED,
                )
                self.assertIs(app._clients["deepseek"], client)
                await ops.set_model("mimo", "b")
                self.assertNotIn("mimo", app._clients)
                self.assertIsNone(ops.thinking)
                self.assertEqual(
                    (await ops.send("again", stream=False)).status, TurnStatus.COMPLETED
                )
                self.assertEqual(len(ops.list_sessions()), 2)
                with self.assertRaises(FileNotFoundError):
                    ops.restore_session("missing")
                self.assertEqual(ops.session.provider, "mimo")
                ops.restore_session(old)
                self.assertEqual(
                    (ops.session.provider, ops.session.model_id), ("deepseek", "a")
                )
                self.assertEqual(ops.message_snapshot(), original)
                self.assertEqual(ops.thinking, "high")
                self.assertEqual(ops.enabled_tools, ("read_file",))
                self.assertEqual(
                    (await ops.send("restored", stream=False)).status,
                    TurnStatus.COMPLETED,
                )
                self.assertIs(app._clients["deepseek"], client)
                self.assertEqual(
                    [(host, key, body["model"]) for host, key, body in received],
                    [
                        ("deepseek.test", "Bearer deepseek-test-key", "a"),
                        ("deepseek.test", "Bearer deepseek-test-key", "plain"),
                        ("mimo.test", "Bearer mimo-test-key", "b"),
                        ("deepseek.test", "Bearer deepseek-test-key", "a"),
                    ],
                )
                self.assertEqual(received[1][2]["reasoning_effort"], "high")
                self.assertNotIn("tools", received[1][2])
                self.assertNotIn("thinking", received[2][2])
                restored_request = received[3][2]
                self.assertEqual(restored_request["reasoning_effort"], "high")
                self.assertEqual(
                    [tool["function"]["name"] for tool in restored_request["tools"]],
                    ["read_file"],
                )
            finally:
                await ops.close()
            self.assertTrue(all(transport.closed for transport in transports.values()))
            with self.assertRaisesRegex(RuntimeError, "已关闭"):
                await ops.send("closed", stream=False)
            with self.assertRaisesRegex(RuntimeError, "已关闭"):
                await ops.set_thinking(None)
            with patch("daphnis.history.Path.home", return_value=root):
                self.assertEqual(
                    History().directory, root.resolve() / ".daphnis/sessions"
                )

    async def test_session_settings_survive_switch_and_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = History(root / "sessions")

            def make_ops():
                return Operations(
                    Application(
                        workspace=root,
                        provider="deepseek",
                        model_id="a",
                        catalog=CATALOG,
                        history=history,
                        connections={
                            name: {
                                "base_url": f"https://{name}.test",
                                "transport": httpx.MockTransport(
                                    lambda _: self.fail("Unexpected model request")
                                ),
                            }
                            for name in ("deepseek", "mimo")
                        },
                    )
                )

            expected = ("high", ("read_file",))
            ops = make_ops()
            try:
                session_a = ops.session
                await ops.set_thinking("high")
                await ops.set_tools(("read_file",))
                history.save(ops._app._runtime.session)
                saved_a = history.load(session_a.id)
                self.assertEqual((saved_a.thinking, saved_a.enabled_tools), expected)

                session_b = await ops.new_session()
                self.assertNotEqual(session_a.id, session_b.id)
                self.assertEqual((ops.thinking, ops.enabled_tools), expected)
                self.assertIn("已恢复服务端默认", await ops.set_model("mimo", "b"))
                self.assertIsNone(ops.thinking)
                self.assertEqual(ops.enabled_tools, ("read_file",))
                await ops.set_thinking("off")
                await ops.set_tools(("list_directory",))
                ops.restore_session(session_a.id)
                self.assertEqual(ops.session, session_a)
                self.assertEqual((ops.thinking, ops.enabled_tools), expected)
            finally:
                await ops.close()

            restarted = make_ops()
            try:
                restarted.restore_session(session_a.id)
                self.assertEqual(restarted.session, session_a)
                self.assertEqual(
                    (restarted.thinking, restarted.enabled_tools), expected
                )

                for settings, error, detail in (
                    ({"thinking": "removed-mode"}, CapabilityError, "removed-mode"),
                    (
                        {"enabled_tools": ("removed-tool",)},
                        ValidationError,
                        "removed-tool",
                    ),
                    ({"model_id": "plain"}, CapabilityError, "unsupported"),
                ):
                    with self.subTest(settings=settings):
                        invalid = replace(saved_a, id=f"invalid-{next(iter(settings))}", **settings)
                        history.save(invalid)
                        path = history.directory / f"{invalid.id}.json"
                        original = path.read_bytes()
                        with self.assertRaisesRegex(error, detail):
                            restarted.restore_session(invalid.id)
                        self.assertEqual(restarted.session, session_a)
                        self.assertEqual(
                            (restarted.thinking, restarted.enabled_tools), expected
                        )
                        self.assertEqual(path.read_bytes(), original)
            finally:
                await restarted.close()

    async def test_generic_thinking_uses_explicit_connection_compatibility(self):
        model = ModelInfo(id="local", provider="compatible", thinking_modes=("high",))
        override = Compatibility(
            field_mappings={"thinking_mode": "vendor_thinking"},
            value_mappings={"thinking_mode": {"high": {"level": 3}}},
        )
        requests = []

        def handler(request):
            body = json.loads(request.content)
            if body["messages"][0]["content"] != TITLE_PROMPT:
                requests.append(body)
            return response({"role": "assistant", "content": "done"})

        with tempfile.TemporaryDirectory() as temporary:
            for compatibility in (None, override):
                app = Application(
                    workspace=temporary,
                    provider=model.provider,
                    model_id=model.id,
                    catalog=(model,),
                    history=History(Path(temporary) / "sessions"),
                    connections={
                        model.provider: {
                            "base_url": "https://local.test",
                            "compatibility": compatibility,
                            "transport": httpx.MockTransport(handler),
                        },
                    },
                )
                ops = Operations(app)
                try:
                    await ops.set_thinking("high")
                    result = await ops.send("think", stream=False)
                finally:
                    await ops.close()
                if compatibility is None:
                    self.assertEqual(result.status, TurnStatus.FAILED)
                    self.assertIn("explicit Compatibility mapping", result.error)
                    self.assertEqual(requests, [])
                else:
                    self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                    self.assertEqual(requests[0]["vendor_thinking"], {"level": 3})
                    self.assertNotIn("thinking_mode", requests[0])

    async def test_noninteractive_denial_and_close_during_approval_are_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data.txt"
            target.write_text("old\n")
            tool = {
                "id": "call-edit",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": json.dumps(
                        {
                            "patch": (
                                "--- a/data.txt\n+++ b/data.txt\n"
                                "@@ -1 +1 @@\n-old\n+new\n"
                            )
                        }
                    ),
                },
            }

            def handler(request):
                messages = json.loads(request.content)["messages"]
                if messages[-1]["role"] == "tool":
                    self.assertEqual(
                        json.loads(messages[-1]["content"])["error"]["code"],
                        "permission_denied",
                    )
                    return response({"role": "assistant", "content": "refused"})
                return response(
                    {"role": "assistant", "content": None, "tool_calls": [tool]},
                    "tool_calls",
                )

            finished = []

            async def listener(event):
                await asyncio.sleep(0)
                finished.append(event)

            for interactive in (False, True):
                app = Application(
                    workspace=root,
                    provider="deepseek",
                    model_id="a",
                    catalog=CATALOG,
                    connections={
                        "deepseek": {
                            "base_url": "https://model.test",
                            "transport": httpx.MockTransport(handler),
                        }
                    },
                    history=History(root / f"history-{interactive}"),
                    interactive=interactive,
                )
                ops = Operations(app)
                try:
                    if not interactive:
                        result = await ops.send("edit", stream=False)
                        self.assertEqual(result.status, TurnStatus.COMPLETED)
                    else:
                        app.events.on("turn.finished", listener)
                        task = asyncio.create_task(ops.send("edit", stream=False))
                        async with asyncio.timeout(2):
                            while ops.pending_approval is None:
                                await asyncio.sleep(0)
                        pending = ops.pending_approval
                        self.assertTrue(ops.running)
                        pending.details.clear()
                        self.assertIn("targets", ops.pending_approval.details)
                        self.assertFalse(ops.resolve_approval("stale", "allow_once"))
                        for change, args in (
                            (ops.new_session, ()),
                            (ops.restore_session, ("missing",)),
                            (ops.set_model, ("deepseek", "plain")),
                            (ops.set_thinking, ("high",)),
                            (ops.set_tools, ((),)),
                            (ops.set_approval_mode, ("full",)),
                        ):
                            with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                                value = change(*args)
                                if isawaitable(value):
                                    await value
                        with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                            await ops.send("concurrent", stream=False)
                        await ops.close()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        self.assertFalse(ops.running)
                        self.assertIsNone(ops.pending_approval)
                        self.assertFalse(ops.resolve_approval(pending.request_id, "allow_once"))
                        messages = ops.message_snapshot()
                        self.assertEqual(messages[0], {"role": "user", "content": "edit"})
                        self.assertEqual(messages[-1][TOOL_STATE], PENDING)
                        self.assertEqual(len(ops.list_sessions()), 1)
                        self.assertEqual(len(finished), 1)
                        self.assertEqual(
                            finished[0].data["status"], TurnStatus.CANCELLED
                        )
                        await app.events.emit(
                            "turn.finished",
                            session_id=pending.session_id,
                            turn_id=pending.turn_id,
                        )
                        await app.events.drain()
                        self.assertEqual(len(finished), 1)
                finally:
                    await ops.close()
                self.assertEqual(target.read_text(), "old\n")


class ConfigTests(unittest.TestCase):
    def test_environment_precedence_and_provider_key_isolation(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "deepseek",
                "LLM_MODEL": "a",
                "LLM_BASE_URL": "https://default.test",
                "LLM_API_KEY": "default-test-key",
                "MIMO_BASE_URL": "https://mimo.test",
            },
            clear=True,
        ):
            provider, model, connections = environment_connections()
            self.assertEqual((provider, model), ("deepseek", "a"))
            self.assertEqual(
                connections,
                {
                    "deepseek": {
                        "base_url": "https://default.test",
                        "api_key": "default-test-key",
                    },
                    "mimo": {"base_url": "https://mimo.test", "api_key": None},
                },
            )
            with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "https://deepseek.test"}):
                _, _, connections = environment_connections()
            self.assertEqual(
                connections["deepseek"],
                {"base_url": "https://deepseek.test", "api_key": None},
            )
