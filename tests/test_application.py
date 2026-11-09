import asyncio
import contextlib
import io
import json
import logging
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
from daphnis.app import Application
from daphnis.config import environment_connections
from daphnis.models import CapabilityError, ModelInfo

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
                await a.set_thinking("high")
                b.restore_session(a.session.id)
                await a.send("first writer", stream=False)
                with self.assertRaisesRegex(RuntimeError, "Session write conflict"):
                    await b.send("stale writer", stream=False)
                self.assertEqual(len(model_calls), 1)
                saved = history.load(a.session.id).messages.snapshot()
                self.assertEqual(saved[0]["content"], "first writer")
                self.assertEqual(len(saved), 2)
            finally:
                await a.close()
                await b.close()

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
                    peer = replace(ops._app._session)
                    with self.assertRaisesRegex(RuntimeError, "Session is busy"):
                        other.save(peer)
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await send
                    self.assertEqual(len(other.load(ops.session.id).messages), 2)
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
            seed = replace(app._session)
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
                self.assertEqual(len(saved.messages), 0)
                self.assertEqual(saved.usage.total_tokens, 5)
                peer.restore_session(seed.id)
                self.assertEqual((await peer.send("retry", stream=False)).status, TurnStatus.COMPLETED)
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
                received.append(
                    (
                        request.url.host,
                        request.headers["authorization"],
                        json.loads(request.content),
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
                self.assertEqual(history.load(fresh.id).enabled_tools, ())
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
                self.assertEqual(history.load(session_a.id).thinking, "high")
                await ops.set_tools(("read_file",))
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
            requests.append(json.loads(request.content))
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
                        self.assertEqual(len(ops.message_snapshot()), 0)
                        self.assertEqual(ops.list_sessions(), ())
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


class CLITests(unittest.TestCase):
    def test_prompt_entry_prints_only_final_answer_and_saves_session(self):
        from daphnis import cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bodies = []

            def handler(request):
                body = json.loads(request.content)
                bodies.append(body)
                if body["messages"][-1]["content"] == "empty answer":
                    return response({"role": "assistant", "content": ""})
                if body["messages"][-1]["content"] == "terminal payload":
                    return response({
                        "role": "assistant",
                        "content": "中文\x1b]52;c;PAYLOAD\x07\x9b31m**OK**\x7f\n\tfin",
                    })
                if body["messages"][-1]["content"] == "terminal error":
                    logging.getLogger("daphnis.events").warning("warning\x1b]52;c;PAYLOAD\x07")
                    raise RuntimeError("failure\x9d52;c;PAYLOAD\x9c")
                return response({"role": "assistant", "content": "CLI ready"})

            connections = {
                "deepseek": {
                    "base_url": "https://model.test",
                    "transport": httpx.MockTransport(handler),
                }
            }

            def factory(**kwargs):
                return Application(
                    **kwargs, history=History(root / "sessions"), catalog=CATALOG
                )

            history = History(root / "sessions")
            with (
                patch.object(
                    cli,
                    "environment_connections",
                    return_value=("deepseek", "a", connections),
                ),
                patch.object(cli, "Application", side_effect=factory),
                patch.object(cli, "load_dotenv"),
            ):

                def run(*args, error=None):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        patch("sys.argv", ["daphnis", *args]),
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        self.assertEqual(cli.main(), 0 if error is None else 1)
                    if error is None:
                        self.assertEqual(stderr.getvalue(), "")
                    else:
                        self.assertIn(error, stderr.getvalue())
                        self.assertNotIn("PAYLOAD", stderr.getvalue())
                    return stdout.getvalue()

                self.assertEqual(
                    run("--workspace", str(root), "-p", "hello"), "CLI ready\n"
                )
                (saved,) = history.list_sessions()
                self.assertEqual(
                    run("--session", saved.id, "--thinking", "high", "-p", "continue"),
                    "CLI ready\n",
                )
                self.assertEqual(bodies[-1]["reasoning_effort"], "high")
                self.assertEqual(bodies[-1]["messages"][-3]["content"], "hello")
                self.assertIn("deepseek: 1. a", run("-p", "/model list"))
                (updated,) = history.list_sessions()
                self.assertEqual(updated.id, saved.id)
                self.assertEqual(updated.message_count, 4)
                self.assertEqual(history.load(updated.id).thinking, "high")
                self.assertEqual(
                    run(
                        "--session", saved.id, "-p", "empty answer",
                        error="no visible text or refusal",
                    ),
                    "",
                )
                self.assertEqual(len(history.load(saved.id).messages), 4)
                self.assertEqual(
                    run("--session", saved.id, "-p", "terminal payload"),
                    "中文**OK**\n\tfin\n",
                )
                self.assertIn(
                    "\x1b]52;c;PAYLOAD\x07",
                    history.load(saved.id).messages.snapshot()[-1]["content"],
                )
                self.assertEqual(run("--session", saved.id, "-p", "terminal error",
                                     error="failure"), "")
