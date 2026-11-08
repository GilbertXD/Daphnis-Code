import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
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
                ops.set_thinking("high")
                with self.assertRaises(CapabilityError):
                    ops.set_thinking("invalid")
                self.assertEqual(ops.thinking, "high")
                self.assertEqual(ops.enabled_tools, ops.all_tools)
                with self.assertRaises(ValidationError):
                    ops.set_tools(("missing_tool",))
                self.assertEqual(ops.enabled_tools, ops.all_tools)
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
                with (
                    patch.object(
                        history, "save", side_effect=OSError("disk unavailable")
                    ),
                    self.assertRaises(OSError),
                ):
                    ops.set_model("deepseek", "plain")
                self.assertEqual((ops.session, ops.thinking, ops.enabled_tools), before)
                self.assertEqual(history.load(old).model_id, "a")
                fresh = ops.new_session()
                self.assertNotEqual(fresh.id, old)
                self.assertEqual(peer.session, fresh)
                self.assertEqual(len(ops.message_snapshot()), 0)
                client = app._clients["deepseek"]
                ops.set_model("deepseek", "plain")
                self.assertEqual(peer.model.id, "plain")
                self.assertEqual(ops.enabled_tools, ())
                with self.assertRaisesRegex(CapabilityError, "unsupported"):
                    ops.set_tools(("read_file",))
                self.assertEqual(
                    (await ops.send("same provider", stream=False)).status,
                    TurnStatus.COMPLETED,
                )
                self.assertIs(app._clients["deepseek"], client)
                ops.set_model("mimo", "b")
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
                ops.set_tools(())
                self.assertEqual(ops.enabled_tools, ())
                ops.set_tools(("read_file",))
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
            finally:
                await ops.close()
            self.assertTrue(all(transport.closed for transport in transports.values()))
            with self.assertRaisesRegex(RuntimeError, "已关闭"):
                await ops.send("closed", stream=False)
            with self.assertRaisesRegex(RuntimeError, "已关闭"):
                ops.set_thinking(None)
            with patch("daphnis.history.Path.home", return_value=root):
                self.assertEqual(
                    History().directory, root.resolve() / ".daphnis/sessions"
                )

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
                    ops.set_thinking("high")
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
                        self.assertFalse(
                            ops.resolve_approval("stale", pending.tool_call_id, True)
                        )
                        self.assertFalse(
                            ops.resolve_approval(pending.turn_id, "stale", True)
                        )
                        for change, args in (
                            (ops.new_session, ()),
                            (ops.restore_session, ("missing",)),
                            (ops.set_model, ("deepseek", "plain")),
                            (ops.set_thinking, ("high",)),
                            (ops.set_tools, ((),)),
                        ):
                            with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                                change(*args)
                        with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                            await ops.send("concurrent", stream=False)
                        await ops.close()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        self.assertFalse(ops.running)
                        self.assertIsNone(ops.pending_approval)
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
                bodies.append(json.loads(request.content))
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

                def run(*args):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        patch("sys.argv", ["daphnis", *args]),
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        self.assertEqual(cli.main(), 0)
                    self.assertEqual(stderr.getvalue(), "")
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
