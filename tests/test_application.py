import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from daphnis import History, TurnStatus
from daphnis.app import Application
from daphnis.models import ModelInfo

CATALOG = (
    ModelInfo(
        id="a", provider="deepseek", supports_tools=True, thinking_modes=("off", "high")
    ),
    ModelInfo(
        id="b", provider="mimo", supports_tools=True, thinking_modes=("off", "on")
    ),
)


def response(message, finish="stop"):
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": message, "finish_reason": finish}]},
    )


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_models_commands_and_thinking_revalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = History(root / "sessions")
            received = []

            def handler(request):
                received.append(
                    (request.headers["authorization"], json.loads(request.content))
                )
                return response({"role": "assistant", "content": "hello"})

            app = Application(
                workspace=root,
                provider="deepseek",
                model_id="a",
                catalog=CATALOG,
                connections={
                    name: {
                        "base_url": f"https://{name}.test",
                        "api_key": f"{name}-test-key",
                        "transport": httpx.MockTransport(handler),
                    }
                    for name in ("deepseek", "mimo")
                },
                history=history,
            )
            try:
                app.set_thinking("high")
                result = await app.send("hello", stream=False)
                self.assertEqual(result.status, TurnStatus.COMPLETED)
                old = app.session.id
                self.assertEqual(
                    history.load(old).messages.snapshot(),
                    app.session.messages.snapshot(),
                )
                app.set_model("mimo", "b")
                self.assertIsNone(app.thinking)
                await app.send("again", stream=False)
                self.assertEqual(
                    [item[0] for item in received],
                    ["Bearer deepseek-test-key", "Bearer mimo-test-key"],
                )
                self.assertNotIn("thinking", received[1][1])
                self.assertEqual((await app.command("/clear")).action, "clear")
                count = len(app.session.messages)
                await app.command("/new")
                self.assertEqual(len(app.session.messages), 0)
                self.assertEqual(len(app.sessions()), 2)
                await app.command(f"/session {old}")
                self.assertEqual(len(app.session.messages), count)
                await app.command("/tools none")
                self.assertEqual(app.enabled_tools, ())
                await app.command("/tools on read_file")
                self.assertEqual(app.enabled_tools, ("read_file",))
            finally:
                await app.close()
            with patch("daphnis.history.Path.home", return_value=root):
                self.assertEqual(
                    History().directory, root.resolve() / ".daphnis/sessions"
                )

    async def test_noninteractive_denial_and_interactive_cancel_are_safe(self):
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
                try:
                    if not interactive:
                        result = await app.send("edit", stream=False)
                        self.assertEqual(result.status, TurnStatus.COMPLETED)
                    else:
                        task = asyncio.create_task(app.send("edit", stream=False))
                        async with asyncio.timeout(2):
                            while app.pending_approval is None:
                                await asyncio.sleep(0)
                        pending = app.pending_approval
                        self.assertFalse(
                            app.resolve_approval("stale", pending.tool_call_id, True)
                        )
                        with self.assertRaises(RuntimeError):
                            app.new_session()
                        await app.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        self.assertIsNone(app.pending_approval)
                        self.assertEqual(len(app.session.messages), 0)
                        self.assertEqual(app.sessions(), ())
                finally:
                    await app.close()
                self.assertEqual(target.read_text(), "old\n")


class CLITests(unittest.TestCase):
    def test_prompt_entry_prints_only_final_answer_and_saves_session(self):
        from daphnis import cli

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            connections = {
                "deepseek": {
                    "base_url": "https://model.test",
                    "transport": httpx.MockTransport(
                        lambda request: response(
                            {"role": "assistant", "content": "CLI ready"}
                        )
                    ),
                }
            }

            def factory(**kwargs):
                return Application(
                    **kwargs, history=History(root / "sessions"), catalog=CATALOG
                )

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.object(
                    cli,
                    "environment_connections",
                    return_value=("deepseek", "a", connections),
                ),
                patch.object(cli, "Application", side_effect=factory),
                patch.object(cli, "load_dotenv"),
                patch("sys.argv", ["daphnis", "--workspace", str(root), "-p", "hello"]),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertEqual(cli.main(), 0)
            self.assertEqual(stdout.getvalue(), "CLI ready\n")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(len(History(root / "sessions").list_sessions()), 1)
