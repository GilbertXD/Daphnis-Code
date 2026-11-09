import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from textual.widgets import OptionList, SelectionList

from daphnis import Event, History, Operations
from daphnis.app import Application
from daphnis.models import ModelInfo
from daphnis.tui.app import ChoiceScreen, DaphnisApp, ToolsScreen
from daphnis.tui.approval import ApprovalScreen
from daphnis.tui.chat import ChatRow, ChatView, ToolGroup, patch_counts
from daphnis.tui.input import PromptInput
from daphnis.tui.model_picker import ModelPicker
from daphnis.tui.status import UsageView

CATALOG = (
    ModelInfo(
        id="a", provider="deepseek", supports_tools=True, thinking_modes=("off", "high")
    ),
    ModelInfo(
        id="b", provider="mimo", supports_tools=True, thinking_modes=("off", "on")
    ),
)


def sse(message, finish="stop", usage=None):
    delta = dict(message)
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {**call, "index": index} for index, call in enumerate(message["tool_calls"])
        ]
    chunks = [{"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}]
    if usage is not None:
        chunks.append({"choices": [], "usage": usage})
    data = (
        "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + "data: [DONE]\n\n"
    )
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=data
    )


def controller(root, handler):
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
        interactive=True,
    )


async def until(pilot, predicate):
    async with asyncio.timeout(4):
        while not predicate():
            await pilot.pause(0.02)
    await pilot.pause()


async def submit(ui, pilot, text):
    ui.query_one(PromptInput).load_text(text)
    await pilot.press("enter")


class TUITests(unittest.IsolatedAsyncioTestCase):
    async def test_input_stream_clear_new_and_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = controller(
                root,
                lambda request: sse(
                    {"role": "assistant", "content": "hello"},
                    usage={
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                    },
                ),
            )
            ops = Operations(app)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                prompt = ui.query_one(PromptInput)
                prompt.load_text("line")
                prompt.move_cursor((0, 4))
                await pilot.press("shift+enter")
                self.assertEqual(prompt.text, "line\n")
                await pilot.press("f2")
                self.assertEqual(prompt.text, "line\n\n")
                await submit(ui, pilot, "hello")
                await until(
                    pilot, lambda: len(ops.message_snapshot()) == 2 and not ui._sending
                )
                chat = ui.query_one(ChatView)
                self.assertEqual(len(chat.assistants), 1)
                self.assertEqual(next(iter(chat.assistants.values())).text, "hello")
                usage = str(ui.query_one(UsageView).render())
                self.assertIn("本轮 in 12 out 8 total 20", usage)
                self.assertIn("会话 in 12 out 8 total 20", usage)
                old = ops.session.id
                await submit(ui, pilot, "/thinking high")
                await until(pilot, lambda: ops.thinking == "high")
                await submit(ui, pilot, "/tools none")
                await until(pilot, lambda: not ops.enabled_tools)
                await submit(ui, pilot, "/clear")
                await pilot.pause()
                self.assertEqual(len(ops.message_snapshot()), 2)
                await submit(ui, pilot, "/new")
                await until(pilot, lambda: ops.session.id != old)
                self.assertEqual(len(ops.message_snapshot()), 0)
                usage = str(ui.query_one(UsageView).render())
                self.assertIn("会话 in 0 out 0 total 0", usage)
                self.assertIn("思考 high", usage)
                self.assertIn("0 tools", usage)
                await submit(ui, pilot, "/thinking default")
                await until(pilot, lambda: ops.thinking is None)
                await submit(ui, pilot, "/tools all")
                await until(pilot, lambda: ops.enabled_tools == ops.all_tools)
                await submit(ui, pilot, "/session")
                await until(pilot, lambda: isinstance(ui.screen, ChoiceScreen))
                sessions = ui.screen.query_one(OptionList)
                sessions.focus()
                sessions.highlighted = list(ui.screen.values.values()).index(old)
                await pilot.press("enter")
                await until(pilot, lambda: ops.session.id == old)
                usage = str(ui.query_one(UsageView).render())
                self.assertIn("会话 in 12 out 8 total 20", usage)
                self.assertIn("思考 high", usage)
                self.assertIn("0 tools", usage)
                self.assertTrue(
                    any(
                        row.text == "hello" and row.title == "助手"
                        for row in chat.query(ChatRow)
                    )
                )
                async with asyncio.timeout(2):
                    await pilot.press("ctrl+c")
                self.assertTrue(app._closed)

    async def test_approval_yes_no_and_cancel_waiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data.txt"
            target.write_text("old\n")

            def handler(request):
                messages = json.loads(request.content)["messages"]
                if messages[-1]["role"] == "tool":
                    return sse({"role": "assistant", "content": "done"})
                call = {
                    "id": "edit",
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
                return sse(
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    "tool_calls",
                )

            app = controller(root, handler)
            ops = Operations(app)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(110, 40)) as pilot:
                await submit(ui, pilot, "edit")
                await until(pilot, lambda: isinstance(ui.screen, ApprovalScreen))
                self.assertEqual(ui.focused.id, "approval-no")
                await pilot.press("n")
                await until(pilot, lambda: not ui._sending)
                self.assertEqual(target.read_text(), "old\n")
                await submit(ui, pilot, "edit again")
                await until(pilot, lambda: isinstance(ui.screen, ApprovalScreen))
                await pilot.press("y")
                await until(pilot, lambda: not ui._sending)
                self.assertEqual(target.read_text(), "new\n")
                committed = ops.message_snapshot()
                await submit(ui, pilot, "/approval ask")
                await until(pilot, lambda: ops.approval_mode == "ask")
                await submit(ui, pilot, "wait then cancel")
                await until(pilot, lambda: isinstance(ui.screen, ApprovalScreen))
                self.assertFalse(ops.pending_approval.can_trust)
                self.assertFalse(ui.screen.query("#approval-trust"))
                await pilot.press("ctrl+c")
                await until(
                    pilot,
                    lambda: (
                        not ui._sending and not isinstance(ui.screen, ApprovalScreen)
                    ),
                )
                self.assertIsNone(ops.pending_approval)
                self.assertEqual(ops.message_snapshot(), committed)
                self.assertFalse(ui.query_one(PromptInput).disabled)

    async def test_provider_columns_thinking_and_tool_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = controller(
                root, lambda request: sse({"role": "assistant", "content": "hello"})
            )
            ops = Operations(app)
            ops.set_thinking("high")
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                await submit(ui, pilot, "/model")
                await until(pilot, lambda: bool(ui.query(ModelPicker)))
                picker = ui.query_one(ModelPicker)
                self.assertNotIsInstance(ui.screen, ChoiceScreen)
                self.assertEqual(set(picker.groups), {"deepseek", "mimo"})
                options = picker.query_one("#models-1", OptionList)
                options.focus()
                options.highlighted = 0
                await pilot.press("enter")
                await until(pilot, lambda: ops.session.provider == "mimo")
                self.assertIsNone(ops.thinking)
                await submit(ui, pilot, "/model 1")
                await until(pilot, lambda: ops.session.provider == "deepseek")
                await submit(ui, pilot, "/model b")
                await until(pilot, lambda: ops.session.provider == "mimo")
                await submit(ui, pilot, "/thinking")
                await until(pilot, lambda: isinstance(ui.screen, ChoiceScreen))
                self.assertNotIn("high", ui.screen.values.values())
                thinking = ui.screen.query_one(OptionList)
                thinking.focus()
                thinking.highlighted = list(ui.screen.values.values()).index("on")
                await pilot.press("enter")
                await until(pilot, lambda: ops.thinking == "on")
                self.assertIn("on", str(ui.query_one(UsageView).render()))
                await submit(ui, pilot, "/tools")
                await until(pilot, lambda: isinstance(ui.screen, ToolsScreen))
                ui.screen.query_one(SelectionList).deselect_all()
                await pilot.click("#tools-apply")
                self.assertEqual(ops.enabled_tools, ())

    async def test_tool_folding_expands_errors_and_shows_real_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = controller(
                Path(temporary), lambda request: self.fail("No model call")
            )
            ops = Operations(app)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                chat = ui.query_one(ChatView)
                for index in range(6):
                    data = {
                        "name": "read_file",
                        "tool_call_id": str(index),
                        "arguments": json.dumps({"path": f"file-{index}.txt"}),
                    }
                    await chat.event(
                        Event("tool.started", ops.session.id, "turn", data)
                    )
                    await chat.event(
                        Event(
                            "tool.finished",
                            ops.session.id,
                            "turn",
                            {**data, "status": "COMPLETED"},
                        )
                    )
                group = chat.query_one(ToolGroup)
                self.assertEqual(group.more.title, "3 more operations")
                self.assertTrue(group.more.collapsed)
                await pilot.pause()
                await pilot.click("CollapsibleTitle")
                self.assertFalse(group.more.collapsed)
                group.more.collapsed = True
                await chat.event(
                    Event(
                        "tool.finished",
                        ops.session.id,
                        "turn",
                        {
                            "name": "read_file",
                            "tool_call_id": "5",
                            "status": "ERROR",
                            "error": {"message": "file missing"},
                        },
                    )
                )
                self.assertFalse(group.more.collapsed)
                await chat.event(
                    Event(
                        "tool.started",
                        ops.session.id,
                        "turn",
                        {
                            "name": "run_command",
                            "tool_call_id": "run",
                            "arguments": json.dumps({"argv": ["example", "--version"]}),
                        },
                    )
                )
                await chat.event(
                    Event(
                        "tool.finished",
                        ops.session.id,
                        "turn",
                        {
                            "name": "run_command",
                            "tool_call_id": "run",
                            "status": "COMPLETED",
                            "content": json.dumps(
                                {"ok": True, "stdout": "actual tool output\n"}
                            ),
                        },
                    )
                )
                self.assertEqual(
                    chat.tools[("turn", "run")].output, "actual tool output"
                )
                await chat.event(
                    Event(
                        "tool.started",
                        ops.session.id,
                        "turn",
                        {
                            "name": "apply_patch",
                            "tool_call_id": "invalid",
                            "arguments": '{"patch": null}',
                        },
                    )
                )
                await chat.event(
                    Event(
                        "tool.finished",
                        ops.session.id,
                        "turn",
                        {
                            "name": "apply_patch",
                            "tool_call_id": "invalid",
                            "status": "ERROR",
                            "error": {"message": "invalid arguments"},
                        },
                    )
                )
                self.assertIn(
                    "invalid arguments", chat.tools[("turn", "invalid")].outcome
                )
                self.assertEqual(
                    patch_counts(
                        json.dumps(
                            {
                                "patch": (
                                    "--- a/a.txt\n+++ b/a.txt\n"
                                    "@@ -1 +1,2 @@\n-old\n+++ new\n+more\n"
                                )
                            }
                        )
                    ),
                    (2, 1),
                )

    async def test_escape_cancels_running_request_and_returns_to_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            entered = asyncio.Event()

            async def handler(request):
                entered.set()
                await asyncio.Event().wait()

            app = controller(Path(temporary), handler)
            ops = Operations(app)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(80, 24)) as pilot:
                await submit(ui, pilot, "wait")
                await asyncio.wait_for(entered.wait(), 2)
                await pilot.press("escape")
                await until(pilot, lambda: not ui._sending)
                self.assertEqual(len(ops.message_snapshot()), 0)
                self.assertFalse(app._closed)
                self.assertFalse(ui.query_one(PromptInput).disabled)
                await pilot.press("escape")
                self.assertIs(ui.focused, ui.query_one(PromptInput))
                entered.clear()
                await submit(ui, pilot, "wait then quit")
                await asyncio.wait_for(entered.wait(), 2)
                async with asyncio.timeout(2):
                    await pilot.press("ctrl+q")
                self.assertTrue(app._closed)
                self.assertFalse(ops.running)
                self.assertEqual(ops.message_snapshot(), [])
