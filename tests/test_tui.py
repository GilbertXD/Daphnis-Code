import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from rich.markdown import Markdown
from rich.syntax import Syntax
from textual.widgets import (
    Button,
    Checkbox,
    OptionList,
    SelectionList,
    Static,
    TextArea,
)

from daphnis import Event, History, Operations, Session
from daphnis.app import Application
from daphnis.models import ModelInfo
from daphnis.tool_state import PENDING, TOOL_STATE
from daphnis.tools import ApprovalDecision, ApprovalRequest
from daphnis.tui.app import ChoiceScreen, DaphnisApp, SystemPromptScreen, ToolsScreen
from daphnis.tui.approval import ApprovalScreen
from daphnis.tui.chat import ChatRow, ChatView, ToolGroup, patch_counts
from daphnis.tui.input import PromptInput
from daphnis.tui.model_picker import ModelPicker
from daphnis.tui.status import UsageView
from daphnis.turn_state import ABANDONED, TURN_STATE

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
    async def test_system_prompt_dialog_saves_multiline_text_and_reset_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = controller(Path(temporary), lambda request: self.fail("Unexpected model call"))
            ops = Operations(app)
            default = ops.get_system_prompt()
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                await submit(ui, pilot, "/system")
                await until(pilot, lambda: isinstance(ui.screen, SystemPromptScreen))
                ui.screen.query_one(TextArea).load_text("First line\nSecond line")
                await pilot.click("#system-save")
                await until(pilot, lambda: ops.get_system_prompt() == "First line\nSecond line")
                await submit(ui, pilot, "/system reset")
                await until(pilot, lambda: ops.get_system_prompt() == default)

    async def test_compact_command_updates_status_without_rendering_summary_as_chat(self):
        summary = json.dumps({
            "goals": "keep the release tag", "progress": "tag recorded",
            "files": "", "next_steps": "continue",
        })

        def handler(request):
            body = json.loads(request.content)
            if body.get("stream"):
                self.assertEqual(body["messages"][1]["reasoning_content"], "")
                self.assertIn("history summary", body["messages"][1]["content"])
                return sse({"role": "assistant", "content": "continued"})
            return httpx.Response(200, json={"choices": [{
                "index": 0, "message": {"role": "assistant", "content": summary},
                "finish_reason": "stop",
            }]})

        with tempfile.TemporaryDirectory() as temporary:
            app = controller(Path(temporary), handler)
            app._session.messages.add_user("old conversation " * 180)
            app._session.messages.add_assistant("old answer")
            app._session.messages.add_user("latest task")
            app._session.messages.add_assistant("latest answer")
            ops = Operations(app)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                await submit(ui, pilot, "/compact")
                await until(pilot, lambda: ops.compacted_through == 2)
                self.assertIn("摘要至 #2", ui.query_one(UsageView).tooltip)
                await submit(ui, pilot, "continue")
                await until(pilot, lambda: len(ops.message_snapshot()) == 6)
                self.assertIn("上下文 ≈", ui.query_one(UsageView).tooltip)
                self.assertNotIn(summary, [row.text for row in ui.query(ChatRow)])

    async def test_terminal_controls_are_removed_after_streaming_and_before_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = controller(Path(temporary), lambda request: self.fail("Unexpected model call"))
            ui = DaphnisApp(Operations(app), app.events)
            async with ui.run_test(size=(100, 36)):
                chat = ui.query_one(ChatView)
                fragments = ("中文\x1b", "]52;c;PAY", "LOAD\x07**OK**\n\tfin")
                for fragment in fragments:
                    await chat.event(Event("model.delta", "session", "turn", {
                        "request_number": 1, "chunk": {"choices": [{
                            "index": 0, "delta": {"content": fragment},
                        }]},
                    }))
                row = chat.assistants["turn", 1]
                with patch("daphnis.tui.chat.Markdown", wraps=Markdown) as markdown:
                    row.redraw()
                    markdown.assert_called_once_with("中文**OK**\n\tfin", code_theme="monokai")
                self.assertEqual(row.text, "".join(fragments))

                result = "data\x90control\x9c\x1b[31m\x00\x7f!"
                await chat.event(Event("tool.finished", "session", "turn", {
                    "tool_call_id": "mcp", "request_number": 1,
                    "name": "mcp__demo__read", "status": "COMPLETED", "content": result,
                }))
                self.assertEqual(chat.tools["turn", 1, "mcp"].output, "data!")

                request = ApprovalRequest(
                    session_id="session", turn_id="turn", tool_call_id="patch",
                    tool_name="apply_patch", workspace=temporary,
                    reason="reason\x1b]52;c;PAYLOAD\x07",
                    details={
                        "patch": "+中文\x9b31m\x7f\n",
                        "arg": "中文\x9d52;c;PAYLOAD",
                        "other": "still visible",
                    },
                )
                screen = ApprovalScreen(request)
                with patch("daphnis.tui.approval.Syntax", wraps=Syntax) as syntax:
                    await ui.push_screen(screen)
                    self.assertEqual(syntax.call_args.args[0], "+中文\n")
                self.assertEqual(str(screen.query_one("#approval-reason", Static).content), "reason")
                details = str(screen.query_one("#approval-details", Static).content)
                self.assertIn("中文", details)
                self.assertIn("still visible", details)
                self.assertNotIn("\x9d", details)
                self.assertEqual(request.details["patch"], "+中文\x9b31m\x7f\n")

    async def test_budget_pause_preserves_results_and_displays_continue_notice(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertIn("tools", body)
            if body["messages"][-1]["role"] == "tool":
                return sse({"role": "assistant", "content": "已恢复任务"})
            self.assertEqual(body["messages"][-1]["role"], "user")
            return sse({"role": "assistant", "tool_calls": [{
                "id": "list-1", "type": "function",
                "function": {"name": "list_directory", "arguments": "{}"},
            }]}, "tool_calls")

        with tempfile.TemporaryDirectory() as temporary:
            app = controller(Path(temporary), handler)
            app._request_limits.update(max_primary_requests=1, max_total_requests=2)
            ops = Operations(app)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                await submit(ui, pilot, "检查工作区")
                await until(pilot, lambda: any("主请求上限（1/1）" in row.text for row in ui.query(ChatRow)))
                self.assertEqual([item["role"] for item in ops.message_snapshot()], ["user", "assistant", "tool"])
                self.assertFalse(ui.query_one(PromptInput).disabled)
                self.assertTrue(any("/continue" in row.text for row in ui.query(ChatRow)))
                await submit(ui, pilot, "/continue")
                await until(pilot, lambda: not ui._sending)
                self.assertEqual([item["role"] for item in ops.message_snapshot()], ["user", "assistant", "tool", "assistant"])
                self.assertTrue(any(row.text == "已恢复任务" for row in ui.query(ChatRow)))

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
                self.assertFalse(ui.screen.query("#approval-trust"))
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
                messages = ops.message_snapshot()
                self.assertEqual(messages[:-2], [*committed, {"role": "user", "content": "wait then cancel"}])
                self.assertEqual(messages[-1][TOOL_STATE], PENDING)
                self.assertFalse(ui.query_one(PromptInput).disabled)

    async def test_approval_trust_checkbox_requires_allow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = controller(root, lambda request: self.fail("No model call"))
            ui = DaphnisApp(Operations(app), app.events)
            request = ApprovalRequest(
                session_id="session",
                turn_id="turn",
                tool_call_id="call",
                tool_name="mcp__demo__read",
                workspace=temporary,
                reason="MCP 工具执行确认",
                details={
                    "server": "demo",
                    "tool": "read",
                    "arguments": {"repoName": "example/project"},
                },
                can_trust=True,
            )
            decisions = []
            async with ui.run_test(size=(100, 36)) as pilot:
                for keys, expected in (
                    (("y",), ApprovalDecision.ALLOW_ONCE),
                    (("t", "t", "y"), ApprovalDecision.ALLOW_ONCE),
                    (("t", "y"), ApprovalDecision.TRUST_SESSION),
                    (("t", "n"), ApprovalDecision.DENY),
                    (("t", "escape"), ApprovalDecision.DENY),
                ):
                    with self.subTest(keys=keys):
                        screen = ApprovalScreen(request)
                        await ui.push_screen(screen, decisions.append)
                        checkbox = screen.query_one("#approval-trust", Checkbox)
                        self.assertFalse(checkbox.value)
                        self.assertEqual(len(screen.query(Button)), 2)
                        note = screen.query_one("#approval-trust-note", Static)
                        self.assertIn("不同的参数", str(note.render()))
                        for key in keys[:-1]:
                            previous = checkbox.value
                            await pilot.press(key)
                            self.assertIs(ui.screen, screen)
                            self.assertNotEqual(checkbox.value, previous)
                        await pilot.press(keys[-1])
                        await until(pilot, lambda screen=screen: ui.screen is not screen)
                        self.assertIs(decisions[-1], expected)

    async def test_tools_selection_can_remove_offline_preferences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = controller(root, lambda request: self.fail("No model call"))
            ops = Operations(app)
            remove, keep = "mcp__demo__read", "mcp__demo__search"
            session = Session(
                workspace=root,
                provider="deepseek",
                model_id="a",
                enabled_tools=("read_file", remove, keep),
            )
            app._history.save(session)
            ops.restore_session(session.id)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                await submit(ui, pilot, "/tools")
                await until(pilot, lambda: isinstance(ui.screen, ToolsScreen))
                choices = ui.screen.query_one(SelectionList)
                self.assertEqual(set(choices.selected), set(session.enabled_tools))
                choices.focus()
                choices.highlighted = len(ops.all_tools)
                option = choices.get_option_at_index(choices.highlighted)
                self.assertIn("离线", str(option.prompt))
                await pilot.press("space")
                await pilot.click("#tools-apply")
                await until(pilot, lambda: ops.enabled_tools == ("read_file", keep))
                self.assertEqual(ops.effective_tools, ("read_file",))
                self.assertEqual(
                    app._history.load(session.id).enabled_tools, ops.enabled_tools
                )
                await submit(ui, pilot, "/tools")
                await until(pilot, lambda: isinstance(ui.screen, ToolsScreen))
                choices = ui.screen.query_one(SelectionList)
                self.assertEqual(choices.option_count, len(ops.all_tools) + 1)

    async def test_provider_columns_thinking_and_tool_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = controller(
                root, lambda request: sse({"role": "assistant", "content": "hello"})
            )
            ops = Operations(app)
            await ops.set_thinking("high")
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

    async def test_tool_ids_reused_across_requests_have_separate_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for path in ("first.txt", "second.txt"):
                (root / path).write_text(path)

            def handler(request):
                messages = json.loads(request.content)["messages"]
                count = sum(message["role"] == "tool" for message in messages)
                if count == 2:
                    return sse({"role": "assistant", "content": "done"})
                name = ("first.txt", "second.txt")[count]
                return sse({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "reused", "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": name})},
                    }],
                }, "tool_calls")

            app = controller(root, handler)
            ops = Operations(app)
            ui = DaphnisApp(ops, app.events)
            async with ui.run_test(size=(100, 36)) as pilot:
                await submit(ui, pilot, "read both")
                await until(pilot, lambda: bool(ops.message_snapshot()) and not ui._sending)
                chat = ui.query_one(ChatView)
                rows = list(chat.query("ChatRow.tool"))
                self.assertEqual([Path(row.text).name for row in rows], ["first.txt", "second.txt"])
                await chat.restore(ops.message_snapshot())
                restored = list(chat.query("ChatRow.tool"))
                self.assertEqual([Path(row.text).name for row in restored], ["first.txt", "second.txt"])

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
                    chat.tools[("turn", 0, "run")].output, "actual tool output"
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
                    "invalid arguments", chat.tools[("turn", 0, "invalid")].outcome
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
                self.assertEqual(ops.message_snapshot(), [{"role": "user", "content": "wait"}])
                self.assertTrue(any("/retry" in row.outcome for row in ui.query(ChatRow)))
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
                self.assertEqual(ops.message_snapshot(), [
                    {"role": "user", "content": "wait", TURN_STATE: ABANDONED},
                    {"role": "user", "content": "wait then quit"},
                ])
