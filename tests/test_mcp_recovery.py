import asyncio
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from test_mcp import DEMO, tool_call

from daphnis import History, Operations, TurnStatus
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.mcp_store import MCPServerConfig, MCPStore
from daphnis.tools.approval import ToolError


def reply(message=None):
    message = message or {"role": "assistant", "content": "Done"}
    return httpx.Response(200, json={"choices": [{"index": 0, "message": message,
        "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})


class MCPRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.options = {"command": sys.executable, "args": [str(DEMO), str(self.root)]}
        self.now = 1000.0

    def application(self, model=lambda _: reply()):
        app = Application(
            workspace=self.root, provider="deepseek", model_id="deepseek-flash",
            history=History(self.root / "sessions"),
            connections={"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(model)}},
        )
        self.addAsyncCleanup(app.close)
        return app, Operations(app)

    def configure(self, **servers):
        MCPStore(self.root / "mcp.json").write({
            name: MCPServerConfig(name, state, self.options) for name, state in servers.items()
        })

    async def test_only_always_starts_and_failed_startup_can_be_recovered_by_model(self):
        self.configure(always="always", lazy="on_demand", off="disabled")
        app, ops = self.application()
        sid = ops.initial_session_id
        fail = AsyncMock(side_effect=RuntimeError("temporary error"))
        with patch.object(app, "_start_mcp", fail), self.assertLogs("daphnis.app", "WARNING"):
            await ops.start_mcp(session_id=sid)
            self.assertEqual(app._mcp_store.attempts("always"), {})
            await ops.send("First", stream=False, session_id=sid)
            await ops.send("Second", stream=False, session_id=sid)
            await ops.start_mcp(session_id=sid)
        self.assertEqual([call.args[0].name for call in fail.await_args_list], ["always"])
        with self.assertRaises(ToolError) as disabled:
            await app._activate_mcp(sid, "global:off")
        self.assertEqual(disabled.exception.code, "mcp_not_available")
        self.assertTrue((await app._activate_mcp(sid, "global:always"))["ok"])
        self.assertIn("mcp__always__add", ops.effective_tools(sid))
        self.assertEqual(ops.list_mcp_servers(sid), ("always",))

    async def test_fault_cycle_and_window_survive_flapping(self):
        self.configure(demo="on_demand")
        app, ops = self.application()
        sid = ops.initial_session_id
        start = app._start_mcp
        fail_remaining = 3

        async def flaky(connection):
            nonlocal fail_remaining
            if fail_remaining:
                fail_remaining -= 1
                raise RuntimeError("temporary error")
            await start(connection)

        spy = AsyncMock(side_effect=flaky)
        with (
            patch("daphnis.app.monotonic", lambda: self.now),
            patch("daphnis.mcp.monotonic", lambda: self.now),
            patch("daphnis.app._MCP_RETRY_DELAYS", (0, 0, 0, 0)),
            patch.object(app, "_start_mcp", spy),
        ):
            await app._activate_mcp(sid, "global:demo")
            self.assertEqual(spy.await_count, 4)
            self.now += 20
            await self.drop(app._mcp["demo"])
            await app._activate_mcp(sid, "global:demo")
            self.assertEqual(spy.await_count, 5)
            self.now += 20
            await self.drop(app._mcp["demo"])
            self.now += 300
            with self.assertRaises(ToolError) as exhausted:
                await app._activate_mcp(sid, "global:demo")
            self.assertEqual((exhausted.exception.code, exhausted.exception.details["reason"]), ("mcp_retry_blocked", "fault_cycle"))
            await ops.send("A new turn", stream=False, session_id=sid)
            with self.assertRaises(ToolError) as still_exhausted:
                await app._activate_mcp(sid, "global:demo")
            self.assertEqual(still_exhausted.exception.code, "mcp_retry_blocked")
            self.assertEqual(spy.await_count, 5)

            await ops.restart_mcp("global:demo", session_id=sid)
            for _ in range(4):
                self.now += 30
                await self.drop(app._mcp["demo"])
                await app._activate_mcp(sid, "global:demo")
            self.now += 30
            await self.drop(app._mcp["demo"])
            with self.assertRaises(ToolError) as limited:
                await app._activate_mcp(sid, "global:demo")
            self.assertEqual((limited.exception.code, limited.exception.details["reason"]), ("mcp_retry_blocked", "time_window"))
            self.now += 180
            await app._activate_mcp(sid, "global:demo")
            self.assertEqual(spy.await_count, 11)

    async def test_session_limit_survives_reload_and_manual_retry_resets_all_sessions(self):
        self.configure(demo="on_demand", other="on_demand")
        app, ops = self.application()
        first = ops.initial_session_id
        second = (await ops.new_session(session_id=first)).id
        await ops.remember_session(first)
        await ops.remember_session(second)
        with (
            patch("daphnis.app.monotonic", lambda: self.now),
            patch("daphnis.mcp.monotonic", lambda: self.now),
        ):
            for _ in range(10):
                await app._activate_mcp(first, "global:demo")
                self.now += 301
                await self.drop(app._mcp["demo"])
            await app._activate_mcp(second, "global:demo")
            await app._activate_mcp(second, "global:other")
            await app.close()
            app, ops = self.application()
            ops.restore_session(first)
            ops.restore_session(second)
            with self.assertRaises(ToolError) as exhausted:
                await app._activate_mcp(first, "global:demo")
            self.assertEqual((exhausted.exception.code, exhausted.exception.details["reason"]), ("mcp_retry_blocked", "session_limit"))
            self.assertEqual(ops.mcp_retry_blocks(first)["demo"], "session_limit")
            self.assertIsNone(await ops.restart_mcp("global:demo", session_id=first))
            self.assertEqual(app._mcp_store.attempts("demo"), {})
            self.assertEqual(app._mcp_store.attempts("other"), {second: 1})
            self.assertNotIn("demo", ops.mcp_retry_blocks(first))
            self.now += 31
            await self.drop(app._mcp["demo"])
            await app._activate_mcp(first, "global:demo")
            self.assertEqual(app._mcp_store.attempts("demo"), {first: 1})

    async def test_concurrent_waiter_does_not_pay_for_shared_physical_attempt(self):
        self.configure(demo="on_demand")
        app, ops = self.application()
        first = ops.initial_session_id
        second = (await ops.new_session(session_id=first)).id
        start = app._start_mcp
        entered, proceed = asyncio.Event(), asyncio.Event()

        async def slow(connection):
            entered.set()
            await proceed.wait()
            await start(connection)

        spy = AsyncMock(side_effect=slow)
        with patch.object(app, "_start_mcp", spy):
            a = asyncio.create_task(app._activate_mcp(first, "global:demo"))
            await entered.wait()
            b = asyncio.create_task(app._activate_mcp(second, "global:demo"))
            await asyncio.sleep(0)
            a.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await a
            proceed.set()
            self.assertTrue((await asyncio.wait_for(b, 5))["ok"])
        self.assertEqual(spy.await_count, 1)
        self.assertEqual(app._mcp_store.attempts("demo"), {first: 1})
        self.assertNotIn("mcp__demo__add", ops.effective_tools(first))
        self.assertIn("mcp__demo__add", ops.effective_tools(second))
        self.assertEqual(app._mcp_starts, {})

        await self.drop(app._mcp["demo"])
        entered.clear()
        proceed.clear()
        with patch.object(app, "_start_mcp", spy):
            last = asyncio.create_task(app._activate_mcp(second, "global:demo"))
            await asyncio.wait_for(entered.wait(), 5)
            last.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await last
        self.assertEqual(app._mcp_starts, {})
        self.assertEqual(app._mcp_store.attempts("demo"), {first: 1, second: 1})

    async def test_tool_disconnect_and_recovery_use_updated_tools_in_same_turn(self):
        self.configure(demo="always")
        add = "mcp__demo__add"
        requests = []

        def model(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                return reply()
            requests.append({tool["function"]["name"] for tool in body.get("tools", [])})
            if len(requests) == 1:
                call = tool_call(add, "lost", a=-1, b=9)
            elif len(requests) == 2:
                self.assertEqual(json.loads(messages[-1]["content"])["error"]["code"], "mcp_disconnected")
                self.assertEqual(json.loads(messages[-1]["content"])["server_id"], "global:demo")
                call = tool_call("activate_mcp", "recover", server_id="global:demo")
            elif len(requests) == 3:
                call = tool_call(add, "after", a=20, b=22)
            else:
                self.assertEqual(json.loads(messages[-1]["content"])["structured_content"], {"sum": 42})
                return reply()
            return reply({"role": "assistant", "tool_calls": [call]})

        app, ops = self.application(model)
        sid = ops.initial_session_id
        await ops.start_mcp(session_id=sid)
        await ops.set_tools((add,), session_id=sid)
        ops.set_approval_mode("full", session_id=sid)
        task = asyncio.create_task(ops.send("Recover during the turn", stream=False, session_id=sid))
        async with asyncio.timeout(5):
            while not (self.root / "calls").exists():
                await asyncio.sleep(0.01)
        failed = app._mcp["demo"]
        await self.drop(failed)
        result = await asyncio.wait_for(task, 10)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertFalse(failed.available)
        self.assertEqual(requests[:3], [{add}, {"activate_mcp"}, {add}])
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["add -1 9", "add 20 22"])

    async def drop(self, connection):
        os.kill(int((self.root / "pid").read_text()), signal.SIGTERM)
        async with asyncio.timeout(5):
            await asyncio.shield(connection._task)
        self.assertFalse(connection.available)
