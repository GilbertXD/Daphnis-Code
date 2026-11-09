import asyncio
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from daphnis import History, Operations, TurnStatus
from daphnis.app import Application
from daphnis.mcp import MCPConnection
from daphnis.tools import ApprovalDecision

DEMO = Path(__file__).with_name("mcp_demo_server.py").resolve()


def tool_call(name, call_id, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_exit_after_discovery_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = root / "broken"
            broken.mkdir()
            app = Application(
                workspace=root,
                provider="mimo",
                model_id="mimo-v2.6-flash",
                connections={"mimo": {"base_url": "https://model.test"}},
                history=History(root / "sessions"),
            )
            ops = Operations(app)
            start = MCPConnection.start

            async def exit_after_discovery(connection, *callbacks):
                tools = await start(connection, *callbacks)
                connection._stop.set()  # Same wakeup used by transport EOF.
                await connection._task
                return tools

            try:
                await ops.connect_mcp("healthy", sys.executable, [str(DEMO), directory])
                available = ops.all_tools
                with (
                    patch.object(MCPConnection, "start", exit_after_discovery),
                    self.assertRaisesRegex(RuntimeError, "意外中断"),
                ):
                    await ops.connect_mcp("broken", sys.executable, [str(DEMO), str(broken)])
                statuses = {status.name: status for status in ops.mcp_status()}
                self.assertEqual(statuses["healthy"].state, "Connected")
                self.assertEqual(statuses["broken"].state, "Error")
                self.assertIn("mcp__healthy__add", ops.list_mcp_tools())
                self.assertEqual(ops.all_tools, available)
            finally:
                await ops.close()

    async def test_stdio_tools_approval_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = asyncio.Queue()
            outcomes = []
            result_bytes = []
            calls = []
            advertised = set()

            def model(request):
                body = json.loads(request.content)
                messages = body["messages"]
                advertised.update(tool["function"]["name"] for tool in body["tools"])
                if messages[-1]["role"] == "tool":
                    result_bytes[:] = [
                        len(item["content"].encode()) for item in messages
                        if item["role"] == "tool"
                    ]
                    outcomes[:] = [
                        json.loads(item["content"])
                        for item in messages
                        if item["role"] == "tool"
                    ]
                    message = {"role": "assistant", "content": "done"}
                    reason = "stop"
                else:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": calls,
                    }
                    reason = "tool_calls"
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": reason}
                        ]
                    },
                )

            app = Application(
                workspace=root,
                provider="mimo",
                model_id="mimo-v2.6-flash",
                history=History(root / "sessions"),
                connections={
                    "mimo": {
                        "base_url": "https://model.test",
                        "transport": httpx.MockTransport(model),
                    }
                },
                interactive=True,
            )
            app.events.on("tool.approval_requested", requests.put_nowait)
            ops = Operations(app)
            log = root / "calls"

            async def approve(
                call_id, decision=ApprovalDecision.ALLOW_ONCE, *, server="demo"
            ):
                async with asyncio.timeout(5):
                    event = await requests.get()
                pending = ops.pending_approval
                self.assertEqual(event.data["tool_call_id"], call_id)
                self.assertEqual(pending.details["server"], server)
                self.assertTrue(
                    ops.resolve_approval(pending.request_id, decision)
                )
                return pending.details

            try:
                with patch.dict(os.environ, {"LLM_API_KEY": "synthetic-test-key"}):
                    await ops.connect_mcp(
                        "demo", sys.executable, [str(DEMO), directory]
                    )
                add, special = ops.list_mcp_tools()
                self.assertEqual(add, "mcp__demo__add")
                self.assertRegex(special, r"^mcp__demo____[0-9a-f]{32}$")
                self.assertTrue({add, special}.issubset(ops.enabled_tools))
                self.assertEqual(ops.list_mcp_servers(), ("demo",))
                await ops.set_tools(("list_directory", add, special))
                saved = ops.session.id
                calls[:] = [
                    tool_call("list_directory", "builtin"),
                    tool_call(add, "approved", a=12, b=30),
                    tool_call(add, "trusted", a=1, b=2),
                    tool_call(special, "denied", a=5, b=9),
                    tool_call(special, "error", a=-2, b=0),
                    tool_call(add, "large-text", a=-3, b=0),
                    tool_call(add, "large-structure", a=-4, b=0),
                ]
                send = asyncio.create_task(ops.send("exercise", stream=False))
                details = await approve("approved", ApprovalDecision.TRUST_SESSION)
                self.assertEqual(details["arguments"], {"a": 12, "b": 30})
                await approve("denied", ApprovalDecision.DENY)
                details = await approve("error")
                self.assertEqual(details["tool"], "math.add")
                self.assertEqual(
                    (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                )
                self.assertEqual(advertised, {"list_directory", add, special})
                self.assertTrue(outcomes[0]["ok"])
                self.assertEqual(outcomes[1]["structured_content"], {"sum": 42})
                self.assertEqual(outcomes[1]["text"], "42")
                self.assertEqual(outcomes[2]["structured_content"], {"sum": 3})
                self.assertEqual(outcomes[3]["error"]["code"], "permission_denied")
                self.assertEqual(outcomes[4]["error"]["code"], "mcp_tool_error")
                for index, omitted in ((5, "text"), (6, "structured_content")):
                    self.assertLessEqual(result_bytes[index], 65_536)
                    self.assertTrue(outcomes[index]["truncated"])
                    self.assertIn(omitted, outcomes[index]["omitted_fields"])
                    self.assertNotIn(omitted, outcomes[index])
                self.assertEqual(outcomes[5]["structured_content"], {"kind": "text"})
                self.assertEqual(outcomes[6]["text"], "summary")
                self.assertEqual(
                    log.read_text().splitlines(),
                    ["add 12 30", "add 1 2", "math.add -2 0", "add -3 0", "add -4 0"],
                )

                ops.set_approval_mode("ask")
                calls[:] = [
                    tool_call(add, "ask-first", a=1, b=1),
                    tool_call(add, "ask-again", a=1, b=2),
                ]
                send = asyncio.create_task(ops.send("ask twice", stream=False))
                await approve("ask-first")
                await approve("ask-again")
                self.assertEqual((await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED)
                ops.set_approval_mode("auto")
                calls[:] = [tool_call(add, "trust-cleared", a=3, b=4)]
                send = asyncio.create_task(ops.send("return to auto", stream=False))
                await approve("trust-cleared", ApprovalDecision.TRUST_SESSION)
                self.assertEqual((await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED)
                ops.set_approval_mode("full")
                calls[:] = [tool_call(special, "full", a=3, b=4)]
                self.assertEqual((await ops.send("full", stream=False)).status, TurnStatus.COMPLETED)
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 7})
                self.assertTrue(requests.empty())
                ops.set_approval_mode("auto")

                await ops.new_session()
                calls[:] = [tool_call(add, "slow", a=-1, b=0)]
                with patch("daphnis.mcp._CALL_TIMEOUT", 0.1):
                    send = asyncio.create_task(ops.send("timeout", stream=False))
                    await approve("slow")
                    self.assertEqual(
                        (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                    )
                self.assertEqual(outcomes[-1]["error"]["code"], "mcp_timeout")
                self.assertIsNone(ops.pending_approval)

                calls[:] = [tool_call(add, "cancel", a=-1, b=1)]
                send = asyncio.create_task(ops.send("cancel", stream=False))
                await approve("cancel")
                async with asyncio.timeout(5):
                    while "add -1 1" not in log.read_text():
                        await asyncio.sleep(0.02)
                with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                    await ops.disconnect_mcp("demo")
                await ops.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await send
                self.assertIsNone(ops.pending_approval)
                pid = int((root / "pid").read_text())
                await ops.set_tools(("list_directory", add))
                snapshot = app._tools.select([add]).bind(
                    session=app._session, events=app.events, turn_id="old-turn"
                )
                await ops.disconnect_mcp("demo")
                self.assertNotIn(add, ops.all_tools)
                self.assertIn(add, ops.enabled_tools)
                self.assertEqual(ops.effective_tools, ("list_directory",))
                offline = await snapshot.execute(tool_call(add, "offline", a=1, b=2))
                self.assertEqual(offline.value["error"]["code"], "mcp_unavailable")
                if os.name == "posix":
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)

                await ops.reconnect_mcp("demo")
                self.assertIn(add, ops.effective_tools)
                self.assertNotIn(special, ops.effective_tools)
                await ops.disconnect_mcp("demo")
                ops.restore_session(saved)
                self.assertIn(add, ops.enabled_tools)
                self.assertNotIn(add, ops.effective_tools)
                await ops.reconnect_mcp("demo")
                calls[:] = [tool_call(add, "reconnected", a=2, b=3)]
                send = asyncio.create_task(ops.send("reconnected", stream=False))
                await approve("reconnected", ApprovalDecision.TRUST_SESSION)
                self.assertEqual(
                    (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                )
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 5})
                (root / "changed").touch()
                await ops.refresh_mcp("demo")
                (root / "changed").unlink()
                await ops.refresh_mcp("demo")
                calls[:] = [tool_call(add, "definition-changed", a=4, b=5)]
                send = asyncio.create_task(ops.send("changed", stream=False))
                await approve("definition-changed", ApprovalDecision.TRUST_SESSION)
                self.assertEqual(
                    (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                )
                pid = int((root / "pid").read_text())
                backup_dir = root / "backup"
                backup_dir.mkdir()
                await ops.connect_mcp(
                    "backup", sys.executable, [str(DEMO), "."], cwd=backup_dir
                )
                backup = "mcp__backup__add"
                backup_pid = int((backup_dir / "pid").read_text())
                self.assertNotIn(backup, ops.enabled_tools)
                await ops.set_tools((*ops.enabled_tools, backup))
                calls[:] = [tool_call(backup, "separate-server", a=2, b=4)]
                send = asyncio.create_task(ops.send("other server", stream=False))
                await approve("separate-server", server="backup")
                self.assertEqual((await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED)
                with self.assertRaisesRegex(RuntimeError, "MCP_MISSING_TEST_KEY"):
                    await ops.connect_mcp(
                        "broken", sys.executable, env={"KEY": "MCP_MISSING_TEST_KEY"}
                    )
                self.assertEqual(ops.mcp_status()[-1].state, "Error")
                if os.name == "posix":
                    os.kill(pid, signal.SIGKILL)
                    async with asyncio.timeout(5):
                        while ops.mcp_status()[0].state != "Error":
                            await asyncio.sleep(0.02)
                    self.assertIn(add, ops.enabled_tools)
                    self.assertNotIn(add, ops.effective_tools)

                await ops.new_session()
                advertised.clear()
                calls[:] = [tool_call(backup, "backup", a=4, b=7)]
                send = asyncio.create_task(ops.send("backup", stream=False))
                await approve("backup", server="backup")
                self.assertEqual(
                    (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                )
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 11})
                if os.name == "posix":
                    self.assertNotIn(add, advertised)
                    await ops.reconnect_mcp("demo")
                    self.assertIn(add, ops.effective_tools)
                    self.assertNotIn((saved, add), app._trusted_tools)
                    pid = int((root / "pid").read_text())
                (root / "changed").touch()
                await ops.refresh_mcp("demo")
                self.assertIn("mcp__demo__sum", ops.list_mcp_tools())
                self.assertNotIn("mcp__demo__sum", ops.enabled_tools)
                self.assertIn(add, ops.enabled_tools)
                self.assertNotIn(add, ops.effective_tools)
                (root / "invalid").touch()
                with self.assertRaisesRegex(RuntimeError, "保留上次目录"):
                    await ops.refresh_mcp("demo")
                self.assertIn("mcp__demo__sum", ops.list_mcp_tools())
                self.assertEqual(ops.mcp_status()[0].state, "Connected")
            finally:
                await ops.close()
            if os.name == "posix":
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(backup_pid, 0)

    @patch("daphnis.mcp._HTTP_HEALTH_INTERVAL", 0.1)
    async def test_streamable_http_auth_and_lifecycle(self):
        import socket

        import uvicorn
        from mcp.server import MCPServer
        from mcp.server.mcpserver import Context
        from mcp.server.subscriptions import InMemorySubscriptionBus

        subscribed = asyncio.Event()
        changed = asyncio.Event()
        finish_call = asyncio.Event()

        class ObservedBus(InMemorySubscriptionBus):
            def subscribe(self, listener):
                unsubscribe = super().subscribe(listener)
                subscribed.set()
                return unsubscribe

        server = MCPServer(
            "http-demo", log_level="CRITICAL", subscriptions=ObservedBus()
        )

        async def double(a: int, b: int) -> dict[str, int]:
            return {"sum": 2 * (a + b)}

        @server.tool()
        async def add(a: int, b: int, ctx: Context) -> dict[str, int]:
            server.remove_tool("add")
            server.add_tool(double)
            await ctx.notify_tools_changed()
            changed.set()
            await finish_call.wait()
            return {"sum": a + b}

        remote = server.streamable_http_app(json_response=True, stateless_http=True)
        received = []
        outcomes = []
        advertised = set()
        target = "mcp__remote__add"
        token = "daphnis-synthetic-token"

        async def protected(scope, receive, send):
            if scope["type"] != "http":
                return await remote(scope, receive, send)
            received.append(scope["path"])
            if scope["path"] == "/redirect":
                location = f"http://localhost:{port}/mcp".encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 307,
                        "headers": [(b"location", location)],
                    }
                )
                await send({"type": "http.response.body", "body": b""})
                return
            if scope["path"] == "/stall":
                while (await receive())["type"] != "http.disconnect":
                    pass
                return
            if (b"authorization", f"Bearer {token}".encode()) not in scope["headers"]:
                await send(
                    {"type": "http.response.start", "status": 401, "headers": []}
                )
                await send({"type": "http.response.body", "body": b""})
                return
            return await remote(scope, receive, send)

        def model(request):
            body = json.loads(request.content)
            last = body["messages"][-1]
            advertised.update(tool["function"]["name"] for tool in body["tools"])
            if last["role"] == "tool":
                outcomes.append(json.loads(last["content"]))
                message = {"role": "assistant", "content": "42"}
                reason = "stop"
            else:
                message = {
                    "role": "assistant",
                    "tool_calls": [tool_call(target, "remote-add", a=12, b=30)],
                }
                reason = "tool_calls"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": reason}
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            http_server = uvicorn.Server(
                uvicorn.Config(
                    protected, log_level="critical", timeout_graceful_shutdown=1
                )
            )
            serving = asyncio.create_task(http_server.serve(sockets=[listener]))
            app = Application(
                workspace=directory,
                provider="mimo",
                model_id="mimo-v2.6-flash",
                history=History(Path(directory) / "sessions"),
                connections={
                    "mimo": {
                        "base_url": "https://model.test",
                        "transport": httpx.MockTransport(model),
                    }
                },
                interactive=True,
            )
            ops = Operations(app)

            async def approve(event):
                pending = ops.pending_approval
                self.assertEqual(pending.details["server"], "remote")
                ops.resolve_approval(
                    pending.request_id, ApprovalDecision.ALLOW_ONCE
                )

            app.events.on("tool.approval_requested", approve)
            address = f"http://127.0.0.1:{port}"
            bearer = {"type": "bearer", "env": "MCP_TEST_TOKEN"}
            try:
                async with asyncio.timeout(5):
                    while not http_server.started:
                        await asyncio.sleep(0.01)
                with (
                    patch.dict(os.environ, {"MCP_TEST_TOKEN": "incorrect"}),
                    self.assertRaisesRegex(RuntimeError, "HTTP 401"),
                ):
                    await ops.connect_mcp(
                        "denied",
                        transport="streamable_http",
                        url=address + "/mcp",
                        auth=bearer,
                    )
                with patch.dict(os.environ, {"MCP_TEST_TOKEN": token}):
                    received.clear()
                    with self.assertRaisesRegex(RuntimeError, "重定向"):
                        await ops.connect_mcp(
                            "redirect",
                            transport="streamable_http",
                            url=address + "/redirect",
                            auth=bearer,
                        )
                    self.assertEqual(received, ["/redirect"])
                    with (
                        patch("daphnis.mcp._CONNECT_TIMEOUT", 0.2),
                        self.assertRaisesRegex(RuntimeError, "超时"),
                    ):
                        await ops.connect_mcp(
                            "slow", transport="streamable_http", url=address + "/stall"
                        )
                    await ops.connect_mcp(
                        "remote",
                        transport="streamable_http",
                        url=address + "/mcp",
                        auth=bearer,
                    )
                self.assertEqual(ops.list_mcp_tools(), ("mcp__remote__add",))
                await ops.set_tools(("mcp__remote__add",))
                await asyncio.wait_for(subscribed.wait(), 5)
                send = asyncio.create_task(ops.send("add", stream=False))
                async with asyncio.timeout(5):
                    await changed.wait()
                    while not app._mcp_pending:
                        await asyncio.sleep(0.01)
                self.assertEqual(ops.list_mcp_tools(), ("mcp__remote__add",))
                with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                    await ops.refresh_mcp("remote")
                finish_call.set()
                turn = await asyncio.wait_for(send, 5)
                self.assertEqual(turn.status, TurnStatus.COMPLETED)
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 42})
                self.assertEqual(advertised, {"mcp__remote__add"})
                target = "mcp__remote__double"
                async with asyncio.timeout(5):
                    while target not in ops.list_mcp_tools():
                        await asyncio.sleep(0.01)
                self.assertEqual(ops.enabled_tools, ("mcp__remote__add",))
                self.assertEqual(ops.effective_tools, ())
                await ops.set_tools((target,), keep_unavailable=True)
                advertised.clear()
                self.assertEqual(
                    (
                        await asyncio.wait_for(ops.send("double", stream=False), 5)
                    ).status,
                    TurnStatus.COMPLETED,
                )
                self.assertEqual(advertised, {target})
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 84})
                http_server.should_exit = True
                await asyncio.wait_for(serving, 5)
                async with asyncio.timeout(5):
                    while ops.mcp_status()[-1].state != "Error":
                        await asyncio.sleep(0.02)
                self.assertNotIn(target, ops.effective_tools)
                self.assertIn(target, ops.enabled_tools)
            finally:
                finish_call.set()
                await ops.close()
                http_server.should_exit = True
                await asyncio.wait_for(serving, 5)
