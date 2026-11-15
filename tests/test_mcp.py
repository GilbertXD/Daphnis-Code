import asyncio
import json
import os
import re
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from daphnis import History, Operations, TurnStatus
from daphnis.app import Application
from daphnis.mcp import MCPConnection, MCPPromptError
from daphnis.runtime import open_operations
from daphnis.tool_state import RUNNING, TOOL_STATE
from daphnis.tools import ApprovalDecision

DEMO = Path(__file__).with_name("mcp_demo_server.py").resolve()


def tool_call(name, call_id, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_protocol_pagination_payloads_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_root = root / "prompts"
            prompt_root.mkdir()
            (prompt_root / "prompts").touch()
            (prompt_root / "prompts-only").touch()
            plain_root = root / "plain"
            plain_root.mkdir()
            both_root = root / "both"
            both_root.mkdir()
            (both_root / "prompts").touch()
            app = Application(
                workspace=root, provider="deepseek", model_id="deepseek-flash",
                connections={"deepseek": {"base_url": "https://model.test"}},
                history=History(root / "sessions"),
            )
            ops = Operations(app)
            try:
                await ops.connect_mcp("plain", sys.executable, [str(DEMO), str(plain_root)])
                with self.assertRaisesRegex(MCPPromptError, "不支持模板"):
                    await app._mcp["plain"].list_prompts()
                self.assertTrue(app._mcp["plain"].available)
                await ops.connect_mcp("prompt-only", sys.executable, [str(DEMO), str(prompt_root)])
                self.assertTrue(app._mcp["prompt-only"].supports_prompts)
                self.assertEqual(app._mcp["prompt-only"].tools, {})
                await ops.connect_mcp("both", sys.executable, [str(DEMO), str(both_root)])
                connection = app._mcp["both"]
                prompts = await connection.list_prompts()
                self.assertEqual([prompt.name for prompt in prompts], ["greet", "bundle", "slow", "fail", "large", "dialogue"])
                self.assertEqual([(arg.name, arg.required) for arg in prompts[0].arguments],
                                 [("who", True), ("tone", False)])
                greeting = await connection.get_prompt("greet", {"who": "Ada", "tone": "casual"})
                self.assertEqual(greeting.messages[0].content.text, "Hello, Ada! (casual)")
                bundle = await connection.get_prompt("bundle", {})
                self.assertEqual([(msg.role, msg.content.type) for msg in bundle.messages],
                                 [("user", "text"), ("assistant", "image")])
                self.assertEqual(bundle.messages[1].content.data, "AA==")
                with patch("daphnis.mcp._PROMPT_TIMEOUT", 0.1), self.assertRaisesRegex(MCPPromptError, "超时"):
                    await connection.get_prompt("slow", {})
                with self.assertRaisesRegex(MCPPromptError, "响应无效"):
                    await connection.get_prompt("fail", {})
                with self.assertRaisesRegex(MCPPromptError, "大小上限"):
                    await connection.get_prompt("large", {})
                self.assertEqual((await connection.call("add", {"a": 2, "b": 3}))["structured_content"], {"sum": 5})
                self.assertEqual(ops.list_mcp_prompt_servers(), ("prompt-only", "both"))
                self.assertEqual(await ops.list_mcp_prompts("both"), prompts)
                with self.assertRaisesRegex(MCPPromptError, "未连接"):
                    await ops.list_mcp_prompts("missing")
                with self.assertRaisesRegex(ValueError, "必填"):
                    await ops.get_mcp_prompt("both", "greet", {})
                with self.assertRaisesRegex(ValueError, "未定义"):
                    await ops.get_mcp_prompt("both", "greet", {"who": "Ada", "extra": "no"})
                with self.assertRaisesRegex(ValueError, "未找到"):
                    await ops.get_mcp_prompt("both", "missing", {})
                self.assertEqual(await ops.get_mcp_prompt("both", "bundle", {}), bundle)
                self.assertEqual(ops.message_snapshot(), [])
                self.assertEqual(ops.list_sessions(), ())
            finally:
                await ops.close()

    async def test_runtime_persists_and_restores_mcp_with_isolated_connection_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {
                "LLM_PROVIDER": "deepseek",
                "LLM_MODEL": "deepseek-flash",
                "LLM_BASE_URL": "https://model.test",
            }
            good, broken = root / "good", root / "broken"
            good.mkdir()
            broken.mkdir()
            (broken / "stderr-exit").touch()

            def server(cwd):
                return {"transport": "stdio", "command": sys.executable,
                        "args": [str(DEMO), "."], "cwd": str(cwd)}

            with (
                patch.dict(os.environ, env, clear=True),
                patch("daphnis.runtime.load_dotenv"),
                patch("daphnis.app.History", return_value=History(root / "sessions")),
            ):
                ops = await open_operations(root)
                try:
                    self.assertIsNone(await ops.save_mcp("good", options=server(good), enabled=True, create=True))
                    self.assertTrue(await ops.save_mcp("broken", options=server(broken), enabled=True, create=True))
                    await ops.save_mcp("off", options=server(root), enabled=False, create=True)
                    self.assertEqual((root / "mcp.json").stat().st_mode & 0o777, 0o600)
                finally:
                    await ops.close()

                (good / "pid").unlink()
                with self.assertLogs("daphnis.runtime", "WARNING"):
                    ops = await open_operations(root)
                try:
                    self.assertIn("mcp__good__add", ops.effective_tools)
                    statuses = {status.name: status for status in ops.mcp_status()}
                    self.assertEqual(statuses["broken"].state, "Error")
                    self.assertTrue(statuses["broken"].error)
                    self.assertNotIn("off", statuses)
                    self.assertFalse((root / "pid").exists())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int((broken / "pid").read_text()), 0)
                finally:
                    await ops.close()
                with self.assertRaises(ProcessLookupError):
                    os.kill(int((good / "pid").read_text()), 0)

    @unittest.skipUnless(os.name == "posix", "Requires POSIX stderr pipes")
    async def test_stdio_stderr_is_drained_and_sanitized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stderr-diagnostics").touch()
            script = """
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from mcp import stdio_client

from daphnis.mcp import MCPConnection

async def main():
    demo, directory = sys.argv[1:]
    root = Path(directory)
    loop = asyncio.get_running_loop()
    connect_read_pipe = loop.connect_read_pipe
    read_pipes, write_pipes = [], []

    async def track_read(factory, pipe):
        result = await connect_read_pipe(factory, pipe)
        read_pipes.append(pipe)
        return result

    @asynccontextmanager
    async def track_write(parameters, *, errlog):
        write_pipes.append(errlog)
        async with stdio_client(parameters, errlog=errlog) as streams:
            yield streams

    async def error(name, detail):
        raise AssertionError(detail)

    async def changed(server):
        pass

    with patch.object(loop, "connect_read_pipe", track_read), patch("daphnis.mcp.stdio_client", track_write):
        for scenario in ("normal", "exit", "cancel", "timeout", "spawn-error"):
            read_pipes.clear()
            write_pipes.clear()
            for marker in ("stderr-exit", "stderr-stall", "stderr-written", "pid"):
                (root / marker).unlink(missing_ok=True)
            if scenario == "exit":
                (root / "stderr-exit").touch()
            elif scenario in ("cancel", "timeout"):
                (root / "stderr-stall").touch()
            command = str(root / "missing-program") if scenario == "spawn-error" else sys.executable
            connection = MCPConnection("audit", command, [demo, directory])
            with patch("daphnis.mcp._CONNECT_TIMEOUT", 1.5 if scenario == "timeout" else 8):
                startup = asyncio.create_task(connection.start(error, changed))
                try:
                    if scenario == "cancel":
                        async with asyncio.timeout(4):
                            while not (root / "stderr-written").exists():
                                await asyncio.sleep(0.01)
                        startup.cancel()
                    tools = await startup
                except asyncio.CancelledError:
                    assert scenario == "cancel"
                except RuntimeError as exc:
                    assert scenario in ("exit", "timeout", "spawn-error")
                    if scenario == "timeout":
                        assert "超时" in str(exc)
                else:
                    assert scenario == "normal" and "mcp__audit__add" in tools
                    connection.state = "Connected"
                    result = await connection.call("add", {"a": 2, "b": 3})
                    assert result["structured_content"] == {"sum": 5}
                finally:
                    if not startup.done():
                        startup.cancel()
                        await asyncio.gather(startup, return_exceptions=True)
                    await connection.aclose()
            assert connection._task.done()
            assert read_pipes and read_pipes[0].closed
            assert len(write_pipes) == 1 and write_pipes[0].closed
            if scenario != "spawn-error":
                assert (root / "stderr-written").exists()
                pid = int((root / "pid").read_text())
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError("MCP child still running")

asyncio.run(main())
"""
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script, str(DEMO), directory,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), 18)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            self.assertEqual(process.returncode, 0, stderr[-1000:])
            self.assertEqual(stderr.count(b"fixture start"), 4)
            self.assertEqual(stderr.count(b" diagnostic "), 4)
            self.assertIsNone(
                re.search(rb"\x1b|\x07|\xc2[\x80-\x9f]|CLIPBOARD|HIDDEN|AFTER-LIMIT", stderr)
            )
            self.assertLess(len(stderr), 80 * 1024)
            self.assertEqual(stderr.count(b"[MCP stderr truncated]"), 4)

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
            executing = {}

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

            def record_execution(event):
                call_id = event.data["tool_call_id"]
                messages = History(root / "sessions").load(event.session_id).messages.snapshot()
                executing[call_id] = next(
                    message.get(TOOL_STATE) for message in messages
                    if message.get("tool_call_id") == call_id
                )

            ops.subscribe("tool.executing", record_execution)
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
                self.assertEqual(executing["approved"], RUNNING)
                self.assertNotIn("denied", executing)
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

        @server.prompt()
        def greeting() -> str:
            return "Hello from MCP"

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
            if scope["method"] == "POST":
                first = await receive()
                if (
                    first["type"] == "http.request" and not first.get("more_body") and first.get("body")
                    and json.loads(first["body"]).get("method") == "prompts/get"
                ):
                    await send({"type": "http.response.start", "status": 307,
                                "headers": [(b"location", b"/unexpected")]})
                    await send({"type": "http.response.body", "body": b""})
                    return

                async def replay():
                    nonlocal first
                    if first is not None:
                        message, first = first, None
                        return message
                    return await receive()

                return await remote(scope, replay, send)
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
                connection = app._mcp["remote"]
                self.assertEqual([prompt.name for prompt in await connection.list_prompts()], ["greeting"])
                with self.assertRaisesRegex(MCPPromptError, "响应无效"):
                    await connection.get_prompt("greeting", {})
                self.assertTrue(connection.available)
                self.assertNotIn("/unexpected", received)
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
