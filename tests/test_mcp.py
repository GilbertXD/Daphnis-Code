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
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.mcp import MCPConnection, MCPPromptError, MCPResourceError
from daphnis.mcp_store import MCPServerConfig, MCPStore
from daphnis.runtime import open_operations
from daphnis.tool_state import RUNNING, TOOL_STATE
from daphnis.tools import ApprovalDecision
from daphnis.tools.approval import ToolError

DEMO = Path(__file__).with_name("mcp_demo_server.py").resolve()


def tool_call(name, call_id, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class MCPStoreTests(unittest.TestCase):
    def test_old_config_migrates_without_changing_options_and_scopes_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            store = MCPStore(workspace / "mcp.json")
            local = {"command": "example", "args": ["a b"], "env": {"TOKEN": "EXAMPLE_TOKEN"}}
            remote = {"transport": "streamable_http", "url": "https://example.com/mcp",
                      "auth": {"type": "bearer", "env": "REMOTE_TOKEN"}}
            old = {"version": 1, "servers": {
                "same": {"enabled": True, "options": local},
                "remote": {"enabled": False, "options": remote},
            }}
            store.path.write_text(json.dumps(old))
            configs = store.read()
            self.assertEqual(configs["same"], MCPServerConfig("same", "always", local))
            self.assertEqual(configs["remote"], MCPServerConfig("remote", "disabled", remote))
            self.assertEqual(json.loads(store.path.read_text()), old)
            project = MCPServerConfig("same", "on_demand", local, workspace, "Local example")
            configs[project.key] = project
            store.write(configs)
            self.assertEqual(store.read(), configs)
            self.assertEqual(project.id, "project:same")
            self.assertEqual(configs["same"].id, "global:same")
            self.assertTrue(configs["same"].applies_to(workspace / "another"))
            self.assertFalse(project.applies_to(workspace / "another"))
            self.assertNotEqual(project.connection().prefix, configs["same"].connection().prefix)


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_lazy_connection_shares_initialization_and_retries_with_current_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = Application(
                workspace=root, provider="deepseek", model_id="deepseek-flash",
                history=History(root / "sessions"), connections={"deepseek": {"base_url": "https://model.test"}},
            )
            self.addAsyncCleanup(app.close)
            ops = Operations(app)
            first = ops.initial_session_id
            second = (await ops.new_session(session_id=first)).id
            options = {"command": sys.executable, "args": [str(DEMO), directory],
                       "env": {"DEMO_CREDENTIAL": "DAPHNIS_TEST_MCP_CREDENTIAL"}}
            await ops.save_mcp("lazy", state="on_demand", options=options, create=True)
            self.assertEqual(ops.mcp_status(), ())
            self.assertFalse((root / "pid").exists())
            with (
                patch.dict(os.environ, {"DAPHNIS_TEST_MCP_CREDENTIAL": ""}),
                self.assertRaisesRegex(ToolError, "缺少 MCP 环境变量"),
            ):
                await app._activate_mcp(first, "global:lazy")

            start = MCPConnection.start
            entered, proceed = asyncio.Event(), asyncio.Event()
            starts = []

            async def delayed(connection, *args):
                starts.append(connection)
                entered.set()
                await proceed.wait()
                return await start(connection, *args)

            with patch.dict(os.environ, {"DAPHNIS_TEST_MCP_CREDENTIAL": "first"}):
                with patch.object(MCPConnection, "start", delayed):
                    a = asyncio.create_task(app._activate_mcp(first, "global:lazy"))
                    await asyncio.wait_for(entered.wait(), 5)
                    b = asyncio.create_task(app._activate_mcp(second, "global:lazy"))
                    await asyncio.sleep(0)
                    a.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await a
                    proceed.set()
                    self.assertEqual((await asyncio.wait_for(b, 5))["tool_count"], 2)
                self.assertEqual(starts, [app._mcp["lazy"]])
                self.assertEqual(app._load_runtime(first).activated_mcp, set())
                self.assertEqual(app._load_runtime(second).activated_mcp, {"lazy"})
            with patch.dict(os.environ, {"DAPHNIS_TEST_MCP_CREDENTIAL": "rotated"}):
                self.assertFalse(starts[0].available)
                await app._activate_mcp(first, "global:lazy")
                self.assertIsNot(app._mcp["lazy"], starts[0])
                await ops.save_mcp("global:lazy", state="disabled")
                self.assertFalse(app._load_runtime(second).activated_mcp)
                with self.assertRaisesRegex(ToolError, "没有可按需激活"):
                    await app._activate_mcp(first, "global:lazy")

    async def test_native_tools_appear_between_requests_and_respect_scope_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sibling = root / "sibling"
            sibling.mkdir()
            native = "mcp__lazy__add"
            requests, approvals, persisted = [], [], []

            def model(request):
                body = json.loads(request.content)
                messages = body["messages"]
                last = messages[-1]
                tools = {item["function"]["name"]: item["function"] for item in body.get("tools", [])}
                if messages[0]["content"] == TITLE_PROMPT:
                    message = {"role": "assistant", "content": "Title"}
                elif last.get("content") == "activate":
                    requests.append((messages[0]["content"], tools))
                    message = {"role": "assistant", "tool_calls": [
                        tool_call("activate_mcp", "activation", server_id="global:lazy"),
                        tool_call(native, "premature", a=1, b=4),
                        tool_call("activate_mcp", "disabled", server_id="global:off"),
                    ]}
                elif last["role"] == "tool" and last.get("tool_call_id") == "disabled":
                    requests.append((messages[0]["content"], tools))
                    message = {"role": "assistant", "tool_calls": [tool_call(native, "native", a=20, b=22)]}
                else:
                    requests.append((messages[0]["content"], tools))
                    message = {"role": "assistant", "content": "Done"}
                return httpx.Response(200, json={"choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}]})

            app = Application(
                workspace=root, provider="deepseek", model_id="deepseek-flash", interactive=True,
                history=History(root / "sessions"),
                connections={"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(model)}},
            )
            self.addAsyncCleanup(app.close)
            ops = Operations(app)
            a = ops.initial_session_id
            project = await ops.create_project("Sibling", sibling, session_id=a)
            b = (await ops.open_project(project.id, session_id=a)).id
            options = {"command": sys.executable, "args": [str(DEMO), directory]}
            await ops.save_mcp("lazy", state="on_demand", description="Add two integers", options=options, create=True)
            await ops.save_mcp("lazy", state="on_demand", scope="project", options=options, create=True, session_id=a)
            await ops.save_mcp("off", state="disabled", options=options, create=True)
            self.assertFalse((root / "pid").exists())
            self.assertEqual({server.id for server in ops.managed_mcp(b)}, {"global:lazy", "global:off"})
            with self.assertRaisesRegex(ValueError, "当前工作区"):
                await ops.restart_mcp("project:lazy", session_id=b)
            ops.set_approval_mode("ask", session_id=a)

            async def approve(event):
                approvals.append(event.data["name"])
                ops.resolve_approval(event.data["request_id"], ApprovalDecision.ALLOW_ONCE, session_id=a)

            def record(event):
                snapshot = app._history.load(event.session_id).messages.snapshot()
                result = next(item for item in snapshot if item.get("tool_call_id") == event.data["tool_call_id"])
                persisted.append((event.data["name"], result.get(TOOL_STATE)))

            ops.subscribe("tool.approval_requested", approve)
            ops.subscribe("tool.executing", record)
            result = await asyncio.wait_for(ops.send("activate", session_id=a, stream=False), 10)
            self.assertEqual(result.status, TurnStatus.COMPLETED)
            self.assertIn("global:lazy", requests[0][0])
            self.assertIn("project:lazy", requests[0][0])
            self.assertNotIn("global:off", requests[0][0])
            self.assertIn("activate_mcp", requests[0][1])
            self.assertNotIn("call_mcp_tool", requests[0][1])
            self.assertNotIn(native, requests[0][1])
            self.assertEqual(requests[1][1][native]["parameters"]["properties"]["a"]["type"], "integer")
            self.assertEqual(approvals, [native])
            self.assertEqual(persisted, [("activate_mcp", RUNNING), ("activate_mcp", RUNNING), (native, RUNNING)])
            results = {item["tool_call_id"]: json.loads(item["content"]) for item in ops.message_snapshot(a)
                       if item["role"] == "tool"}
            self.assertEqual(results["activation"], {"ok": True, "server_id": "global:lazy", "tool_count": 2})
            self.assertEqual(results["premature"]["error"]["code"], "unknown_tool")
            self.assertEqual(results["disabled"]["error"]["code"], "mcp_not_available")
            self.assertEqual(results["native"]["structured_content"], {"sum": 42})
            self.assertNotIn(native, ops.effective_tools(b))
            await ops.send("second", session_id=b, stream=False)
            self.assertNotIn(native, requests[-1][1])
            self.assertNotIn("project:lazy", requests[-1][0])
            await ops.set_tools(tuple(name for name in ops.enabled_tools(a) if name != native), session_id=a)
            self.assertIn(native, app._history.load(a).disabled_mcp_tools)
            await ops.restart_mcp("global:lazy", session_id=a)
            self.assertNotIn(native, ops.effective_tools(a))
            await ops.save_mcp("global:lazy", state="disabled", session_id=a)
            await ops.save_mcp("global:lazy", state="on_demand", session_id=a)
            await app._activate_mcp(a, "global:lazy")
            self.assertNotIn(native, ops.effective_tools(a))
            await ops.save_mcp("auto", state="always", scope="project", options=options, create=True, session_id=b)
            auto = next(name for name in ops.available_tools(b) if name.endswith("_auto__add"))
            self.assertNotIn(auto, ops.available_tools(a))
            await ops.set_tools((auto,), session_id=b)
            await ops.send("always", session_id=b, stream=False)
            self.assertIn(auto, requests[-1][1])
            self.assertEqual(len(ops.list_mcp_servers(b)), 2)
            self.assertEqual(ops.list_mcp_servers(a), ("lazy",))

    async def test_gateway_keeps_models_tools_fixed_and_applies_target_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = []
            approvals = []
            running = []

            def gateway(call_id, tool="add", **arguments):
                return tool_call(
                    "call_mcp_tool", call_id, server_id="global:lazy", tool_name=tool, arguments=arguments,
                )

            def model(request):
                body = json.loads(request.content)
                messages = body["messages"]
                last = messages[-1]
                calls = []
                if messages[0]["content"] == TITLE_PROMPT:
                    pass
                else:
                    primary.append(body["tools"])
                    if last["content"] == "first":
                        calls = [
                            tool_call("activate_mcp", "activate", server_id="global:lazy"),
                            gateway("same_batch", a=1, b=2),
                            tool_call("activate_mcp", "disabled", server_id="global:off"),
                        ]
                    elif last.get("tool_call_id") == "disabled":
                        calls = [
                            gateway("invalid", a="20", b=22),
                            gateway("unknown", "missing", a=20, b=22),
                            tool_call("mcp__lazy__add", "hidden", a=20, b=22),
                            gateway("good", a=20, b=22),
                            gateway("trusted", a=1, b=2),
                            gateway("other", "math.add", a=2, b=3),
                        ]
                    elif last["content"] == "new session":
                        calls = [gateway("isolated", a=20, b=22)]
                    elif last["content"] == "disabled tool":
                        calls = [gateway("disabled_tool", a=20, b=22), gateway("ask", "math.add", a=4, b=6)]
                    elif last["content"] == "cancel tool":
                        calls = [gateway("cancelled", "math.add", a=-1, b=0)]
                message = {"role": "assistant", "tool_calls": calls} if calls else {
                    "role": "assistant", "content": "Done",
                }
                return httpx.Response(200, json={"choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if calls else "stop"}]})

            app = Application(
                workspace=root, provider="deepseek", model_id="deepseek-flash", interactive=True,
                mcp_tool_loading="gateway", history=History(root / "sessions"),
                connections={"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(model)}},
            )
            self.addAsyncCleanup(app.close)
            ops = Operations(app)
            first = ops.initial_session_id
            options = {"command": sys.executable, "args": [str(DEMO), directory]}
            await ops.save_mcp("lazy", state="on_demand", options=options, create=True)
            await ops.save_mcp("off", state="disabled", options=options, create=True)
            self.assertFalse((root / "pid").exists())

            async def approve(event):
                request = event.data["request"]
                approvals.append((request["tool_name"], request["tool_call_id"], request["can_trust"]))
                decision = (
                    ApprovalDecision.TRUST_SESSION if request["tool_call_id"] == "good" else
                    ApprovalDecision.DENY if request["tool_call_id"] == "other" else ApprovalDecision.ALLOW_ONCE
                )
                self.assertTrue(ops.resolve_approval(event.data["request_id"], decision, session_id=event.session_id))

            def record(event):
                call_id = event.data["tool_call_id"]
                messages = app._history.load(event.session_id).messages.snapshot()
                result = next(item for item in messages if item.get("tool_call_id") == call_id)
                running.append((call_id, result.get(TOOL_STATE)))

            def results(session_id):
                return {item["tool_call_id"]: json.loads(item["content"])
                        for item in ops.message_snapshot(session_id) if item["role"] == "tool"}

            ops.subscribe("tool.approval_requested", approve)
            ops.subscribe("tool.executing", record)
            outcome = await asyncio.wait_for(ops.send("first", session_id=first, stream=False), 10)
            self.assertEqual(outcome.status, TurnStatus.COMPLETED, outcome.error)
            self.assertEqual(len(primary), 3)
            self.assertEqual(primary[0], primary[1])
            self.assertEqual(primary[1], primary[2])
            names = {item["function"]["name"] for item in primary[0]}
            self.assertTrue({"activate_mcp", "call_mcp_tool"}.issubset(names))
            self.assertFalse(any(name.startswith("mcp__") for name in names))
            data = results(first)
            self.assertEqual([tool["name"] for tool in data["activate"]["tools"]], ["add", "math.add"])
            self.assertEqual(data["activate"]["tools"][0]["parameters"], app._mcp["lazy"].tools["mcp__lazy__add"].input_schema)
            for call_id, code in (
                ("same_batch", "mcp_tool_not_available"), ("disabled", "mcp_not_available"),
                ("invalid", "arguments_schema_mismatch"), ("unknown", "mcp_tool_not_available"),
                ("hidden", "unknown_tool"), ("other", "permission_denied"),
            ):
                self.assertEqual(data[call_id]["error"]["code"], code)
            self.assertEqual(data["good"]["structured_content"], {"sum": 42})
            self.assertEqual(data["trusted"]["structured_content"], {"sum": 3})
            self.assertEqual([(call_id, can_trust) for _, call_id, can_trust in approvals], [("good", True), ("other", True)])
            self.assertTrue(all(name.startswith("mcp__lazy__") for name, _, _ in approvals))
            self.assertEqual(running, [(call_id, RUNNING) for call_id in ("activate", "disabled", "good", "trusted")])

            second = (await ops.new_session(session_id=first)).id
            ops.set_approval_mode("full", session_id=second)
            await ops.send("new session", session_id=second, stream=False)
            self.assertEqual(results(second)["isolated"]["error"]["code"], "mcp_tool_not_available")
            await ops.set_tools(tuple(name for name in ops.enabled_tools(first) if name != "mcp__lazy__add"), session_id=first)
            ops.set_approval_mode("ask", session_id=first)
            await ops.send("disabled tool", session_id=first, stream=False)
            self.assertEqual(results(first)["disabled_tool"]["error"]["code"], "mcp_tool_not_available")
            self.assertEqual(results(first)["ask"]["structured_content"], {"sum": 10})
            self.assertEqual(approvals[-1][1:], ("ask", False))

            ops.set_approval_mode("full", session_id=first)
            entered = asyncio.Event()
            original_call = MCPConnection.call

            async def observe_call(connection, name, arguments):
                if arguments["a"] == -1:
                    entered.set()
                return await original_call(connection, name, arguments)

            with patch.object(MCPConnection, "call", observe_call):
                pending = asyncio.create_task(ops.send("cancel tool", session_id=first, stream=False))
                await asyncio.wait_for(entered.wait(), 5)
                await ops.cancel(first)
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertEqual(running[-1], ("cancelled", RUNNING))
                self.assertEqual(approvals[-1][1], "ask")
                recovery = ops.recovery(first)
                resumed = await ops.continue_interrupted_turn(recovery.key, session_id=first, stream=False)
                self.assertEqual(resumed.status, TurnStatus.COMPLETED, resumed.error)
                self.assertEqual(running.count(("cancelled", RUNNING)), 1)

    async def test_sessions_share_one_connection_and_select_their_own_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            advertised = {}

            def handler(request):
                body = json.loads(request.content)
                messages = body["messages"]
                if messages[0]["content"] == TITLE_PROMPT:
                    message, finish = {"role": "assistant", "content": "Title"}, "stop"
                elif messages[-1]["role"] == "tool":
                    result = json.loads(messages[-1]["content"])
                    message, finish = {"role": "assistant", "content": str(result["structured_content"]["sum"])}, "stop"
                else:
                    prompt = messages[-1]["content"]
                    advertised[prompt] = [item["function"]["name"] for item in body["tools"]]
                    call = tool_call(advertised[prompt][0], "same-id", a=-1 if prompt == "A" else 30, b=12)
                    message, finish = {"role": "assistant", "tool_calls": [call]}, "tool_calls"
                return httpx.Response(200, json={"choices": [{
                    "index": 0, "message": message, "finish_reason": finish,
                }]})

            app = Application(
                workspace=root, provider="deepseek", model_id="deepseek-flash", interactive=True,
                history=History(root / "sessions"),
                connections={"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(handler)}},
            )
            ops = Operations(app)
            a = ops.initial_session_id
            b = (await ops.new_session(session_id=a)).id
            await ops.connect_mcp("shared", sys.executable, [str(DEMO), directory])
            add, special = ops.list_mcp_tools()
            self.assertNotIn(add, ops.effective_tools(a))
            self.assertNotIn(add, ops.effective_tools(b))
            await ops.set_tools((add,), session_id=a)
            ops.subscribe("tool.approval_requested", lambda event: self._approve_event(ops, event))
            started = []

            def record_start(event):
                if event.data["request_number"] == 1 and event.data.get("purpose") != "title":
                    content = ops.message_snapshot(event.session_id)[-1]["content"]
                    started.append((event.session_id, content))

            ops.subscribe("model.started", record_start)
            client = app._clients["deepseek"]
            turn = asyncio.create_task(ops.send("A", stream=False, session_id=a))
            try:
                async with asyncio.timeout(5):
                    while not (root / "calls").exists():
                        await asyncio.sleep(0.01)
                    with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                        await ops.set_tools((), session_id=a)
                    with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                        await ops.delete_session(a)
                    with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                        await ops.refresh_mcp("shared")
                    await ops.set_model("deepseek", "deepseek-flash", session_id=b)
                    await ops.set_tools((special,), session_id=b)
                    self.assertEqual((await ops.send("B", stream=False, session_id=b)).status, TurnStatus.COMPLETED)
                    self.assertFalse(turn.done())
                    self.assertIs(app._clients["deepseek"], client)
                    self.assertEqual(advertised, {"A": [add], "B": [special]})
                    self.assertEqual(ops.message_snapshot(b)[-1]["content"], "42")
                    self.assertEqual(started, [(a, "A"), (b, "B")])
                    project = await ops.create_project("Other", session_id=b)
                    runtime = app._load_runtime(b)
                    await ops.move_session(b, project.id)
                    self.assertEqual(ops.restore_session(b).project_id, project.id)
                    self.assertIs(app._load_runtime(b), runtime)
                    await ops.delete_session(b)
                    with self.assertRaises(FileNotFoundError):
                        ops.restore_session(b)
                    await ops.cancel(a)
                    with self.assertRaises(asyncio.CancelledError):
                        await turn
            finally:
                await ops.close()
                await asyncio.gather(turn, return_exceptions=True)

    @staticmethod
    async def _approve_event(ops, event):
        ops.resolve_approval(event.data["request_id"], ApprovalDecision.ALLOW_ONCE, session_id=event.session_id)

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
                self.assertEqual([prompt.name for prompt in prompts], [
                    "greet", "bundle", "slow", "fail", "large", "dialogue",
                    "embedded", "embedded-large", "embedded-oversize",
                ])
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
                large = await connection.get_prompt("embedded-large", {})
                self.assertEqual(len(large.messages[1].content.resource.text), 70_000)
                with self.assertRaisesRegex(MCPPromptError, "128 KiB"):
                    await connection.get_prompt("embedded-oversize", {})
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
                self.assertEqual(ops.message_snapshot(ops.initial_session_id), [])
                self.assertEqual(ops.list_sessions(), ())
            finally:
                await ops.close()

    async def test_resources_use_catalog_templates_and_text_only_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = root / "plain"
            plain.mkdir()
            (root / "resources").touch()
            ops = Operations(Application(
                workspace=root, provider="deepseek", model_id="deepseek-flash",
                connections={"deepseek": {"base_url": "https://model.test"}},
                history=History(root / "sessions"),
            ))
            self.addAsyncCleanup(ops.close)
            await ops.connect_mcp("demo", sys.executable, [str(DEMO), str(root)])
            await ops.connect_mcp("plain", sys.executable, [str(DEMO), str(plain)])
            self.assertEqual(ops.list_mcp_resource_servers(), ("demo",))
            with self.assertRaisesRegex(MCPResourceError, "不支持资源"):
                await ops.list_mcp_resources("plain")
            resources = await ops.list_mcp_resources("demo")
            self.assertEqual(len(resources), 6)
            templates = await ops.list_mcp_resource_templates("demo")
            template = templates[0].uri_template
            item, = await ops.read_mcp_resource("demo", template, {"name": "中文/a ?%"})
            self.assertEqual(item.uri, "notes://pages/%E4%B8%AD%E6%96%87%2Fa%20%3F%25")
            with self.assertRaisesRegex(ValueError, "模板参数"):
                await ops.read_mcp_resource("demo", template, {"wrong": "a"})
            item, = await ops.read_mcp_resource("demo", "notes://architecture")
            self.assertEqual((item.name, item.mime_type), ("Architecture", "text/markdown"))
            self.assertEqual(len(await ops.read_mcp_resource("demo", "notes://bundle")), 2)
            for name, error in (("blob", "Blob"), ("fail", "响应无效"), ("bad-mime", "MIME")):
                with self.subTest(name=name), self.assertRaisesRegex(MCPResourceError, error):
                    await ops.read_mcp_resource("demo", f"notes://{name}")
            with patch("daphnis.mcp._RESOURCE_TIMEOUT", 0.15), self.assertRaisesRegex(MCPResourceError, "超时"):
                await ops.read_mcp_resource("demo", "notes://slow")
            with self.assertRaisesRegex(ValueError, "未找到"):
                await ops.read_mcp_resource("demo", "file:///not-in-catalog")
            (root / "resource-cursor").touch()
            with self.assertRaisesRegex(MCPResourceError, "Cursor"):
                await ops.list_mcp_resources("demo")
            await ops.disconnect_mcp("demo")
            with self.assertRaisesRegex(MCPResourceError, "未连接"):
                await ops.list_mcp_resources("demo")

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
                with self.assertLogs("daphnis.app", "WARNING"):
                    ops = await open_operations(root)
                try:
                    self.assertIn("mcp__good__add", ops.list_mcp_tools())
                    self.assertNotIn("mcp__good__add", ops.effective_tools(ops.initial_session_id))
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
                advertised.update(tool["function"]["name"] for tool in body.get("tools", []))
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
            session_id = ops.initial_session_id
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
                pending = ops.pending_approval(session_id)
                self.assertEqual(event.data["tool_call_id"], call_id)
                self.assertEqual(pending.details["server"], server)
                self.assertTrue(
                    ops.resolve_approval(pending.request_id, decision, session_id=session_id)
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
                self.assertTrue({add, special}.isdisjoint(ops.enabled_tools(session_id)))
                self.assertEqual(ops.list_mcp_servers(), ("demo",))
                await ops.set_tools(("list_directory", add, special), session_id=session_id)
                saved = ops.session(session_id).id
                calls[:] = [
                    tool_call("list_directory", "builtin"),
                    tool_call(add, "approved", a=12, b=30),
                    tool_call(add, "trusted", a=1, b=2),
                    tool_call(special, "denied", a=5, b=9),
                    tool_call(special, "error", a=-2, b=0),
                    tool_call(add, "large-text", a=-3, b=0),
                    tool_call(add, "large-structure", a=-4, b=0),
                ]
                send = asyncio.create_task(ops.send("exercise", stream=False, session_id=session_id))
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

                ops.set_approval_mode("ask", session_id=session_id)
                calls[:] = [
                    tool_call(add, "ask-first", a=1, b=1),
                    tool_call(add, "ask-again", a=1, b=2),
                ]
                send = asyncio.create_task(ops.send("ask twice", stream=False, session_id=session_id))
                await approve("ask-first")
                await approve("ask-again")
                self.assertEqual((await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED)
                ops.set_approval_mode("auto", session_id=session_id)
                calls[:] = [tool_call(add, "trust-cleared", a=3, b=4)]
                send = asyncio.create_task(ops.send("return to auto", stream=False, session_id=session_id))
                await approve("trust-cleared", ApprovalDecision.TRUST_SESSION)
                self.assertEqual((await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED)
                ops.set_approval_mode("full", session_id=session_id)
                calls[:] = [tool_call(special, "full", a=3, b=4)]
                self.assertEqual((await ops.send("full", stream=False, session_id=session_id)).status, TurnStatus.COMPLETED)
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 7})
                self.assertTrue(requests.empty())
                ops.set_approval_mode("auto", session_id=session_id)

                session_id = (await ops.new_session(session_id=session_id)).id
                calls[:] = [tool_call(add, "slow", a=-1, b=0)]
                with patch("daphnis.mcp._CALL_TIMEOUT", 0.1):
                    send = asyncio.create_task(ops.send("timeout", stream=False, session_id=session_id))
                    await approve("slow")
                    self.assertEqual(
                        (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                    )
                self.assertEqual(outcomes[-1]["error"]["code"], "mcp_timeout")
                self.assertIsNone(ops.pending_approval(session_id))

                calls[:] = [tool_call(add, "cancel", a=-1, b=1)]
                send = asyncio.create_task(ops.send("cancel", stream=False, session_id=session_id))
                await approve("cancel")
                async with asyncio.timeout(5):
                    while "add -1 1" not in log.read_text():
                        await asyncio.sleep(0.02)
                with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                    await ops.disconnect_mcp("demo")
                await ops.cancel(session_id)
                with self.assertRaises(asyncio.CancelledError):
                    await send
                self.assertIsNone(ops.pending_approval(session_id))
                pid = int((root / "pid").read_text())
                await ops.set_tools(("list_directory", add), session_id=session_id)
                snapshot = app._tools.select([add]).bind(
                    session=app._runtimes[session_id].session, events=app.events, turn_id="old-turn"
                )
                await ops.disconnect_mcp("demo")
                self.assertNotIn(add, ops.all_tools)
                self.assertIn(add, ops.enabled_tools(session_id))
                self.assertEqual(ops.effective_tools(session_id), ("list_directory",))
                offline = await snapshot.execute(tool_call(add, "offline", a=1, b=2))
                self.assertEqual(offline.value["error"]["code"], "mcp_unavailable")
                if os.name == "posix":
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)

                await ops.reconnect_mcp("demo")
                self.assertIn(add, ops.effective_tools(session_id))
                self.assertNotIn(special, ops.effective_tools(session_id))
                await ops.disconnect_mcp("demo")
                session_id = ops.restore_session(saved).id
                self.assertIn(add, ops.enabled_tools(session_id))
                self.assertNotIn(add, ops.effective_tools(session_id))
                await ops.reconnect_mcp("demo")
                calls[:] = [tool_call(add, "reconnected", a=2, b=3)]
                send = asyncio.create_task(ops.send("reconnected", stream=False, session_id=session_id))
                await approve("reconnected", ApprovalDecision.TRUST_SESSION)
                self.assertEqual(
                    (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                )
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 5})
                (root / "invalid").touch()
                with self.assertRaisesRegex(RuntimeError, "保留上次目录"):
                    await ops.refresh_mcp("demo")
                calls[:] = [tool_call(add, "unverified", a=1, b=3)]
                send = asyncio.create_task(ops.send("unverified", stream=False, session_id=session_id))
                self.assertEqual((await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED)
                self.assertEqual(outcomes[-1]["error"]["code"], "unknown_tool")
                self.assertIsNone(ops.pending_approval(session_id))
                self.assertNotIn((session_id, add), app._trusted_tools)
                (root / "invalid").unlink()
                (root / "changed").touch()
                await ops.refresh_mcp("demo")
                (root / "changed").unlink()
                await ops.refresh_mcp("demo")
                calls[:] = [tool_call(add, "definition-changed", a=4, b=5)]
                send = asyncio.create_task(ops.send("changed", stream=False, session_id=session_id))
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
                self.assertNotIn(backup, ops.enabled_tools(session_id))
                await ops.set_tools((*ops.enabled_tools(session_id), backup), session_id=session_id)
                calls[:] = [tool_call(backup, "separate-server", a=2, b=4)]
                send = asyncio.create_task(ops.send("other server", stream=False, session_id=session_id))
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
                    self.assertIn(add, ops.enabled_tools(session_id))
                    self.assertNotIn(add, ops.effective_tools(session_id))

                session_id = (await ops.new_session(session_id=session_id)).id
                advertised.clear()
                calls[:] = [tool_call(backup, "backup", a=4, b=7)]
                send = asyncio.create_task(ops.send("backup", stream=False, session_id=session_id))
                await approve("backup", server="backup")
                self.assertEqual(
                    (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                )
                self.assertEqual(outcomes[-1]["structured_content"], {"sum": 11})
                if os.name == "posix":
                    self.assertNotIn(add, advertised)
                    await ops.reconnect_mcp("demo")
                    self.assertIn(add, ops.effective_tools(session_id))
                    self.assertNotIn((saved, add), app._trusted_tools)
                    pid = int((root / "pid").read_text())
                (root / "changed").touch()
                await ops.refresh_mcp("demo")
                self.assertIn("mcp__demo__sum", ops.list_mcp_tools())
                self.assertNotIn("mcp__demo__sum", ops.enabled_tools(session_id))
                self.assertIn(add, ops.enabled_tools(session_id))
                self.assertNotIn(add, ops.effective_tools(session_id))
                (root / "invalid").touch()
                with self.assertRaisesRegex(RuntimeError, "保留上次目录"):
                    await ops.refresh_mcp("demo")
                self.assertNotIn("mcp__demo__sum", ops.list_mcp_tools())
                self.assertIn("mcp__demo__sum", app._mcp["demo"].tools)
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
            advertised.update(tool["function"]["name"] for tool in body.get("tools", []))
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
                pending = ops.pending_approval(ops.initial_session_id)
                self.assertEqual(pending.details["server"], "remote")
                ops.resolve_approval(
                    pending.request_id, ApprovalDecision.ALLOW_ONCE,
                    session_id=ops.initial_session_id
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
                with patch.dict(os.environ, {"MCP_TEST_TOKEN": token}):
                    self.assertEqual(ops.list_mcp_tools(), ("mcp__remote__add",))
                    connection = app._mcp["remote"]
                    self.assertEqual([prompt.name for prompt in await connection.list_prompts()], ["greeting"])
                    with self.assertRaisesRegex(MCPPromptError, "响应无效"):
                        await connection.get_prompt("greeting", {})
                    self.assertTrue(connection.available)
                    self.assertNotIn("/unexpected", received)
                    await ops.set_tools(("mcp__remote__add",), session_id=ops.initial_session_id)
                    await asyncio.wait_for(subscribed.wait(), 5)
                    send = asyncio.create_task(ops.send("add", stream=False, session_id=ops.initial_session_id))
                    replacement = "mcp__remote__double"
                    async with asyncio.timeout(5):
                        while not changed.is_set():
                            self.assertFalse(send.done(), (outcomes, advertised, received))
                            await asyncio.sleep(0.01)
                        while replacement not in ops.list_mcp_tools():
                            await asyncio.sleep(0.01)
                    self.assertFalse(send.done())
                    self.assertEqual(ops.list_mcp_tools(), (replacement,))
                    with self.assertRaisesRegex(RuntimeError, "当前 Turn"):
                        await ops.refresh_mcp("remote")
                    finish_call.set()
                    turn = await asyncio.wait_for(send, 5)
                    self.assertEqual(turn.status, TurnStatus.COMPLETED)
                    self.assertEqual(outcomes[-1]["structured_content"], {"sum": 42})
                    self.assertEqual(advertised, {"mcp__remote__add"})
                    self.assertEqual(ops.enabled_tools(ops.initial_session_id), ("mcp__remote__add",))
                    self.assertEqual(ops.effective_tools(ops.initial_session_id), ())
                    target = replacement
                    await ops.set_tools((target,), keep_unavailable=True, session_id=ops.initial_session_id)
                    advertised.clear()
                    self.assertEqual(
                        (
                            await asyncio.wait_for(ops.send("double", stream=False, session_id=ops.initial_session_id), 5)
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
                    self.assertNotIn(target, ops.effective_tools(ops.initial_session_id))
                    self.assertIn(target, ops.enabled_tools(ops.initial_session_id))
            finally:
                finish_call.set()
                await ops.close()
                http_server.should_exit = True
                await asyncio.wait_for(serving, 5)
