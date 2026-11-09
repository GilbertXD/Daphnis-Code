import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from saturnus import ValidationError

from daphnis import History, Operations, TurnStatus
from daphnis.app import Application

DEMO = Path(__file__).with_name("mcp_demo_server.py").resolve()


def tool_call(name, call_id, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_tools_approval_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = asyncio.Queue()
            outcomes = []
            calls = []
            advertised = set()

            def model(request):
                body = json.loads(request.content)
                messages = body["messages"]
                advertised.update(tool["function"]["name"] for tool in body["tools"])
                if messages[-1]["role"] == "tool":
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

            async def approve(call_id, approved=True):
                async with asyncio.timeout(5):
                    event = await requests.get()
                pending = ops.pending_approval
                self.assertEqual(event.data["tool_call_id"], call_id)
                self.assertEqual(pending.details["server"], "demo")
                self.assertTrue(
                    ops.resolve_approval(pending.turn_id, call_id, approved)
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
                self.assertNotIn(add, ops.enabled_tools)
                self.assertEqual(ops.list_mcp_servers(), ("demo",))
                ops.set_tools(("list_directory", add, special))
                saved = ops.session.id
                calls[:] = [
                    tool_call("list_directory", "builtin"),
                    tool_call(add, "approved", a=12, b=30),
                    tool_call(special, "denied", a=5, b=9),
                    tool_call(special, "error", a=-2, b=0),
                ]
                send = asyncio.create_task(ops.send("exercise", stream=False))
                details = await approve("approved")
                self.assertEqual(details["arguments"], {"a": 12, "b": 30})
                await approve("denied", False)
                details = await approve("error")
                self.assertEqual(details["tool"], "math.add")
                self.assertEqual(
                    (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                )
                self.assertEqual(advertised, {"list_directory", add, special})
                self.assertTrue(outcomes[0]["ok"])
                self.assertEqual(outcomes[1]["structured_content"], {"sum": 42})
                self.assertEqual(outcomes[1]["text"], "42")
                self.assertEqual(outcomes[2]["error"]["code"], "permission_denied")
                self.assertEqual(outcomes[3]["error"]["code"], "mcp_tool_error")
                self.assertEqual(
                    log.read_text().splitlines(), ["add 12 30", "math.add -2 0"]
                )

                ops.new_session()
                calls[:] = [tool_call(add, "slow", a=-1, b=0)]
                with patch("daphnis.mcp._CALL_TIMEOUT", 0.1):
                    send = asyncio.create_task(ops.send("timeout", stream=False))
                    await approve("slow")
                    self.assertEqual(
                        (await asyncio.wait_for(send, 5)).status, TurnStatus.COMPLETED
                    )
                self.assertEqual(outcomes[-1]["error"]["code"], "mcp_timeout")

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
                pid = int((root / "pid").read_text())
                await ops.disconnect_mcp("demo")
                self.assertNotIn(add, ops.all_tools)
                self.assertEqual(ops.enabled_tools, ("list_directory",))
                with self.assertRaisesRegex(ValidationError, "not registered"):
                    ops.restore_session(saved)
                if os.name == "posix":
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)

                await ops.connect_mcp("demo", sys.executable, [str(DEMO), directory])
                ops.restore_session(saved)
                self.assertIn(add, ops.enabled_tools)
                pid = int((root / "pid").read_text())
            finally:
                await ops.close()
            if os.name == "posix":
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
