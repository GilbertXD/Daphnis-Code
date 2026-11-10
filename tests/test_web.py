import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from daphnis import History, Operations
from daphnis.app import Application
from daphnis.history import SessionLease
from daphnis.web.bridge import Bridge
from daphnis.web.redaction import Redactor
from daphnis.web.server import create_app

ORIGIN = "http://127.0.0.1:8765"
ACCESS = "test-web-access-credential"
API_KEY = "test-model-key-with-distinct-fragments"


def chunk(delta, finish=None, usage=None):
    return (
        "data: " + json.dumps({
            "id": "stream-test",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **({"usage": usage} if usage else {}),
        }) + "\n\n"
    ).encode()


class PausedReply(httpx.AsyncByteStream):
    def __init__(self, partial, resume, remainder):
        self.partial, self.resume, self.remainder = partial, resume, remainder

    async def __aiter__(self):
        yield self.partial
        await self.resume.wait()
        yield self.remainder


class WebTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.workspace = root / "work"
        self.workspace.mkdir()
        self.history = History(root / "sessions")
        self.requests = []

        async def provider(request):
            body = json.loads(request.content)
            self.requests.append(body)
            last = body["messages"][-1]
            if last.get("content") == "pause":
                self.resume = asyncio.Event()
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=PausedReply(
                    chunk({"role": "assistant", "content": "prefix " + API_KEY[:17]}),
                    self.resume,
                    chunk({"content": API_KEY[17:]}, "stop") + b"data: [DONE]\n\n",
                ))
            if last.get("content") == "force-error":
                return httpx.Response(429, json={"error": {"api_key": API_KEY, "message": "Rate limit"}})
            if last.get("content") == "test patch":
                payload = chunk({"role": "assistant", "tool_calls": [{
                    "index": 0, "id": "call", "type": "function", "function": {
                        "name": "apply_patch", "arguments": json.dumps({
                            "patch": "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+approved\n",
                        }),
                    },
                }]}, "tool_calls")
            else:
                payload = chunk(
                    {"role": "assistant", "content": "done", "reasoning_content": "check"},
                    "stop", {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                content=payload + b"data: [DONE]\n\n",
            )

        async def open_bridge():
            connections = {
                name: {"base_url": "https://model.test/v1", "api_key": API_KEY,
                       "transport": httpx.MockTransport(provider)}
                for name in ("deepseek", "mimo")
            }
            ops = Operations(Application(
                workspace=self.workspace, provider="deepseek", model_id="deepseek-flash",
                history=self.history, connections=connections, interactive=True, observe_model_io=True,
            ))
            self.bridge = Bridge(ops, Redactor((ACCESS, API_KEY)))
            return self.bridge

        self.client = self.enterContext(TestClient(
            create_app(open_bridge, token=ACCESS, port=8765, assets=root),
            base_url=ORIGIN,
        ))
        self.headers = {"Origin": ORIGIN, "Authorization": "Bearer " + ACCESS}

    def command(self, action, session_id, expected=200, **args):
        response = self.client.post("/api/command", headers=self.headers, json={
            "action": action, "session_id": session_id, **args,
        })
        self.assertEqual(response.status_code, expected, response.text)

    def snapshot(self):
        with self.client.websocket_connect(ORIGIN.replace("http:", "ws:") + "/api/events", headers={"Origin": ORIGIN}) as socket:
            socket.send_text(ACCESS)
            return socket.receive_json()

    def wait_for(self, condition):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            frame = self.snapshot()
            if condition(frame):
                return frame
            time.sleep(0.01)
        self.fail("Expected Web state did not appear")

    def test_access_session_and_configuration(self):
        with (
            self.assertRaises(WebSocketDisconnect),
            self.client.websocket_connect(ORIGIN.replace("http:", "ws:") + "/api/events", headers={"Origin": "https://other.test"}),
        ):
            pass
        with self.client.websocket_connect(ORIGIN.replace("http:", "ws:") + "/api/events", headers={"Origin": ORIGIN}) as socket:
            socket.send_text("wrong")
            with self.assertRaises(WebSocketDisconnect) as error:
                socket.receive_json()
            self.assertEqual(error.exception.code, 4401)
        frame = self.snapshot()
        first = frame["session_id"]
        command = {"action": "new", "session_id": first}
        self.assertEqual(self.client.post("/api/command", headers={"Origin": ORIGIN}, json=command).status_code, 401)
        self.assertEqual(self.client.post("/api/command", headers={**self.headers, "Origin": "https://other.test"},
                                          json=command).status_code, 403)
        self.assertEqual(self.client.post("/api/command", headers={**self.headers, "Host": "other.test"},
                                          json=command).status_code, 403)
        self.command("set_approval_mode", first, 400, mode="full")
        self.command("tools", first, 400, names=[{}])
        self.command("rename", first, id=first, title="Original")
        self.command("new", first)
        second = self.snapshot()["session_id"]
        self.assertNotEqual(first, second)
        self.command("send", first, 409, prompt="stale")
        self.command("rename", second, id=first, title="Inactive")
        self.command("restore", second, id=first)
        self.assertEqual(self.snapshot()["state"]["session"]["title"], "Inactive")
        self.command("model", first, provider="mimo", model_id="mimo-v2.6-pro")
        self.command("thinking", first, mode="on")
        self.command("tools", first, names=[])
        state = self.snapshot()["state"]
        self.assertEqual(state["model"]["thinking_modes"], ["off", "on"])
        self.assertEqual(state["thinking"], "on")
        self.assertEqual(state["enabled_tools"], [])
        with SessionLease(self.history.directory / f"{first}.json"):
            self.command("delete", first, 409, id=first)
            self.assertEqual(self.snapshot()["session_id"], first)
        self.command("delete", first, id=first)
        state = self.snapshot()["state"]
        self.assertNotEqual(state["session"]["id"], first)
        self.assertEqual([row["id"] for row in state["sessions"]], [second])
        with self.assertRaises(FileNotFoundError):
            self.history.load(first)

    def test_approval_reconnect_and_exact_model_json(self):
        session_id = self.snapshot()["session_id"]
        self.command("thinking", session_id, mode="high")
        self.command("send", session_id, 202, prompt="test patch")
        waiting = self.wait_for(lambda frame: frame["status"]["approval"])
        approval = waiting["status"]["approval"]
        self.assertFalse((self.workspace / "new.txt").exists())
        self.assertEqual(self.snapshot()["status"]["approval"]["request_id"], approval["request_id"])
        self.command("approve", session_id, 409, request_id="stale", decision="allow_once")
        self.command("approve", session_id, request_id=approval["request_id"], decision="allow_once")
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual((self.workspace / "new.txt").read_text(), "approved\n")
        self.assertEqual(done["state"]["messages"][-1]["content"], "done")
        self.assertEqual(done["status"]["session_usage"]["total_tokens"], 5)
        events = done["events"]
        self.assertEqual([e["id"] for e in events], sorted({e["id"] for e in events}))
        self.assertEqual(
            [e["data"]["body"] for e in events if e["name"] == "model.request"],
            self.requests,
        )
        self.assertEqual(self.requests[0]["reasoning_effort"], "high")
        self.assertTrue(any(e["name"] == "tool.finished" for e in events))
        self.command("new", session_id)
        next_frame = self.snapshot()
        self.assertEqual(next_frame["events"], [])
        self.command("restore", next_frame["session_id"], id=session_id)
        self.assertEqual(self.snapshot()["events"], events)

    def test_stream_redaction_stop_and_provider_error(self):
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="pause")
        partial = self.wait_for(lambda frame: frame["live"])
        self.assertEqual(partial["live"]["1"]["content"], "prefix ")
        self.command("cancel", session_id, 409, run_id="old")
        self.command("cancel", session_id, run_id=partial["run"]["id"])
        stopped = self.snapshot()
        self.assertEqual(stopped["state"]["messages"], [])
        self.assertEqual(stopped["events"][-1]["data"]["status"], "CANCELLED")

        self.command("send", session_id, 202, prompt="pause")
        self.wait_for(lambda frame: frame["live"])
        self.client.portal.call(self.resume.set)
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(done["state"]["messages"][-1]["content"], "prefix [REDACTED]")
        self.assertNotIn(API_KEY, json.dumps(done))

        self.command("send", session_id, 202, prompt="force-error")
        failed = self.wait_for(lambda frame: not frame["status"]["running"])
        response = next(e for e in failed["events"] if e["name"] == "model.response_error")
        self.assertEqual(response["data"]["http_status"], 429)
        self.assertEqual(response["data"]["body"]["error"]["api_key"], "[REDACTED]")
        self.assertNotIn(API_KEY, json.dumps(failed))
        self.assertEqual(failed["state"]["messages"], done["state"]["messages"])
