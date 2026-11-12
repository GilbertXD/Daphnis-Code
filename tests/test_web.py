import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import httpx
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from daphnis import History, Operations, Session
from daphnis.app import Application
from daphnis.history import SessionLease
from daphnis.tool_state import PENDING, TOOL_STATE
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
        self.overflowed = False

        async def provider(request):
            body = json.loads(request.content)
            self.requests.append(body)
            last = body["messages"][-1]
            if "Summarize the coding conversation" in body["messages"][0]["content"]:
                return httpx.Response(200, json={"choices": [{
                    "index": 0, "message": {"role": "assistant", "content": json.dumps({
                        "goals": "continue", "progress": "recorded old state",
                        "files": "", "next_steps": "reply",
                    })}, "finish_reason": "stop",
                }]})
            if last.get("content") == "overflow" and not self.overflowed:
                self.overflowed = True
                return httpx.Response(400, json={"error": {"code": "context_window_exceeded"}})
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

    def test_context_status_and_diagnostics_preserve_the_full_transcript(self):
        initial = self.snapshot()
        sid = initial["session_id"]
        old_prompt = "long historical text " * 140
        self.command("send", sid, 202, prompt=old_prompt)
        self.wait_for(lambda frame: not frame["status"]["running"] and len(frame["state"]["messages"]) == 2)
        self.command("send", sid, 202, prompt="overflow")
        frame = self.wait_for(
            lambda frame: not frame["status"]["running"] and frame["status"]["compacted_through"] == 2
        )
        self.assertEqual(frame["state"]["messages"][0]["content"], old_prompt)
        self.assertEqual(len(frame["state"]["messages"]), 4)
        self.assertGreater(frame["status"]["context_budget"]["input_tokens"], 0)
        self.assertFalse(frame["status"]["compacting"])
        names = [event["name"] for event in frame["events"]]
        self.assertIn("context.overflow", names)
        self.assertIn("compaction.committed", names)
        summary_requests = [
            event for event in frame["events"]
            if event["name"] == "model.request" and event["data"].get("purpose") == "compaction"
        ]
        self.assertEqual(len(summary_requests), 1)
        self.assertIn("history summary", self.requests[-1]["messages"][1]["content"])

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
        self.assertIn("write_file", frame["state"]["all_tools"])
        self.assertIn("write_file", frame["state"]["enabled_tools"])
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
        frame = self.snapshot()
        second = frame["session_id"]
        self.assertNotEqual(first, second)
        self.assertEqual({row["id"] for row in frame["state"]["sessions"]}, {first, second})
        self.assertEqual(frame["state"]["history_issues"], [])
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

    def test_history_lists_and_restores_valid_sessions_alongside_invalid_files(self):
        self.history.directory.mkdir()
        invalid_files = {
            "legacy.json": b'{"version": 1, "messages": ["private legacy content"]}',
            "legacy-unversioned.json": b"{}",
        }
        for name, content in invalid_files.items():
            (self.history.directory / name).write_bytes(content)

        frame = self.snapshot()
        first = frame["session_id"]
        self.assertEqual(frame["state"]["sessions"], [])
        self.assertEqual(len(frame["state"]["history_issues"]), 2)
        self.assertIsNone(frame["state"]["history_error"])
        self.assertEqual(self.history.list_sessions(), ())

        invalid_files["broken.json"] = b'{"messages": ["private broken content"'
        (self.history.directory / "broken.json").write_bytes(invalid_files["broken.json"])
        archived_workspace = self.workspace.parent / "archived-workspace"
        archived_workspace.mkdir()
        archived = Session(archived_workspace, "deepseek", "deepseek-flash")
        archived.messages.add_user("archived prompt")
        archived.messages.add_assistant("archived reply")
        self.history.save(archived)
        archived_path = self.history.directory / f"{archived.id}.json"
        os.utime(archived_path, (1, 1))
        invalid_files["bad-message.json"] = json.dumps({
            **json.loads(archived_path.read_bytes()),
            "id": "bad-message",
            "messages": [None],
        }).encode()
        (self.history.directory / "bad-message.json").write_bytes(invalid_files["bad-message.json"])

        self.command("send", first, 202, prompt="current prompt")
        frame = self.wait_for(lambda frame: not frame["status"]["running"] and frame["state"]["messages"])
        state = frame["state"]
        self.assertEqual([row["id"] for row in state["sessions"]], [first, archived.id])
        self.assertEqual([row.id for row in self.history.list_sessions()], [first, archived.id])
        self.assertEqual(
            {issue["filename"]: issue["error"] for issue in state["history_issues"]},
            {
                "legacy.json": "UnsupportedSessionVersionError",
                "legacy-unversioned.json": "UnsupportedSessionVersionError",
                "broken.json": "JSONDecodeError",
                "bad-message.json": "ValidationError",
            },
        )
        self.assertIsNone(state["history_error"])
        encoded_state = json.dumps(state)
        self.assertNotIn("private legacy content", encoded_state)
        self.assertNotIn("private broken content", encoded_state)
        self.assertNotIn("archived prompt", encoded_state)

        self.command("new", first)
        frame = self.snapshot()
        fresh = frame["session_id"]
        self.assertNotEqual(fresh, first)
        self.assertEqual({row["id"] for row in frame["state"]["sessions"]}, {fresh, first, archived.id})
        self.command("restore", fresh, 409, id="legacy")
        self.assertEqual(self.snapshot()["session_id"], fresh)
        self.command("restore", fresh, id=first)
        frame = self.snapshot()
        self.assertEqual(frame["session_id"], first)
        self.assertEqual(frame["state"]["messages"], state["messages"])
        self.command("restore", first, id=archived.id)
        frame = self.snapshot()
        self.assertEqual(frame["session_id"], archived.id)
        self.assertEqual(frame["state"]["messages"], archived.messages.snapshot())
        self.assertEqual(frame["state"]["session"]["workspace"], str(archived_workspace.resolve()))
        for name, content in invalid_files.items():
            self.assertEqual((self.history.directory / name).read_bytes(), content)

    def test_history_directory_error_is_reported_separately(self):
        self.history.directory.touch()
        with self.assertRaises(NotADirectoryError):
            self.history.list_sessions()
        state = self.snapshot()["state"]
        self.assertEqual(state["sessions"], [])
        self.assertEqual(state["history_issues"], [])
        self.assertIn("NotADirectoryError", state["history_error"])

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

    def test_budget_pause_and_continue_use_saved_history(self):
        self.bridge.ops._app._request_limits.update(max_primary_requests=5, max_total_requests=1)
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="test patch")
        waiting = self.wait_for(lambda frame: frame["status"]["approval"])
        self.assertEqual(waiting["run"]["base_count"], 0)
        waiting_messages = waiting["state"]["messages"]
        self.assertEqual(waiting_messages[0], {"role": "user", "content": "test patch"})
        self.assertEqual(waiting_messages[-1][TOOL_STATE], PENDING)
        self.command("approve", session_id, request_id=waiting["status"]["approval"]["request_id"], decision="allow_once")
        paused = self.wait_for(lambda frame: not frame["status"]["running"])
        outcome = paused["events"][-1]["data"]
        self.assertEqual(outcome["status"], "PAUSED_LIMIT")
        self.assertIn("总请求上限（1/1）", outcome["notice"])
        self.assertIsNone(outcome["error"])
        self.assertEqual(len(self.requests), 1)
        self.assertIn("tools", self.requests[0])
        messages = paused["state"]["messages"]
        self.assertEqual(messages, self.history.load(session_id).messages.snapshot())
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "tool"])
        self.assertTrue(json.loads(messages[-1]["content"])["ok"])

        self.command("new", session_id)
        self.command("restore", self.snapshot()["session_id"], id=session_id)
        recovery = self.snapshot()["state"]["recovery"]
        self.assertEqual(recovery["kind"], "INTERRUPTED_TURN")
        self.command("send", session_id, 409, prompt="继续")
        self.command("continue", session_id, 202, key=recovery["key"])
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(done["events"][-1]["data"]["status"], "COMPLETED")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[-1]["messages"][1:], messages)
        self.assertIn("tools", self.requests[-1])
        self.command("terminate", session_id, 409, key=recovery["key"])

    def test_failed_user_retry_and_terminate_actions_use_the_current_recovery_key(self):
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="force-error")
        failed = self.wait_for(lambda frame: frame["state"]["recovery"] and not frame["status"]["running"])
        recovery = failed["state"]["recovery"]
        self.assertEqual((recovery["kind"], recovery["message_index"]), ("FAILED_USER", 0))
        self.command("retry", session_id, 202, key=recovery["key"])
        retry = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(len(retry["state"]["messages"]), 1)
        self.assertEqual(self.requests[0]["messages"], self.requests[1]["messages"])
        self.command("retry", session_id, 409, key=recovery["key"])

        self.command("send", session_id, 202, prompt="test patch")
        waiting = self.wait_for(lambda frame: frame["status"]["approval"])
        self.command("cancel", session_id, run_id=waiting["run"]["id"])
        paused = self.wait_for(lambda frame: not frame["status"]["running"])
        recovery = paused["state"]["recovery"]
        self.command("terminate", session_id, 202, key=recovery["key"])
        terminated = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertIsNone(terminated["state"]["recovery"])
        self.assertEqual(terminated["events"][-1]["data"]["status"], "TERMINATED")
        self.assertEqual(len(self.requests), 3)
        result = json.loads(terminated["state"]["messages"][-1]["content"])
        self.assertEqual(result["error"]["code"], "execution_skipped")

    def test_system_prompt_can_be_saved_and_reset(self):
        initial = self.snapshot()
        session_id = initial["session_id"]
        default = initial["state"]["system_prompt"]
        custom = "First line\nSecond line"
        self.command("system", session_id, prompt=custom)
        self.assertEqual(self.snapshot()["state"]["system_prompt"], custom)
        self.assertEqual(self.history.load(session_id).system_prompt, custom)
        self.command("system", session_id, 409, prompt=" ")
        self.command("system", session_id, prompt=None)
        self.assertEqual(self.snapshot()["state"]["system_prompt"], default)
        self.assertIsNone(self.history.load(session_id).system_prompt)

    def test_stream_redaction_stop_and_provider_error(self):
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="pause")
        partial = self.wait_for(lambda frame: frame["live"])
        self.assertEqual(partial["live"]["1"]["content"], "prefix ")
        self.command("cancel", session_id, 409, run_id="old")
        self.command("cancel", session_id, run_id=partial["run"]["id"])
        stopped = self.snapshot()
        self.assertEqual(stopped["state"]["messages"], [{"role": "user", "content": "pause"}])
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
        self.assertEqual(failed["state"]["messages"], [
            *done["state"]["messages"], {"role": "user", "content": "force-error"},
        ])
