import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from daphnis import History, Operations, Session
from daphnis.agent import TITLE_PROMPT
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
        self.title_requests = []
        self.titles_at_reply = []
        self.overflowed = False
        self.resource_failed = False

        async def provider(request):
            body = json.loads(request.content)
            if body["messages"][0]["content"] == TITLE_PROMPT:
                self.title_requests.append(body)
                source = body["messages"][-1]["content"]
                if source.startswith("[REDACTED]"):
                    return httpx.Response(200, json={
                        "choices": [{
                            "index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "整理首条消息"},
                        }],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                    })
                return httpx.Response(503, json={"error": {"message": "Title unavailable"}})
            self.requests.append(body)
            self.titles_at_reply.append(self.history.load(self.bridge.ops.session.id).title)
            last = body["messages"][-1]
            if "Summarize the conversation" in body["messages"][0]["content"]:
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
            if last.get("content") == "resource-retry" and not self.resource_failed:
                self.resource_failed = True
                return httpx.Response(400, json={"error": {"message": "Fixture first resource turn fails."}})
            if last.get("content") == "force-error":
                return httpx.Response(429, json={"error": {"api_key": API_KEY, "message": "Rate limit"}})
            if last.get("content") == "test mcp":
                payload = chunk({"role": "assistant", "tool_calls": [{
                    "index": 0, "id": "mcp-call", "type": "function", "function": {
                        "name": "mcp__demo__add", "arguments": '{"a":2,"b":3}',
                    },
                }]}, "tool_calls")
            elif last.get("content") == "test patch":
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
        return response.json()

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
        budget = frame["status"]["context_budget"]
        self.assertEqual(budget["hard_limit"], budget["context_window"] * 95 // 100)
        self.assertTrue(budget["auto_enabled"])
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
        self.assertEqual(frame["state"]["approval_mode"], "auto")
        with self.client.websocket_connect(ORIGIN.replace("http:", "ws:") + "/api/events", headers={"Origin": ORIGIN}) as socket:
            socket.send_text(ACCESS)
            socket.receive_json()
            for mode in ("ask", "auto"):
                self.command("set_approval_mode", first, mode=mode)
                self.assertEqual(socket.receive_json()["state"]["approval_mode"], mode)
        self.command("set_approval_mode", first, 400, mode="invalid")
        self.command("set_approval_mode", first, 400, mode="full")
        self.assertEqual(self.snapshot()["state"]["approval_mode"], "auto")
        self.command("confirm_full", first)
        self.assertEqual(self.snapshot()["state"]["approval_mode"], "full")
        self.command("tools", first, 400, names=[{}])
        self.command("send", first, 202, prompt="Save this session")
        self.wait_for(lambda frame: not frame["status"]["running"])
        self.command("rename", first, id=first, title="Original")
        self.command("new", first)
        frame = self.snapshot()
        second = frame["session_id"]
        self.assertNotEqual(first, second)
        self.assertEqual(frame["state"]["approval_mode"], "auto")
        self.assertEqual({row["id"] for row in frame["state"]["sessions"]}, {first})
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
        self.command("confirm_full", first)
        with SessionLease(self.history.directory / f"{first}.json"):
            self.command("delete", first, 409, id=first)
            self.assertEqual(self.snapshot()["session_id"], first)
        self.command("delete", first, id=first)
        state = self.snapshot()["state"]
        self.assertNotEqual(state["session"]["id"], first)
        self.assertEqual(state["approval_mode"], "auto")
        self.assertEqual(state["sessions"], [])
        with self.assertRaises(FileNotFoundError):
            self.history.load(first)

    def test_project_sidebar_commands_keep_session_scope_and_workspace(self):
        first = self.snapshot()["session_id"]
        project_root = self.workspace / "other work"
        project_root.mkdir()
        self.command("project_create", first, 409, name="Missing", workspace="missing")
        self.command("project_open", first, 404, project_id="0" * 32)
        self.command("project_create", first, name="Other work", workspace="other work")
        frame = self.snapshot()
        self.assertEqual(frame["session_id"], first)
        self.assertIsNone(frame["state"]["session"]["project_id"])
        [project] = frame["state"]["projects"]
        self.assertEqual(project["name"], "Other work")
        self.assertEqual(Path(project["workspace"]), project_root.resolve())

        self.command("confirm_full", first)
        self.command("project_open", first, project_id=project["id"])
        state = self.snapshot()["state"]
        opened = state["session"]
        self.assertNotEqual(opened["id"], first)
        self.assertEqual(opened["project_id"], project["id"])
        self.assertEqual(Path(opened["workspace"]), project_root.resolve())
        self.assertEqual(state["approval_mode"], "auto")
        self.assertEqual(state["sessions"], [])
        self.assertEqual(state["messages"], [])
        self.assertIsNone(opened["title"])
        self.command("new", opened["id"])
        fresh = self.snapshot()["state"]["session"]
        self.assertEqual(fresh["project_id"], project["id"])

        self.command("project_open", fresh["id"], project_id=None)
        unscoped = self.snapshot()["state"]["session"]
        self.assertIsNone(unscoped["project_id"])
        self.assertEqual(unscoped["workspace"], opened["workspace"])
        self.assertEqual(self.snapshot()["state"]["sessions"], [])
        self.command("project_open", unscoped["id"], project_id=project["id"])
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="First project message")
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual({row["id"]: row["project_id"] for row in done["state"]["sessions"]}, {
            session_id: project["id"],
        })
        self.assertEqual(self.history.load(session_id).workspace, project_root.resolve())
        self.command("delete", session_id, id=session_id)
        self.assertEqual(self.snapshot()["state"]["session"]["project_id"], project["id"])

    def test_remove_project_keeps_active_and_saved_sessions(self):
        first = self.snapshot()["session_id"]
        project_root = self.workspace / "project"
        project_root.mkdir()
        file = project_root / "keep.txt"
        file.write_text("keep")
        self.command("project_create", first, name="Project", workspace="project")
        [project] = self.snapshot()["state"]["projects"]
        project_id = project["id"]
        self.command("project_open", first, project_id=project_id)
        saved_id = self.snapshot()["session_id"]
        self.command("send", saved_id, 202, prompt="Keep this conversation")
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        messages = done["state"]["messages"]
        self.command("new", saved_id)
        active_id = self.snapshot()["session_id"]

        self.command("project_remove", active_id, project_id=project_id)
        state = self.snapshot()["state"]
        self.assertEqual(state["projects"], [])
        self.assertEqual(state["session"]["id"], active_id)
        self.assertIsNone(state["session"]["project_id"])
        self.assertTrue(all(row["project_id"] is None for row in state["sessions"]))
        self.assertEqual(file.read_text(), "keep")
        self.command("project_remove", active_id, 404, project_id=project_id)

        self.command("restore", active_id, id=saved_id)
        state = self.snapshot()["state"]
        self.assertEqual(state["messages"], messages)
        self.assertIsNone(state["session"]["project_id"])
        self.assertEqual(Path(state["session"]["workspace"]), project_root.resolve())

    def test_move_session_updates_workspace_and_preserves_history(self):
        session_id = self.snapshot()["session_id"]
        other = self.workspace / "other project"
        other.mkdir()
        self.command("project_create", session_id, name="Local", workspace=None)
        self.command("project_create", session_id, name="Other", workspace=other.name)
        local, target = self.snapshot()["state"]["projects"]

        self.command("move", session_id, id=session_id, project_id=local["id"], workspace=local["workspace"])
        state = self.snapshot()["state"]
        self.assertEqual(state["session"]["project_id"], local["id"])
        self.assertEqual(state["sessions"], [])
        self.command("send", session_id, 202, prompt="Keep this conversation")
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        saved = self.history.load(session_id)

        self.command("move", session_id, 409, id=session_id, project_id=target["id"], workspace=local["workspace"])
        self.assertEqual(self.history.load(session_id).project_id, local["id"])
        self.command("new", session_id)
        current_id = self.snapshot()["session_id"]
        with SessionLease(self.history.directory / f"{session_id}.json"):
            self.command("move", current_id, 409, id=session_id, project_id=target["id"], workspace=target["workspace"])
        self.command("move", current_id, id=session_id, project_id=target["id"], workspace=target["workspace"])
        self.assertEqual(self.snapshot()["session_id"], current_id)
        self.command("restore", current_id, id=session_id)
        frame = self.snapshot()
        self.assertEqual(frame["state"]["session"]["project_id"], target["id"])
        self.assertEqual(Path(frame["state"]["session"]["workspace"]), other.resolve())
        self.assertEqual(frame["state"]["messages"], done["state"]["messages"])

        self.command("move", session_id, id=session_id, project_id=None, workspace=None)
        moved = self.history.load(session_id)
        self.assertIsNone(moved.project_id)
        self.assertEqual(moved.workspace, other.resolve())
        self.assertEqual(moved.title, saved.title)
        self.assertEqual(moved.messages.snapshot(), saved.messages.snapshot())
        self.assertEqual(moved.usage, saved.usage)

    def test_project_metadata_does_not_block_session_restore(self):
        first = self.snapshot()["session_id"]
        project_root = self.workspace / "project"
        project_root.mkdir()
        self.command("project_create", first, name="Project", workspace="project")
        [project] = self.snapshot()["state"]["projects"]
        self.command("project_open", first, project_id=project["id"])
        saved_id = self.snapshot()["session_id"]
        self.command("send", saved_id, 202, prompt="Keep this conversation")
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        messages = done["state"]["messages"]
        path = self.history.directory / "projects" / f"{project['id']}.json"

        for metadata in ("{", json.dumps({**project, "workspace": str(self.workspace)})):
            with self.subTest(metadata=metadata):
                path.write_text(metadata)
                state = self.snapshot()["state"]
                self.assertIsNone(state["session"]["project_id"])
                self.assertIsNone(state["sessions"][0]["project_id"])
                self.command("project_open", saved_id, project_id=None)
                self.command("restore", self.snapshot()["session_id"], id=saved_id)
                state = self.snapshot()["state"]
                self.assertEqual(state["messages"], messages)
                self.assertIsNone(state["session"]["project_id"])
                self.assertEqual(Path(state["session"]["workspace"]), project_root.resolve())

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
        self.assertEqual({row["id"] for row in frame["state"]["sessions"]}, {first, archived.id})
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
        self.command("confirm_full", session_id, 409)
        self.assertEqual(self.snapshot()["state"]["approval_mode"], "auto")
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
            [e["data"]["body"] for e in events if e["name"] == "model.request" and e["data"].get("purpose") != "title"],
            self.requests,
        )
        self.assertEqual(self.requests[0]["reasoning_effort"], "high")
        self.assertTrue(any(e["name"] == "tool.finished" for e in events))
        self.command("new", session_id)
        next_frame = self.snapshot()
        self.assertEqual(next_frame["events"], [])
        self.command("restore", next_frame["session_id"], id=session_id)
        self.assertEqual(self.snapshot()["events"], events)
        self.command("confirm_full", next_frame["session_id"], 409)
        self.command("confirm_full", session_id)
        (self.workspace / "new.txt").unlink()
        self.command("send", session_id, 202, prompt="test patch")
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual((self.workspace / "new.txt").read_text(), "approved\n")
        self.assertIsNone(done["status"]["approval"])
        self.assertEqual(sum(e["name"] == "tool.approval_requested" for e in done["events"]), 1)

    def test_mcp_management_updates_tools_and_preserves_approval(self):
        sid = self.snapshot()["session_id"]
        options = {
            "transport": "stdio", "command": sys.executable, "cwd": str(self.workspace),
            "args": [str(Path(__file__).with_name("mcp_demo_server.py").resolve()), "."],
            "env": {"DEMO_TOKEN": "PATH"},
        }
        self.command("mcp_save", sid, 409, name="demo", enabled=False, create=True,
                     options={"transport": "streamable_http", "url": f"https://mcp.example/{API_KEY}"})
        self.command("mcp_save", sid, name="demo", enabled=False, options=options, create=True)
        (server,) = self.snapshot()["state"]["mcp_servers"]
        self.assertFalse(server["enabled"])
        self.assertEqual(server["options"]["env"], options["env"])
        self.assertEqual(server["state"], "Disconnected")
        self.command("mcp_enable", sid, name="demo", enabled=True)
        state = self.snapshot()["state"]
        self.assertEqual(state["mcp_servers"][0]["tool_count"], 2)
        self.assertIn("mcp__demo__add", state["all_tools"])
        self.command("tools", sid, names=["mcp__demo__add"])
        self.command("send", sid, 202, prompt="test mcp")
        approval = self.wait_for(lambda frame: frame["status"]["approval"])["status"]["approval"]
        self.command("mcp_enable", sid, 409, name="demo", enabled=False)
        self.command("approve", sid, request_id=approval["request_id"], decision="allow_once")
        self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual((self.workspace / "calls").read_text(), "add 2 3\n")
        self.command("mcp_restart", sid, name="demo")
        self.command("mcp_enable", sid, name="demo", enabled=False)
        state = self.snapshot()["state"]
        self.assertIn("mcp__demo__add", state["enabled_tools"])
        self.assertNotIn("mcp__demo__add", state["all_tools"])
        self.command("mcp_save", sid, name="demo", enabled=False, create=False,
                     options={"transport": "streamable_http", "url": "https://mcp.example/mcp"})
        self.assertEqual(self.snapshot()["state"]["mcp_servers"][0]["options"]["transport"], "streamable_http")
        self.command("mcp_delete", sid, name="demo")
        self.assertEqual(self.snapshot()["state"]["mcp_servers"], [])

    def test_mcp_prompt_preview_preserves_message_order_without_changing_session(self):
        sid = self.snapshot()["session_id"]
        (self.workspace / "prompts").touch()
        self.command("mcp_save", sid, name="demo", enabled=True, create=True, options={
            "transport": "stdio", "command": sys.executable, "cwd": str(self.workspace),
            "args": [str(Path(__file__).with_name("mcp_demo_server.py").resolve()), "."],
        })
        before = self.snapshot()
        self.assertEqual(before["state"]["mcp_prompt_servers"], ["demo"])
        prompts = self.command("mcp_list_prompts", sid, server="demo")["mcp_prompts"]
        self.assertEqual(prompts[0]["name"], "greet")
        self.assertTrue(prompts[0]["arguments"][0]["required"])
        result = self.command("mcp_get_prompt", sid, server="demo", name="greet",
                              arguments={"who": API_KEY})["mcp_prompt_result"]
        self.assertTrue(result["redacted"])
        self.assertEqual(result["messages"], [
            {"role": "user", "content": {"type": "text", "text": "Hello, [REDACTED]! (neutral)"}},
        ])
        bundle = self.command("mcp_get_prompt", sid, server="demo", name="bundle",
                              arguments={})["mcp_prompt_result"]
        self.assertEqual(bundle["messages"], [
            {"role": "user", "content": {"type": "text", "text": "first"}},
            {"role": "assistant", "content": {"type": "image"}},
        ])
        self.assertIsNone(bundle["preview_id"])
        self.command("mcp_run_prompt", sid, 409, preview_id="a" * 32, texts=["first"])
        self.command("mcp_get_prompt", sid, 409, server="demo", name="greet", arguments={"who": 7})
        after = self.snapshot()
        self.assertEqual(after["state"]["messages"], before["state"]["messages"])
        self.assertEqual(after["state"]["enabled_tools"], before["state"]["enabled_tools"])
        self.assertEqual(self.requests, [])
        self.command("mcp_run_prompt", sid, 400, preview_id=result["preview_id"], texts=[API_KEY])
        visible_text = result["messages"][0]["content"]["text"]
        self.command("mcp_run_prompt", sid, 202, preview_id=result["preview_id"], texts=[visible_text])
        self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(self.requests[0]["messages"][1:], [{"role": "user", "content": visible_text}])

        dialogue = self.command("mcp_get_prompt", sid, server="demo", name="dialogue",
                                arguments={})["mcp_prompt_result"]
        preview_id = dialogue["preview_id"]
        texts = ["Review this code.", "I will look for regressions.", "Begin."]
        self.command("mcp_run_prompt", sid, 400, preview_id=preview_id, texts=texts[:2])
        self.command("mcp_run_prompt", sid, 202, preview_id=preview_id, texts=texts)
        saved = self.history.load(sid)
        group = saved.prompt_groups[-1]
        self.assertEqual((group.server, group.prompt, group.edited), ("demo", "dialogue", True))
        expected = [{"role": role, "content": text} for role, text in zip(("user", "assistant", "user"), texts)]
        self.assertEqual(saved.messages.snapshot()[group.start:group.end], expected)
        self.command("mcp_run_prompt", sid, 202, preview_id=preview_id, texts=texts)
        self.command("mcp_run_prompt", sid, 409, preview_id=preview_id, texts=[*texts[:2], "Different."])
        reconnected = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[-1]["messages"][-3:], expected)
        self.assertEqual(reconnected["state"]["prompt_groups"][-1]["id"], preview_id)

        old_preview = self.command("mcp_get_prompt", sid, server="demo", name="dialogue",
                                    arguments={})["mcp_prompt_result"]["preview_id"]
        self.command("new", sid)
        self.command("mcp_list_prompts", sid, 409, server="demo")
        self.command("mcp_run_prompt", self.snapshot()["session_id"], 409, preview_id=old_preview, texts=texts)

    def test_embedded_text_resource_keeps_role_and_uses_snapshot_on_retry(self):
        sid = self.snapshot()["session_id"]
        for name in ("prompts", "resources"):
            (self.workspace / name).touch()
        text = self.workspace / "resource-text"
        text.write_text("original material " + API_KEY)
        self.command("mcp_save", sid, name="demo", enabled=True, create=True, options={
            "transport": "stdio", "command": sys.executable, "cwd": str(self.workspace),
            "args": [str(Path(__file__).with_name("mcp_demo_server.py").resolve()), "."],
        })
        preview = self.command("mcp_get_prompt", sid, server="demo", name="embedded",
                               arguments={})["mcp_prompt_result"]
        self.assertTrue(preview["redacted"])
        expected = "original material [REDACTED]"
        self.assertEqual(preview["messages"][1]["content"]["resource"]["text"], expected)
        fields = {"preview_id": preview["preview_id"], "texts": ["Review this resource.", "", "resource-retry"]}
        self.command("mcp_run_prompt", sid, 400, **{**fields, "texts": ["Review", "forged", "resource-retry"]})
        self.command("mcp_run_prompt", sid, 202, **fields)
        failed = self.wait_for(lambda frame: not frame["status"]["running"])
        saved = self.history.load(sid)
        item, = saved.attachment_refs[1].files
        self.assertEqual((item.source, item.server, item.uri), ("embedded", "demo", "notes://embedded-source"))
        self.assertEqual(self.history.attachments.read(sid, item), expected)
        self.assertEqual([m["role"] for m in saved.messages.snapshot()], ["user", "assistant", "user"])
        projected = self.requests[0]["messages"][-2]
        self.assertEqual(projected["role"], "assistant")
        self.assertIn(expected, projected["content"])
        self.assertNotIn(item.uri, projected["content"])
        text.write_text("new remote material")
        self.command("new", sid)
        self.command("restore", self.snapshot()["session_id"], id=sid)
        self.command("retry", sid, 202, key=failed["state"]["recovery"]["key"])
        self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(self.requests[-1]["messages"], self.requests[0]["messages"])
        self.assertFalse((self.workspace / "resource-reads").exists())
        self.command("mcp_run_prompt", sid, 202, **fields)
        self.assertEqual(len(self.requests), 2)

    def test_standalone_resource_draft_is_session_scoped_and_replay_safe(self):
        sid = self.snapshot()["session_id"]
        (self.workspace / "resources").touch()
        remote = self.workspace / "resource-text"
        remote.write_text("standalone material")
        self.command("mcp_save", sid, name="demo", enabled=True, create=True, options={
            "transport": "stdio", "command": sys.executable, "cwd": str(self.workspace),
            "args": [str(Path(__file__).with_name("mcp_demo_server.py").resolve()), "."],
        })
        self.assertEqual(self.snapshot()["state"]["mcp_resource_servers"], ["demo"])
        catalog = self.command("mcp_list_resources", sid, server="demo")["mcp_resource_catalog"]
        template = next(entry for entry in catalog["entries"] if entry["variables"])
        self.assertEqual(template["variables"], ["name"])
        reads = self.workspace / "resource-reads"
        self.assertFalse(reads.exists())
        preview, = self.command(
            "mcp_read_resource", sid, server="demo", uri=template["uri"], arguments={"name": "Review/二"},
        )["mcp_resource_preview"]
        self.assertEqual(preview["text"], "standalone material")
        self.assertEqual(preview["uri"], "notes://pages/Review%2F%E4%BA%8C")
        self.command("mcp_read_resource", sid, 409, server="demo", uri="notes://blob", arguments=None)
        preview_reads = reads.read_text()
        self.assertEqual(self.snapshot()["state"]["messages"], [])
        self.command("send", sid, 202, prompt="Start.")
        self.wait_for(lambda frame: not frame["status"]["running"])
        payload = {
            "action": "send_attachments", "session_id": sid, "submission_id": "b" * 32, "prompt": "Read these.",
            "attachments": [{"resource_id": preview["id"]}, {"name": "local.txt", "text": "uploaded material"}],
        }
        self.command("new", sid)
        other = self.snapshot()["session_id"]
        path = "/api/attachments/send"
        response = self.client.post(path, headers=self.headers, json={**payload, "session_id": other})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.snapshot()["state"]["messages"], [])
        self.command("restore", other, id=sid)
        remote.write_text("changed remote content")
        response = self.client.post(path, headers=self.headers, json=payload)
        self.assertEqual(response.status_code, 202, response.text)
        saved = self.history.load(sid)
        (index, batch), = saved.attachment_refs.items()
        self.assertEqual(saved.messages.get(index)["content"], "Read these.")
        item, uploaded = batch.files
        self.assertEqual((item.source, item.uri), ("standalone", preview["uri"]))
        self.assertEqual(self.history.attachments.read(sid, item), "standalone material")
        self.assertEqual(self.history.attachments.read(sid, uploaded), "uploaded material")
        self.wait_for(lambda frame: not frame["status"]["running"])
        content = self.requests[-1]["messages"][-1]["content"]
        self.assertIn("standalone material", content)
        self.assertNotIn(item.uri, content)
        self.bridge.resource_drafts.clear()
        repeated = self.client.post(path, headers=self.headers, json=payload)
        self.assertEqual(repeated.status_code, 202, repeated.text)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(reads.read_text(), preview_reads)

    def test_attachment_send_is_committed_before_ack_and_deduplicated(self):
        sid = self.snapshot()["session_id"]
        path = "/api/attachments/send"
        marker = "upload-private-observation-marker"
        payload = {
            "action": "send_attachments", "session_id": sid, "submission_id": "a" * 32,
            "prompt": "", "attachments": [{"name": "notes.txt", "text": marker}],
        }
        invalid = {**payload, "attachments": [{"name": "../notes.txt", "text": marker}]}
        rejected = self.client.post(path, headers=self.headers, json=invalid)
        self.assertEqual(rejected.status_code, 409, rejected.text)
        self.assertEqual(self.snapshot()["state"]["messages"], [])
        wrong_route = self.client.post("/api/command", headers=self.headers, json=payload)
        self.assertEqual(wrong_route.status_code, 400)
        response = self.client.post(path, headers=self.headers, json=payload)
        self.assertEqual(response.status_code, 202, response.text)
        saved = self.history.load(sid)
        item = saved.attachment_refs[0].files[0]
        self.assertEqual(self.history.attachments.read(sid, item), marker)
        repeated = self.client.post(path, headers=self.headers, json=payload)
        self.assertEqual(repeated.status_code, 202, repeated.text)
        conflict = self.client.post(path, headers=self.headers, json={**payload, "prompt": "different"})
        self.assertEqual(conflict.status_code, 409, conflict.text)
        frame = self.wait_for(lambda item: item["run"] is None)
        self.assertEqual(len(frame["state"]["messages"]), 2)
        self.assertEqual(len(self.requests), 1)
        self.assertIn(marker, self.requests[0]["messages"][-1]["content"])
        self.assertEqual(frame["state"]["attachment_refs"], {"0": [{"id": item.id, "name": "notes.txt"}]})
        self.assertNotIn(marker, json.dumps(frame))

        oversized_command = self.client.post("/api/command", headers=self.headers, json={
            "action": "send", "session_id": sid, "prompt": "x" * (256 * 1024),
        })
        self.assertEqual(oversized_command.status_code, 413)
        large = {**payload, "submission_id": "b" * 32, "attachments": [
            {"name": "one.txt", "text": "A" * (128 * 1024)},
            {"name": "two.txt", "text": "B" * (128 * 1024)},
        ]}
        uploaded = self.client.post(path, headers=self.headers, json=large)
        self.assertEqual(uploaded.status_code, 202, uploaded.text)
        self.wait_for(lambda item: item["run"] is None)
        self.assertEqual(len(self.history.load(sid).attachment_refs[2].files), 2)
        self.assertNotIn("A" * 1024, json.dumps(self.requests[-1]))

    def test_budget_pause_and_continue_use_saved_history(self):
        self.bridge.ops._app._request_limits.update(max_primary_requests=5, max_total_requests=1)
        session_id = self.snapshot()["session_id"]
        project_root = self.workspace / "different workspace"
        project_root.mkdir()
        self.command("project_create", session_id, name="Other", workspace=project_root.name)
        [project] = self.snapshot()["state"]["projects"]
        move = {"id": session_id, "project_id": project["id"], "workspace": project["workspace"]}
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
        self.assertEqual(paused["state"]["sessions"][0]["recovery_kind"], "INTERRUPTED_TURN")
        self.command("move", session_id, 409, **move)

        self.command("new", session_id)
        fresh = self.snapshot()["session_id"]
        self.command("move", fresh, 409, **move)
        self.assertEqual(self.history.load(session_id).workspace, self.workspace.resolve())
        self.command("restore", fresh, id=session_id)
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
        self.command("move", session_id, **move)
        self.assertEqual(self.history.load(session_id).workspace, project_root.resolve())

    def test_failed_user_retry_and_terminate_actions_use_the_current_recovery_key(self):
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="force-error")
        failed = self.wait_for(lambda frame: frame["state"]["recovery"] and not frame["status"]["running"])
        recovery = failed["state"]["recovery"]
        self.assertEqual(failed["state"]["session"]["title"], "force-error")
        self.assertEqual((recovery["kind"], recovery["message_index"]), ("FAILED_USER", 0))
        self.command("retry", session_id, 202, key=recovery["key"])
        retry = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(len(retry["state"]["messages"]), 1)
        self.assertEqual(retry["state"]["session"]["title"], "force-error")
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

    def test_session_title_uses_first_prompt_and_preserves_manual_names(self):
        session_id = self.snapshot()["session_id"]
        prompt = f"  {API_KEY}\n\t{'一' * 80}"
        self.command("thinking", session_id, mode="high")
        self.command("send", session_id, 202, prompt=prompt)
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        expected = "整理首条消息"
        self.assertEqual(done["state"]["session"]["title"], expected)
        self.assertEqual(done["state"]["sessions"][0]["title"], expected)
        self.assertEqual(self.history.load(session_id).title, expected)
        self.assertEqual(
            [message["content"] for message in done["state"]["messages"]],
            [prompt.replace(API_KEY, "[REDACTED]"), "done"],
        )
        [title_request] = self.title_requests
        self.assertEqual(title_request["reasoning_effort"], "none")
        self.assertNotIn("tools", title_request)
        self.assertEqual(title_request["messages"][1]["content"], prompt.replace(API_KEY, "[REDACTED]").strip())
        self.assertEqual(self.requests[0]["reasoning_effort"], "high")
        self.assertEqual(self.titles_at_reply, ["[REDACTED] " + "一" * 28 + "…"])
        self.assertNotIn(expected, json.dumps(self.requests[0], ensure_ascii=False))
        started = [e for e in done["events"] if e["name"] == "model.started"]
        self.assertEqual([e["data"].get("purpose") for e in started], [None, "title"])
        finished = next(e for e in done["events"] if e["name"] == "turn.finished")
        self.assertLess(finished["id"], started[1]["id"])
        reply_finished, title_finished = [e["data"] for e in done["events"] if e["name"] == "model.finished"]
        self.assertGreater(reply_finished["chunks"], 0)
        self.assertNotIn("chunks", title_finished)
        self.assertEqual(done["status"]["turn_usage"]["total_tokens"], 5)
        self.assertEqual(done["status"]["session_usage"]["total_tokens"], 17)

        self.command("new", session_id)
        fresh = self.snapshot()["session_id"]
        self.command("rename", fresh, id=fresh, title="手动命名")
        self.command("send", fresh, 202, prompt="首条消息")
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(done["state"]["session"]["title"], "手动命名")
        self.assertEqual(self.history.load(fresh).title, "手动命名")
        self.assertEqual(len(self.title_requests), 1)

    def test_failed_title_request_keeps_truncated_first_prompt(self):
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="  修复\n\t" + "字" * 60)
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        expected = "修复 " + "字" * 36 + "…"
        self.assertEqual(done["state"]["session"]["title"], expected)
        self.assertEqual(self.titles_at_reply, [expected])
        self.assertEqual(done["state"]["messages"][-1]["content"], "done")

    def test_new_session_settings_are_saved_with_first_message(self):
        initial = self.snapshot()
        session_id = initial["session_id"]
        default = initial["state"]["system_prompt"]
        custom = "First line\nSecond line"
        self.command("model", session_id, provider="mimo", model_id="mimo-v2.6-pro")
        self.command("thinking", session_id, mode="on")
        self.command("tools", session_id, names=["read_file"])
        self.command("system", session_id, prompt=custom)
        state = self.snapshot()["state"]
        self.assertEqual(state["system_prompt"], custom)
        self.assertEqual(state["sessions"], [])
        self.assertEqual(self.history.list_sessions(), ())
        self.command("send", session_id, 202, prompt="First message")
        self.wait_for(lambda frame: not frame["status"]["running"])
        saved = self.history.load(session_id)
        self.assertEqual((saved.provider, saved.model_id), ("mimo", "mimo-v2.6-pro"))
        self.assertEqual(saved.thinking, "on")
        self.assertEqual(saved.enabled_tools, ("read_file",))
        self.assertEqual(saved.system_prompt, custom)
        self.assertEqual(saved.messages.snapshot()[0]["content"], "First message")
        self.command("system", session_id, 409, prompt=" ")
        self.command("system", session_id, prompt=None)
        self.assertEqual(self.snapshot()["state"]["system_prompt"], default)
        self.assertIsNone(self.history.load(session_id).system_prompt)

    def test_stream_redaction_stop_and_provider_error(self):
        session_id = self.snapshot()["session_id"]
        self.command("send", session_id, 202, prompt="pause")
        partial = self.wait_for(lambda frame: frame["live"])
        self.assertEqual(partial["live"]["1"]["content"], "prefix ")
        self.assertEqual(partial["state"]["session"]["title"], "pause")
        self.command("cancel", session_id, 409, run_id="old")
        self.command("cancel", session_id, run_id=partial["run"]["id"])
        stopped = self.snapshot()
        self.assertEqual(stopped["state"]["messages"], [{"role": "user", "content": "pause"}])
        self.assertEqual(stopped["events"][-1]["data"]["status"], "CANCELLED")
        self.assertEqual(self.title_requests, [])

        self.command("send", session_id, 202, prompt="pause")
        self.wait_for(lambda frame: frame["live"])
        self.client.portal.call(self.resume.set)
        done = self.wait_for(lambda frame: not frame["status"]["running"])
        self.assertEqual(done["state"]["messages"][-1]["content"], "prefix [REDACTED]")
        self.assertNotIn(API_KEY, json.dumps(done))

        self.command("send", session_id, 202, prompt="force-error")
        failed = self.wait_for(lambda frame: not frame["status"]["running"])
        response = next(e for e in failed["events"] if e["name"] == "model.response_error" and e["data"].get("purpose") != "title")
        self.assertEqual(response["data"]["http_status"], 429)
        self.assertEqual(response["data"]["body"]["error"]["api_key"], "[REDACTED]")
        self.assertNotIn(API_KEY, json.dumps(failed))
        self.assertEqual(failed["state"]["messages"], [
            *done["state"]["messages"], {"role": "user", "content": "force-error"},
        ])
