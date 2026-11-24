import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from starlette.testclient import TestClient
from test_agent_core import CATALOG, reply, tool_call

from daphnis import History, Operations, TurnStatus
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.config import environment_embedding
from daphnis.events import EventBus
from daphnis.memory import EmbeddingClient, EmbeddingConfig, MemoryStore
from daphnis.redaction import Redactor
from daphnis.session import Session
from daphnis.tools import ApprovalDecision, ApprovalMode, create_tools
from daphnis.web.bridge import Bridge
from daphnis.web.redaction import Redactor as WebRedactor
from daphnis.web.server import create_app


class MemorySystemTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_session_hybrid_rag_and_session_trust(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        workspace_a = root / "project_a"
        workspace_b = root / "project_b"
        workspace_a.mkdir()
        workspace_b.mkdir()
        history = History(root / "state" / "sessions")
        secret = "sk-secretvalue123456"

        embed_inputs: list[list[str]] = []

        def embed_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            items = body["input"]
            embed_inputs.append(items)
            data = []
            for idx, text in enumerate(items):
                lower = text.casefold()
                if "instruct:" in lower and "缓存过期" in lower:
                    vec = [1.0, 0.0, 0.0]
                elif "redis ttl" in lower:
                    vec = [0.98, 0.1, 0.0]
                elif "简洁中文" in lower:
                    vec = [0.0, 1.0, 0.0]
                else:
                    vec = [0.0, 0.0, 1.0]
                data.append({"index": idx, "embedding": vec})
            return httpx.Response(200, json={"data": data})

        saved_project_id: str | None = None
        turn_step = 0

        def model_handler(request: httpx.Request) -> httpx.Response:
            nonlocal turn_step
            body = json.loads(request.content)
            messages = body["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                return httpx.Response(200, json=reply("Memory test"))
            turn_step += 1
            if turn_step == 1:
                return httpx.Response(200, json=reply(None, finish="tool_calls", calls=[
                    tool_call("w1", name="memory_write", arguments=json.dumps({
                        "scope": "project",
                        "title": "Redis TTL 约定",
                        "content": f"订单会话键统一设置 900 秒过期，凭据 {secret}",
                        "tags": ["cache", "redis"],
                    })),
                ]))
            if turn_step == 2:
                first_res = json.loads(messages[-1]["content"])
                self.assertTrue(first_res["ok"])
                nonlocal saved_project_id
                saved_project_id = first_res["entry"]["id"]
                return httpx.Response(200, json=reply(None, finish="tool_calls", calls=[
                    tool_call("w2", name="memory_write", arguments=json.dumps({
                        "scope": "global",
                        "title": "回复语言偏好",
                        "content": "默认使用简洁中文回复。",
                        "tags": ["preference"],
                    })),
                ]))
            if turn_step == 3:
                second_res = json.loads(messages[-1]["content"])
                self.assertTrue(second_res["ok"])
                return httpx.Response(200, json=reply("已保存两条记忆。"))
            if turn_step == 4:
                return httpx.Response(200, json=reply(None, finish="tool_calls", calls=[
                    tool_call("r1", name="retrieval", arguments=json.dumps({
                        "query": "缓存过期策略怎么配",
                        "source": "memory",
                    })),
                ]))
            if turn_step == 5:
                search_res = json.loads(messages[-1]["content"])
                self.assertEqual(search_res["mode"], "hybrid")
                self.assertEqual(search_res["matches"][0]["id"], saved_project_id)
                self.assertNotIn(secret, search_res["matches"][0]["excerpt"])
                return httpx.Response(200, json=reply(None, finish="tool_calls", calls=[
                    tool_call("m1", name="memory_read", arguments=json.dumps({"id": saved_project_id})),
                ]))
            page = json.loads(messages[-1]["content"])
            self.assertEqual(page["entry"]["id"], saved_project_id)
            self.assertNotIn(secret, page["entry"]["content"])
            return httpx.Response(200, json=reply("900 秒"))

        embedding = EmbeddingConfig(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3-embedding:8b",
            transport=httpx.MockTransport(embed_handler),
        )
        app = Application(
            workspace=workspace_a,
            provider="compatible",
            model_id="model-a",
            catalog=CATALOG,
            history=history,
            interactive=True,
            embedding=embedding,
            connections={
                "compatible": {
                    "base_url": "https://model.test",
                    "api_key": secret,
                    "transport": httpx.MockTransport(model_handler),
                },
            },
        )
        self.addAsyncCleanup(app.close)
        ops = Operations(app)

        approvals: list[dict] = []

        async def auto_trust(event):
            req = event.data["request"]
            approvals.append(req)
            self.assertTrue(req["can_trust"])
            self.assertNotIn(secret, json.dumps(req, ensure_ascii=False))
            ops.resolve_approval(
                req["request_id"], ApprovalDecision.TRUST_SESSION, session_id=event.session_id,
            )

        ops.subscribe("tool.approval_requested", auto_trust)

        session_1 = ops.initial_session_id
        res1 = await ops.send("记住缓存与语言偏好", stream=False, session_id=session_1)
        self.assertEqual(res1.status, TurnStatus.COMPLETED, res1.error)
        # Second memory_write in the same session did not prompt again due to TRUST_SESSION.
        self.assertEqual(len(approvals), 1)

        # Cross-session recall in a fresh session on the same workspace.
        session_2 = (await ops.new_session(session_id=session_1)).id
        res2 = await ops.send("之前定的缓存过期策略是多少？", stream=False, session_id=session_2)
        self.assertEqual(res2.status, TurnStatus.COMPLETED, res2.error)
        self.assertTrue(any(
            any(item.startswith("Instruct:") for item in call) for call in embed_inputs
        ))

        # Workspace isolation: another workspace sees global memory, not project_a's memory.
        other_list, _ = await app._memory.list_entries(workspace_b, scope="all", redactor=Redactor([secret]))
        self.assertEqual([item.scope for item in other_list], ["global"])

        # Session trust does not leak across sessions: updating/deleting in session_2 prompts again.
        approvals.clear()

        async def run_update_and_delete():
            tools = app._tools.select(
                ["memory_write", "memory_delete"], mode=ApprovalMode.AUTO,
            ).bind(session=app._runtimes[session_2].session, events=app.events, turn_id="t2")
            upd = await tools.execute({"id": "u1", "type": "function", "function": {
                "name": "memory_write",
                "arguments": json.dumps({
                    "id": saved_project_id,
                    "scope": "project",
                    "title": "Redis TTL 约定",
                    "content": "订单会话键改为 1800 秒过期。",
                    "tags": ["cache"],
                }),
            }})
            self.assertEqual(json.loads(upd.content)["action"], "updated")
            rem = await tools.execute({"id": "d1", "type": "function", "function": {
                "name": "memory_delete",
                "arguments": json.dumps({"id": saved_project_id, "scope": "project"}),
            }})
            self.assertTrue(json.loads(rem.content)["ok"])

        await asyncio.wait_for(run_update_and_delete(), timeout=5)
        # Prompted once for memory_write (then trusted in session_2) and once for memory_delete (distinct tool name).
        self.assertEqual([item["tool_name"] for item in approvals], ["memory_write", "memory_delete"])

    async def test_embedding_failure_falls_back_to_bm25(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        workspace = root / "ws"
        workspace.mkdir()

        def broken_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Ollama offline")

        client = EmbeddingClient(EmbeddingConfig(
            base_url="http://127.0.0.1:11434/v1",
            model="qwen3-embedding:8b",
            transport=httpx.MockTransport(broken_handler),
        ))
        self.addAsyncCleanup(client.aclose)
        store = MemoryStore(root / "memory", embedding=client)
        redactor = Redactor(())

        entry, created = await store.write_entry(
            workspace,
            scope="project",
            title="uv 构建注意事项",
            content="前端先运行 npm --prefix web run build 再启动 daphnis-web。",
            tags=["build"],
            redactor=redactor,
        )
        self.assertTrue(created)
        result = await store.search(workspace, "daphnis-web build", redactor=redactor)
        self.assertEqual(result["mode"], "bm25_fallback")
        self.assertEqual([item["id"] for item in result["matches"]], [entry.id])

    async def test_memory_directory_blocked_from_file_and_search_tools(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        memory_dir = root / "memory"
        store = MemoryStore(memory_dir)
        await store.write_entry(
            root, scope="global", title="Secret note", content="canary-memory-token", redactor=Redactor(()),
        )
        session = Session(workspace=root, provider="compatible", model_id="fixture")
        tools = create_tools(memory_store=store).bind(session=session, events=EventBus(), turn_id="t1")

        target_file = str(memory_dir / "global.json")
        for call in (
            {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": target_file})}},
            {"id": "2", "type": "function", "function": {"name": "search", "arguments": json.dumps({"query": "canary", "path": str(memory_dir)})}},
            {"id": "3", "type": "function", "function": {"name": "grep", "arguments": json.dumps({"pattern": "canary", "path": str(memory_dir)})}},
        ):
            res = json.loads((await tools.execute(call)).content)
            self.assertFalse(res["ok"])
            self.assertEqual(res["error"]["code"], "sensitive_path")

        # Root workspace search/grep must also skip the nested memory directory.
        search_res = json.loads((await tools.execute({
            "id": "4", "type": "function",
            "function": {"name": "search", "arguments": json.dumps({"query": "canary-memory-token"})},
        })).content)
        self.assertEqual(search_res["matches"], [])
        grep_res = json.loads((await tools.execute({
            "id": "5", "type": "function",
            "function": {"name": "grep", "arguments": json.dumps({"pattern": "canary-memory-token"})},
        })).content)
        self.assertEqual(grep_res["matches"], [])

    def test_environment_embedding_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(environment_embedding())
        with (
            patch.dict(os.environ, {"EMBEDDING_BASE_URL": "http://127.0.0.1:11434/v1"}, clear=True),
            self.assertRaises(ValueError),
        ):
            environment_embedding()
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_BASE_URL": "http://127.0.0.1:11434/v1",
                "EMBEDDING_MODEL": "qwen3-embedding:8b",
            },
            clear=True,
        ):
            cfg = environment_embedding()
            assert cfg is not None
            self.assertEqual(cfg.model, "qwen3-embedding:8b")
            self.assertIsNone(cfg.api_key)

    def test_web_memory_commands_and_shared_state(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        ws_a = root / "ws_a"
        ws_b = root / "ws_b"
        ws_a.mkdir()
        ws_b.mkdir()
        assets = root / "assets"
        assets.mkdir()
        (assets / "index.html").write_text("<!doctype html>")
        token = "web-token-123"
        secret = "sk-websecret999999"

        async def open_bridge():
            app = Application(
                workspace=ws_a,
                provider="compatible",
                model_id="model-a",
                catalog=CATALOG,
                history=History(root / "sessions"),
                interactive=True,
                connections={
                    "compatible": {
                        "base_url": "https://model.test",
                        "api_key": secret,
                        "transport": httpx.MockTransport(lambda req: httpx.Response(200, json=reply("ok"))),
                    },
                },
            )
            return Bridge(Operations(app), WebRedactor([secret]))

        web_app = create_app(open_bridge, token=token, port=8765, assets=assets)
        headers = {
            "Host": "127.0.0.1:8765",
            "Origin": "http://127.0.0.1:8765",
            "Authorization": f"Bearer {token}",
        }
        ws_headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}

        def cmd(client: TestClient, action: str, session_id: str, **fields):
            res = client.post("/api/command", headers=headers, json={"action": action, "session_id": session_id, **fields})
            self.assertEqual(res.status_code, 200, res.text)
            return res.json()

        client = self.enterContext(TestClient(web_app, base_url="http://127.0.0.1:8765"))
        with client.websocket_connect("/api/events", headers=ws_headers) as sock_a:
            sock_a.send_text(token)
            initial = sock_a.receive_json()
            session_a = initial["session_id"]
            self.assertEqual(initial["state"]["memories"], [])

            cmd(client, "project_create", session_a, name="Project B", workspace=str(ws_b))
            frame = sock_a.receive_json()
            project_b = (frame.get("state") or frame["shared"])["projects"][0]["id"]
            session_b = cmd(client, "project_open", session_a, project_id=project_b)["session_id"]

            with client.websocket_connect(f"/api/events?session_id={session_b}", headers=ws_headers) as sock_b:
                sock_b.send_text(token)
                self.assertEqual(sock_b.receive_json()["state"]["memories"], [])

                # Global memory appears in both workspaces; secret is redacted.
                cmd(
                    client, "memory_save", session_a,
                    id=None, scope="global", title="全局风格", content=f"简洁中文 {secret}", tags=["style"],
                )
                update_b = sock_b.receive_json()
                shared_b = update_b.get("state") or update_b["shared"]
                self.assertEqual(len(shared_b["memories"]), 1)
                self.assertNotIn(secret, shared_b["memories"][0]["content"])

            # Project memory in ws_a is isolated from ws_b, can be edited and deleted.
            cmd(
                client, "memory_save", session_a,
                id=None, scope="project", title="构建步骤", content="先跑 npm run build", tags=["web"],
            )
            with client.websocket_connect(f"/api/events?session_id={session_a}", headers=ws_headers) as check_a:
                check_a.send_text(token)
                mems_a = check_a.receive_json()["state"]["memories"]
            self.assertEqual({item["scope"] for item in mems_a}, {"project", "global"})
            proj_entry = next(item for item in mems_a if item["scope"] == "project")

            with client.websocket_connect(f"/api/events?session_id={session_b}", headers=ws_headers) as check_b2:
                check_b2.send_text(token)
                self.assertEqual([item["scope"] for item in check_b2.receive_json()["state"]["memories"]], ["global"])

            # Update in place & delete.
            cmd(
                client, "memory_save", session_a,
                id=proj_entry["id"], scope="project", title="构建步骤", content="已更新构建说明", tags=["web", "ci"],
            )
            cmd(client, "memory_remove", session_a, id=proj_entry["id"], scope="project")
            with client.websocket_connect(f"/api/events?session_id={session_a}", headers=ws_headers) as check_a2:
                check_a2.send_text(token)
                self.assertEqual([item["scope"] for item in check_a2.receive_json()["state"]["memories"]], ["global"])
