import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import httpx2
from saturnus import Messages
from test_agent_core import CATALOG, reply, tool_call

from daphnis import History, Operations, TurnStatus
from daphnis.app import Application
from daphnis.compaction import CompactionCheckpoint
from daphnis.config import environment_context_options
from daphnis.events import EventBus
from daphnis.models import ModelInfo
from daphnis.redaction import Redactor
from daphnis.session import Session
from daphnis.tools import create_tools
from daphnis.tools.history import prepare_history
from daphnis.tools.retrieval import HttpEmbedder, RetrievalEngine


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_uses_configured_retrieval_after_compaction(self):
        with patch.dict(os.environ, {"DAPHNIS_RETRIEVAL": "bm25"}, clear=True):
            options = environment_context_options()
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            messages = body["messages"]
            if len(requests) == 1:
                self.assertIn("Use retrieval", messages[0]["content"])
                names = {tool["function"]["name"] for tool in body["tools"]}
                self.assertIn("retrieval", names)
                self.assertNotIn("history_search", names)
                response = reply(None, finish="tool_calls", calls=[
                    tool_call("find", name="retrieval", arguments='{"query":"orchid budget"}'),
                ])
            elif len(requests) == 2:
                hit = json.loads(messages[-1]["content"])["matches"][0]
                self.assertEqual(hit["message_id"], 0)
                response = reply(None, finish="tool_calls", calls=[
                    tool_call("read", name="history_read", arguments=json.dumps(hit["history_read"])),
                ])
            else:
                page = json.loads(messages[-1]["content"])
                self.assertEqual(page["content"], "The orchid budget was 62500.")
                response = reply("62500")
            return httpx.Response(200, json=response)

        history = History(directory / "history")
        state = Session(directory, "compatible", "model-a", enabled_tools=("history_search", "history_read"))
        state.messages.add_user("The orchid budget was 62500.")
        state.messages.add_assistant("Recorded.")
        state.messages.add_user("Next topic.")
        state.messages.add_assistant("Ready.")
        state.compaction = CompactionCheckpoint(2, '{"goals":"budget","progress":"","files":"","next_steps":""}')
        history.save(state)
        app = Application(
            workspace=directory, provider="compatible", model_id="model-a", catalog=CATALOG,
            history=history, retrieval=options["retrieval"],
            connections={"compatible": {"base_url": "https://model.test", "transport": httpx.MockTransport(handler)}},
        )
        self.addAsyncCleanup(app.close)
        ops = Operations(app)
        ops.restore_session(state.id)
        result = await ops.send("Find the old amount.", stream=False)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(len(requests), 3)

    async def test_configured_embedding_endpoint_receives_redacted_text_and_malformed_rows_fall_back(self):
        with patch.dict(os.environ, {
            "DAPHNIS_RETRIEVAL": "hybrid",
            "DAPHNIS_EMBEDDING_URL": "https://embedding.test/embeddings",
            "DAPHNIS_EMBEDDING_MODEL": "fixture",
            "DAPHNIS_EMBEDDING_DIMENSIONS": "2",
        }, clear=True):
            engine = environment_context_options()["retrieval"]
        self.assertIsInstance(engine.embedder, HttpEmbedder)
        secret = "synthetic-embedding-secret"
        bodies = []

        def handler(request):
            self.assertEqual(str(request.url), "https://embedding.test/embeddings")
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                return httpx2.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})
            return httpx2.Response(200, json={"data": [{"index": 0, "embedding": None}]})

        clients = [httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) for _ in range(2)]
        redactor = Redactor([secret])
        messages = Messages([
            {"role": "system", "content": "hidden instructions"},
            {"role": "user", "content": f"orchid budget {secret}"},
            {"role": "user", "content": "current question"},
        ])
        with patch("daphnis.tools.retrieval.httpx.AsyncClient", side_effect=clients):
            directory = self.enterContext(tempfile.TemporaryDirectory())
            state = Session(directory, "compatible", "fixture", messages=messages)
            tools = create_tools(redactor=redactor, retrieval=engine).bind(
                session=state, events=EventBus(), turn_id="test",
            )
            result = await tools.execute(tool_call(
                "find", name="retrieval", arguments=json.dumps({"query": f"orchid {secret}"}),
            ))
        self.assertEqual(bodies, [
            {"model": "fixture", "input": ["orchid budget [REDACTED]"]},
            {"model": "fixture", "input": ["orchid [REDACTED]"]},
        ])
        self.assertEqual((result.value["strategy"], result.value["fallback"]), ("bm25", "embedding_unavailable"))
        self.assertEqual(result.value["matches"][0]["message_id"], 1)

    async def test_offsets_filtering_cache_and_bound_tool(self):
        engine = RetrievalEngine()
        secret = "known-local-secret"
        redactor = Redactor([secret])
        messages = Messages([
            {"role": "system", "content": "orchid hidden system"},
            {"role": "user", "content": "padding " * 110 + "orchid approved budget 62500 " + secret},
            {"role": "assistant", "content": "<think>orchid hidden thought</think>"},
            {"role": "assistant", "tool_calls": [{"id": "protected", "type": "function", "function": {
                "name": "read_file", "arguments": '{"path":".env"}',
            }}]},
            {"role": "tool", "tool_call_id": "protected", "content": '{"content":"orchid blocked file"}'},
            {"role": "user", "content": "current orchid prompt"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            session = Session(workspace=Path(directory), provider="compatible", model_id="fixture", messages=messages)
            index = engine.index(session.id, messages, redactor)
            self.assertIs(engine.index(session.id, messages, redactor), index)
            self.assertIsNot(engine.index("another", messages, redactor), index)
            tools = create_tools(redactor=redactor, retrieval=engine).bind(
                session=session, events=EventBus(), turn_id="test",
            )
            names = {tool["function"]["name"] for tool in tools.definitions()}
            self.assertIn("retrieval", names)
            self.assertNotIn("history_search", names)
            reply = await tools.execute({"id": "lookup", "type": "function", "function": {
                "name": "retrieval", "arguments": '{"query":"orchid approved budget"}',
            }})
            data = json.loads(reply.content)
            self.assertEqual(data["session_id"], session.id)
            self.assertEqual([hit["message_id"] for hit in data["matches"]], [1])
            hit = data["matches"][0]
            self.assertNotIn(secret, hit["excerpt"])
            page = await prepare_history("history_read", hit["history_read"], messages, redactor).run()
            self.assertEqual(page["content"], hit["excerpt"])
            messages.add_assistant("another prior fact")
            messages.add_user("next question")
            self.assertIsNot(engine.index(session.id, messages, redactor), index)

    async def test_vectors_and_fallback(self):
        class Vectors:
            model = "fixture"
            dimensions = 2

            async def embed(self, texts):
                return [[1, 0] if "orchid" in text or "flower" in text else [0, 1] for text in texts]

        messages = Messages([{"role": "user", "content": "orchid label A"},
                             {"role": "assistant", "content": "unrelated shipping"}])
        for strategy in ("semantic", "hybrid"):
            engine = RetrievalEngine(strategy, Vectors())
            index = engine.index("session", messages, Redactor(()), active=False)
            result = await engine.search(index, "session", "flower", 1)
            self.assertEqual(result["strategy"], strategy)
            self.assertEqual(result["matches"][0]["message_id"], 0)
            self.assertEqual((await engine.search(index, "session", "flower", 1))["embedding_inputs"], 1)

        engine = RetrievalEngine("hybrid")
        index = engine.index("session", messages, Redactor(()), active=False)
        fallback = await engine.search(index, "session", "orchid", 1)
        self.assertEqual((fallback["strategy"], fallback["fallback"]), ("bm25", "embedding_not_configured"))
        with self.assertRaises(ValueError):
            await engine.search(index, "another-session", "orchid", 1)

    async def test_saved_tool_preferences_switch_both_ways(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "history")
            model = "fixture"
            connection = {"compatible": {"base_url": "https://offline.invalid"}}
            catalog = (ModelInfo(model, "compatible", supports_tools=True),)
            for old, new, engine in (("history_search", "retrieval", RetrievalEngine()),
                                     ("retrieval", "history_search", None)):
                session = Session(workspace=root, provider="compatible", model_id=model, enabled_tools=(old,))
                history.save(session)
                app = Application(workspace=root, provider="compatible", model_id=model, connections=connection,
                                  history=history, retrieval=engine, catalog=catalog)
                try:
                    ops = Operations(app)
                    ops.restore_session(session.id)
                    self.assertEqual(ops.effective_tools, (new,))
                    self.assertEqual(history.load(session.id).enabled_tools, (old,))
                finally:
                    await app.close()
