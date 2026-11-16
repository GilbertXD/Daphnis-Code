import json
import tempfile
import unittest
from pathlib import Path

import httpx
from saturnus import Messages
from test_agent_core import CATALOG, reply, tool_call

from daphnis import History, Operations, TurnStatus
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.compaction import CompactionCheckpoint
from daphnis.events import EventBus
from daphnis.redaction import Redactor
from daphnis.session import Session
from daphnis.tools import create_tools
from daphnis.tools.history import prepare_history
from daphnis.tools.retrieval import RetrievalEngine


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_uses_retrieval_after_compaction(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        requests = []

        def handler(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                return httpx.Response(200, json=reply("Orchid budget"))
            requests.append(body)
            if len(requests) == 1:
                self.assertIn("use retrieval", messages[0]["content"])
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
            history=history,
            connections={"compatible": {"base_url": "https://model.test", "transport": httpx.MockTransport(handler)}},
        )
        self.addAsyncCleanup(app.close)
        ops = Operations(app)
        ops.restore_session(state.id)
        self.assertEqual(ops.effective_tools(state.id), ("retrieval", "history_read"))
        self.assertEqual(history.load(state.id).enabled_tools, ("history_search", "history_read"))
        result = await ops.send("Find the old amount.", stream=False, session_id=state.id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(len(requests), 3)
        self.assertEqual(history.load(state.id).enabled_tools, ("retrieval", "history_read"))

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
            tools = create_tools(redactor=redactor).bind(
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
