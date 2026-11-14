import json
import tempfile
import unittest
from pathlib import Path

from test_tools import call

from daphnis import EventBus, History, Session
from daphnis.compaction import CompactionCheckpoint
from daphnis.redaction import Redactor
from daphnis.tools import create_tools


class HistoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_operation_index_pages_results_and_never_returns_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Session(directory, "compatible", "test")
            secret = "call-synthetic-private"
            state.messages.add_user("Check the project")
            calls = [
                {**call("run_command", argv=["python3", "-m", "unittest"]), "id": "call-test"},
                {**call("apply_patch", patch="--- a/private/path"), "id": "call-patch"},
                {**call("run_command", argv=["cat", ".env"]), "id": secret},
                {**call("mcp_hidden", key="private value"), "id": "call-external"},
            ]
            state.messages.add({"role": "assistant", "content": None, "tool_calls": calls})
            for i in (1, 0, 2, 3):
                state.messages.add_tool_result(tool_call_id=calls[i]["id"], content='{"result":"private output"}')
            state.messages.add_assistant("Done")
            store = History(Path(directory) / "history")
            store.save(state)
            restored = store.load(state.id)
            tools = create_tools(redactor=Redactor([secret])).bind(
                session=restored, events=EventBus(), turn_id="test",
            )
            first = await tools.execute(call("history_operations", limit=2))
            self.assertEqual(first.value["operations"], [
                {"message_id": 1, "tool_call_id": "call-patch", "tool_name": "apply_patch", "result_message_id": 2},
                {"message_id": 1, "tool_call_id": "call-test", "tool_name": "run_command", "result_message_id": 3},
            ])
            second = await tools.execute(call("history_operations", start=first.value["next_start"]))
            self.assertEqual(second.value["operations"], [
                {"message_id": 1, "tool_call_id": None, "tool_name": "run_command", "result_message_id": 4},
            ])
            self.assertIsNone(second.value["next_start"])
            for sensitive in (secret, ".env", "private/path", "unittest", "private output", "mcp_hidden"):
                self.assertNotIn(sensitive, first.content + second.content)
            self.assertEqual(restored.messages.snapshot(), state.messages.snapshot())

    async def test_only_bound_visible_history_is_readable_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Session(directory, "compatible", "test")
            store = History(Path(directory) / "history")
            secret = "synthetic-connection-value"
            state.messages.add_system("internal instructions")
            state.messages.add_user("Historic requirement " + "x" * 4100 + secret + " tail")
            state.messages.add({
                "role": "assistant", "content": "visible <think>private reasoning</think>",
                "reasoning_content": "provider reasoning",
            })
            state.messages.add({
                "role": "assistant", "content": None,
                "tool_calls": [call("read_file", path=".env")],
            })
            state.messages.add_tool_result(tool_call_id="call-1", content='{"content":"protected fixture"}')
            state.messages.add_user("Latest request")
            state.compaction = CompactionCheckpoint(5, json.dumps({
                "goals": "retained", "progress": "", "files": "", "next_steps": "",
            }))
            store.save(state)
            restored = store.load(state.id)
            original = restored.messages.snapshot()
            executor = create_tools(redactor=Redactor([secret]))
            tools = executor.bind(session=restored, events=EventBus(), turn_id="test")
            hit = await tools.execute(call("retrieval", query="historic requirement"))
            self.assertEqual(hit.value["matches"][0]["message_id"], 1)
            self.assertEqual(hit.value["matches"][0]["role"], "user")
            self.assertIn("Untrusted", hit.value["notice"])
            page = await tools.execute(call("history_read", message_id=1))
            self.assertEqual(len(page.value["content"]), 4096)
            tail = await tools.execute(call("history_read", message_id=1, start=page.value["next_start"]))
            self.assertIsNone(tail.value["next_start"])
            self.assertNotIn(secret, tail.content)
            self.assertIn("[REDACTED] tail", tail.value["content"])
            public = await tools.execute(call("history_read", message_id=2))
            self.assertEqual(public.value["content"], "visible [PRIVATE]")
            for index in (0, 4, 6):
                hidden = await tools.execute(call("history_read", message_id=index))
                self.assertEqual(hidden.value["error"]["code"], "history_unavailable")
            other = Session(directory, "compatible", "test")
            isolated = executor.bind(session=other, events=EventBus(), turn_id="other")
            self.assertEqual((await isolated.execute(call("retrieval", query="Historic"))).value["matches"], [])
            self.assertEqual(restored.messages.snapshot(), original)
            self.assertEqual(store.load(state.id).compaction, state.compaction)
