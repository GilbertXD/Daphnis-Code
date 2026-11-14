import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from saturnus import Messages
from test_agent_core import (
    CATALOG,
    make_client,
    registry,
    reply,
    session,
    stream_reply,
    tool_call,
)

from daphnis import Agent, Context, History, Operations, Session, TurnStatus
from daphnis.app import Application
from daphnis.compaction import (
    CompactionCheckpoint,
    CompactionError,
    Compactor,
    _summary_text,
    safe_boundaries,
)
from daphnis.models import ModelInfo
from daphnis.terminal_commands import terminal_command
from daphnis.token_budget import request_budget
from daphnis.tools import create_tools
from daphnis.turn_state import ABANDONED, CONTINUE, TERMINATED, TURN_STATE

SUMMARY = json.dumps({
    "goals": "Update a file", "progress": "Inspected the workspace",
    "files": "app.py", "next_steps": "Run checks",
})


class ManualCompactionTests(unittest.IsolatedAsyncioTestCase):
    def test_projection_preserves_omitted_user_task_and_original_reply(self):
        raw = [
            {"role": "user", "content": "写一篇1000字的小说"},
            {"role": "assistant", "content": "# 《无法投递》\n" + "小说正文" * 100},
            {"role": "user", "content": "已作废的要求", TURN_STATE: ABANDONED},
            {"role": "user", "content": "做一个扫雷游戏"},
            {"role": "assistant", "content": "扫雷完成。"},
            {"role": "user", "content": "我们做过什么？"},
        ]
        checkpoint = CompactionCheckpoint(5, json.dumps({
            "goals": "制作小游戏", "progress": "扫雷完成", "files": "minesweeper.html", "next_steps": "",
        }))
        projected = Context().build(Messages(raw), checkpoint)
        summary = projected[1]["content"]
        self.assertIn('[message_id=0] "写一篇1000字的小说"', summary)
        self.assertIn('[message_id=1] "# 《无法投递》"', summary)
        self.assertNotIn("已作废的要求", summary)
        self.assertNotIn("小说正文" * 100, summary)
        self.assertEqual(projected[-1], raw[-1])

    def test_summary_parser_accepts_one_json_object_with_optional_fence(self):
        canonical = _summary_text(SUMMARY)
        for value in (
            f"  {SUMMARY}\n",
            f"\n```JSON  \r\n{SUMMARY}\r\n```  ",
            f"```\n{SUMMARY}\n```",
            f"~~~JsOn\n{SUMMARY}\n~~~",
        ):
            with self.subTest(value=value):
                self.assertEqual(_summary_text(value), canonical)

    def test_summary_parser_rejects_ambiguous_and_invalid_output(self):
        for value, reason in (
            (f"Here is the summary: {SUMMARY}", "summary_json"),
            (f"{SUMMARY}\n{SUMMARY}", "summary_json"),
            (f"```json\n{SUMMARY}\n```\nadditional text", "summary_json"),
            ("[]", "summary_schema"),
            ('{"goals":"partial"}', "summary_schema"),
            (SUMMARY.replace('"next_steps": "Run checks"', '"next_steps": 1'), "summary_schema"),
            (SUMMARY.replace('"next_steps": "Run checks"', '"next_steps": NaN'), "summary_json"),
            (SUMMARY[:-1] + ', "goals": "ambiguous"}', "summary_schema"),
            (SUMMARY[:-1] + ', "extra": "field"}', "summary_schema"),
        ):
            with self.subTest(value=value), self.assertRaises(CompactionError) as failure:
                _summary_text(value)
            self.assertEqual(failure.exception.reason, reason)

    async def test_summary_retries_invalid_format_with_the_same_source(self):
        raw = [
            {"role": "user", "content": "Check the tests."},
            {"role": "assistant", "content": "Checks passed."},
            {"role": "user", "content": "Next task."},
        ]
        prompts = []

        async def summarize(messages, limit):
            self.assertFalse(request_budget(messages, {"output_limit": limit}, 1200).needs_compaction)
            prompts.append([message["content"] for message in messages])
            return '{"goals":["Check the tests"]}' if len(prompts) == 1 else SUMMARY

        checkpoint = await Compactor(1200).summarize(raw, None, 2, summarize)
        self.assertEqual(checkpoint.summary, _summary_text(SUMMARY))
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0][1], prompts[1][1])
        self.assertIn("failed validation", prompts[1][0])
        self.assertIn("Missing history summary fields: progress, files, next_steps.", prompts[1][0])

    async def test_summary_preserves_structured_notes_and_extra_facts(self):
        raw = [
            {"role": "user", "content": "Verify the files."},
            {"role": "assistant", "content": "Verification pending."},
        ]

        async def summarize(messages, limit):
            return json.dumps({
                "goals": "Verify the files",
                "progress": {"checked": ["main.py"]},
                "files": ["main.py", "app.py"],
                "next_steps": [],
                "tests": {"marker": "failure-204"},
            })

        checkpoint = await Compactor(1200).summarize(raw, None, 2, summarize)
        summary = json.loads(checkpoint.summary)
        self.assertEqual(set(summary), {"goals", "progress", "files", "next_steps"})
        self.assertIn('{"checked":["main.py"]}', summary["progress"])
        self.assertIn('{"tests":{"marker":"failure-204"}}', summary["progress"])
        self.assertEqual(json.loads(summary["files"]), ["main.py", "app.py"])
        self.assertEqual(summary["next_steps"], "")

    async def test_history_range_hint_requires_checkpoint_and_both_selected_tools(self):
        for compacted, names, expected in (
            (False, ["history_search", "history_read"], False),
            (True, ["history_search"], False),
            (True, ["history_search", "history_read"], True),
        ):
            with self.subTest(compacted=compacted, names=names), tempfile.TemporaryDirectory() as directory:
                state = Session(directory, "compatible", "model-a")
                state.messages.add_user("earlier original text")
                state.messages.add_assistant("earlier response")
                state.messages.add_user("recent user turn")
                if compacted:
                    state.compaction = CompactionCheckpoint(2, SUMMARY)
                sent = []

                def handler(request, sent=sent):
                    sent.extend(json.loads(request.content)["messages"])
                    return httpx.Response(200, json=reply())

                async with make_client(handler) as client:
                    agent = Agent(
                        client=client, provider="compatible", catalog=CATALOG,
                        tools=create_tools().select(names),
                    )
                    result = await agent.run(session=state, prompt="Continue.")
                self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                self.assertEqual(
                    any("Compacted message range: [0, 2)." in message["content"] for message in sent),
                    expected,
                )
                self.assertNotIn("Compacted message range", json.dumps(state.messages.snapshot()))

    async def test_summary_fragments_cover_the_full_tool_result(self):
        content = "HEAD\n" + "多行 data\n" * 200 + "MIDDLE" + "x" * 4000 + "\nTAIL"
        raw = [
            {"role": "user", "content": "Inspect logs."},
            reply(None, finish="tool_calls", calls=[tool_call()])["choices"][0]["message"],
            {"role": "tool", "tool_call_id": "call-1", "content": content},
            {"role": "user", "content": "Continue."},
        ]
        fragments = []

        async def summarize(messages, limit):
            self.assertFalse(request_budget(messages, {"output_limit": limit}, 1200).needs_compaction)
            fragments.extend(json.loads(line) for line in messages[1]["content"].split(
                "History records:\n", 1,
            )[1].splitlines())
            return SUMMARY

        checkpoint = await Compactor(1200).summarize(raw, None, 3, summarize)
        checkpoint.validate(raw)
        parts = [part for part in fragments if part["message_index"] == 2]
        self.assertGreater(len(parts), 1)
        offset = 0
        for part in parts:
            self.assertEqual(part["tool_call_id"], "call-1")
            self.assertEqual(part["start"], offset)
            self.assertEqual(part["content"], content[offset:part["end"]])
            offset = part["end"]
        self.assertEqual(offset, len(content))
        self.assertEqual(raw[2]["content"], content)

    async def test_summary_folds_only_repetitive_command_diagnostics(self):
        log = {
            "ok": False, "error": {"code": "command_failed", "message": "nonzero exit"},
            "stdout": (
                "".join(f"[{index:04}] TRACE checking cached inputs\n" for index in range(80))
                + "ERROR ERR_PAYLOAD_CHECKSUM failure_id=BLD-7841\n"
                + "".join(f"[{index:04}] TRACE checking cached inputs\n" for index in range(80, 160))
            ),
            "stderr": "WARNING followup needed\n" + "DEBUG check complete\n" * 10,
            "exit_code": 23, "timed_out": False, "terminated": False,
            "stdout_truncated": False, "stderr_truncated": True,
        }
        content = json.dumps(log)
        unique = json.dumps({
            **log,
            "ok": True, "exit_code": 0, "error": None,
            "stdout": "".join(f"[{index:04}] INFO record id={index}\n" for index in range(20)),
            "stderr": "",
        })
        raw = Messages()
        raw.add_user("Build and collect diagnostics.")
        raw.add_assistant(extra={"tool_calls": [
            tool_call("command", name="run_command"),
            tool_call("external", name="mcp_example"),
            tool_call("unique", name="run_command"),
        ]})
        raw.add_tool_result(tool_call_id="external", content=content)
        raw.add_tool_result(tool_call_id="command", content=content)
        raw.add_tool_result(tool_call_id="unique", content=unique)
        raw.add_user("Follow up.")
        original = raw.snapshot()
        records = []

        async def summarize(messages, limit):
            records.extend(
                json.loads(line)
                for line in messages[1]["content"].split("History records:\n", 1)[1].splitlines()
            )
            return SUMMARY

        await Compactor(None).summarize(original, None, 5, summarize)
        results = {record["tool_call_id"]: record for record in records if record["role"] == "tool"}
        self.assertEqual(results["external"]["content"], content)
        self.assertEqual(results["unique"]["content"], unique)
        record = results["command"]
        self.assertEqual(record["message_index"], 3)
        self.assertEqual(record["view"], "folded_command_log")
        self.assertEqual(record["end"], len(record["content"]))
        reduced = json.loads(record["content"])
        self.assertEqual(
            {key: value for key, value in reduced.items() if key not in {"stdout", "stderr"}},
            {key: value for key, value in log.items() if key not in {"stdout", "stderr"}},
        )
        self.assertEqual(reduced["stdout"].splitlines(), [
            "[0000] TRACE checking cached inputs",
            "[Daphnis: 78 repeated log lines folded.]",
            "[0079] TRACE checking cached inputs",
            "ERROR ERR_PAYLOAD_CHECKSUM failure_id=BLD-7841",
            "[0080] TRACE checking cached inputs",
            "[Daphnis: 78 repeated log lines folded.]",
            "[0159] TRACE checking cached inputs",
        ])
        self.assertIn("WARNING followup needed\n", reduced["stderr"])
        self.assertIn("[Daphnis: 8 repeated log lines folded.]", reduced["stderr"])
        self.assertEqual(raw.snapshot(), original)

    async def test_recovery_facts_survive_compaction_when_the_model_omits_them(self):
        def result(code, *, source="daphnis_recovery"):
            return json.dumps({"ok": False, "source": source, "error": {"code": code}})

        raw = Messages()
        raw.add_user("Run the first batch", extra={TURN_STATE: TERMINATED})
        raw.add_assistant(extra={"tool_calls": [tool_call("a"), tool_call("b"), tool_call("c")]})
        raw.add_tool_result(tool_call_id="a", content=result("execution_result_unknown"))
        raw.add_tool_result(tool_call_id="b", content=result("execution_skipped"))
        raw.add_tool_result(tool_call_id="c", content=result("execution_result_unknown", source="other"))
        raw.add_user("Check the workspace")
        raw.add_assistant("Workspace checked")
        raw.add_user("Next task")
        original = raw.snapshot()
        expected = [
            {"message_index": 0, "turn_terminated_by_user": True},
            {"message_index": 2, "tool_call_id": "a", "error_code": "execution_result_unknown"},
            {"message_index": 3, "tool_call_id": "b", "error_code": "execution_skipped"},
        ]
        records = []

        async def summarize(messages, limit):
            for line in messages[1]["content"].split("History records:\n", 1)[1].splitlines():
                record = json.loads(line)
                if record["field"] == "recovery":
                    records.append({"message_index": record["message_index"], **json.loads(record["content"])})
            return SUMMARY

        compactor = Compactor(None)
        first = await compactor.summarize(original, None, 5, summarize)
        self.assertEqual(records, expected)
        second = await compactor.summarize(original, first, 7, summarize)
        for checkpoint in (first, second):
            effective = Context().build(raw, checkpoint)
            summary, facts = effective[1]["content"].rsplit("\n", 1)
            self.assertIn(checkpoint.summary, summary)
            self.assertEqual(json.loads(facts), expected)
            self.assertEqual(effective[2:], original[checkpoint.through:])

        continued = Messages([{**original[0], TURN_STATE: CONTINUE}, *original[1:]])
        effective = Context().build(continued, first)
        summary, facts = effective[1]["content"].rsplit("\n", 1)
        self.assertIn("A turn_terminated_by_user record means", summary)
        self.assertEqual(json.loads(facts), expected[1:])
        self.assertEqual(raw.snapshot(), original)

    async def test_manual_compaction_is_atomic_and_restores_the_effective_context(self):
        bodies = []
        summary_content = SUMMARY

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            if "Summarize" in body["messages"][0]["content"]:
                self.assertNotIn("tools", body)
                response = (
                    reply('{"goals":"unfinished', finish="length")
                    if len(bodies) == 1
                    else reply(summary_content)
                )
                response["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                response = reply("done")
            return httpx.Response(200, json=response)

        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory) / "history")
            app = Application(
                workspace=directory, provider="compatible", model_id="model-a",
                connections={"compatible": {
                    "base_url": "https://model.test/v1", "transport": httpx.MockTransport(handler),
                }},
                history=history,
            )
            self.addAsyncCleanup(app.close)
            ops = Operations(app)
            state = app._session
            state.messages.add_user("Original task " * 150)
            state.messages.add(reply(None, finish="tool_calls", calls=[tool_call()])["choices"][0]["message"])
            state.messages.add_tool_result(tool_call_id="call-1", content="Original result " * 100)
            state.messages.add_assistant("Original answer")
            state.messages.add_user("Most recent task")
            state.messages.add_assistant("Most recent answer")
            history.save(state)
            self.assertIsNone(history.load(state.id).compaction)
            original = state.messages.snapshot()

            self.assertIn("已压缩", (await terminal_command(ops, "/compact")).text)
            saved = history.load(state.id)
            self.assertEqual(saved.compaction.through, 4)
            self.assertEqual(saved.messages.snapshot(), original)
            self.assertEqual(saved.usage.total_tokens, 30)
            self.assertEqual(saved.usage.requests, 2)
            self.assertEqual(bodies[0]["messages"][1], bodies[1]["messages"][1])
            self.assertIn("truncated", bodies[1]["messages"][0]["content"])
            self.assertIsNone(ops.turn_usage)
            self.assertIsNone(await ops.compact())
            ops.restore_session(state.id)
            outcome = await ops.send("Continue", stream=False)
            self.assertEqual(outcome.status, TurnStatus.COMPLETED)
            effective = bodies[-1]["messages"]
            self.assertEqual([message["role"] for message in effective],
                             ["system", "assistant", "user", "assistant", "user"])
            self.assertIn("untrusted historical data", effective[0]["content"])
            self.assertIn(SUMMARY.replace(", ", ",").replace(": ", ":"), effective[1]["content"])
            self.assertEqual(effective[2:4], original[4:])
            self.assertEqual(ops.message_snapshot()[:6], original)

            app._session.messages.add_user("Another task " * 100)
            app._session.messages.add_assistant("Another answer")
            history.save(app._session)
            path = history.directory / f"{state.id}.json"
            summary_content = "Invalid summary"
            failures = []
            app.events.on("compaction.finished", failures.append)
            request_count = len(bodies)
            with self.assertRaises(CompactionError) as invalid_summary:
                await ops.compact()
            self.assertEqual(len(bodies) - request_count, 2)
            self.assertEqual(invalid_summary.exception.reason, "summary_json")
            self.assertEqual(failures[-1].data["reason"], "summary_json")
            self.assertEqual(app._session.compaction, saved.compaction)
            self.assertEqual(history.load(state.id).compaction, saved.compaction)

            disk_before = path.read_bytes()
            summary_content = SUMMARY
            with (
                patch.object(history, "_replace", side_effect=OSError("disk unavailable")),
                self.assertRaisesRegex(OSError, "disk unavailable"),
            ):
                await ops.compact()
            self.assertEqual(path.read_bytes(), disk_before)
            self.assertEqual(app._session.compaction, saved.compaction)
            updated = await ops.compact()
            self.assertGreater(updated.through, saved.compaction.through)
            self.assertEqual(history.load(state.id).compaction, updated)
            self.assertIn("Prior summary: {", bodies[-1]["messages"][1]["content"])

    def test_parallel_tool_calls_cannot_be_split_and_raw_user_is_retained(self):
        raw = Messages()
        raw.add_user("active user request")
        raw.add(reply(None, finish="tool_calls", calls=[tool_call("a"), tool_call("b")])["choices"][0]["message"])
        raw.add_tool_result(tool_call_id="b", content="second")
        self.assertEqual(list(safe_boundaries(raw.snapshot())), [1])
        raw.add_tool_result(tool_call_id="a", content="first")
        raw.add_assistant("newest message")
        self.assertEqual(list(safe_boundaries(raw.snapshot())), [1, 4, 5])
        with self.assertRaisesRegex(ValueError, "incomplete exchange"):
            CompactionCheckpoint(3, SUMMARY).validate(raw.snapshot())
        checkpoint = CompactionCheckpoint(4, SUMMARY)
        checkpoint.validate(raw.snapshot())
        effective = Context().build(raw, checkpoint)
        self.assertEqual(effective[-2:], [raw.get(0), raw.get(4)])

    def test_recent_budget_prefers_whole_user_turns(self):
        raw = [
            {"role": role, "content": content}
            for role, content in (
                ("user", "large past " * 300), ("assistant", "old answer"),
                ("user", "previous question"), ("assistant", "previous answer"),
                ("user", "new question"), ("assistant", "new answer"),
            )
        ]
        compactor = Compactor(1200, recent_tokens=100)
        self.assertEqual(compactor.cut(raw, None, within_turn=True), 2)
        self.assertEqual(compactor.cut(raw, None, within_turn=True, available_tokens=60), 4)


SMALL_MODEL = ModelInfo("model-a", "compatible", context_window=1200, supports_tools=True)


class AutomaticCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_maintenance_exhaustion_keeps_primary_allowance_and_raw_checkpoint(self):
        state = session()
        state.messages.add_user("past data " * 1200)
        state.messages.add_assistant("old answer")
        original = state.messages.snapshot()
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=reply(SUMMARY))

        async with make_client(handler) as client:
            agent = Agent(
                client=client, provider="compatible", catalog=(SMALL_MODEL,),
                max_primary_requests=2, max_maintenance_requests=1, max_total_requests=3,
            )
            outcome = await agent.run(session=state, prompt="continue")
        self.assertEqual((outcome.primary_requests, outcome.maintenance_requests, outcome.model_requests), (0, 1, 1))
        self.assertEqual(len(seen), 1)
        self.assertEqual(outcome.status, TurnStatus.PAUSED_LIMIT)
        self.assertIsNone(outcome.error)
        self.assertIn("用于上下文压缩的请求额度不足", outcome.notice)
        self.assertEqual(state.messages.snapshot(), [*original, {"role": "user", "content": "continue"}])
        self.assertIsNone(state.compaction)

    async def test_long_tool_loop_trims_then_summarizes_and_commits_raw_history(self):
        state = session()
        tool_output = "tool-data " * 150
        calls, requests, events = [], [], []

        def execute(text):
            calls.append(text)
            return tool_output

        def handler(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if "Summarize" in messages[0]["content"]:
                self.assertNotIn("tools", body)
                return httpx.Response(200, json=reply(SUMMARY))
            requests.append(messages)
            if len(requests) <= 7:
                return httpx.Response(200, json=reply(
                    None, finish="tool_calls", calls=[tool_call(str(len(requests)))],
                ))
            return httpx.Response(200, json=reply("finished"))

        async with make_client(handler) as client:
            agent = Agent(
                client=client, provider="compatible", catalog=(SMALL_MODEL,),
                tools=registry(execute),
            )
            for event in ("compaction.finished", "compaction.committed", "context.budget"):
                agent.events.on(event, events.append)
            result = await agent.run(session=state, prompt="active user request")
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(len(calls), 7)
        raw = state.messages.snapshot()
        self.assertEqual([item["content"] for item in raw if item["role"] == "tool"], [tool_output] * 7)
        self.assertIsNotNone(state.compaction)
        state.compaction.validate(raw)
        self.assertTrue(any(event.data.get("status") == "trimmed" for event in events))
        self.assertTrue(any(event.data.get("status") == "staged" for event in events))
        self.assertTrue(any(event.name == "compaction.committed" for event in events))
        for sent in requests:
            self.assertIn(raw[0], sent)
            self.assertIn(len(sent), safe_boundaries(sent))
            self.assertNotIn("Tool output excerpt", sent[-1].get("content", "") or "")
        budgets = [event.data for event in events if event.name == "context.budget"]
        self.assertFalse(any(budget["needs_compaction"] for budget in budgets))

    async def test_cancel_or_failure_after_staging_keeps_committed_checkpoint(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                state = session()
                state.messages.add_user("old request")
                state.messages.add_assistant("old answer")
                state.messages.add_user("older raw data " * 350)
                state.messages.add_assistant("older response")
                prior = CompactionCheckpoint(2, SUMMARY)
                state.compaction = prior
                original = state.messages.snapshot()
                entered, discarded = asyncio.Event(), []

                async def handler(request, entered=entered, cancel=cancel):
                    body = json.loads(request.content)
                    if "Summarize" in body["messages"][0]["content"]:
                        return httpx.Response(200, json=reply(SUMMARY))
                    entered.set()
                    if cancel:
                        await asyncio.Future()
                    return httpx.Response(500, json={"error": "server failure"})

                async with make_client(handler) as client:
                    agent = Agent(client=client, provider="compatible", catalog=(SMALL_MODEL,))
                    agent.events.on("compaction.discarded", discarded.append)
                    task = asyncio.create_task(agent.run(session=state, prompt="continue"))
                    await asyncio.wait_for(entered.wait(), 1)
                    self.assertEqual(state.compaction, prior)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        self.assertEqual((await task).status, TurnStatus.FAILED)
                self.assertEqual(state.messages.snapshot(), [*original, {"role": "user", "content": "continue"}])
                self.assertEqual(state.compaction, prior)
                self.assertEqual(discarded[0].data["through"], 4)

    async def test_disk_failure_keeps_each_history_and_checkpoint_pair_consistent(self):
        def handler(request):
            body = json.loads(request.content)
            summary = "Summarize" in body["messages"][0]["content"]
            return httpx.Response(200, json=reply(SUMMARY if summary else "done"))

        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory) / "history")
            app = Application(
                workspace=directory, provider="compatible", model_id="model-a",
                connections={"compatible": {
                    "base_url": "https://model.test/v1", "transport": httpx.MockTransport(handler),
                }}, history=history, catalog=(SMALL_MODEL,),
            )
            self.addAsyncCleanup(app.close)
            ops = Operations(app)
            await ops.set_tools(())
            app._session.messages.add_user("original " * 600)
            app._session.messages.add_assistant("answer")
            history.save(app._session)
            before = history.load(ops.session.id)
            replace_file = history._replace

            def fail_final_save(path, data):
                if json.loads(data)["messages"][-1]["role"] == "assistant":
                    raise OSError("disk unavailable")
                replace_file(path, data)

            with patch.object(history, "_replace", side_effect=fail_final_save):
                result = await ops.send("continue", stream=False)
            self.assertEqual(result.status, TurnStatus.FAILED)
            self.assertIn("disk unavailable", result.error)
            disk = history.load(ops.session.id)
            self.assertEqual(disk.messages.snapshot(), [
                *before.messages.snapshot(), {"role": "user", "content": "continue"},
            ])
            self.assertIsNone(disk.compaction)
            self.assertEqual(disk.messages.snapshot(), ops.message_snapshot())
            self.assertIsNone(app._session.compaction)
            self.assertEqual((await ops.send("continue again", stream=False)).status, TurnStatus.COMPLETED)
            restored = history.load(ops.session.id)
            self.assertEqual(restored.messages.snapshot(), ops.message_snapshot())
            self.assertEqual(restored.compaction, app._session.compaction)


class OverflowRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_window_recovers_http_overflow_without_reexecuting_tools(self):
        state = session()
        requested, executed, recoveries = [], [], []

        def execute(text):
            executed.append(text)
            return "large result " * 160

        def handler(request):
            body = json.loads(request.content)
            requested.append(body["messages"])
            if len(requested) == 1:
                return httpx.Response(200, json=reply(
                    None, finish="tool_calls", calls=[tool_call()],
                ))
            if len(requested) == 2:
                return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
            self.assertEqual(requested[-1][-1]["tool_call_id"], "call-1")
            self.assertIn("Tool output excerpt", requested[-1][-1]["content"])
            return httpx.Response(200, json=reply())

        async with make_client(handler) as client:
            agent = Agent(client=client, provider="compatible", catalog=CATALOG, tools=registry(execute),
                          max_total_requests=3)
            agent.events.on("context.overflow", recoveries.append)
            result = await agent.run(session=state, prompt="use a tool")
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(result.model_requests, 3)
        self.assertEqual(len(executed), 1)
        self.assertTrue(recoveries[0].data["retrying"])
        self.assertEqual(state.messages.get(2)["content"], "large result " * 160)

    async def test_streaming_overflow_retries_once_and_discards_checkpoint_on_failure(self):
        state = session()
        state.messages.add_user("large prior history " * 150)
        state.messages.add_assistant("prior answer")
        original = state.messages.snapshot()
        events, requested = [], []

        def handler(request):
            body = json.loads(request.content)
            requested.append(body)
            if not body.get("stream"):
                return httpx.Response(200, json=reply(SUMMARY))
            return stream_reply({"error": {"message": "This model's maximum context length is 1000 tokens."}})

        async with make_client(handler) as client:
            agent = Agent(client=client, provider="compatible")
            agent.events.on("context.overflow", events.append)
            result = await agent.run(session=state, prompt="continue", stream=True)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("context window was exceeded", result.error)
        self.assertEqual([bool(body.get("stream")) for body in requested], [True, False, True])
        self.assertEqual([event.data["retrying"] for event in events], [True, False])
        self.assertEqual(events[-1].data["reason"], "retry_exhausted")
        self.assertIsNone(state.compaction)
        self.assertEqual(state.messages.snapshot(), [*original, {"role": "user", "content": "continue"}])

    async def test_only_explicit_overflow_retries_and_requires_reduction(self):
        cases = (
            (429, {"error": {"code": "rate_limit", "message": "Rate limit exceeded"}}, None),
            (400, {"error": {"message": "Invalid tool input"}}, None),
            (400, {"error": {"message": "Unable to determine maximum context length"}}, None),
            (200, reply("truncated", finish="length"), None),
            (400, {"message": "Input token exceed the limit"}, "no_reduction"),
        )
        for status, payload, reason in cases:
            with self.subTest(status=status, reason=reason):
                events, requests = [], []

                def handler(request, requests=requests, status=status, payload=payload):
                    requests.append(request)
                    return httpx.Response(status, json=payload)

                async with make_client(handler) as client:
                    agent = Agent(client=client, provider="compatible")
                    agent.events.on("context.overflow", events.append)
                    result = await agent.run(session=session(), prompt="cannot reduce this")
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertEqual(len(requests), 1)
                self.assertEqual([event.data["reason"] for event in events], [reason] if reason else [])

    async def test_stream_with_visible_output_is_not_retried(self):
        state = session()
        state.messages.add_user("older data " * 150)
        state.messages.add_assistant("older response")
        requested, events = [], []

        def handler(request):
            requested.append(request)
            return stream_reply(
                {"choices": [{"index": 0, "delta": {"content": "visible output"}, "finish_reason": None}]},
                {"error": {"type": "context_window_exceeded"}},
            )

        async with make_client(handler) as client:
            agent = Agent(client=client, provider="compatible")
            agent.events.on("context.overflow", events.append)
            result = await agent.run(session=state, prompt="continue", stream=True)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertEqual(len(requested), 1)
        self.assertEqual(events[0].data["reason"], "stream_started")
