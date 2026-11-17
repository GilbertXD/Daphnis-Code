import json
import os
import tempfile
import unittest
from pathlib import Path

from daphnis.attachments import (
    TextAttachmentInput,
    TextResourceInput,
    prepare_attachments,
)
from daphnis.history import History, SessionBusyError
from daphnis.messages import Messages
from daphnis.session import Session


class AttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_snapshot_history_round_trip_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "sessions")
            session = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            uploads = [TextAttachmentInput("notes.txt", "first\n"), TextAttachmentInput("notes.txt", "second")]
            batch = prepare_attachments(uploads, "a" * 32)
            self.assertNotEqual(batch.files[0].id, batch.files[1].id)
            session.messages.add_user("Read these")
            async with history.lease(session) as lease:
                history.attachments.write(session.id, batch, uploads)
                session.attachment_refs = {0: batch}
                await history.save_async(session, lease=lease)
                with self.assertRaises(SessionBusyError):
                    await history.delete(history.load(session.id))
            reopened = History(history.directory).load(session.id)
            self.assertEqual(reopened.attachment_refs, session.attachment_refs)
            self.assertEqual(reopened.messages.get(0)["content"], "Read these")
            raw = json.loads((history.directory / f"{session.id}.json").read_text())
            self.assertEqual(raw["version"], 2)
            self.assertNotIn("text", raw["attachment_refs"]["0"]["files"][0])
            self.assertEqual([history.attachments.read(session.id, item) for item in batch.files], ["first\n", "second"])
            store = root / "attachments" / session.id
            snapshot = store / (batch.files[0].id + ".txt")
            self.assertEqual(snapshot.stat().st_mode & 0o777, 0o400)
            other = root / "hardlink"
            os.link(snapshot, other)
            with self.assertRaises(OSError):
                history.attachments.read(session.id, batch.files[0])
            other.unlink()
            snapshot.unlink()
            snapshot.symlink_to(store / (batch.files[1].id + ".txt"))
            with self.assertRaises(OSError):
                history.attachments.read(session.id, batch.files[0])
            with self.assertRaises(FileNotFoundError):
                History(root / "custom").attachments.read(session.id, batch.files[1])
            await history.delete(reopened)
            self.assertFalse(store.exists())
            with self.assertRaises(FileNotFoundError):
                history.load(session.id)

    def test_validation_rejects_invalid_text_and_upload_sizes(self):
        for upload in (
            TextAttachmentInput("../notes.txt", "text"),
            TextAttachmentInput("credentials.json", "{}"),
            TextAttachmentInput("image.png", "text"),
            TextAttachmentInput("notes.txt", "\0binary"),
            TextAttachmentInput("notes.txt", "\ud800"),
            TextAttachmentInput("notes.txt", "界" * 44_000),
        ):
            with self.subTest(name=upload.name), self.assertRaises(ValueError):
                prepare_attachments([upload], None)
        with self.assertRaisesRegex(ValueError, "10 个"):
            prepare_attachments([TextAttachmentInput("notes.txt", "a")] * 11, None)
        with self.assertRaisesRegex(ValueError, "256 KiB"):
            prepare_attachments([TextAttachmentInput("notes.txt", "a" * 100_000)] * 3, None)

    async def test_inline_retry_reference_and_failure_boundaries(self):
        from unittest.mock import patch

        import httpx

        from daphnis import Operations, TurnStatus
        from daphnis.agent import TITLE_PROMPT
        from daphnis.app import Application

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "sessions")
            requests = []
            failed = False

            def provider(request):
                nonlocal failed
                body = json.loads(request.content)
                if body["messages"][0]["content"] != TITLE_PROMPT:
                    requests.append(body)
                    if "retry-marker" in body["messages"][-1]["content"] and not failed:
                        failed = True
                        return httpx.Response(503, json={"error": {"message": "unavailable"}})
                return httpx.Response(200, json={"choices": [{
                    "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"},
                }]})

            def runtime():
                return Operations(Application(
                    workspace=root, provider="deepseek", model_id="deepseek-flash", history=history,
                    connections={"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(provider)}},
                ))

            ops = runtime()
            try:
                result = await ops.send("Explain", attachments=[TextAttachmentInput("code.py", "retry-marker\nExact bytes")],
                                        stream=False, submission_id="b" * 32, session_id=ops.initial_session_id)
                self.assertEqual(result.status, TurnStatus.FAILED)
                sid = ops.initial_session_id
                refs = history.load(sid).attachment_refs
                self.assertIn("retry-marker\nExact bytes", requests[-1]["messages"][-1]["content"])
                self.assertEqual(ops.message_snapshot(ops.initial_session_id), [{"role": "user", "content": "Explain"}])
            finally:
                await ops.close()

            ops = runtime()
            try:
                ops.restore_session(sid)
                self.assertTrue(ops.attachment_submitted("b" * 32, "Explain", [TextAttachmentInput("code.py", "retry-marker\nExact bytes")], session_id=sid))
                result = await ops.retry_failed_user(ops.recovery(sid).key, stream=False, session_id=sid)
                self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                self.assertEqual(history.load(sid).attachment_refs, refs)
                self.assertIn("retry-marker\nExact bytes", requests[-1]["messages"][-1]["content"])
                await ops.send("Next", stream=False, session_id=sid)
                self.assertIn(refs[0].files[0].id, requests[-1]["messages"][1]["content"])
                self.assertNotIn("retry-marker", requests[-1]["messages"][1]["content"])
                from daphnis.compaction import CompactionCheckpoint
                from daphnis.context import Context, attachment_note

                old = history.load(sid)
                compacted = Context().build(old.messages, CompactionCheckpoint(2, json.dumps({
                    "goals": "Earlier task", "progress": "Answered", "files": "code.py", "next_steps": "Continue",
                })),
                                            attachment_notes={0: attachment_note(refs[0])})
                self.assertNotIn("retry-marker", json.dumps(compacted))
                self.assertEqual(history.attachments.read(sid, refs[0].files[0]), "retry-marker\nExact bytes")

                large = "large-marker " * 5500
                await ops.set_tools(["read_attachment", "search_attachment"], session_id=sid)
                result = await ops.send("", attachments=[TextAttachmentInput("large.txt", large)], stream=False, session_id=sid)
                self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                self.assertIn("read_attachment", requests[-1]["messages"][-1]["content"])
                self.assertNotIn("large-marker", requests[-1]["messages"][-1]["content"])

                await ops.set_tools(["read_attachment"], session_id=sid)
                before, count = ops.message_snapshot(sid), len(requests)
                result = await ops.send("Reject", attachments=[TextAttachmentInput("large.txt", large)], stream=False, session_id=sid)
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertIn("无法安全内联", result.error)
                self.assertEqual(ops.message_snapshot(sid), before)
                self.assertEqual(len(requests), count)
                with patch.object(history.attachments, "write", side_effect=OSError("disk full")):
                    result = await ops.send("Reject", attachments=[TextAttachmentInput("small.txt", "small")], stream=False, session_id=sid)
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertEqual(ops.message_snapshot(sid), before)
                with patch.object(history, "save", side_effect=OSError("disk full")):
                    result = await ops.send("Reject", attachments=[TextAttachmentInput("small.txt", "small")], stream=False, session_id=sid)
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertEqual(history.load(sid).messages.snapshot(), before)
                self.assertEqual(set(history.load(sid).attachment_refs), {0, 4})
            finally:
                await ops.close()

    async def test_resources_are_readable_only_by_snapshot_id_in_the_owning_session(self):
        from daphnis.compaction import CompactionCheckpoint
        from daphnis.events import EventBus
        from daphnis.prompt_groups import PromptInput
        from daphnis.tools import ApprovalMode, create_tools

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "sessions")
            session = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            resource = TextResourceInput(
                "Architecture", "资源 needle", "demo", "notes://architecture", "text/markdown", "embedded",
            )
            prompt = PromptInput("resource-turn", "demo", "review", (
                ("assistant", resource), ("user", "Find the marker."),
            ))
            session.messages = Messages([*prompt.snapshot(), {"role": "assistant", "content": "Done."},
                                         {"role": "user", "content": "Next task."}])
            session.prompt_groups = (prompt.group(0),)
            batch = prepare_attachments((resource,), None)
            session.attachment_refs = {0: batch}
            summary = '{"goals":"Review resources","progress":"","files":"","next_steps":""}'
            session.compaction = CompactionCheckpoint(1, summary)
            with self.assertRaisesRegex(ValueError, "Prompt input group"):
                history.save(session)
            session.compaction = CompactionCheckpoint(3, summary)
            history.save(session)
            history.attachments.write(session.id, batch, (resource,))
            saved = history.load(session.id)
            item = saved.attachment_refs[0].files[0]
            registry = create_tools(attachment_store=history.attachments)

            async def call(owner, name, arguments):
                executor = registry.select([name], mode=ApprovalMode.AUTO).bind(
                    session=owner, events=EventBus(), turn_id="test",
                )
                result = await executor.execute({"id": "call", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments),
                }})
                return result.value

            page = await call(saved, "read_attachment", {"id": item.id})
            self.assertEqual(page["lines"][0]["text"], resource.text)
            hits = await call(saved, "search_attachment", {"id": item.id, "query": "needle"})
            self.assertEqual(hits["matches"][0]["column"], 4)
            history_page = await call(saved, "history_read", {"message_id": 0})
            self.assertEqual(history_page["attachments"][0]["id"], item.id)
            self.assertNotIn("notes://", json.dumps(history_page))
            other = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            blocked = await call(other, "read_attachment", {"id": item.id})
            self.assertEqual(blocked["error"]["code"], "attachment_unavailable")

    async def test_attachment_paging_search_and_permissions(self):
        from daphnis.events import EventBus
        from daphnis.tools import ApprovalDecision, ApprovalMode, create_tools

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "sessions")
            history.directory.mkdir()
            session = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            source = "页🙂needle " + "界\\" * 4000 + "\nneedle second\nneedle third\n"
            (root / "source.txt").write_text(source, encoding="utf-8")
            uploads = [TextAttachmentInput("source.txt", source)]
            batch = prepare_attachments(uploads, None)
            history.attachments.write(session.id, batch, uploads)
            session.messages.add_user("Read these")
            session.attachment_refs = {0: batch}
            item = batch.files[0]
            approvals = []

            async def deny(request):
                approvals.append(request.tool_name)
                return ApprovalDecision.DENY

            registry = create_tools(approval=deny, attachment_store=history.attachments)

            async def call(mode, name, arguments, *, owner=session):
                executor = registry.select([name], mode=mode).bind(session=owner, events=EventBus(), turn_id="test")
                return await executor.execute({"id": "call", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments),
                }})

            ask = await call(ApprovalMode.ASK, "read_attachment", {"id": item.id})
            self.assertEqual(ask.value["error"]["code"], "permission_denied")
            self.assertEqual(approvals, ["read_attachment"])
            page = await call(ApprovalMode.AUTO, "read_attachment", {"id": item.id})
            self.assertLessEqual(len(page.content.encode()), 8192)
            self.assertEqual(page.value["next_line"], 1)
            self.assertGreater(page.value["next_column"], 1)
            rows: dict[int, str] = {}
            for _ in range(10):
                self.assertLessEqual(len(page.content.encode()), 8192)
                for row in page.value["lines"]:
                    previous = rows.get(row["line"], "")
                    self.assertEqual(row["column"], len(previous) + 1)
                    rows[row["line"]] = previous + row["text"]
                if page.value["next_line"] is None:
                    self.assertIsNone(page.value["next_column"])
                    break
                page = await call(ApprovalMode.FULL, "read_attachment", {
                    "id": item.id, "start_line": page.value["next_line"], "start_column": page.value["next_column"],
                })
            else:
                self.fail("Attachment pagination did not finish.")
            self.assertEqual(list(rows.items()), list(enumerate(source.split("\n"), 1)))
            workspace = await call(ApprovalMode.AUTO, "read_file", {"path": "source.txt"})
            self.assertEqual(workspace.value["content"], "\n".join(rows.values()))
            workspace_range = await call(ApprovalMode.AUTO, "read_file", {
                "path": "source.txt", "start_line": 2, "end_line": 2,
            })
            self.assertEqual(workspace_range.value["content"], rows[2] + "\n")
            self.assertEqual(workspace_range.value["end_line"], 2)
            workspace_capped = await call(ApprovalMode.AUTO, "read_file", {
                "path": "source.txt", "max_bytes": 4,
            })
            self.assertEqual(workspace_capped.value["content"], "页")
            self.assertEqual(workspace_capped.value["scanned_bytes"], 4)
            self.assertTrue(workspace_capped.value["truncated"])
            search = await call(ApprovalMode.AUTO, "search_attachment", {"query": "needle", "limit": 2})
            self.assertEqual([hit["line"] for hit in search.value["matches"]], [1, 2])
            self.assertEqual([hit["column"] for hit in search.value["matches"]], [3, 1])
            self.assertTrue(search.value["truncated"])
            exact = await call(ApprovalMode.AUTO, "search_attachment", {"query": "NEEDLE"})
            self.assertEqual(exact.value["matches"], [])
            workspace_search = await call(ApprovalMode.AUTO, "search", {
                "query": "NEEDLE", "case_sensitive": False, "mode": "content", "max_results": 2,
            })
            self.assertEqual([(hit["path"], hit["line"]) for hit in workspace_search.value["matches"]],
                             [("source.txt", 1), ("source.txt", 2)])
            self.assertTrue(workspace_search.value["truncated"])
            history_page = await call(ApprovalMode.AUTO, "history_read", {"message_id": 0})
            self.assertEqual(history_page.value["attachments"][0]["id"], item.id)
            other = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            blocked = await call(ApprovalMode.FULL, "read_attachment", {"id": item.id}, owner=other)
            self.assertEqual(blocked.value["error"]["code"], "attachment_unavailable")
            raw = await call(ApprovalMode.FULL, "read_file", {
                "path": str(history.attachments.directory / session.id / f"{item.id}.txt"),
            })
            self.assertEqual(raw.value["error"]["code"], "sensitive_path")
            raw_search = await call(ApprovalMode.FULL, "search", {
                "path": str(history.attachments.directory), "query": "needle", "mode": "content",
            })
            self.assertEqual(raw_search.value["error"]["code"], "sensitive_path")
            self.assertEqual(approvals, ["read_attachment"])

    async def test_aggregate_budget_and_interrupted_tool_continue_after_restart(self):
        import asyncio

        import httpx

        from daphnis import Operations, TurnStatus
        from daphnis.agent import TITLE_PROMPT
        from daphnis.app import Application
        from daphnis.models import ModelInfo
        from daphnis.tools import ApprovalMode
        from daphnis.turn_state import RecoveryKind

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "sessions")
            tiny = ModelInfo(id="tiny", provider="deepseek", context_window=8192,
                             default_output_tokens=512, supports_tools=False)
            large = ModelInfo(id="large", provider="deepseek", context_window=131_072,
                              default_output_tokens=512, supports_tools=True)
            uploads = [TextAttachmentInput("first.txt", "alpha " * 1200),
                       TextAttachmentInput("second.txt", "beta " * 1250)]
            bodies = []

            def provider(request):
                body = json.loads(request.content)
                content = body["messages"][-1]["content"]
                calls = []
                if body["messages"][0]["content"] != TITLE_PROMPT:
                    bodies.append(body)
                    if body["messages"][-1]["role"] == "user":
                        first = next(json.loads(line) for line in content.splitlines() if line.startswith('{"name":'))
                        self.assertNotIn("alpha alpha alpha", content)
                        self.assertNotIn("beta beta beta", content)
                        calls.append({"id": "read-first", "type": "function", "function": {
                            "name": "read_attachment", "arguments": json.dumps({"id": first["id"]}),
                        }})
                    else:
                        page = json.loads(content)
                        self.assertTrue(page["ok"])
                        self.assertIn("alpha alpha alpha", page["lines"][0]["text"])
                message = {"role": "assistant", "content": "done"}
                if calls:
                    message = {"role": "assistant", "content": None, "tool_calls": calls}
                return httpx.Response(200, json={"choices": [{
                    "index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message,
                }]})

            def runtime(model_id):
                return Operations(Application(
                    workspace=root, provider="deepseek", model_id=model_id, history=history,
                    catalog=(tiny, large), interactive=True,
                    connections={"deepseek": {"base_url": "https://model.test", "transport": httpx.MockTransport(provider)}},
                ))

            ops = runtime("tiny")
            try:
                # Smaller than 4,000 tokens, but over five percent of the tiny window's available room.
                failed = await ops.send("read", attachments=[TextAttachmentInput("short.txt", "abc " * 400)], stream=False, session_id=ops.initial_session_id)
                self.assertIn("无法安全内联", failed.error)
                self.assertEqual(ops.message_snapshot(ops.initial_session_id), [])
                self.assertEqual(bodies, [])
            finally:
                await ops.close()

            ops = runtime("large")
            try:
                approved = asyncio.Event()
                ops.subscribe("tool.approval_requested", lambda event: approved.set())
                ops.set_approval_mode(ApprovalMode.ASK, session_id=ops.initial_session_id)
                task = asyncio.create_task(ops.send("read", attachments=uploads, stream=False, session_id=ops.initial_session_id))
                try:
                    await asyncio.wait_for(approved.wait(), 3)
                finally:
                    await ops.cancel(ops.initial_session_id)
                with self.assertRaises(asyncio.CancelledError):
                    await task
                sid = ops.initial_session_id
                self.assertEqual(ops.recovery(ops.initial_session_id).kind, RecoveryKind.INTERRUPTED_TURN)
                refs = history.load(sid).attachment_refs
            finally:
                await ops.close()

            ops = runtime("large")
            try:
                ops.restore_session(sid)
                result = await ops.continue_interrupted_turn(ops.recovery(sid).key, stream=False, session_id=sid)
                self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                self.assertEqual(history.load(sid).attachment_refs, refs)
                self.assertEqual(len(list((history.attachments.directory / sid).glob("*.txt"))), 2)
                self.assertEqual(len(bodies), 2)
            finally:
                await ops.close()
