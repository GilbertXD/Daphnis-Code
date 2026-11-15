import json
import os
import tempfile
import unittest
from pathlib import Path

from daphnis.attachments import TextAttachmentInput, prepare_attachments
from daphnis.history import History, SessionBusyError
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
                                        stream=False, submission_id="b" * 32)
                self.assertEqual(result.status, TurnStatus.FAILED)
                sid = ops.session.id
                refs = history.load(sid).attachment_refs
                self.assertIn("retry-marker\nExact bytes", requests[-1]["messages"][-1]["content"])
                self.assertEqual(ops.message_snapshot(), [{"role": "user", "content": "Explain"}])
            finally:
                await ops.close()

            ops = runtime()
            try:
                ops.restore_session(sid)
                self.assertTrue(ops.attachment_submitted("b" * 32, "Explain", [TextAttachmentInput("code.py", "retry-marker\nExact bytes")]))
                result = await ops.retry_failed_user(ops.recovery.key, stream=False)
                self.assertEqual(result.status, TurnStatus.COMPLETED)
                self.assertEqual(history.load(sid).attachment_refs, refs)
                self.assertIn("retry-marker\nExact bytes", requests[-1]["messages"][-1]["content"])
                await ops.send("Next", stream=False)
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
                await ops.set_tools(["read_attachment", "search_attachment"])
                result = await ops.send("", attachments=[TextAttachmentInput("large.txt", large)], stream=False)
                self.assertEqual(result.status, TurnStatus.COMPLETED)
                self.assertIn("read_attachment", requests[-1]["messages"][-1]["content"])
                self.assertNotIn("large-marker", requests[-1]["messages"][-1]["content"])

                await ops.set_tools(["read_attachment"])
                before, count = ops.message_snapshot(), len(requests)
                result = await ops.send("Reject", attachments=[TextAttachmentInput("large.txt", large)], stream=False)
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertIn("无法安全内联", result.error)
                self.assertEqual(ops.message_snapshot(), before)
                self.assertEqual(len(requests), count)
                with patch.object(history.attachments, "write", side_effect=OSError("disk full")):
                    result = await ops.send("Reject", attachments=[TextAttachmentInput("small.txt", "small")], stream=False)
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertEqual(ops.message_snapshot(), before)
                with patch.object(history, "save", side_effect=OSError("disk full")):
                    result = await ops.send("Reject", attachments=[TextAttachmentInput("small.txt", "small")], stream=False)
                self.assertEqual(result.status, TurnStatus.FAILED)
                self.assertEqual(history.load(sid).messages.snapshot(), before)
                self.assertEqual(set(history.load(sid).attachment_refs), {0, 4})
            finally:
                await ops.close()

    async def test_attachment_paging_search_and_permissions(self):
        from daphnis.events import EventBus
        from daphnis.tools import ApprovalDecision, ApprovalMode, create_tools

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "sessions")
            history.directory.mkdir()
            session = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            source = "needle " + "界" * 4000 + "\nneedle second\nneedle third"
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
            rest = await call(ApprovalMode.FULL, "read_attachment", {
                "id": item.id, "start_line": 1, "start_column": page.value["next_column"],
            })
            self.assertEqual(page.value["lines"][0]["text"] + rest.value["lines"][0]["text"], source.split("\n")[0])
            search = await call(ApprovalMode.AUTO, "search_attachment", {"query": "needle", "limit": 2})
            self.assertEqual([hit["line"] for hit in search.value["matches"]], [1, 2])
            self.assertTrue(search.value["truncated"])
            history_page = await call(ApprovalMode.AUTO, "history_read", {"message_id": 0})
            self.assertEqual(history_page.value["attachments"][0]["id"], item.id)
            other = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            blocked = await call(ApprovalMode.FULL, "read_attachment", {"id": item.id}, owner=other)
            self.assertEqual(blocked.value["error"]["code"], "attachment_unavailable")
            raw = await call(ApprovalMode.FULL, "read_file", {
                "path": str(history.attachments.directory / session.id / f"{item.id}.txt"),
            })
            self.assertEqual(raw.value["error"]["code"], "sensitive_path")
            self.assertEqual(approvals, ["read_attachment"])
