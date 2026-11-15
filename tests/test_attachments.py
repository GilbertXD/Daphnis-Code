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
