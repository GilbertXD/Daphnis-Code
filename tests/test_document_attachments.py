import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from daphnis import History, Session
from daphnis.attachments import (
    DOCUMENT_MIME_TYPES,
    MAX_DOCUMENT_BYTES,
    DocumentAttachment,
    DocumentAttachmentInput,
    ImageAttachmentInput,
    TextAttachmentInput,
    prepare_attachments,
)

PDF = b"%PDF-1.4\n% synthetic test snapshot\n%%EOF\n"


def document(extension=".pdf", *, mode="text", name="report"):
    if extension == ".pdf":
        data = PDF
    else:
        parts = {".docx": "word/document.xml", ".pptx": "ppt/presentation.xml", ".xlsx": "xl/workbook.xml"}
        output = BytesIO()
        with ZipFile(output, "w") as archive:
            archive.writestr("[Content_Types].xml", "test")
            archive.writestr(parts[extension], "test")
        data = output.getvalue()
    return DocumentAttachmentInput(name + extension, data, DOCUMENT_MIME_TYPES[extension], mode)


class DocumentAttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_signatures_mixed_snapshot_round_trip_and_delete(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        history = History(root / "sessions")
        state = Session(root, "deepseek", "deepseek-flash")
        encoded = BytesIO()
        with Image.new("RGB", (8, 8), "blue") as image:
            image.save(encoded, "PNG")
        image = ImageAttachmentInput("image.png", encoded.getvalue(), "image/png")
        expected = []
        for ordinal, extension in enumerate(DOCUMENT_MIME_TYPES):
            upload = document(extension)
            batch = prepare_attachments((TextAttachmentInput("notes.txt", "note"), upload, image), None)
            state.messages.add_user(f"Inspect {ordinal}")
            state.attachment_refs[ordinal] = batch
            history.attachments.write(state.id, batch, (TextAttachmentInput("notes.txt", "note"), upload, image))
            self.assertEqual([item.name for item in batch.files], ["notes.txt", upload.name, "image.png"])
            expected.append(upload)
        history.save(state)
        restored = history.load(state.id)
        self.assertEqual(restored.attachment_refs, state.attachment_refs)
        files = history.attachments.directory / state.id
        for index, upload in enumerate(expected):
            item, = restored.attachment_refs[index].documents
            self.assertIsInstance(item, DocumentAttachment)
            self.assertEqual(history.attachments.read_document(state.id, item), upload.data)
            self.assertTrue((files / (item.id + Path(upload.name).suffix)).is_file())
        item, = restored.attachment_refs[0].documents
        path = files / (item.id + ".pdf")
        path.chmod(0o600)
        path.write_bytes(b"x" + PDF[1:])
        path.chmod(0o400)
        with self.assertRaisesRegex(OSError, "校验"):
            history.attachments.read_document(state.id, item)
        await history.delete(restored)
        self.assertFalse(files.exists())

    def test_invalid_document_format_size_reading_mode_and_per_turn_count(self):
        for upload in (
            replace(document(), data=b""),
            replace(document(), data=PDF[:-6]),
            replace(document(), data=b"x" * (MAX_DOCUMENT_BYTES + 1)),
            replace(document(), name="../file.pdf"),
            replace(document(), mime_type="application/zip"),
            replace(document(), name="file.txt"),
            replace(document(), reading_mode="other"),
            document(".xlsx", mode="visual"),
            replace(document(".pptx"), data=document(".docx").data),
            replace(document(".docx"), data=b"PK\x03\x04invalid"),
        ):
            with self.subTest(name=upload.name), self.assertRaises(ValueError):
                prepare_attachments((upload,), None)
        with self.assertRaisesRegex(ValueError, "一份文档"):
            prepare_attachments((document(), document()), None)
