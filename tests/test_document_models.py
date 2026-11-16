import asyncio
import json
import tempfile
import unittest
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from time import time
from unittest.mock import patch
from uuid import uuid4

import httpx
from test_document_attachments import document

from daphnis import History, Operations, TurnStatus
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.attachments import prepare_attachments
from daphnis.models.deepseek_files import DeepSeekFiles
from daphnis.models.documents import DocumentInputError

SUMMARY = '{"goals":"Review","progress":"Recorded findings","files":"","next_steps":""}'


def completion(content):
    return httpx.Response(200, json={
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    })


def file_ids(messages):
    return [part["file_id"] for message in messages if isinstance(message.get("content"), list)
            for part in message["content"] if part["type"] == "file"]


class FileService:
    def __init__(self):
        self.uploads = []
        self.saved = []
        self.status = 200
        self.lose = False
        self.lookup_count = 0
        self.started = asyncio.Event()
        self.hold = False

    async def __call__(self, request):
        if request.method == "GET":
            self.lookup_count += 1
            return httpx.Response(200, json={"data": self.saved, "has_more": False})
        assert request.url.path == "/files"
        multipart = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
        )
        fields = list(multipart.iter_parts())
        assert [part.get_param("name", header="content-disposition") for part in fields] == [
            "purpose", "expires_after[anchor]", "expires_after[seconds]", "file",
        ]
        purpose, anchor, lifetime, data = (item.get_payload(decode=True) for item in fields)
        assert anchor == b"created_at" and lifetime == b"2592000"
        self.uploads.append((purpose.decode(), data))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "SENSITIVE UPSTREAM BODY"}})
        created = int(time())
        obj = {"id": "file-api-" + uuid4().hex, "filename": fields[-1].get_filename(),
               "bytes": 17 if purpose == b"user_data" else len(data),
               "purpose": purpose.decode(), "created_at": created, "expires_at": created + 2592000}
        self.saved.append(obj)
        self.started.set()
        if self.hold:
            await asyncio.Future()
        if self.lose:
            raise httpx.ReadTimeout("SENSITIVE UPSTREAM BODY", request=request)
        return httpx.Response(200, json=obj)


class DocumentAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.service = FileService()
        self.connection = {"base_url": "https://deepseek.test/v1", "api_key": "fixture-secret",
                           "transport": httpx.MockTransport(self.service)}

    def adapter(self, **overrides):
        adapter = DeepSeekFiles({**self.connection, **overrides}, self.root)
        self.addAsyncCleanup(adapter.aclose)
        return adapter

    async def prepare(self, adapter, upload):
        item, = prepare_attachments((upload,), None).documents
        messages = [{"role": "user", "content": "Read"}]
        prepared = await adapter.prepare(messages, {0: (item,)}, lambda _: upload.data)
        self.assertEqual(messages, [{"role": "user", "content": "Read"}])
        return file_ids(prepared.messages)[0]

    async def test_purposes_cache_scope_restart_and_expiry(self):
        original = self.adapter()
        for extension, mode, purpose in (
            (".pdf", "text", "text_recognition"), (".docx", "visual", "rich_layout_recognition"),
            (".pptx", "text", "text_recognition"), (".xlsx", "text", "user_data"),
        ):
            upload = document(extension, mode=mode)
            self.assertTrue(await self.prepare(original, upload))
            self.assertEqual(self.service.uploads[-1], (purpose, upload.data))
        pdf = document()
        old = await self.prepare(original, pdf)
        reopened = self.adapter()
        self.assertEqual(await self.prepare(reopened, pdf), old)
        self.assertEqual(len(self.service.uploads), 4)
        self.assertNotEqual(await self.prepare(reopened, document(mode="visual")), old)
        self.assertNotEqual(await self.prepare(self.adapter(api_key="other"), pdf), old)
        self.assertNotEqual(await self.prepare(self.adapter(base_url="https://other.test"), pdf), old)
        store = self.root / "deepseek-files.sqlite3"
        self.assertNotIn(b"fixture-secret", store.read_bytes())
        self.assertEqual(store.stat().st_mode & 0o777, 0o600)
        # The mock server's clock remains at the original time.
        with (
            patch("daphnis.models.deepseek_files.time", return_value=time() + 2592001),
            self.assertRaisesRegex(DocumentInputError, "已过期"),
        ):
            await self.prepare(reopened, pdf)
        self.assertEqual(len(self.service.uploads), 8)

    async def test_known_http_failure_and_unknown_completion_reconcile(self):
        adapter = self.adapter()
        for status, error in ((429, "过多"), (504, "超时"), (400, "无法解析")):
            self.service.status = status
            with self.assertRaisesRegex(DocumentInputError, error) as error:
                await self.prepare(adapter, document())
            self.assertNotIn("SENSITIVE", str(error.exception))
        self.service.status = 200
        self.service.lose = True
        with self.assertRaisesRegex(DocumentInputError, "尚未确认"):
            await self.prepare(adapter, document())
        self.service.lose = False
        self.assertEqual(await self.prepare(self.adapter(), document()), self.service.saved[-1]["id"])
        self.assertEqual(self.service.lookup_count, 1)
        self.assertEqual(len(self.service.uploads), 4)

    async def test_cancellation_and_concurrent_attempt_keep_one_upload(self):
        adapter = self.adapter()
        self.service.hold = True
        active = asyncio.create_task(self.prepare(adapter, document()))
        await asyncio.wait_for(self.service.started.wait(), 3)
        duplicate = await self.prepare(self.adapter(), document())
        active.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active
        self.assertEqual(len(self.service.uploads), 1)
        self.assertEqual(duplicate, self.service.saved[0]["id"])


class DocumentOperationsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.history = History(self.root / "sessions")
        self.service = FileService()
        self.wire = []
        self.invalid = False
        self.fail_next = False
        self.overflow = False

    def runtime(self):
        async def handle(request):
            if request.url.path == "/files":
                return await self.service(request)
            body = json.loads(request.content)
            if body["messages"][0]["content"] == TITLE_PROMPT:
                return completion("Documents")
            self.wire.append(body)
            identifiers = file_ids(body["messages"])
            if "response_format" in body:
                self.assertFalse(identifiers)
                return completion(SUMMARY)
            if self.overflow:
                return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
            if self.fail_next:
                self.fail_next = False
                return httpx.Response(503, json={"error": {"message": "sensitive fixture error"}})
            if self.invalid:
                self.invalid = False
                return httpx.Response(400, json={"error": {
                    "message": ".messages: the following file_ids do not exist or are not created under your account: "
                    + identifiers[0],
                }})
            return completion("Done." + (" Echo " + identifiers[0] if identifiers else ""))

        ops = Operations(Application(
            workspace=self.root, history=History(self.history.directory), provider="deepseek", model_id="deepseek-flash",
            connections={name: {"base_url": f"https://{name}.test", "transport": httpx.MockTransport(handle)}
                         for name in ("deepseek", "mimo")},
            recent_tokens=0, observe_model_io=True,
        ))
        self.addAsyncCleanup(ops.close)
        return ops

    async def test_confirmed_missing_reference_retry_and_model_failure_preserve_snapshot(self):
        ops = self.runtime()
        self.fail_next = True
        events = []
        for name in ("model.request", "model.finished", "context.budget"):
            ops.subscribe(name, events.append)
        upload = document()
        failed = await ops.send("Read", attachments=[upload], stream=False, session_id=ops.initial_session_id)
        self.assertEqual(failed.status, TurnStatus.FAILED)
        session = ops.initial_session_id
        refs = self.history.load(session).attachment_refs
        original = self.service.saved[0]["id"]
        self.invalid = True
        result = await ops.retry_failed_user(ops.recovery(ops.initial_session_id).key, stream=False, session_id=ops.initial_session_id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual([file_ids(body["messages"]) for body in self.wire],
                         [[original], [original], [self.service.saved[1]["id"]]])
        self.assertEqual(len(self.service.uploads), 2)
        self.assertEqual(self.history.load(session).attachment_refs, refs)
        self.assertEqual(len(ops.message_snapshot(ops.initial_session_id)), 2)
        self.assertTrue(any(e.name == "context.budget" and e.data["unknown_documents"] == 1 for e in events))
        data = (self.history.directory / f"{session}.json").read_text() + json.dumps([e.data for e in events])
        self.assertNotIn("file-api-", data)
        self.assertNotIn("sensitive fixture error", data)

    async def test_compaction_restore_and_switch_preserve_only_effective_documents(self):
        first = self.runtime()
        session = first.initial_session_id
        a, b = document(name="alpha"), document(".docx", name="beta")
        for prompt, attachments in (("Read A", [a]), ("About A", []), ("Read B", [b])):
            result = await first.send(prompt, attachments=attachments, stream=False, session_id=session)
            self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        original = self.history.load(session)
        a_ref, b_ref = (item["id"] for item in self.service.saved)
        self.assertEqual(file_ids(self.wire[2]["messages"]), [a_ref, b_ref])
        checkpoint = await first.compact(session)
        self.assertEqual(checkpoint.through, 4)
        self.assertNotIn("file-api-", json.dumps(self.wire[-1]))
        self.assertIn("alpha.pdf", json.dumps(self.wire[-1]))
        self.assertIn("Document metadata is not evidence", json.dumps(self.wire[-1]))
        await first.close()

        reopened = self.runtime()
        reopened.restore_session(session)
        result = await reopened.send("About B", stream=False, session_id=session)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(file_ids(self.wire[-1]["messages"]), [b_ref])
        self.assertEqual(len(self.service.uploads), 2)
        await reopened.set_model("mimo", "mimo-v2.6-flash", session_id=session)
        count = len(self.wire)
        blocked = await reopened.send("Try another model", stream=False, session_id=session)
        self.assertEqual(blocked.status, TurnStatus.FAILED)
        self.assertIn("不支持有效上下文中的文档", blocked.error)
        self.assertEqual(len(self.wire), count)
        await reopened.compact(session)
        result = await reopened.retry_failed_user(reopened.recovery(session).key, stream=False, session_id=session)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(self.wire[-1]["model"], "mimo-v2.6-flash")
        self.assertEqual(file_ids(self.wire[-1]["messages"]), [])
        after = self.history.load(session)
        self.assertEqual(after.attachment_refs, original.attachment_refs)
        for index, upload in ((0, a), (4, b)):
            self.assertEqual(self.history.attachments.read_document(session, after.attachment_refs[index].documents[0]),
                             upload.data)

    async def test_cancellation_during_upload_never_calls_chat_and_latest_document_overflow_is_explicit(self):
        ops = self.runtime()
        self.service.hold = True
        running = asyncio.create_task(ops.send("Read", attachments=[document()], stream=False, session_id=ops.initial_session_id))
        await asyncio.wait_for(self.service.started.wait(), 3)
        await ops.cancel(ops.initial_session_id)
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertEqual(self.wire, [])
        self.overflow = True
        result = await ops.retry_failed_user(ops.recovery(ops.initial_session_id).key, stream=False, session_id=ops.initial_session_id)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("文档和当前上下文超过模型上限", result.error)
        self.assertEqual(len(self.service.uploads), 1)
        self.assertEqual(len(self.wire), 1)
        self.assertTrue(self.history.load(ops.initial_session_id).attachment_refs)
