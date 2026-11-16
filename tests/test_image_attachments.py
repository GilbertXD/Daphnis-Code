import asyncio
import json
import tempfile
import unittest
from base64 import b64encode
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image
from saturnus import APIError

from daphnis import CompactionSettings, Operations, TurnStatus
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.attachments import (
    MAX_IMAGE_BYTES,
    AttachmentBatch,
    ImageAttachmentInput,
    TextAttachmentInput,
    prepare_attachments,
)
from daphnis.history import History
from daphnis.models import MODELS
from daphnis.session import Session
from daphnis.token_budget import image_token_reserve, mimo_image_tokens, request_budget
from daphnis.web.bridge import Bridge
from daphnis.web.redaction import Redactor


def image_upload(name="screen.png", fmt="PNG", color="red", size=(24, 16), **options):
    data = BytesIO()
    with Image.new("RGB", size, color) as image:
        image.save(data, fmt, **options)
    return ImageAttachmentInput(name, data.getvalue(), f"image/{fmt.lower()}")


def image_url(upload):
    return f"data:{upload.mime_type};base64,{b64encode(upload.data).decode('ascii')}"


def image_urls(messages):
    return [
        part["image_url"]["url"]
        for message in messages if isinstance(message.get("content"), list)
        for part in message["content"] if part["type"] == "image_url"
    ]


def reply(text, **extra):
    return httpx.Response(200, json={"choices": [{
        "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text},
    }], **extra})


class ImageBudgetTests(unittest.TestCase):
    def test_provider_estimates_and_mimo_grid_boundaries(self):
        for dims, expected in (
            ((1, 1), 9), ((32, 224), 16), ((64, 96), 12), ((64, 128), 8),
            ((512, 512), 256), ((1024, 256), 256), ((4096, 2048), 8192),
            ((4097, 2048), 8192), ((4128, 2048), 8064), ((4000, 4000), 8100), ((1, 20_000_000), 8192),
        ):
            with self.subTest(dims=dims):
                self.assertEqual(mimo_image_tokens(*dims), expected)
                self.assertEqual(mimo_image_tokens(*reversed(dims)), expected)
                self.assertEqual(image_token_reserve("mimo", *dims), expected + (expected + 9) // 10)
                self.assertEqual(image_token_reserve("deepseek", *dims), 1024)
        self.assertEqual(image_token_reserve("compatible", 512, 512), 2048)


class ImageStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_formats_permissions_legacy_metadata_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = History(root / "sessions")
            session = Session(workspace=root, provider="deepseek", model_id="deepseek-flash")
            session.messages.add_user("Inspect these files")
            uploads = (
                image_upload("untrusted-suffix.txt"),
                TextAttachmentInput("notes.txt", "legacy text"),
                image_upload("photo", "JPEG"),
                image_upload("screen.webp", "WEBP"),
            )
            batch = prepare_attachments(uploads, None)
            session.attachment_refs = {0: batch}
            history.attachments.write(session.id, batch, uploads)
            history.save(session)

            saved = History(history.directory).load(session.id)
            self.assertEqual(saved.attachment_refs, {0: batch})
            metadata = batch.to_json()
            self.assertNotIn("type", metadata["files"][1])
            self.assertEqual([item["type"] for item in metadata["files"] if "type" in item],
                             ["image", "image", "image"])
            metadata["files"][0].pop("type")
            with self.assertRaises(TypeError):
                AttachmentBatch.from_json(metadata)

            store = history.attachments
            root = store.directory / session.id
            for image, upload, suffix in zip(batch.images, uploads[0:1] + uploads[2:],
                                              (".png", ".jpg", ".webp"), strict=True):
                self.assertEqual((image.width, image.height), (24, 16))
                self.assertEqual(store.read_image(session.id, image), upload.data)
                self.assertEqual((root / (image.id + suffix)).stat().st_mode & 0o777, 0o400)
            self.assertEqual(store.read(session.id, batch.texts[0]), "legacy text")
            # An existing UUID is immutable even if a caller tries to publish different bytes.
            replacement = image_upload(color="blue")
            updated = replace(prepare_attachments((replacement,), None).images[0], id=batch.images[0].id)
            with self.assertRaisesRegex(OSError, "校验|不可用"):
                store.write(session.id, AttachmentBatch((updated,)), (replacement,))
            self.assertEqual(store.read_image(session.id, batch.images[0]), uploads[0].data)
            await history.delete(saved)
            self.assertFalse(root.exists())

    def test_invalid_content_and_limits(self):
        original = image_upload()
        with Image.new("RGB", (24, 16), "blue") as frame:
            animated = image_upload(save_all=True, append_images=[frame])
        for upload, error in (
            (replace(original, mime_type="image/jpeg"), "MIME"),
            (replace(original, mime_type="image/svg+xml"), "PNG、JPEG"),
            (replace(original, data=original.data[:-4]), "损坏"),
            (replace(original, data=b"invalid"), "损坏"),
            (replace(original, data=b"x" * (MAX_IMAGE_BYTES + 1)), "5 MiB"),
            (replace(original, data=b""), "不能为空"),
            (image_upload(size=(5000, 4001)), "2000 万"),
            (animated, "动图"),
            (replace(original, name="../image.png"), "名称无效"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                prepare_attachments((upload,), None)
        with self.assertRaisesRegex(ValueError, "4 张"):
            prepare_attachments((original,) * 5, None)


class ImageOperationsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.history = History(self.root / "sessions")
        self.red = image_upload("old.png")

    def runtime(
        self, handler, *, provider="deepseek", model_id="deepseek-flash", catalog=MODELS, compaction_settings=None,
    ):
        ops = Operations(Application(
            workspace=self.root, provider=provider, model_id=model_id, catalog=catalog,
            history=History(self.history.directory),
            connections={provider: {"base_url": "https://model.test", "transport": httpx.MockTransport(handler)}},
            observe_model_io=True, recent_tokens=0,
            output_limits={(provider, model_id): 128},
            compaction_settings=compaction_settings,
        ))
        self.addAsyncCleanup(ops.close)
        return ops

    def saved(self, session_id):
        return (self.history.directory / f"{session_id}.json").read_text(encoding="utf-8")

    async def test_projection_restart_mixed_upload_and_manual_compaction(self):
        primary, summaries, previews = [], [], []
        summary_mode = "ok"
        summary_started = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                self.assertFalse(image_urls(messages))
                return reply("Image review")
            if "response_format" in body:
                summaries.append(messages)
                if summary_mode == "error":
                    return httpx.Response(503, json={"error": {"message": "unavailable"}})
                if summary_mode == "block":
                    summary_started.set()
                    await asyncio.Future()
                return reply('{"goals":"Image review","progress":"Done","files":"old.png","next_steps":""}')
            primary.append(messages)
            return reply("Done")

        ops = self.runtime(handler)
        ops.subscribe("model.request", previews.append)
        system = ops.get_system_prompt(ops.initial_session_id)
        result = await ops.send("Inspect old", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(primary[-1][0], {"role": "system", "content": system})
        self.assertEqual(primary[-1][1], {"role": "user", "content": [
            {"type": "text", "text": "Inspect old"},
            {"type": "image_url", "image_url": {"url": image_url(self.red)}},
        ]})

        blue = image_upload("latest.webp", "WEBP", "blue")
        green = image_upload("latest.jpg", "JPEG", "green")
        uploads = [blue, TextAttachmentInput("notes.txt", "inline notes"), green]
        submission = "a" * 32
        result = await ops.send("Compare", attachments=uploads, stream=False, submission_id=submission, session_id=ops.initial_session_id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        newest = primary[-1][-1]["content"]
        self.assertIn("inline notes", newest[0]["text"])
        self.assertEqual(image_urls([primary[-1][-1]]), [image_url(blue), image_url(green)])
        self.assertTrue(previews)
        self.assertTrue(all(event.data.get("omitted") for event in previews))
        self.assertNotIn("data:image/", json.dumps([event.data for event in previews]))
        session_id = ops.initial_session_id
        self.assertNotIn("base64", self.saved(session_id))
        await ops.close()

        resumed = self.runtime(handler)
        resumed.restore_session(session_id)
        self.assertTrue(resumed.attachment_submitted(submission, "Compare", uploads, session_id=session_id))
        self.assertEqual([message["content"] for message in resumed.message_snapshot(session_id)],
                         ["Inspect old", "Done", "Compare", "Done"])
        checkpoint = await resumed.compact(session_id)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.through, 2)
        summary_source = json.dumps(summaries, ensure_ascii=False)
        for field in ("old.png", "image/png", "width", "height", "Image metadata is not visual evidence"):
            self.assertIn(field, summary_source)
        self.assertNotIn("base64", summary_source)
        result = await resumed.send("Describe the images in the current context", stream=False, session_id=session_id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(image_urls(primary[-1]), [image_url(blue), image_url(green)])
        self.assertEqual(len(resumed.message_snapshot(session_id)), 6)
        unchanged = self.history.load(session_id)

        summary_mode = "error"
        with self.assertRaises(APIError):
            await resumed.compact(session_id)
        self.assertEqual(self.history.load(session_id).compaction, checkpoint)
        summary_mode = "block"
        pending = asyncio.create_task(resumed.compact(session_id))
        await asyncio.wait_for(summary_started.wait(), timeout=5)
        await resumed.cancel(session_id)
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(self.history.load(session_id).compaction, checkpoint)

        summary_mode = "ok"
        second = await resumed.compact(session_id)
        self.assertEqual(second.through, 4)
        after = self.history.load(session_id)
        self.assertEqual(after.compaction, second)
        self.assertEqual(after.messages.snapshot(), unchanged.messages.snapshot())
        self.assertEqual(after.attachment_refs, unchanged.attachment_refs)
        stored = (item for batch in after.attachment_refs.values() for item in batch.images)
        for item, upload in zip(stored, (self.red, blue, green), strict=True):
            self.assertEqual(self.history.attachments.read_image(session_id, item), upload.data)
        await resumed.close()

        reopened = self.runtime(handler)
        reopened.restore_session(session_id)
        self.assertEqual(reopened.compacted_through(session_id), 4)
        result = await reopened.send("Continue from the summary", stream=False, session_id=session_id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(image_urls(primary[-1]), [])
        self.assertEqual(self.history.load(session_id).attachment_refs, unchanged.attachment_refs)
        self.assertNotIn("base64", self.saved(session_id))
        self.assertNotIn("base64", json.dumps(summaries))

    async def test_auto_compaction_uses_image_tokens_and_overflow_reuses_the_budget(self):
        blue = image_upload("blue.png", color="blue")
        green = image_upload("green.png", color="green")
        for cause in ("pressure", "overflow"):
            with self.subTest(cause=cause):
                calls, summaries, budgets, compactions = [], [], [], []

                def handler(request, *, calls=calls, summaries=summaries, cause=cause):
                    body = json.loads(request.content)
                    messages = body["messages"]
                    if messages[0]["content"] == TITLE_PROMPT:
                        return reply("Images")
                    if "response_format" in body:
                        summaries.append(messages)
                        return reply('{"goals":"Review images","progress":"Done","files":"old.png","next_steps":""}')
                    urls = image_urls(messages)
                    calls.append(messages)
                    if cause == "overflow" and len(urls) == 3:
                        return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
                    return reply("Done")

                settings = CompactionSettings(threshold_mode="tokens", threshold_value=2600) if cause == "pressure" else None
                ops = self.runtime(handler, compaction_settings=settings)
                ops.subscribe("context.budget", budgets.append)
                ops.subscribe("compaction.finished", compactions.append)
                await ops.set_system_prompt("Inspect images.", session_id=ops.initial_session_id)
                await ops.set_tools([], session_id=ops.initial_session_id)
                first = await ops.send("Review old", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
                self.assertEqual(first.status, TurnStatus.COMPLETED, first.error)
                second = await ops.send("Review new", attachments=[blue, green], stream=False, session_id=ops.initial_session_id)
                self.assertEqual(second.status, TurnStatus.COMPLETED, second.error)
                self.assertEqual([event.data["trigger"] for event in compactions], [cause])
                self.assertEqual(ops.compacted_through(ops.initial_session_id), 2)
                self.assertEqual(image_urls(calls[-1]), [image_url(blue), image_url(green)])
                self.assertEqual(len(calls), 2 if cause == "pressure" else 3)
                self.assertEqual(len(summaries), 1)
                self.assertNotIn("base64", json.dumps(summaries))
                self.assertEqual(len(self.history.load(ops.initial_session_id).attachment_refs), 2)
                self.assertIn("old.png", json.dumps(summaries))
                text = [*calls[-1][:-1], {**calls[-1][-1], "content": "Review new"}]
                self.assertEqual(budgets[-1].data["input_tokens"],
                                 request_budget(text, {}, None).input_tokens + 2048)
                await ops.close()

    async def test_byte_pressure_compacts_old_images_but_keeps_or_rejects_the_newest(self):
        blue = image_upload("blue.png", color="blue")
        calls, summary_requests, candidates = [], [], []

        def handler(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                return reply("Images")
            if "response_format" in body:
                summary_requests.append(messages)
                return reply('{"goals":"Review images","progress":"Done","files":"old.png","next_steps":""}')
            calls.append(image_urls(messages))
            return reply("Done")

        ops = self.runtime(handler, provider="mimo", model_id="mimo-v2.6-flash")
        ops.subscribe("compaction.candidate", candidates.append)
        # Exercise the 32 MiB path without generating or decoding large fixtures.
        cap = max(len(self.red.data), len(blue.data))
        with patch("daphnis.agent.MAX_IMAGE_REQUEST_BYTES", cap):
            first = await ops.send("Review old", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
            self.assertEqual(first.status, TurnStatus.COMPLETED, first.error)
            second = await ops.send("Review new", attachments=[blue], stream=False, session_id=ops.initial_session_id)
            self.assertEqual(second.status, TurnStatus.COMPLETED, second.error)
            self.assertEqual(calls, [[image_url(self.red)], [image_url(blue)]])
            self.assertEqual(len(summary_requests), 1)
            self.assertEqual(ops.compacted_through(ops.initial_session_id), 2)
            candidate = candidates[-1].data
            self.assertEqual(candidate["reason"], "image_bytes_reduced")
            self.assertGreaterEqual(candidate["after_tokens"], candidate["before_tokens"])
            rejected = await ops.send("Review both", attachments=[blue, self.red], stream=False, session_id=ops.initial_session_id)
            self.assertEqual(rejected.status, TurnStatus.FAILED)
            self.assertIn("图片超过 32 MiB", rejected.error)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(summary_requests), 1)
            self.assertEqual(ops.compacted_through(ops.initial_session_id), 2)
            self.assertEqual(len(self.history.load(ops.initial_session_id).attachment_refs[4].images), 2)

    async def test_model_switch_checks_target_capability_for_history(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            if body["messages"][0]["content"] != TITLE_PROMPT:
                requests.append(body)
            return reply("Done")

        original = next(model for model in MODELS if model.id == "deepseek-flash")
        compatible = replace(original, id="custom-vision")
        disabled = replace(original, id="custom-text", supports_image=False)
        ops = self.runtime(handler, catalog=(*MODELS, compatible, disabled))
        first = await ops.send("Review", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
        self.assertEqual(first.status, TurnStatus.COMPLETED, first.error)
        for model_id, reason in (("deepseek-v4-pro", "图片能力未确认"), ("custom-text", "明确不支持图片")):
            await ops.set_model("deepseek", model_id, session_id=ops.initial_session_id)
            rejected = await ops.send("Follow up", stream=False, session_id=ops.initial_session_id)
            self.assertEqual(rejected.status, TurnStatus.FAILED)
            self.assertIn(reason, rejected.error)
        self.assertEqual(len(requests), 1)
        await ops.set_model("deepseek", "custom-vision", session_id=ops.initial_session_id)
        result = await ops.send("Follow up", stream=False, session_id=ops.initial_session_id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(requests[-1]["model"], "custom-vision")
        self.assertEqual(image_urls(requests[-1]["messages"]), [image_url(self.red)])

    async def test_reopen_without_compaction_and_stream_echo_is_private(self):
        payload = b64encode(self.red.data).decode("ascii")
        observed_urls, events, frames = [], [], []

        def handler(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                return reply("Inspect")
            observed_urls.append(image_urls(messages))
            if not body.get("stream"):
                return reply("Done", provider_debug={"echo": image_url(self.red)})
            pieces = ["Still visible. data:im", "age/png;base64," + payload[:13],
                      payload[13:] + " and " + payload[:7], payload[7:]]
            chunks = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": piece}}]}
                for piece in pieces
            ]
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            stream = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(200, text=stream + "data: [DONE]\n\n",
                                  headers={"Content-Type": "text/event-stream"})

        ops = self.runtime(handler)
        ops.subscribe("model.finished", events.append)
        result = await ops.send("", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(ops.message_snapshot(ops.initial_session_id)[0]["content"], "已附上图片。")
        sid = ops.initial_session_id
        await ops.close()

        resumed = self.runtime(handler)
        resumed.restore_session(sid)
        bridge = Bridge(resumed, Redactor(()))
        bridge.session_id = sid
        for name in ("model.request", "model.delta", "model.finished"):
            resumed.subscribe(name, events.append)
        resumed.subscribe("model.delta", lambda _: frames.append(bridge.frame()))
        bridge.start("Describe the previous image")
        await bridge.task
        self.assertEqual(observed_urls, [[image_url(self.red)], [image_url(self.red)]])
        self.assertEqual(len(resumed.message_snapshot(sid)), 4)
        visible = json.dumps([self.saved(sid), bridge.frame(), *frames, *(event.data for event in events)])
        self.assertIn("Still visible.", visible)
        self.assertNotIn("base64", visible)
        self.assertNotIn(payload, visible)
        self.assertTrue(frames)
        self.assertEqual([message["role"] for message in resumed.message_snapshot(sid)], ["user", "assistant"] * 2)

    async def test_persistence_failure_http_failure_cancel_and_corrupt_snapshot(self):
        started = asyncio.Event()
        requested = []
        mode = "error"

        async def handler(request):
            messages = json.loads(request.content)["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                return reply("Image review")
            requested.append(image_urls(messages))
            if mode == "error":
                return httpx.Response(503, json={"error": {"message": image_url(self.red)}})
            if mode == "block":
                started.set()
                await asyncio.Future()
            return reply("Recovered")

        ops = self.runtime(handler)
        sid = ops.initial_session_id
        with patch("daphnis.attachments.AttachmentStore.write", side_effect=OSError("disk full")):
            result = await ops.send("Save this", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("持久化", result.error)
        self.assertEqual(ops.attachment_snapshot(ops.initial_session_id), {})
        self.assertFalse((self.history.directory / f"{sid}.json").exists())
        self.assertEqual(requested, [])

        result = await ops.send("Save this", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
        self.assertEqual(result.error, "模型请求失败。")
        refs = self.history.load(sid).attachment_refs
        self.assertEqual(len(ops.message_snapshot(ops.initial_session_id)), 1)
        self.assertEqual(len(refs[0].images), 1)
        mode = "block"
        retry = asyncio.create_task(ops.retry_failed_user(ops.recovery(ops.initial_session_id).key, stream=False, session_id=ops.initial_session_id))
        await asyncio.wait_for(started.wait(), timeout=5)
        await ops.cancel(ops.initial_session_id)
        with self.assertRaises(asyncio.CancelledError):
            await retry
        self.assertEqual(self.history.load(sid).attachment_refs, refs)
        self.assertEqual(len(requested), 2)
        await ops.close()

        ops = self.runtime(handler)
        ops.restore_session(sid)
        mode = "ok"
        item = refs[0].images[0]
        path = self.history.attachments.directory / sid / f"{item.id}.png"
        data = path.read_bytes()
        path.chmod(0o600)
        path.write_bytes(data[:-1] + bytes((data[-1] ^ 1,)))
        path.chmod(0o400)
        with self.assertRaisesRegex(OSError, "校验"):
            self.history.attachments.read_image(sid, item)
        result = await ops.retry_failed_user(ops.recovery(sid).key, stream=False, session_id=sid)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("图片附件快照不可用", result.error)
        self.assertEqual(len(requested), 2)
        self.assertEqual(self.history.load(sid).attachment_refs, refs)
        self.assertNotIn("base64", self.saved(sid))
        path.chmod(0o600)
        path.write_bytes(data)
        path.chmod(0o400)
        result = await ops.retry_failed_user(ops.recovery(sid).key, stream=False, session_id=sid)
        self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
        self.assertEqual(requested, [[image_url(self.red)]] * 3)
        self.assertEqual(len(ops.message_snapshot(sid)), 2)
        self.assertEqual(self.history.load(sid).attachment_refs, refs)

    async def test_unsupported_model_and_unsafe_context_do_not_send(self):
        def unexpected_request(_):
            self.fail("Image request reached HTTP")

        ops = self.runtime(unexpected_request, model_id="deepseek-v4-pro")
        with self.assertRaisesRegex(ValueError, "不支持 Daphnis 图片"):
            await ops.send("Inspect", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
        self.assertEqual(ops.message_snapshot(ops.initial_session_id), [])
        await ops.close()

        for context_window, error in ((None, "上限未知"), (1_000, "超过上下文安全上限")):
            catalog = tuple(replace(model, context_window=context_window) if model.id == "deepseek-flash"
                            else model for model in MODELS)
            ops = self.runtime(unexpected_request, catalog=catalog)
            result = await ops.send("Inspect", attachments=[self.red], stream=False, session_id=ops.initial_session_id)
            self.assertEqual(result.status, TurnStatus.FAILED)
            self.assertIn(error, result.error)
            self.assertTrue(self.history.load(ops.initial_session_id).attachment_refs)
            await ops.close()
