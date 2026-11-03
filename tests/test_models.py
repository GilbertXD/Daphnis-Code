import json
import unittest

import httpx
from saturnus import Compatibility

from daphnis.models import LLM, CapabilityError, ModelInfo, get_model

MESSAGES = [{"role": "user", "content": "hello"}]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_time", "parameters": {"type": "object"}},
    }
]
RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    "vendor_metadata": {"kept": True},
}


class ModelsTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_routing_and_parameter_mappings(self):
        cases = (
            (
                "deepseek",
                "deepseek-flash",
                "low",
                "max_tokens",
                {"reasoning_effort": "low"},
            ),
            (
                "mimo",
                "mimo-v2.6-pro",
                "off",
                "max_completion_tokens",
                {"thinking": {"type": "disabled"}},
            ),
            ("compatible", "my-local-model", None, "max_tokens", {}),
        )
        for provider, model_id, mode, output_field, thinking_fields in cases:
            with self.subTest(provider=provider):

                def handler(request):
                    body = json.loads(request.content)
                    self.assertEqual(request.url.host, f"{provider}.test")
                    self.assertEqual(request.url.path, "/v1/chat/completions")
                    self.assertEqual(
                        request.headers["authorization"], "Bearer test-key"
                    )
                    self.assertEqual(request.headers["x-request"], "test")
                    self.assertEqual(body["model"], model_id)
                    self.assertEqual(body[output_field], 512)
                    self.assertEqual(body["vendor_option"], {"keep": True})
                    self.assertNotIn("output_limit", body)
                    self.assertNotIn("thinking_mode", body)
                    for field, value in thinking_fields.items():
                        self.assertEqual(body[field], value)
                    if mode is None:
                        self.assertNotIn("thinking", body)
                        self.assertNotIn("reasoning_effort", body)
                    return httpx.Response(200, json=RESPONSE)

                async with LLM(
                    provider=provider,
                    model_id=model_id,
                    base_url=f"https://{provider}.test/v1",
                    api_key="test-key",
                    transport=httpx.MockTransport(handler),
                ) as llm:
                    response = await llm.create(
                        messages=MESSAGES,
                        thinking_mode=mode,
                        output_limit=512,
                        extra_body={"vendor_option": {"keep": True}},
                        extra_headers={"x-request": "test"},
                    )
                    self.assertEqual(response, RESPONSE)

    async def test_capability_rejection_and_model_switch(self):
        catalog = (
            ModelInfo(
                id="shared",
                provider="deepseek",
                supports_tools=True,
                thinking_modes=("off", "high"),
            ),
            ModelInfo(
                id="shared",
                provider="compatible",
                supports_tools=False,
                thinking_modes=(),
            ),
            ModelInfo(
                id="plain", provider="deepseek", supports_tools=False, thinking_modes=()
            ),
        )
        self.assertTrue(get_model("deepseek", "shared", catalog=catalog).supports_tools)
        self.assertFalse(
            get_model("compatible", "shared", catalog=catalog).supports_tools
        )

        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["tools"], TOOLS)
            self.assertEqual(body["reasoning_effort"], "high")
            return httpx.Response(200, json=RESPONSE)

        async with LLM(
            provider="deepseek",
            model_id="shared",
            catalog=catalog,
            base_url="https://deepseek.test/v1",
            transport=httpx.MockTransport(handler),
        ) as llm:
            await llm.create(messages=MESSAGES, tools=TOOLS, thinking_mode="high")
            llm.select_model("plain")
            with self.assertRaisesRegex(CapabilityError, "unsupported"):
                await llm.create(messages=MESSAGES, tools=TOOLS)
            with self.assertRaisesRegex(CapabilityError, "unsupported"):
                await llm.create(messages=MESSAGES, thinking_mode="high")

        def unexpected_request(request):
            self.fail("Invalid requests must be rejected before network I/O.")

        async with LLM(
            provider="compatible",
            model_id="unregistered",
            base_url="https://local.test/v1",
            transport=httpx.MockTransport(unexpected_request),
        ) as llm:
            self.assertIsNone(llm.model.supports_tools)
            self.assertIsNone(llm.model.thinking_modes)
            with self.assertRaisesRegex(CapabilityError, "unknown"):
                await llm.create(messages=MESSAGES, tools=TOOLS)
            with self.assertRaisesRegex(CapabilityError, "unknown"):
                await llm.create(messages=MESSAGES, thinking_mode="on")
            with self.assertRaisesRegex(ValueError, "explicit request arguments"):
                await llm.create(
                    messages=MESSAGES, extra_body={"thinking": {"type": "enabled"}}
                )

    async def test_explicit_generic_capabilities_and_compatibility(self):
        catalog = (
            ModelInfo(
                id="custom",
                provider="compatible",
                supports_tools=True,
                thinking_modes=("off", "high"),
            ),
        )

        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["reasoning_effort"], "high")
            self.assertEqual(body["max_completion_tokens"], 128)
            self.assertEqual(body["tools"], TOOLS)
            return httpx.Response(200, json=RESPONSE)

        async with LLM(
            provider="compatible",
            model_id="custom",
            catalog=catalog,
            base_url="https://custom.test/v1",
            transport=httpx.MockTransport(handler),
            compatibility=Compatibility(
                field_mappings={
                    "thinking_mode": "reasoning_effort",
                    "output_limit": "max_completion_tokens",
                }
            ),
        ) as llm:
            await llm.create(
                messages=MESSAGES,
                thinking_mode="high",
                tools=TOOLS,
                output_limit=128,
            )

    async def test_stream_preserves_raw_chunks_and_closes_on_exit(self):
        chunk = {
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
            "vendor_metadata": {"kept": True},
        }

        class TrackedStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                self.closed = True

        for early_exit in (False, True):
            tracked = TrackedStream()

            def handler(request):
                body = json.loads(request.content)
                self.assertIs(body["stream"], True)
                self.assertEqual(body["thinking"], {"type": "enabled"})
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=tracked,
                )

            async with LLM(
                provider="mimo",
                model_id="mimo-v2.6-flash",
                base_url="https://mimo.test/v1",
                transport=httpx.MockTransport(handler),
            ) as llm:
                received = []
                async with llm.stream(messages=MESSAGES, thinking_mode="on") as chunks:
                    async for item in chunks:
                        received.append(item)
                        if early_exit:
                            break
                self.assertTrue(tracked.closed)
                self.assertEqual(received, [chunk])

    async def test_unknown_model_and_duplicate_key_are_rejected(self):
        with self.assertRaises(KeyError):
            LLM(
                provider="deepseek",
                model_id="not-registered",
                base_url="https://test.test",
            )
        duplicate = ModelInfo(id="duplicate", provider="compatible")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            LLM(
                provider="compatible",
                model_id="duplicate",
                base_url="https://test.test",
                catalog=(duplicate, duplicate),
            )


if __name__ == "__main__":
    unittest.main()
