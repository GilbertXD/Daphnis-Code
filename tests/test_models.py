import json
import unittest

import httpx
from saturnus import ChatClient, Compatibility, ValidationError

from daphnis import Agent
from daphnis.models import (
    CapabilityError,
    ModelInfo,
    get_model,
    prepare_request_options,
    provider_compatibility,
    resolve_model,
)
from daphnis.models.requests import require_image_input

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
    def test_image_capability_is_declared_in_the_catalog(self):
        for provider, model_id in (
            ("mimo", "mimo-v2.6-flash"), ("mimo", "mimo-v2.6-pro"), ("deepseek", "deepseek-flash"),
        ):
            require_image_input(get_model(provider, model_id))
        custom = ModelInfo("vision", "compatible", supports_image=True)
        require_image_input(resolve_model("compatible", "vision", catalog=(custom,)))
        self.assertIsNone(get_model("deepseek", "deepseek-v4-pro").supports_image)
        with self.assertRaisesRegex(CapabilityError, "图片能力未确认"):
            require_image_input(resolve_model("compatible", "unregistered"))
        with self.assertRaisesRegex(CapabilityError, "明确不支持图片"):
            require_image_input(ModelInfo("text", "compatible", supports_image=False))
        with self.assertRaisesRegex(ValueError, "supports_image"):
            ModelInfo("invalid", "compatible", supports_image=1)

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
                "deepseek",
                "deepseek-flash",
                "off",
                "max_tokens",
                {"reasoning_effort": "none"},
            ),
            (
                "deepseek",
                "deepseek-v4-pro",
                "on",
                "max_tokens",
                {"reasoning_effort": "high"},
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
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=RESPONSE)

        for provider, model_id, mode, output_field, thinking_fields in cases:
            with self.subTest(provider=provider, mode=mode):
                model = resolve_model(provider, model_id)
                extra_body = {"vendor_option": {"keep": True}}
                options = prepare_request_options(
                    model=model,
                    thinking_mode=mode,
                    output_limit=512,
                    extra_body=extra_body,
                )
                async with ChatClient(
                    base_url=f"https://{provider}.test/v1",
                    api_key="test-key",
                    compatibility=provider_compatibility(provider),
                    transport=httpx.MockTransport(handler),
                ) as client:
                    response = await client.create(
                        model=model.id,
                        messages=MESSAGES,
                        extra_body=options,
                        extra_headers={"x-request": "test"},
                    )
                self.assertEqual(response, RESPONSE)
                self.assertEqual(extra_body, {"vendor_option": {"keep": True}})
                request = requests.pop()
                body = json.loads(request.content)
                self.assertEqual(request.url.host, f"{provider}.test")
                self.assertEqual(request.url.path, "/v1/chat/completions")
                self.assertEqual(request.headers["authorization"], "Bearer test-key")
                self.assertEqual(request.headers["x-request"], "test")
                self.assertEqual(
                    body,
                    {
                        "model": model_id,
                        "messages": MESSAGES,
                        output_field: 512,
                        **extra_body,
                        **thinking_fields,
                    },
                )

    def test_capability_rejection_and_provider_scoped_catalog(self):
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
        supported = get_model("deepseek", "shared", catalog=catalog)
        self.assertEqual(
            prepare_request_options(model=supported, tools=TOOLS, thinking_mode="high"),
            {"tools": TOOLS, "thinking_mode": "high"},
        )
        self.assertFalse(
            get_model("compatible", "shared", catalog=catalog).supports_tools
        )

        plain = resolve_model("deepseek", "plain", catalog=catalog)
        unknown = resolve_model("compatible", "unregistered", catalog=catalog)
        self.assertIsNone(unknown.supports_tools)
        self.assertIsNone(unknown.thinking_modes)
        for model, error in ((plain, "unsupported"), (unknown, "unknown")):
            with self.subTest(model=model.id):
                with self.assertRaisesRegex(CapabilityError, error):
                    prepare_request_options(model=model, tools=TOOLS)
                with self.assertRaisesRegex(CapabilityError, error):
                    prepare_request_options(model=model, thinking_mode="high")
                with self.assertRaisesRegex(CapabilityError, error):
                    prepare_request_options(
                        model=model, extra_body={"tool_choice": "auto"}
                    )
                self.assertEqual(
                    prepare_request_options(
                        model=model, extra_body={"tool_choice": "none"}
                    ),
                    {"tool_choice": "none"},
                )

    async def test_explicit_generic_capabilities_and_compatibility(self):
        model = ModelInfo(
            id="custom",
            provider="compatible",
            supports_tools=True,
            thinking_modes=("off", "high"),
        )
        with self.assertRaisesRegex(ValueError, "explicit Compatibility mapping"):
            prepare_request_options(model=model, thinking_mode="high")

        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["reasoning_effort"], "high")
            self.assertEqual(body["max_completion_tokens"], 128)
            self.assertEqual(body["tools"], TOOLS)
            return httpx.Response(200, json=RESPONSE)

        override = Compatibility(
            field_mappings={
                "thinking_mode": "reasoning_effort",
                "output_limit": "max_completion_tokens",
            }
        )
        async with ChatClient(
            base_url="https://custom.test/v1",
            transport=httpx.MockTransport(handler),
            compatibility=provider_compatibility("compatible", override=override),
        ) as client:
            await client.create(
                model=model.id,
                messages=MESSAGES,
                extra_body=prepare_request_options(
                    model=model,
                    thinking_mode="high",
                    tools=TOOLS,
                    output_limit=128,
                    custom_compatibility=True,
                ),
            )

    async def test_managed_fields_and_invalid_output_limits_are_rejected(self):
        model = get_model("deepseek", "deepseek-flash")
        for field in (
            "model",
            "messages",
            "stream",
            "tools",
            "thinking_mode",
            "thinking",
            "reasoning_effort",
            "output_limit",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "explicit request arguments"),
            ):
                prepare_request_options(model=model, extra_body={field: None})
        for limit in (0, -1, True, "512"):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                prepare_request_options(model=model, output_limit=limit)
        with self.assertRaises(TypeError):
            prepare_request_options(model=model, extra_body=[])
        with self.assertRaisesRegex(ValueError, "exceeds the model maximum"):
            prepare_request_options(model=model, output_limit=model.max_output_tokens + 1)

        async with ChatClient(
            base_url="https://model.test",
            compatibility=provider_compatibility(model.provider),
            transport=httpx.MockTransport(
                lambda request: self.fail("unexpected request")
            ),
        ) as client:
            with self.assertRaisesRegex(ValidationError, "max_tokens.*already present"):
                await client.create(
                    model=model.id,
                    messages=MESSAGES,
                    extra_body=prepare_request_options(
                        model=model, output_limit=512, extra_body={"max_tokens": 128}
                    ),
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

            def handler(request, tracked=tracked):
                body = json.loads(request.content)
                self.assertIs(body["stream"], True)
                self.assertEqual(body["thinking"], {"type": "enabled"})
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=tracked,
                )

            model = get_model("mimo", "mimo-v2.6-flash")
            async with ChatClient(
                base_url="https://mimo.test/v1",
                compatibility=provider_compatibility(model.provider),
                transport=httpx.MockTransport(handler),
            ) as client:
                received = []
                async with client.stream(
                    model=model.id,
                    messages=MESSAGES,
                    extra_body=prepare_request_options(model=model, thinking_mode="on"),
                ) as chunks:
                    async for item in chunks:
                        received.append(item)
                        if early_exit:
                            break
                self.assertTrue(tracked.closed)
                self.assertEqual(received, [chunk])

    async def test_unknown_model_and_duplicate_key_are_rejected(self):
        with self.assertRaises(KeyError):
            resolve_model("deepseek", "not-registered")
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            provider_compatibility("unknown")
        duplicate = ModelInfo(id="duplicate", provider="compatible")
        async with ChatClient(base_url="https://test.test") as client:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                Agent(
                    client=client, provider="compatible", catalog=(duplicate, duplicate)
                )


if __name__ == "__main__":
    unittest.main()
