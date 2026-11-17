import json
import unittest
from unittest.mock import patch

import httpx

from daphnis.errors import (
    APIError,
    CommunicationError,
    ResponseError,
    TransportError,
    ValidationError,
)
from daphnis.llm.client import ChatClient
from daphnis.models.compatibility import Compatibility

BASE_URL = "https://model.test/v1"
REQUEST = {"model": "example", "messages": [{"role": "user", "content": "hello"}]}
SSE_HEADERS = {"Content-Type": "text/event-stream"}


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, parts, *, error=None):
        self.parts = parts
        self.error = error
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            yield part
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


def mock_client(handler, **options):
    return ChatClient(
        base_url=BASE_URL, transport=httpx.MockTransport(handler), **options
    )


class ChatClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_custom_endpoint_auth_and_mapping_in_both_modes(self):
        requests = []
        expected = {"choices": [], "vendor_data": {"trace": True}}

        def handler(request):
            requests.append(request)
            if json.loads(request.content).get("stream"):
                return httpx.Response(
                    200,
                    headers=SSE_HEADERS,
                    text=f"data: {json.dumps(expected)}\n\ndata: [DONE]\n\n",
                )
            return httpx.Response(200, json=expected)

        auth = {"X-Api-Key": "fixture-key"}
        async with ChatClient(
            base_url="https://model.test/gateway%2Fteam/v1///?version=1&path=a%2Fb",
            endpoint_path="deployment/chat/completions",
            auth_headers=auth,
            timeout=4,
            compatibility=Compatibility(field_mappings={"output_limit": "max_tokens"}),
            transport=httpx.MockTransport(handler),
        ) as client:
            auth.clear()
            options = {
                **REQUEST,
                "extra_body": {"output_limit": 32, "vendor_option": True},
                "extra_headers": {"X-Trace": "fixture-trace"},
            }
            self.assertEqual(await client.create(**options), expected)
            async with client.stream(**options) as chunks:
                self.assertEqual([chunk async for chunk in chunks], [expected])

        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertEqual(
                request.url.raw_path,
                b"/gateway%2Fteam/v1/deployment/chat/completions?version=1&path=a%2Fb",
            )
            self.assertEqual(request.headers["x-api-key"], "fixture-key")
            self.assertEqual(request.headers["x-trace"], "fixture-trace")
            self.assertNotIn("authorization", request.headers)
            self.assertEqual(request.extensions["timeout"]["read"], 4)
        body = {**REQUEST, "max_tokens": 32, "vendor_option": True}
        self.assertEqual(
            [json.loads(request.content) for request in requests],
            [body, {**body, "stream": True}],
        )

    def test_invalid_connection_config_is_rejected_before_creating_client(self):
        cases = (
            ({"base_url": BASE_URL + "#fragment"}, "fragment"),
            ({"endpoint_path": "deployment/../chat"}, "dot segments"),
            (
                {"api_key": "fixture-key", "auth_headers": {}},
                "cannot be provided together",
            ),
            ({"api_key": "invalid key"}, "ASCII string without whitespace"),
            ({"timeout": False}, "finite positive number"),
            ({"transport": object()}, "AsyncBaseTransport"),
        )
        with patch("daphnis.llm.client.httpx.AsyncClient") as http_client:
            for options, error in cases:
                with (
                    self.subTest(error=error),
                    self.assertRaisesRegex(ValidationError, error),
                ):
                    ChatClient(**{"base_url": BASE_URL, **options})
            http_client.assert_not_called()

    async def test_invalid_request_and_protected_headers_never_reach_transport(self):
        cases = (
            ({"extra_headers": {"x-aPi-kEy": "override"}}, "controlled by ChatClient"),
            ({"extra_headers": {"Host": "other.test"}}, "controlled by ChatClient"),
            (
                {"extra_headers": {"X-Trace": "one", "x-trace": "two"}},
                "Duplicate header name",
            ),
            ({"extra_body": {"stream": True}}, "conflicts with reserved fields"),
            ({"messages": [{"content": "missing role"}]}, r"messages\[0\]\.role"),
            ({"extra_body": {"vendor": (1, 2)}}, "JSON arrays must use lists"),
        )
        async with mock_client(
            lambda request: self.fail("Unexpected request"),
            auth_headers={"X-Api-Key": "fixture"},
        ) as client:
            for options, error in cases:
                with (
                    self.subTest(options=options),
                    self.assertRaisesRegex(ValidationError, error),
                ):
                    await client.create(**{**REQUEST, **options})

    async def test_failure_responses_remain_readable_after_close(self):
        cases = (
            (False, 503, APIError, "HTTP 503"),
            (True, 503, APIError, "HTTP 503"),
            (True, 200, ResponseError, "text/event-stream"),
        )
        for stream, status, error_type, message in cases:
            with self.subTest(stream=stream, status=status):
                source = TrackedStream(
                    [b'{"error":', b'{"message":"fixture failure"}}']
                )
                response = httpx.Response(
                    status, headers={"Content-Type": "application/json"}, stream=source
                )
                async with mock_client(
                    lambda request, response=response: response
                ) as client:
                    with self.assertRaisesRegex(error_type, message) as caught:
                        if stream:
                            async with client.stream(**REQUEST):
                                self.fail("Unexpected usable stream")
                        else:
                            await client.create(**REQUEST)

                    error = caught.exception
                    self.assertIsInstance(error, CommunicationError)
                    self.assertIs(error.response, response)
                    self.assertEqual(error.response.status_code, status)
                    self.assertEqual(
                        error.response.json(), {"error": {"message": "fixture failure"}}
                    )
                    self.assertTrue(source.closed)
                    if isinstance(error, APIError):
                        self.assertIsInstance(error.__cause__, httpx.HTTPStatusError)
                    else:
                        self.assertIsNone(error.event_data)

    async def test_create_rejects_invalid_json_and_non_object_responses(self):
        for body, message, cause in (
            (b"broken", "invalid JSON", ValueError),
            (b"[]", "JSON object", type(None)),
        ):
            with self.subTest(body=body):
                response = httpx.Response(200, content=body)
                async with mock_client(
                    lambda request, response=response: response
                ) as client:
                    with self.assertRaisesRegex(ResponseError, message) as caught:
                        await client.create(**REQUEST)
                self.assertIs(caught.exception.response, response)
                self.assertIsNone(caught.exception.event_data)
                self.assertIsInstance(caught.exception.__cause__, cause)

    async def test_connection_errors_preserve_the_original_cause(self):
        failure = httpx.ConnectError("mock connection failure")

        def handler(request):
            raise failure

        async with mock_client(handler) as client:
            with self.assertRaises(TransportError) as plain:
                await client.create(**REQUEST)
            with self.assertRaises(TransportError) as stream:
                async with client.stream(**REQUEST):
                    self.fail("Unexpected usable stream")
        self.assertIs(plain.exception.__cause__, failure)
        self.assertIs(stream.exception.__cause__, failure)

    async def test_sse_decodes_split_utf8_multiline_data_and_line_endings(self):
        wire = (
            '\ufeff: keep alive\r\nevent: ignored\rdata: {"choices": [],\r\n'
            'data: "vendor": "你好\u2028世界"}\n\rdata: [DONE]\r\n\r\n'
            "data: ignored after done\n\n"
        ).encode()
        source = TrackedStream(bytes([byte]) for byte in wire)
        response = httpx.Response(
            200,
            headers={"Content-Type": "Text/Event-Stream; charset=latin-1"},
            stream=source,
        )
        async with mock_client(lambda request: response) as client:
            async with client.stream(**REQUEST) as chunks:
                self.assertEqual(
                    [chunk async for chunk in chunks],
                    [{"choices": [], "vendor": "你好\u2028世界"}],
                )
            self.assertTrue(source.closed)

    async def test_sse_errors_retain_event_data_and_close_response(self):
        cases = (
            (b"data: not-json\n\n", "invalid JSON", "not-json", []),
            (b"data: []\n\n", "JSON object", "[]", []),
            (
                b'data: {"choices": []}\n\ndata: [DONE]\r',
                r"before the \[DONE\] marker",
                "[DONE]",
                [{"choices": []}],
            ),
        )
        for wire, message, event_data, expected in cases:
            with self.subTest(event_data=event_data):
                source = TrackedStream([wire])
                response = httpx.Response(200, headers=SSE_HEADERS, stream=source)
                received = []
                async with mock_client(
                    lambda request, response=response: response
                ) as client:
                    with self.assertRaisesRegex(ResponseError, message) as caught:
                        async with client.stream(**REQUEST) as chunks:
                            async for chunk in chunks:
                                received.append(chunk)
                    self.assertTrue(source.closed)
                self.assertIs(caught.exception.response, response)
                self.assertEqual(caught.exception.event_data, event_data)
                self.assertEqual(received, expected)

    async def test_interrupted_stream_and_error_body_close_and_preserve_cause(self):
        for status, message, expected in (
            (200, "streaming response could not be read", [{"choices": []}]),
            (503, "server error response could not be read", []),
        ):
            with self.subTest(status=status):
                failure = httpx.ReadError("fixture disconnected")
                source = TrackedStream([b'data: {"choices": []}\n\n'], error=failure)
                response = httpx.Response(status, headers=SSE_HEADERS, stream=source)
                received = []
                async with mock_client(
                    lambda request, response=response: response
                ) as client:
                    with self.assertRaisesRegex(TransportError, message) as caught:
                        async with client.stream(**REQUEST) as chunks:
                            async for chunk in chunks:
                                received.append(chunk)
                    self.assertTrue(source.closed)
                self.assertIs(caught.exception.__cause__, failure)
                self.assertEqual(received, expected)

    async def test_shared_client_isolates_streams_headers_and_bodies(self):
        requests = []
        sources = {}

        def handler(request):
            body = json.loads(request.content)
            model = body["model"]
            requests.append((dict(request.headers), body))
            wire = f"data: {json.dumps({'model': model})}\r\n\r\n".encode()
            source = sources[model] = TrackedStream([wire, b"data: [DONE]\r\n\r\n"])
            return httpx.Response(200, headers=SSE_HEADERS, stream=source)

        transport = httpx.MockTransport(handler)
        client = ChatClient(
            base_url=BASE_URL, transport=transport, api_key="fixture-key"
        )
        with patch.object(transport, "aclose", wraps=transport.aclose) as close:
            async with client:
                async with client.stream(
                    model="first",
                    messages=[{"role": "user", "content": "one"}],
                    extra_headers={"X-Trace": "first", "X-One": "only here"},
                    extra_body={"limit": 1},
                ) as first:
                    self.assertEqual(await anext(first), {"model": "first"})
                    async with client.stream(
                        model="second",
                        messages=[{"role": "user", "content": "two"}],
                        extra_headers={"X-Trace": "second"},
                        extra_body={"limit": 2},
                    ) as second:
                        self.assertEqual(
                            [chunk async for chunk in second], [{"model": "second"}]
                        )
                    self.assertTrue(sources["second"].closed)
                    self.assertFalse(sources["first"].closed)
                self.assertTrue(sources["first"].closed)
                close.assert_not_awaited()
            close.assert_awaited_once()

        (first_headers, first_body), (second_headers, second_body) = requests
        self.assertEqual(
            first_headers["authorization"], second_headers["authorization"]
        )
        self.assertEqual(first_headers["x-trace"], "first")
        self.assertEqual(second_headers["x-trace"], "second")
        self.assertEqual(first_headers["x-one"], "only here")
        self.assertNotIn("x-one", second_headers)
        self.assertEqual(
            first_body,
            {
                "model": "first",
                "messages": [{"role": "user", "content": "one"}],
                "limit": 1,
                "stream": True,
            },
        )
        self.assertEqual(
            second_body,
            {
                "model": "second",
                "messages": [{"role": "user", "content": "two"}],
                "limit": 2,
                "stream": True,
            },
        )
