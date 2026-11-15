import asyncio
import socket
import unittest
from unittest.mock import patch

import httpx2

from daphnis.mcp.public_http import public_transport
from daphnis.mcp_registry import RegistryDiscovery, RegistryError

NAME = "com.example/docs"
ENTRY = {
    "server": {
        "name": NAME, "version": "1.2.3", "description": "Example documentation",
        "remotes": [
            {"type": "streamable-http", "url": "https://docs.example.com/mcp"},
            {"type": "streamable-http", "url": "https://127.0.0.1/mcp"},
            {"type": "streamable-http", "url": "https://other.example.com/mcp",
             "headers": [{"name": "X-API-Key", "isRequired": True}]},
        ],
        "packages": [{"registryType": "npm", "identifier": "@example/docs", "version": "1.2.3"}],
    },
    "_meta": {"io.modelcontextprotocol.registry/official": {"status": "active"}},
}


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_pages_and_details_validate_registry_metadata(self):
        def handler(request):
            self.assertEqual(request.url.host, "registry.modelcontextprotocol.io")
            if request.url.path == "/v0.1/servers":
                self.assertEqual(request.url.params["version"], "latest")
                self.assertEqual(request.url.params["search"], "docs")
                if request.url.params.get("cursor") == "next/+?":
                    return httpx2.Response(200, json={"servers": [], "metadata": {"count": 0}})
                return httpx2.Response(200, json={
                    "servers": [ENTRY, {"server": {"name": "invalid"}}],
                    "metadata": {"nextCursor": "next/+?"},
                })
            self.assertEqual(request.url.raw_path, b"/v0.1/servers/com.example%2Fdocs/versions/latest")
            return httpx2.Response(200, json=ENTRY)

        discovery = RegistryDiscovery(transport=httpx2.MockTransport(handler))
        page = await discovery.search("docs", None)
        self.assertEqual([server["name"] for server in page["servers"]], [NAME])
        self.assertEqual(await discovery.search("docs", page["cursor"]), {"servers": [], "cursor": None})
        detail = await discovery.detail(NAME, "latest")
        self.assertEqual(detail["version"], "1.2.3")
        self.assertEqual(detail["packages"][0]["identifier"], "@example/docs")
        self.assertEqual([remote["auth"] for remote in detail["remotes"]], ["none", "unsupported", "unsupported"])

    async def test_timeouts_and_invalid_registry_responses_are_safe(self):
        async def handler(request):
            if request.url.params["search"] == "timeout":
                raise httpx2.ReadTimeout("upstream details")
            return httpx2.Response(200, json={"servers": {"invalid": "body"}})

        discovery = RegistryDiscovery(transport=httpx2.MockTransport(handler))
        with self.assertRaisesRegex(RegistryError, "Registry 不可用"):
            await discovery.search("timeout", None)
        with self.assertRaisesRegex(RegistryError, "无效的搜索结果"):
            await discovery.search("invalid", None)

    async def test_public_transport_checks_dns_and_pins_the_dialed_address(self):
        url = httpx2.URL("https://docs.example.com/mcp")
        loop = asyncio.get_running_loop()
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("9.9.9.9", 443))
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 443))
        with (
            patch.object(loop, "getaddrinfo", return_value=[public, private]),
            self.assertRaisesRegex(ValueError, "非公网"),
        ):
            await public_transport(url)
        with patch.object(loop, "getaddrinfo", return_value=[public]) as lookup:
            transport = await public_transport(url)
            async with transport:
                with patch.object(httpx2.AsyncHTTPTransport, "handle_async_request") as send:
                    await transport.handle_async_request(httpx2.Request("POST", url))
            lookup.assert_awaited_once()
            sent = send.await_args.args[0]
            self.assertEqual(sent.url.host, "9.9.9.9")
            self.assertEqual(sent.headers["host"], "docs.example.com")
            self.assertEqual(sent.extensions["sni_hostname"], "docs.example.com")
