import json
import unittest
from types import SimpleNamespace

import httpx

from daphnis import EventBus, TurnStatus
from daphnis.usage import UsageTotals
from experiments.mcp_tool_loading.run import Trace, digest


class MCPExperimentTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_wire_summary_and_latest_usage_keep_requests_separate(self):
        trace = Trace("deepseek", "deepseek-flash", "native", "T1", "smoke", 1)
        bus = EventBus()
        for name in ("model.started", "model.usage", "model.finished"):
            bus.on(name, trace.observe)
        fields = {"session_id": "s", "turn_id": "primary", "request_number": 1}
        await bus.emit("model.started", **fields)
        tools = [{"function": {"name": "activate_mcp"}}, {"function": {"name": "read_file"}}]
        request = httpx.Request("POST", "https://example.test", headers={"Authorization": "private-token"}, json={
            "messages": [{"role": "system", "content": "private-system"}],
            "tools": tools, "model": "deepseek-flash", "max_tokens": 768,
        })
        await trace.request(request)
        await trace.response(httpx.Response(200))
        for value in (
            {"prompt_tokens": 10, "completion_tokens": 1, "prompt_cache_hit_tokens": 8},
            {"prompt_tokens": 20, "completion_tokens": 2, "prompt_cache_hit_tokens": 0,
             "prompt_cache_miss_tokens": 20, "prompt_tokens_details": {"cached_tokens": 0}},
        ):
            await bus.emit("model.usage", **fields, usage=value)
        await bus.emit("model.finished", **fields)
        title_fields = fields | {"turn_id": "title", "purpose": "title"}
        await bus.emit("model.started", **title_fields)
        await bus.emit("model.usage", **title_fields, usage={"prompt_tokens": 5, "completion_tokens": 1})
        main, title = trace.requests.values()
        self.assertEqual((main["tool_schema_hash"], main["tool_schema_bytes"]), digest(tools))
        self.assertEqual(main["tool_names"], ["activate_mcp", "read_file"])
        self.assertEqual(main["request_options"], {"model": "deepseek-flash", "max_tokens": 768})
        self.assertEqual((main["purpose"], main["phase"], main["http_status"]), ("primary", "before_activation", 200))
        self.assertEqual((main["prompt_tokens"], main["cached_tokens"], main["uncached_prompt_tokens"]), (20, 0, 20))
        self.assertTrue(main["cache_reported"])
        self.assertEqual(main["cache_fields"]["prompt_tokens_details"], {"cached_tokens": 0})
        self.assertEqual((title["purpose"], title["prompt_tokens"], title["cached_tokens"]), ("title", 5, None))
        self.assertFalse(title["cache_reported"])
        summary = trace.result(SimpleNamespace(
            message={"content": "hot cool cold archive offline https://learn.microsoft.com/en-us/azure/storage/blobs/access-tiers-overview"},
            status=TurnStatus.COMPLETED, turn_id="primary", primary_requests=1, maintenance_requests=0,
        ), UsageTotals())
        self.assertFalse(summary["success"])
        saved = json.dumps([main, title, summary])
        self.assertNotIn("private-token", saved)
        self.assertNotIn("private-system", saved)
        self.assertNotIn("https://learn.microsoft.com", saved)
