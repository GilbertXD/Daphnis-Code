import json
import unittest

import httpx
from test_agent_core import make_client, registry, reply, session

from daphnis import Agent, Context
from daphnis.models import ModelInfo
from daphnis.token_budget import request_budget


class TokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_capacity_accounts_for_cjk_tools_and_output_reserve(self):
        plain = [{"role": "user", "content": "a" * 200}]
        cjk = [{"role": "user", "content": "中文" * 100}]
        self.assertGreater(
            request_budget(cjk, {}, None).input_tokens,
            request_budget(plain, {}, None).input_tokens,
        )
        body = {"tools": registry().definitions(), "output_limit": 100}
        budget = request_budget(cjk, body, 400)
        self.assertGreater(budget.input_tokens, request_budget(cjk, {}, None).input_tokens)
        self.assertEqual(budget.output_reserve, 100)
        self.assertEqual(budget.compaction_threshold, 280)
        self.assertTrue(budget.needs_compaction)
        unknown = request_budget(cjk, body, None)
        self.assertIsNone(unknown.compaction_threshold)
        self.assertFalse(unknown.needs_compaction)

    async def test_budget_observation_leaves_model_request_unchanged(self):
        state = session()
        model = ModelInfo("model-a", "compatible", context_window=80, supports_tools=True)
        tools = registry()
        bodies, events = [], []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=reply())

        async with make_client(handler) as client:
            agent = Agent(
                client=client, provider="compatible", catalog=(model,), tools=tools,
                context=Context(system_prompt="system"),
            )
            agent.events.on("context.budget", events.append)
            await agent.run(session=state, prompt="中文" * 50, output_limit=30)
        self.assertEqual(bodies[0]["messages"], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "中文" * 50},
        ])
        self.assertEqual(bodies[0]["tools"], tools.definitions())
        self.assertEqual(bodies[0]["max_tokens"], 30)
        self.assertEqual(events[0].data["request_number"], 1)
        self.assertTrue(events[0].data["needs_compaction"])
