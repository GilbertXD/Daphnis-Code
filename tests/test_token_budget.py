import json
import unittest

import httpx
from test_agent_core import make_client, registry, reply, session

from daphnis import Agent, Context
from daphnis.models import ModelInfo
from daphnis.token_budget import TokenCounter, request_budget


class TokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_optional_counter_counts_request_components_and_labels_local_values(self):
        seen = []

        def count(text):
            seen.append(json.loads(text))
            return 10

        messages = [{"role": "user", "content": "text"}]
        tools = registry().definitions()
        result = request_budget(
            messages, {"tools": tools}, 200, counter=TokenCounter(count, "local_tokenizer"),
        )
        self.assertEqual(seen, [messages, tools])
        self.assertEqual(result.input_tokens, 27)
        self.assertEqual(result.count_source, "local_tokenizer")

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

    def test_deepseek_reserves_the_default_completion_space_for_each_mode(self):
        for mode, reserve in ((None, 65_536), ("off", 8192), ("max", 131_072)):
            with self.subTest(mode=mode):
                body = {"thinking_mode": mode} if mode else {}
                budget = request_budget([], body, 1_048_576, provider="deepseek")
                self.assertEqual(budget.output_reserve, reserve)
        budget = request_budget([], {"output_limit": 512}, 1_048_576, provider="deepseek")
        self.assertEqual(budget.output_reserve, 512)

    async def test_budget_observation_leaves_model_request_unchanged(self):
        state = session()
        tools = registry()
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "中文" * 50},
        ]
        body = {"tools": tools.definitions(), "output_limit": 30}
        window = request_budget(messages, body, None).input_tokens + 30
        model = ModelInfo("model-a", "compatible", context_window=window, supports_tools=True)
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
        self.assertEqual(bodies[0]["messages"], messages)
        self.assertEqual(bodies[0]["tools"], tools.definitions())
        self.assertEqual(bodies[0]["max_tokens"], 30)
        self.assertEqual(events[0].data["request_number"], 1)
        self.assertTrue(events[0].data["needs_compaction"])

    async def test_request_that_cannot_fit_fails_before_transport(self):
        state = session()
        model = ModelInfo("model-a", "compatible", context_window=80)
        async with make_client(lambda _: self.fail("Unexpected model request")) as client:
            agent = Agent(client=client, provider="compatible", catalog=(model,))
            result = await agent.run(session=state, prompt="中文" * 50, output_limit=30)
        self.assertIn("cannot fit", result.error)
        self.assertEqual(result.model_requests, 0)
        self.assertEqual(state.messages.snapshot(), [])
