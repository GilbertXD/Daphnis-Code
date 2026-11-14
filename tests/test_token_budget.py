import json
import unittest

import httpx
from test_agent_core import make_client, registry, reply, session

from daphnis import Agent, Context
from daphnis.compaction_settings import CompactionSettings
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

    def test_fixed_threshold_counts_reserve_even_if_window_is_unknown(self):
        settings = CompactionSettings(threshold_mode="tokens", threshold_value=131_072)
        budget = request_budget(
            [], {"max_completion_tokens": 8192}, None, settings=settings,
            default_output_tokens=131_072, counter=TokenCounter(lambda _: 122_877),
        )
        self.assertEqual(budget.output_reserve, 8192)
        self.assertEqual(budget.compaction_threshold, 122_880)
        self.assertTrue(budget.needs_compaction)
        self.assertTrue(budget.fits)

        blocked = request_budget([], {}, None, settings=settings, default_output_tokens=131_072)
        self.assertEqual(blocked.output_reserve, 131_072)
        self.assertTrue(blocked.threshold_blocked_by_reserve)
        self.assertFalse(blocked.needs_compaction)

    def test_percent_recomputes_on_model_switch_and_hard_limit_stays_enabled(self):
        settings = CompactionSettings(auto_enabled=False, threshold_value=0.70)
        body = {"output_limit": 20}
        at_limit = request_budget([], body, 100, settings=settings, counter=TokenCounter(lambda _: 72))
        self.assertEqual(at_limit.compaction_threshold, 50)
        self.assertFalse(at_limit.needs_compaction)
        self.assertTrue(at_limit.fits)
        self.assertEqual(at_limit.hard_limit, 95)
        self.assertEqual(at_limit.hard_input_limit, 75)
        above_limit = request_budget([], body, 100, settings=settings, counter=TokenCounter(lambda _: 73))
        self.assertTrue(above_limit.needs_safety_compaction)
        larger_model = request_budget([], body, 200, settings=settings)
        self.assertEqual(larger_model.compaction_threshold, 120)
        capped = request_budget([], body, 100, settings=CompactionSettings(threshold_value=1))
        self.assertEqual(capped.trigger_tokens, 95)

    async def test_budget_observation_leaves_model_request_unchanged(self):
        state = session()
        tools = registry()
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "中文" * 50},
        ]
        body = {"tools": tools.definitions(), "output_limit": 30}
        total = request_budget(messages, body, None).input_tokens + 30
        window = (total * 100 + 94) // 95
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
        prompt = "中文" * 15
        total = request_budget([{"role": "user", "content": prompt}], {"output_limit": 30}, None)
        model = ModelInfo("model-a", "compatible", context_window=total.input_tokens + 30)
        async with make_client(lambda _: self.fail("Unexpected model request")) as client:
            agent = Agent(
                client=client, provider="compatible", catalog=(model,),
                compaction_settings=CompactionSettings(auto_enabled=False),
            )
            result = await agent.run(session=state, prompt=prompt, output_limit=30)
        self.assertIn("安全上限", result.error)
        self.assertEqual(result.model_requests, 0)
        self.assertEqual(state.messages.snapshot(), [{"role": "user", "content": prompt}])
