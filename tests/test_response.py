import unittest
from copy import deepcopy

import httpx
from test_agent_core import CATALOG, make_agent, make_client

from daphnis.agent import Agent
from daphnis.compaction import CompactionError
from daphnis.context import DOCUMENT_REFS, IMAGE_REFS
from daphnis.errors import ValidationError
from daphnis.response import ResponseAccumulator
from daphnis.tool_state import TOOL_STATE
from daphnis.turn_state import TURN_ATTEMPT, TURN_STATE


def chunk(delta, *, finish_reason=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


class ResponseAccumulatorTests(unittest.TestCase):
    def test_finalize_combines_choices_text_reasoning_and_usage(self):
        accumulator = ResponseAccumulator()
        accumulator.add_chunk(
            {
                "id": "response-1",
                "created": 5,
                "choices": [
                    {
                        "index": 1,
                        "delta": {"content": "Separate"},
                        "finish_reason": "length",
                    },
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "Hello ",
                            "reasoning_content": "First ",
                        },
                    },
                ],
            }
        )
        accumulator.add_chunk(
            {
                "model": "example",
                "usage": {"prompt_tokens": 8},
                **chunk(
                    {"content": "world", "reasoning_content": "step"},
                    finish_reason="stop",
                ),
            }
        )
        usage = {
            "total_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
        accumulator.add_chunk({"choices": [], "usage": usage})
        accumulator.add_chunk({"choices": [], "usage": None})
        usage["completion_tokens_details"].clear()

        expected = {
            "id": "response-1",
            "created": 5,
            "model": "example",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello world",
                        "reasoning_content": "First step",
                    },
                },
                {
                    "index": 1,
                    "finish_reason": "length",
                    "message": {"content": "Separate"},
                },
            ],
            "usage": {
                "total_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }
        result = accumulator.finalize()
        self.assertEqual(result, expected)
        result["usage"]["completion_tokens_details"].clear()
        result["choices"].clear()
        self.assertEqual(accumulator.finalize(), expected)
        with self.assertRaisesRegex(ValidationError, "after finalization"):
            accumulator.add_chunk({"choices": []})

    def test_finalize_collects_indexed_tools_and_argument_fragments(self):
        accumulator = ResponseAccumulator()
        accumulator.add_chunk(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "second",
                            "type": "function",
                            "function": {"arguments": '{"text":'},
                        },
                        {
                            "index": 0,
                            "id": "first",
                            "type": "function",
                            "function": {"name": "first_tool", "arguments": "{}"},
                        },
                    ]
                }
            )
        )
        with self.assertRaisesRegex(ValidationError, "identity conflict.*name"):
            accumulator.add_chunk(
                chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "different_name"}},
                        ]
                    }
                )
            )
        accumulator.add_chunk(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "function": {
                                "name": "second_tool",
                                "arguments": '"hello"}',
                            },
                        },
                        {"index": 0, "function": {"name": "first_tool"}},
                    ]
                },
                finish_reason="tool_calls",
            )
        )

        (choice,) = accumulator.finalize()["choices"]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(
            choice["message"]["tool_calls"],
            [
                {
                    "id": "first",
                    "type": "function",
                    "function": {"name": "first_tool", "arguments": "{}"},
                },
                {
                    "id": "second",
                    "type": "function",
                    "function": {
                        "name": "second_tool",
                        "arguments": '{"text":"hello"}',
                    },
                },
            ],
        )

    def test_invalid_chunk_and_failed_finalization_leave_state_unsealed(self):
        accumulator = ResponseAccumulator()
        accumulator.add_chunk(chunk({"content": "Keep"}))
        with self.assertRaisesRegex(ValidationError, "no finish reason"):
            accumulator.finalize()
        with self.assertRaisesRegex(
            ValidationError, "delta.content must be a string or null"
        ):
            accumulator.add_chunk(
                {"usage": {"total_tokens": 99}, **chunk({"content": 42})}
            )
        accumulator.add_chunk(chunk({"content": " going"}, finish_reason="stop"))
        self.assertEqual(
            accumulator.finalize(),
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "Keep going"},
                        "finish_reason": "stop",
                    },
                ]
            },
        )

    def test_finalize_rejects_incomplete_tool_call(self):
        accumulator = ResponseAccumulator()
        accumulator.add_chunk(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "example"},
                        },
                    ]
                },
                finish_reason="tool_calls",
            )
        )
        with self.assertRaisesRegex(ValidationError, "Incomplete tool call"):
            accumulator.finalize()


class AgentResponseTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_response_copies_message_and_cleans_internal_fields(self):
        message = {
            "content": [{"type": "text", "text": "visible"}],
            "vendor": {"items": [1]},
        }
        internal = dict.fromkeys(
            (TOOL_STATE, TURN_STATE, TURN_ATTEMPT, IMAGE_REFS, DOCUMENT_REFS), "ignored"
        )
        response = {
            "choices": [
                {
                    "index": 0,
                    "message": {**message, **internal},
                    "finish_reason": "stop",
                }
            ]
        }
        original = deepcopy(response)

        parsed, reason = Agent._parse_response(response)

        self.assertEqual((parsed, reason), ({**message, "role": "assistant"}, "stop"))
        parsed["vendor"]["items"].append(2)
        self.assertEqual(response, original)

    async def test_invalid_summary_response_keeps_compaction_error(self):
        response = {"choices": [{"index": 0, "message": None, "finish_reason": "stop"}]}
        async with make_client(
            lambda request: httpx.Response(200, json=response)
        ) as client:
            with self.assertRaises(CompactionError) as caught:
                await make_agent(client)._request_summary(
                    CATALOG[0],
                    [{"role": "user", "content": "Summarize"}],
                    32,
                    {"session_id": "session", "turn_id": "turn", "request_number": 1},
                )
        self.assertEqual(caught.exception.reason, "summary_response")
        self.assertIsInstance(caught.exception.__cause__, ValidationError)
