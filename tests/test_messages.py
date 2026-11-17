import unittest

from daphnis.errors import ValidationError
from daphnis.messages import Messages


class MessagesTests(unittest.TestCase):
    def test_input_get_and_snapshot_are_isolated(self):
        raw = {"role": "user", "content": [{"type": "text", "text": "original"}]}
        messages = Messages([raw])
        messages.add(raw)

        raw["content"][0]["text"] = "changed input"
        messages.get(-1)["content"].clear()
        messages.snapshot()[0]["content"][0]["text"] = "changed snapshot"

        self.assertEqual(
            [message["content"][0]["text"] for message in messages.snapshot()],
            ["original", "original"],
        )

    def test_editing_and_tool_result_preserve_message_order(self):
        messages = Messages()
        messages.add_user("question")
        messages.add_tool_result(
            tool_call_id="call-1", content="result", extra={"vendor": 1}
        )
        messages.insert(-1, {"role": "assistant", "reasoning": "work"})
        messages.replace(-2, {"role": "assistant", "content": None})
        messages.remove(0)

        self.assertEqual(
            messages.snapshot(),
            [
                {"role": "assistant", "content": None},
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "result",
                    "vendor": 1,
                },
            ],
        )

    def test_invalid_message_or_tool_result_leaves_history_intact(self):
        messages = Messages([{"role": "user", "content": "question"}])
        original = messages.snapshot()

        with self.assertRaisesRegex(ValidationError, "arrays must use lists"):
            messages.replace(0, {"role": "user", "extra": (1, 2)})
        with self.assertRaisesRegex(ValidationError, "tool_call_id"):
            messages.add_tool_result(tool_call_id=" ", content="result")
        self.assertEqual(messages.snapshot(), original)
