import unittest

from daphnis.attachments import (
    ATTACHMENT_ONLY_PROMPT,
    TextResourceInput,
    prepare_attachments,
)
from daphnis.redaction import Redactor
from daphnis.title import build_title_source, fallback_title, normalize_title


class TitleTests(unittest.TestCase):
    def test_sources_keep_roles_and_prioritize_task_text(self):
        text = "A " * 2000
        upload = TextResourceInput("Architecture", "small excerpt", "demo", "notes://arch", "text/plain", "embedded")
        batch = prepare_attachments((upload,), None)
        source = build_title_source([
            {"role": "user", "content": "Initial request " + text},
            {"role": "assistant", "content": "Proposed change"},
            {"role": "user", "content": "Final goal"},
            {"role": "tool", "content": "tool result"},
        ], {1: batch}, max_chars=390, excerpts={batch.files[0].id: "small excerpt"})
        self.assertLessEqual(len(source), 390)
        self.assertTrue(source.startswith("User: Initial request"))
        self.assertIn("Assistant: Proposed change\nAssistant: 附件: Architecture (text/plain, 13 字节)", source)
        self.assertIn("small excerpt", source)
        self.assertTrue(source.endswith("User: Final goal"))
        self.assertNotIn("tool result", source)

        resource_only = build_title_source([{"role": "user", "content": ATTACHMENT_ONLY_PROMPT}], {0: batch})
        self.assertEqual(fallback_title(resource_only), "Architecture")
        self.assertNotIn(ATTACHMENT_ONLY_PROMPT, resource_only)

    def test_redaction_precedes_truncation(self):
        secret = "title-secret-0123456789"
        source = build_title_source([{"role": "user", "content": "Review " + secret}], {},
                                    redactor=Redactor((secret,)), max_chars=27)
        self.assertIn("[REDACTED]", source)
        self.assertNotIn("title-secret", source)

    def test_normalizes_single_line_output(self):
        redact = Redactor(())
        for output, expected in (
            ("本会话的标题是：“XXX”", "XXX"),
            ("# **Title: “Review   API”**。", "Review API"),
            ("标题： 更新 C++ 工具链。", "更新 C++ 工具链"),
            ("", None), ("标题：", None), ("one\ntwo", None), ("one\rtwo", None),
            ("字" * 41, None), ("[Review](https://example.org)", None), ("```text```", None),
        ):
            with self.subTest(output=output):
                self.assertEqual(normalize_title(output, redact), expected)
