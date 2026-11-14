import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from daphnis.config import environment_context_options, load_context_options


class ContextConfigTests(unittest.TestCase):
    def test_model_preferences_inherit_and_deployment_limits_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "daphnis.toml"
            path.write_text(
                """
                [compaction]
                threshold_value = 0.75
                recent_tokens = 1024

                [models."mimo:mimo-v2.6-flash"]
                output_limit = 8192

                [models."mimo:mimo-v2.6-flash".compaction]
                threshold_mode = "tokens"
                threshold_value = 131072

                [models."compatible:proxy/model:version"]
                context_window = 64000

                [models."compatible:proxy/model:version".compaction]
                auto_enabled = false
                """,
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DAPHNIS_CONFIG": str(path)}, clear=True):
                options = environment_context_options()

        settings = options["compaction_settings"]
        self.assertEqual((settings.threshold_mode, settings.threshold_value), ("percent", 0.75))
        overrides = options["model_compaction_settings"]
        mimo = ("mimo", "mimo-v2.6-flash")
        proxy = ("compatible", "proxy/model:version")
        self.assertEqual(overrides[mimo].threshold_mode, "tokens")
        self.assertEqual(overrides[mimo].threshold_value, 131072)
        self.assertEqual(overrides[mimo].recent_tokens, 1024)
        self.assertEqual(overrides[proxy].threshold_value, 0.75)
        self.assertFalse(overrides[proxy].auto_enabled)
        self.assertEqual(options["output_limits"], {mimo: 8192})
        catalog = {(model.provider, model.id): model for model in options["catalog"]}
        self.assertEqual(catalog[proxy].context_window, 64000)
        self.assertEqual(catalog[mimo].max_output_tokens, 131072)

    def test_rejects_invalid_configuration(self):
        cases = (
            ("[compaction]\nthreshold_mode = 'tokens'", "threshold_value"),
            ("[compaction]\nthreshold_value = 75", "percent compaction threshold"),
            ("[models.'compatible:proxy']\ncontext_window = true", "context_window"),
            ("[models.'mimo:mimo-v2.6-flash']\noutput_limit = 131073", "output_limit"),
            ("[models.'mimo:typo']", "模型目录"),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "daphnis.toml"
            for content, reason in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, reason):
                        load_context_options(path)
