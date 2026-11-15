import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from saturnus import Messages

from daphnis import History, Session
from daphnis.prompt_groups import PromptInput


class PromptGroupHistoryTests(unittest.TestCase):
    def test_optional_group_round_trip_and_validation(self):
        prompt = PromptInput("group-01", "example", "review", (
            ("user", "Review this project."),
            ("assistant", "I will check concurrency."),
            ("user", "Start now."),
        ), edited=True)
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory) / "sessions")
            session = Session(directory, "mimo", "mimo-v2.6-flash")
            history.save(session)
            self.assertEqual(history.load(session.id).prompt_groups, ())
            session = replace(session, messages=Messages(prompt.snapshot()), prompt_groups=(prompt.group(0),))
            history.save(session)
            loaded = history.load(session.id)
            self.assertEqual(loaded.prompt_groups, session.prompt_groups)
            self.assertEqual(loaded.messages.snapshot(), prompt.snapshot())

            for changes in (
                {"prompt_groups": (prompt.group(1),)},
                {"prompt_groups": (prompt.group(0), prompt.group(0))},
                {"messages": Messages([{"role": "system", "content": "forged"}, *prompt.snapshot()[1:]])},
                {"messages": Messages([prompt.snapshot()[0], {**prompt.snapshot()[1], "tool_calls": []},
                                       prompt.snapshot()[2]])},
            ):
                with self.assertRaises(ValueError):
                    replace(loaded, **changes)
