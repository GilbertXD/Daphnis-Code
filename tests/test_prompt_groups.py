import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
from saturnus import Messages
from test_agent_core import make_agent, make_client, reply, session

from daphnis import History, Session, TurnStatus
from daphnis.prompt_groups import PromptInput
from daphnis.turn_state import ABANDONED, TURN_STATE, RecoveryAction, RecoveryKind

MESSAGES = (("user", "review"), ("assistant", "sample exchange"), ("user", "begin"))


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


class PromptGroupTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_atomic_commit_retry_and_abandonment(self):
        for abandon in (False, True):
            with self.subTest(abandon=abandon), tempfile.TemporaryDirectory() as directory:
                history = History(Path(directory) / "sessions")
                task = session()
                template = PromptInput("stable", "source-server", "source-prompt", MESSAGES)
                requests = []

                def model(request, requests=requests, history=history, task=task, template=template):
                    messages = json.loads(request.content)["messages"]
                    requests.append(messages)
                    if len(requests) == 1:
                        saved = history.load(task.id)
                        self.assertEqual(saved.messages.snapshot(), template.snapshot())
                        self.assertEqual(saved.prompt_groups, (template.group(0),))
                        return httpx.Response(400, json={"error": {"message": "failure"}})
                    return httpx.Response(200, json=reply())

                async def persist(candidate, history=history):
                    history.save(candidate)

                async with make_client(model) as client:
                    agent = make_agent(client)
                    failed = await agent.run(session=task, prompt_input=template, persist=persist)
                    self.assertEqual(failed.status, TurnStatus.FAILED)
                    restored = history.load(task.id)
                    self.assertEqual(restored.recovery.message_index, 0)
                    self.assertEqual(restored.recovery.kind, RecoveryKind.FAILED_USER)
                    duplicate = await agent.run(session=restored, prompt_input=template, persist=persist)
                    self.assertEqual(duplicate.status, TurnStatus.FAILED)
                    self.assertEqual(len(requests), 1)
                    if abandon:
                        result = await agent.run(session=restored, prompt="new task", persist=persist)
                        self.assertEqual(requests[-1], [{"role": "user", "content": "new task"}])
                        self.assertTrue(all(item.get(TURN_STATE) == ABANDONED for item in
                                            restored.messages.snapshot()[:3]))
                    else:
                        result = await agent.run(
                            session=restored, recovery_action=RecoveryAction.RETRY,
                            recovery_key=restored.recovery.key, persist=persist,
                        )
                        self.assertEqual(requests[-1], requests[0])
                        self.assertEqual(len(restored.messages), 4)
                    self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
                    self.assertEqual(requests[0], template.snapshot())

    async def test_assistant_tail_is_recognized_but_requires_user_input(self):
        async with make_client(lambda _: self.fail("unexpected model request")) as client:
            agent = make_agent(client)
            task = session()
            prompt = PromptInput("tail", "example", "review", MESSAGES[:2])
            result = await agent.run(session=task, prompt_input=prompt)
            self.assertEqual(result.status, TurnStatus.FAILED)
            self.assertIn("补充一条 User", result.error)
            self.assertEqual(task.messages.snapshot(), [])
            task = replace(task, messages=Messages(prompt.snapshot()), prompt_groups=(prompt.group(0),))
            self.assertEqual(task.recovery.kind, RecoveryKind.FAILED_USER)
            self.assertEqual(task.recovery.message_index, 0)
