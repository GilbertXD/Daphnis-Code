import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx
from test_agent_core import make_agent, make_client, reply, session

from daphnis import History, Session, TurnStatus
from daphnis.messages import Messages
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

    async def test_failed_persistence_leaves_no_partial_input(self):
        async def persist(_):
            raise OSError("save failed")

        async with make_client(lambda _: self.fail("unexpected model request")) as client:
            task = session()
            prompt = PromptInput("failed-write", "example", "review", MESSAGES)
            result = await make_agent(client).run(session=task, prompt_input=prompt, persist=persist)
            self.assertEqual(result.status, TurnStatus.FAILED)
            self.assertEqual(task.messages.snapshot(), [])
            self.assertEqual(task.prompt_groups, ())

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

    async def test_compaction_keeps_atomic_boundaries_and_reloads_failed_group(self):
        from daphnis import Context
        from daphnis.compaction import CompactionCheckpoint, Compactor, safe_boundaries

        completed = PromptInput("previous", "example", "previous", MESSAGES)
        failed = PromptInput("failed", "example", "failed", (
            ("assistant", "another example"), ("user", "next task"),
        ))
        old_reply = {"role": "assistant", "content": "actual reply"}
        messages = [*completed.snapshot(), old_reply, *failed.snapshot()]
        groups = (completed.group(0), failed.group(4))
        summary = '{"goals":"review","progress":"","files":"","next_steps":""}'
        compactor = Compactor(8192)
        self.assertEqual(list(safe_boundaries(messages, groups)), [3, 4, 6])
        self.assertEqual(compactor.cut(messages, None, prompt_groups=groups), 4)
        with self.assertRaises(ValueError):
            CompactionCheckpoint(2, summary).validate(messages, groups)

        async def summarize(request, _):
            self.assertIn("Never infer file changes", request[0]["content"])
            self.assertIn('"preloaded_prompt_message":true', request[1]["content"])
            self.assertIn("actual reply", request[1]["content"])
            self.assertNotIn("example", request[1]["content"])
            return summary

        checkpoint = await compactor.summarize(messages, None, 4, summarize, prompt_groups=groups)
        # A cut just before a model reply retains the entire original input, including its assistant.
        tail = Context().build(Messages(messages), CompactionCheckpoint(3, summary), prompt_groups=groups)
        self.assertEqual(tail[2:], messages)
        with tempfile.TemporaryDirectory() as directory:
            history = History(Path(directory) / "sessions")
            task = replace(session(), messages=Messages(messages), compaction=checkpoint, prompt_groups=groups)
            history.save(task)
            restored = history.load(task.id)
            self.assertEqual(restored.recovery.message_index, 4)
            self.assertEqual(restored.messages.snapshot(), messages)
            requests = []

            def model(request):
                requests.append(json.loads(request.content)["messages"])
                return httpx.Response(200, json=reply())

            async with make_client(model) as client:
                result = await make_agent(client).run(
                    session=restored, recovery_action=RecoveryAction.RETRY, recovery_key=restored.recovery.key,
                )
            self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
            self.assertEqual(requests[0][2:], failed.snapshot())
            self.assertNotIn("sample exchange", str(requests[0]))
