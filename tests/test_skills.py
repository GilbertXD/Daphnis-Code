import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from test_application import response
from test_compaction import SUMMARY
from test_skill_store import package
from test_tools import call

from daphnis import History, Operations, Session, TurnStatus
from daphnis.agent import TITLE_PROMPT
from daphnis.app import Application
from daphnis.events import EventBus
from daphnis.models import ModelInfo
from daphnis.skills import SkillScope, SkillState, SkillStore
from daphnis.token_budget import TokenCounter
from daphnis.tools import ApprovalMode, create_tools


class SkillsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.history = History(self.root / "data" / "sessions")

    def application(self, handler, *, model=None):
        model = model or ModelInfo("mock", "deepseek", 100_000, supports_tools=True, thinking_modes=("off",))
        app = Application(
            workspace=self.workspace, provider=model.provider, model_id=model.id, catalog=(model,),
            history=self.history, interactive=True, connections={model.provider: {
                "base_url": "https://model.test/v1", "transport": httpx.MockTransport(handler),
            }},
        )
        self.addAsyncCleanup(app.close)
        return Operations(app)

    async def test_primary_requests_freeze_skills_and_read_tool_uses_approval_and_history(self):
        entered = asyncio.Event()
        proceed = asyncio.Event()
        approvals = asyncio.Queue()
        primary = []
        titles = []

        async def handler(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if messages[0]["content"] == TITLE_PROMPT:
                titles.append(body)
                return response({"role": "assistant", "content": "Skills test"})
            primary.append(body)
            if len(primary) == 1:
                entered.set()
                await proceed.wait()
                skill_call = call("read_skill", skill_id="global:review", path="references/guide.md")
                return response({"role": "assistant", "tool_calls": [skill_call]}, "tool_calls")
            return response({"role": "assistant", "content": "Done"})

        ops = self.application(handler)
        sid = ops.initial_session_id
        await ops.set_tools([], session_id=sid)
        ops.set_approval_mode(ApprovalMode.ASK, session_id=sid)
        ops.subscribe("tool.approval_requested", approvals.put_nowait)
        started, finished = [], []
        ops.subscribe("tool.executing", started.append)
        ops.subscribe("tool.finished", finished.append)
        source = package(self.root, "review", body="ON_DEMAND_BODY")
        (source / "references").mkdir()
        (source / "references" / "guide.md").write_text("GUIDE_FROM_TOOL")
        ondemand = await ops.import_skill(source, "global", session_id=sid)
        always = await ops.import_skill(package(self.root, "testing", body="ALWAYS_BODY"), "project", session_id=sid)
        await ops.set_skill_state(ondemand.id, "on_demand", session_id=sid)
        await ops.set_skill_state(always.id, "always", session_id=sid)

        turn = asyncio.create_task(ops.send("Review the work", session_id=sid, stream=False))
        try:
            async with asyncio.timeout(5):
                await entered.wait()
                await ops.set_skill_state(ondemand.id, "disabled", session_id=sid)
                await ops.set_skill_state(always.id, "disabled", session_id=sid)
                proceed.set()
                approval = await approvals.get()
                self.assertEqual(approval.data["name"], "read_skill")
                self.assertFalse(started)
                self.assertTrue(ops.resolve_approval(approval.data["request_id"], "allow_once", session_id=sid))
                self.assertEqual((await turn).status, TurnStatus.COMPLETED)
        finally:
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)

        self.assertEqual([entry["function"]["name"] for entry in primary[0]["tools"]], ["read_skill"])
        self.assertEqual(len(started), 1)
        self.assertEqual(len(finished), 1)
        system = primary[0]["messages"][0]
        self.assertEqual(system, primary[1]["messages"][0])
        self.assertIn("ALWAYS_BODY", system["content"])
        self.assertIn("global:review (on_demand)", system["content"])
        self.assertNotIn("ON_DEMAND_BODY", system["content"])
        result = json.loads(primary[1]["messages"][-1]["content"])
        self.assertEqual(result["lines"][0]["text"], "GUIDE_FROM_TOOL")
        saved = self.history.load(sid).messages.snapshot()
        self.assertNotIn("ALWAYS_BODY", json.dumps(saved))
        self.assertIn("GUIDE_FROM_TOOL", json.dumps(saved))
        self.assertEqual(len(titles), 1)
        self.assertNotIn("ALWAYS_BODY", json.dumps(titles))
        self.assertNotIn("GUIDE_FROM_TOOL", json.dumps(titles))

        self.assertEqual((await ops.send("Next task", session_id=sid, stream=False)).status, TurnStatus.COMPLETED)
        self.assertNotIn("tools", primary[-1])
        for text in ("ALWAYS_BODY", "GUIDE_FROM_TOOL", "ON_DEMAND_BODY"):
            self.assertNotIn(text, json.dumps(primary[-1]))

    async def test_workspaces_and_non_tool_model_get_correct_catalogues(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return response({"role": "assistant", "content": "OK"})

        ops = self.application(handler, model=ModelInfo("plain", "deepseek", supports_tools=False))
        sid = ops.initial_session_id
        other = self.root / "other"
        other.mkdir()
        project = await ops.create_project("Other workspace", other, session_id=sid)
        second = await ops.open_project(project.id, session_id=sid)
        await ops.import_skill(package(self.root, "common", body="PRIVATE_BODY"), "global", session_id=sid)
        local = await ops.import_skill(package(self.root, "local", body="WORKSPACE_BODY"), "project", session_id=second.id)
        await ops.set_skill_state("global:common", "on_demand", session_id=sid)
        await ops.set_skill_state(local.id, "always", session_id=second.id)
        self.assertIsNone(ops.session(sid).project_id)
        self.assertEqual([item.id for item in ops.list_skills(sid)], ["global:common"])
        self.assertEqual([item.id for item in ops.list_skills(second.id)], ["global:common", "project:local"])
        self.assertEqual((await ops.send("First", session_id=sid, stream=False)).status, TurnStatus.COMPLETED)
        self.assertEqual((await ops.send("Second", session_id=second.id, stream=False)).status, TurnStatus.COMPLETED)
        await ops.move_session(second.id, None)
        self.assertIsNone(ops.session(second.id).project_id)
        self.assertIn(local.id, [item.id for item in ops.list_skills(second.id)])
        await ops.move_session(sid, project.id)
        self.assertIn(local.id, [item.id for item in ops.list_skills(sid)])
        primary = [body for body in requests if body["messages"][0]["content"] != TITLE_PROMPT]
        self.assertNotIn("WORKSPACE_BODY", json.dumps(primary[0]))
        self.assertIn("WORKSPACE_BODY", json.dumps(primary[1]))
        for body in primary:
            self.assertIn("global:common (on_demand)", body["messages"][0]["content"])
            self.assertNotIn("PRIVATE_BODY", json.dumps(body))
            self.assertNotIn("tools", body)

    async def test_fixed_skill_over_budget_fails_before_request_or_history_commit(self):
        def handler(_request):
            self.fail("An over-budget Skill must not reach the model.")

        model = ModelInfo("small", "deepseek", 6000, supports_tools=False,
                          token_counter=TokenCounter(len), default_output_tokens=100)
        ops = self.application(handler, model=model)
        sid = ops.initial_session_id
        skill = await ops.import_skill(package(self.root, "large", body="long " * 4000), "global", session_id=sid)
        await ops.set_skill_state(skill.id, "always", session_id=sid)
        result = await ops.send("Small request", session_id=sid, stream=False)
        self.assertEqual(result.status, TurnStatus.FAILED)
        self.assertIn("Skill 目录、固定正文", result.error)
        self.assertEqual(ops.message_snapshot(sid), [])

    async def test_summary_request_does_not_load_skills(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if "Summarize the conversation" in body["messages"][0]["content"]:
                return response({"role": "assistant", "content": SUMMARY})
            return response({"role": "assistant", "content": "Done"})

        ops = self.application(handler)
        sid = ops.initial_session_id
        await ops.set_tools([], session_id=sid)
        skill = await ops.import_skill(package(self.root, "summary", body="FIXED_SKILL_BODY"), "global", session_id=sid)
        await ops.set_skill_state(skill.id, "always", session_id=sid)
        await ops.send("Original task " * 150, session_id=sid, stream=False)
        await ops.send("Recent task", session_id=sid, stream=False)
        self.assertIsNotNone(await ops.compact(sid))
        summaries = [body for body in requests if "Summarize the conversation" in body["messages"][0]["content"]]
        self.assertTrue(summaries)
        self.assertNotIn("FIXED_SKILL_BODY", json.dumps(summaries))
        self.assertNotIn("tools", summaries[0])

    async def test_read_skill_pagination_scope_changes_and_state_file_protection(self):
        store = SkillStore(self.workspace / ".daphnis")
        source = package(self.root, "paged")
        resource = "界" * 3500 + "\nend"
        (source / "guide.txt").write_text(resource, encoding="utf-8")
        item = store.import_skill(source, SkillScope.PROJECT, self.workspace)
        store.set_state(item.id, SkillState.ON_DEMAND, self.workspace)
        store.import_skill(package(self.root, "disabled"), SkillScope.GLOBAL, self.workspace)
        session = Session(self.workspace, "deepseek", "mock")
        tools = create_tools(skill_state_paths=(store.state_path, store.lock_path)).select(
            ["read_skill", "replace_text"], mode=ApprovalMode.FULL,
        ).bind(session=session, events=EventBus(), turn_id="turn", skills=store.snapshot(self.workspace))
        page = (await tools.execute(call("read_skill", skill_id=item.id, path="guide.txt"))).value
        self.assertLessEqual(len(json.dumps(page, ensure_ascii=False).encode()), 8192)
        self.assertEqual(page["next_line"], 1)
        rest = (await tools.execute(call(
            "read_skill", skill_id=item.id, path="guide.txt", start_line=page["next_line"],
            start_column=page["next_column"],
        ))).value
        self.assertIsNone(rest["next_line"])
        self.assertEqual(page["lines"][0]["text"] + "\n".join(row["text"] for row in rest["lines"]), resource)
        disabled = await tools.execute(call("read_skill", skill_id="global:disabled"))
        self.assertEqual(disabled.value["error"]["code"], "skill_unavailable")
        protected = await tools.execute(call("replace_text", path=".daphnis/skills.json", old_text="on_demand", new_text="always"))
        self.assertEqual(protected.value["error"]["code"], "sensitive_path")
        directory = Path(page["directory"])
        (directory / "guide.txt").write_text("Changed version")
        changed = await tools.execute(call("read_skill", skill_id=item.id, path="guide.txt"))
        self.assertEqual(changed.value["error"]["code"], "skill_changed")
