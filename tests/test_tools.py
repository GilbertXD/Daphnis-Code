import asyncio
import difflib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from saturnus import ChatClient, ToolCallValidationError

from daphnis import Agent, EventBus, Session, TurnStatus
from daphnis.models import ModelInfo
from daphnis.tools import ApprovalDecision, ApprovalMode, create_tools


def call(name, **arguments):
    return {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def diff(path, old, new):
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def bound(workspace, *, approval=None, names=None, events=None):
    state = Session(workspace=workspace, provider="compatible", model_id="test")
    tools = create_tools(approval=approval)
    if names is not None:
        tools = tools.select(names)
    return tools.bind(session=state, events=events or EventBus(), turn_id="turn-test")


async def approve(request):
    return ApprovalDecision.ALLOW_ONCE


class ToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_reads_listing_search_and_external_path_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            workspace = base / "workspace"
            workspace.mkdir()
            (workspace / "hello.txt").write_text(
                "first\n你好 needle\nlast\n", encoding="utf-8"
            )
            (workspace / ".git").mkdir()
            (workspace / ".git" / "hidden.txt").write_text("needle", encoding="utf-8")
            external = base / "outside.txt"
            external.write_text("external", encoding="utf-8")
            (workspace / "link.txt").symlink_to(external)
            tools = bound(workspace)
            result = await tools.execute(
                call("read_file", path="hello.txt", start_line=2, end_line=2)
            )
            self.assertEqual(result.value["content"], "你好 needle\n")
            short = await tools.execute(
                call("read_file", path="hello.txt", max_bytes=8)
            )
            self.assertTrue(short.value["truncated"])
            listing = await tools.execute(call("list_directory", max_entries=1))
            self.assertTrue(listing.value["truncated"])
            found = await tools.execute(call("search", query="needle", mode="content"))
            self.assertEqual(
                [item["path"] for item in found.value["matches"]], ["hello.txt"]
            )
            for path in (str(external), "../outside.txt", "link.txt"):
                denied = await tools.execute(call("read_file", path=path))
                self.assertEqual(denied.value["error"]["code"], "permission_denied")
            requests = []

            async def inspect(request):
                requests.append(request)
                return ApprovalDecision.ALLOW_ONCE

            allowed = bound(workspace, approval=inspect)
            result = await allowed.execute(call("read_file", path="link.txt"))
            self.assertEqual(result.value["content"], "external")
            self.assertIn(str(external), requests[0].details["targets"])
            read = call("read_file", path="hello.txt")
            ask = tools.select(["read_file"], mode=ApprovalMode.ASK)
            self.assertEqual(
                (await ask.execute(read)).value["error"]["code"], "permission_denied"
            )
            full = allowed.select(["read_file"], mode=ApprovalMode.FULL)
            self.assertTrue((await full.execute(read)).value["ok"])
            self.assertEqual(len(requests), 1)
            result = await full.execute(call("read_file", path="link.txt"))
            self.assertTrue(result.value["ok"])
            self.assertEqual(len(requests), 2)
            self.assertFalse(requests[-1].can_trust)
            allowed._trusted.add((allowed._binding.session_id, "read_file"))
            result = await allowed.execute(call("read_file", path="link.txt"))
            self.assertTrue(result.value["ok"])
            self.assertEqual(len(requests), 3)
            self.assertFalse(requests[-1].can_trust)
            # Resolve symlink/.. in filesystem order, not by lexical abspath first.
            (base / "external-dir").mkdir()
            (workspace / "jump").symlink_to(
                base / "external-dir", target_is_directory=True
            )
            denied = await tools.execute(call("read_file", path="jump/../outside.txt"))
            self.assertEqual(denied.value["error"]["code"], "permission_denied")
            (workspace / "loop-a").symlink_to("loop-b")
            (workspace / "loop-b").symlink_to("loop-a")
            invalid = await tools.execute(call("read_file", path="loop-a"))
            self.assertEqual(invalid.value["error"]["code"], "invalid_path")

    async def test_patch_creation_modification_deletion_and_selected_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            tools = bound(workspace)
            add = "--- /dev/null\n+++ b/sub/new.txt\n@@ -0,0 +1 @@\n+new\n"
            result = await tools.execute(call("apply_patch", patch=add))
            self.assertTrue(result.value["ok"], result.value)
            target = workspace / "sub/new.txt"
            self.assertEqual(target.read_text(), "new\n")
            tools._trusted.add((tools._binding.session_id, "apply_patch"))
            change = diff("sub/new.txt", "new\n", "changed\n")
            denied = await tools.execute(call("apply_patch", patch=change))
            self.assertEqual(denied.value["error"]["code"], "permission_denied")
            self.assertEqual(target.read_text(), "new\n")
            allowed = bound(workspace, approval=approve)
            result = await allowed.execute(call("apply_patch", patch=change))
            self.assertTrue(result.value["ok"], result.value)
            deletion = "--- a/sub/new.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-changed\n"
            denied = await tools.execute(call("apply_patch", patch=deletion))
            self.assertEqual(denied.value["error"]["code"], "permission_denied")
            full = tools.select(["apply_patch"], mode=ApprovalMode.FULL)
            result = await full.execute(call("apply_patch", patch=deletion))
            self.assertTrue(result.value["ok"], result.value)
            self.assertFalse(target.exists())
            selected = allowed.select(["read_file"])
            self.assertEqual(
                [item["function"]["name"] for item in selected.definitions()],
                ["read_file"],
            )
            with self.assertRaises(ToolCallValidationError):
                await selected.execute(call("apply_patch", patch=add))
            bypass = await allowed._registry.execute(call("apply_patch", patch=add))
            self.assertEqual(bypass.value["error"]["code"], "permission_denied")
            self.assertFalse(target.exists())

    async def test_patch_preflight_races_and_partial_failure_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            a, b = workspace / "a.txt", workspace / "b.txt"
            a.write_text("a\n")
            b.write_text("b\n")
            valid = diff("a.txt", "a\n", "A\n")
            invalid = diff("b.txt", "wrong\n", "B\n")
            tools = bound(workspace, approval=approve)
            result = await tools.execute(call("apply_patch", patch=valid + invalid))
            self.assertEqual(result.value["error"]["code"], "patch_mismatch")
            self.assertEqual(a.read_text(), "a\n")

            async def race(request):
                a.write_text("someone else's change\n")
                return ApprovalDecision.ALLOW_ONCE

            result = await bound(workspace, approval=race).execute(
                call("apply_patch", patch=valid)
            )
            self.assertEqual(result.value["error"]["code"], "target_changed")
            self.assertEqual(a.read_text(), "someone else's change\n")
            a.write_text("a\n")
            real_replace = os.replace
            calls = []

            def fail_second(*args, **kwargs):
                calls.append(args)
                if len(calls) == 2:
                    raise OSError("injected disk failure")
                return real_replace(*args, **kwargs)

            with patch("daphnis.tools.files.os.replace", side_effect=fail_second):
                result = await tools.execute(
                    call("apply_patch", patch=valid + diff("b.txt", "b\n", "B\n"))
                )
            self.assertFalse(result.value["ok"])
            self.assertTrue(result.value["partial"])
            self.assertEqual(result.value["applied_files"], [str(a)])
            self.assertEqual(a.read_text(), "A\n")
            self.assertEqual(b.read_text(), "b\n")
            self.assertEqual(list(workspace.glob(".daphnis-*.tmp")), [])
            real_link = os.link

            def concurrent_creation(*args, **kwargs):
                (workspace / "new.txt").write_text("created by someone else\n")
                return real_link(*args, **kwargs)

            with patch("daphnis.tools.files.os.link", side_effect=concurrent_creation):
                result = await tools.execute(
                    call(
                        "apply_patch",
                        patch="--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+ours\n",
                    )
                )
            self.assertEqual(result.value["error"]["code"], "target_changed")
            self.assertEqual(
                (workspace / "new.txt").read_text(), "created by someone else\n"
            )

    async def test_dangerous_commands_never_reach_approval_or_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            alias = workspace / "innocent-name"
            alias.symlink_to(shutil.which("rm"))
            callback = AsyncMock(return_value=ApprovalDecision.ALLOW_ONCE)
            tools = bound(workspace, approval=callback).select(
                ["run_command"], mode=ApprovalMode.FULL
            )
            vectors = [
                ["rm", "-rf", "."],
                ["rm", "-fr", "."],
                ["rm", "--recursive", "--force", "."],
                ["rm", "-r", "/"],
                ["rm", "-r", str(Path("/etc").resolve())],
                [str(alias), "-rf", "."],
                ["git", "-C", str(workspace), "reset", "--hard"],
                ["git", "reset", "--ha"],
                ["git", "clean", "-fdx"],
                ["git", "push", "--force-with-lease"],
                ["git", "push", "origin", "+main"],
                ["sudo", "ls"],
                ["sh", "-c", "echo forbidden"],
                ["env", "sh", "-c", "echo forbidden"],
                ["find", ".", "-delete"],
                ["rsync", "--delete", "a", "b"],
                ["mkfs.ext4", "/dev/example"],
                ["diskutil", "eraseDisk", "unused"],
                ["dd", "of=/dev/example"],
                [sys.executable, "-X", "dev", "-c", "print('blocked')"],
                ["node", "--eval=console.log('blocked')"],
            ]
            with patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock
            ) as spawn:
                for argv in vectors:
                    with self.subTest(argv=argv):
                        result = await tools.execute(call("run_command", argv=argv))
                        self.assertEqual(
                            result.value["error"]["code"], "command_blocked"
                        )
                spawn.assert_not_awaited()
            callback.assert_not_awaited()

    async def test_shell_readonly_nonzero_and_output_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            tools = bound(workspace)
            result = await tools.execute(call("run_command", argv=["pwd"]))
            self.assertTrue(result.value["ok"], result.value)
            self.assertEqual(Path(result.value["stdout"].strip()).resolve(), workspace)
            script = workspace / "output.py"
            script.write_text(
                "import sys\nprint('x' * 2000)\n"
                "print('failure', file=sys.stderr)\nsys.exit(7)\n"
            )
            denied = await tools.execute(
                call("run_command", argv=[sys.executable, str(script)])
            )
            self.assertEqual(denied.value["error"]["code"], "permission_denied")
            result = await bound(workspace, approval=approve).execute(
                call(
                    "run_command",
                    argv=[sys.executable, str(script)],
                    max_output_bytes=100,
                )
            )
            self.assertEqual(result.value["exit_code"], 7)
            self.assertEqual(result.value["error"]["code"], "command_failed")
            self.assertTrue(result.value["stdout_truncated"])
            self.assertLessEqual(len(result.value["stdout"].encode()), 100)
            self.assertIn("failure", result.value["stderr"])

    @unittest.skipUnless(os.name == "posix", "Process-group test requires POSIX")
    async def test_shell_timeout_and_cancel_terminate_child_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            child = workspace / "child.py"
            child.write_text(
                "import pathlib, signal, sys, time\n"
                "ready, stopped = map(pathlib.Path, sys.argv[1:])\n"
                "def stop(*args):\n    stopped.write_text('terminated')\n"
                "    sys.exit(0)\n"
                "signal.signal(signal.SIGTERM, stop)\nready.write_text('ready')\n"
                "while True: time.sleep(1)\n"
            )
            parent = workspace / "parent.py"
            parent.write_text(
                "import subprocess, sys, time\n"
                "subprocess.Popen([sys.executable, *sys.argv[1:]])\n"
                "while True: time.sleep(1)\n"
            )
            tools = bound(workspace, approval=approve)
            for cancel in (False, True):
                ready = workspace / f"ready-{cancel}"
                stopped = workspace / f"stopped-{cancel}"
                operation = call(
                    "run_command",
                    argv=[
                        sys.executable,
                        str(parent),
                        str(child),
                        str(ready),
                        str(stopped),
                    ],
                    timeout=0.6 if not cancel else 10,
                )
                task = asyncio.create_task(tools.execute(operation))
                async with asyncio.timeout(4):
                    while not ready.exists():
                        await asyncio.sleep(0.01)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    result = await task
                    self.assertTrue(result.value["timed_out"])
                    self.assertTrue(result.value["terminated"])
                self.assertEqual(stopped.read_text(), "terminated")

    async def test_agent_read_create_approve_modify_and_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            (workspace / "input.txt").write_text("source\n")
            requests, approvals = [], []
            events = EventBus()
            requested_events = []
            events.on("tool.approval_requested", requested_events.append)

            async def approve_change(request):
                approvals.append(request)
                self.assertEqual(request.tool_name, "apply_patch")
                self.assertFalse(request.can_trust)
                self.assertEqual((workspace / "result.txt").read_text(), "created\n")
                return ApprovalDecision.ALLOW_ONCE

            steps = [
                call("read_file", path="input.txt"),
                call(
                    "apply_patch",
                    patch="--- /dev/null\n+++ b/result.txt\n@@ -0,0 +1 @@\n+created\n",
                ),
                call(
                    "apply_patch", patch=diff("result.txt", "created\n", "modified\n")
                ),
            ]

            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                index = len(requests) - 1
                if index:
                    self.assertTrue(json.loads(body["messages"][-1]["content"])["ok"])
                if index < len(steps):
                    step = {**steps[index], "id": f"call-{index}"}
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [step],
                    }
                    finish = "tool_calls"
                else:
                    message, finish = {"role": "assistant", "content": "done"}, "stop"
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": finish}
                        ]
                    },
                )

            async with ChatClient(
                base_url="https://model.test",
                transport=httpx.MockTransport(handler),
            ) as client:
                state = Session(
                    workspace=workspace, provider="compatible", model_id="test"
                )
                tools = create_tools(approval=approve_change).select(
                    ["read_file", "apply_patch"]
                )
                result = await Agent(
                    client=client,
                    provider="compatible",
                    catalog=(
                        ModelInfo(
                            id="test", provider="compatible", supports_tools=True
                        ),
                    ),
                    tools=tools,
                    events=events,
                ).run(session=state, prompt="read, create, modify")
            self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
            self.assertEqual(result.tool_calls, 3)
            self.assertEqual(len(approvals), 1)
            self.assertEqual(len(requested_events), 1)
            self.assertEqual((workspace / "result.txt").read_text(), "modified\n")
            self.assertEqual(len(state.messages), 8)


if __name__ == "__main__":
    unittest.main()
