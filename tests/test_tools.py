import asyncio
import difflib
import json
import os
import shutil
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from saturnus import ChatClient, ToolCallValidationError

from daphnis import Agent, EventBus, History, Operations, Session, TurnStatus
from daphnis.app import Application
from daphnis.models import ModelInfo
from daphnis.tools import ApprovalDecision, ApprovalMode, create_tools
from daphnis.tools.files import MAX_FILE_BYTES


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
    async def test_credentials_do_not_reach_model_history_or_child_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            secret = "synthetic-P9-credential"
            (workspace / ".env").write_text(f"MIMO_API_KEY={secret}\n")
            (workspace / ".env.local").write_text(secret)
            (workspace / ".env.example").write_text("MIMO_API_KEY=example\n")
            (workspace / "alias").symlink_to(".env")
            (workspace / "hardlink").hardlink_to(workspace / ".env")
            (workspace / "private.pem").write_text(secret)
            (workspace / "code.py").write_text("print('ordinary code')\n")
            script = workspace / "child.py"
            script.write_text("import os\nprint(os.getenv('MIMO_API_KEY', 'absent'))\n")
            calls = [
                call("read_file", path="./.env"),
                call("read_file", path="alias"),
                call("read_file", path="hardlink"),
                call("search", query="synthetic-P9", mode="content"),
                call("read_file", path=".env.example"),
                call("read_file", path="code.py"),
            ]
            returned = []

            def model(request):
                messages = json.loads(request.content)["messages"]
                if messages[-1]["role"] == "tool":
                    returned.extend(
                        json.loads(item["content"]) for item in messages
                        if item["role"] == "tool"
                    )
                    message, finish = {"role": "assistant", "content": "done"}, "stop"
                else:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {**item, "id": str(index)}
                            for index, item in enumerate(calls)
                        ],
                    }
                    finish = "tool_calls"
                return httpx.Response(200, json={
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}]
                })

            history = History(workspace / "history")
            app = Application(
                workspace=workspace, provider="mimo", model_id="mimo-v2.6-flash",
                connections={"mimo": {
                    "base_url": "https://model.test", "transport": httpx.MockTransport(model)
                }},
                history=history,
            )
            try:
                ops = Operations(app)
                self.assertEqual((await ops.send("inspect", stream=False)).status, TurnStatus.COMPLETED)
                saved = (history.directory / f"{ops.session.id}.json").read_text()
            finally:
                await app.close()
            with patch.dict(os.environ, {"MIMO_API_KEY": secret}):
                full = bound(workspace, approval=approve).select(
                    ["run_command"], mode=ApprovalMode.FULL
                )
                denied = await full.execute(call("run_command", argv=["cat", "alias"]))
                self.assertEqual(denied.value["error"]["code"], "sensitive_path")
                child = await full.execute(
                    call("run_command", argv=[sys.executable, str(script)])
                )
            self.assertNotIn(secret, json.dumps(returned))
            self.assertNotIn(secret, saved)
            self.assertEqual(
                [item["error"]["code"] for item in returned[:3]], ["sensitive_path"] * 3
            )
            self.assertEqual(returned[3]["matches"], [])
            self.assertIn("example", returned[4]["content"])
            self.assertIn("ordinary code", returned[5]["content"])
            self.assertEqual(child.value["stdout"].strip(), "absent")

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

    async def test_write_file_schema_approval_and_empty_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            approval = AsyncMock(return_value=ApprovalDecision.DENY)
            tools = bound(workspace, approval=approval).select(["write_file"], mode=ApprovalMode.FULL)
            definition = tools.definitions()[0]["function"]
            self.assertEqual(definition["name"], "write_file")
            self.assertEqual(definition["parameters"]["required"], ["path", "content"])
            for arguments in ({"path": "empty.txt"}, {"path": "empty.txt", "content": 1},
                              {"path": "empty.txt", "content": "", "overwrite": True}):
                with self.subTest(arguments=arguments), self.assertRaises(ToolCallValidationError):
                    await tools.execute(call("write_file", **arguments))
            approval.assert_not_awaited()

            request = call("write_file", path="empty.txt", content="")
            denied = await tools.execute(request)
            self.assertEqual(denied.value["error"]["code"], "permission_denied")
            self.assertFalse(approval.call_args.args[0].can_trust)
            self.assertEqual(list(workspace.iterdir()), [])

            approval.return_value = ApprovalDecision.ALLOW_ONCE
            created = await tools.execute(request)
            target = workspace / "empty.txt"
            self.assertEqual(created.value, {"ok": True, "path": str(target), "bytes_written": 0})
            self.assertEqual(target.read_bytes(), b"")

    async def test_write_file_rejects_unsafe_paths_and_oversized_utf8(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (workspace / "jump").symlink_to(outside, target_is_directory=True)
            (workspace / "dangling").symlink_to(outside / "new.txt")
            approval = AsyncMock(return_value=ApprovalDecision.ALLOW_ONCE)
            tools = bound(workspace, approval=approval)
            for path, code in (
                ("../outside/new.txt", "permission_denied"),
                (str(outside / "new.txt"), "permission_denied"),
                ("jump/new.txt", "invalid_path"),
                ("dangling", "file_exists"),
                ("missing/new.txt", "file_not_found"),
            ):
                with self.subTest(path=path):
                    result = await tools.execute(call("write_file", path=path, content="text"))
                    self.assertEqual(result.value["error"]["code"], code)
            result = await tools.execute(call(
                "write_file", path="large.txt", content="中" * (MAX_FILE_BYTES // 3 + 1),
            ))
            self.assertEqual(result.value["error"]["code"], "file_too_large")
            approval.assert_not_awaited()
            self.assertEqual(list(outside.iterdir()), [])
            self.assertEqual({path.name for path in workspace.iterdir()}, {"jump", "dangling"})

    async def test_write_file_atomic_creation_and_directory_races(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            parent = workspace / "sub"
            parent.mkdir(parents=True)
            tools = bound(workspace, approval=approve)
            request = call("write_file", path="sub/new.txt", content="complete file\n")
            target = parent / "new.txt"

            with patch("daphnis.tools.files.os.fsync", side_effect=OSError("disk failure")):
                result = await tools.execute(request)
            self.assertEqual(result.value["error"]["code"], "io_error")
            self.assertEqual(list(parent.iterdir()), [])

            real_sync = os.fsync
            moved = root / "moved"

            def move_after_staging(fd):
                real_sync(fd)
                parent.rename(moved)
                parent.symlink_to(moved, target_is_directory=True)

            with patch("daphnis.tools.files.os.fsync", side_effect=move_after_staging):
                result = await tools.execute(request)
            self.assertEqual(result.value["error"]["code"], "target_changed")
            self.assertEqual(list(moved.iterdir()), [])
            parent.unlink()
            parent.mkdir()

            real_link = os.link

            def create_before_publish(*args, **kwargs):
                target.write_bytes(b"concurrent file")
                return real_link(*args, **kwargs)

            with patch("daphnis.tools.files.os.link", side_effect=create_before_publish):
                result = await tools.execute(request)
            self.assertEqual(result.value["error"]["code"], "file_exists")
            self.assertEqual(target.read_bytes(), b"concurrent file")
            self.assertEqual(list(parent.iterdir()), [target])

    async def test_patch_creation_modification_deletion_and_selected_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            tools = bound(workspace)
            add = "--- /dev/null\n+++ b/sub/new.txt\n@@ -0,0 +1 @@\n+new\n"
            result = await tools.execute(call("apply_patch", patch=add))
            self.assertEqual(result.value["error"]["code"], "permission_denied")
            allowed = bound(workspace, approval=approve)
            result = await allowed.execute(call("apply_patch", patch=add))
            self.assertTrue(result.value["ok"], result.value)
            target = workspace / "sub/new.txt"
            self.assertEqual(target.read_text(), "new\n")
            tools._trusted.add((tools._binding.session_id, "apply_patch"))
            change = diff("sub/new.txt", "new\n", "changed\n")
            denied = await tools.execute(call("apply_patch", patch=change))
            self.assertEqual(denied.value["error"]["code"], "permission_denied")
            self.assertEqual(target.read_text(), "new\n")
            result = await allowed.execute(call("apply_patch", patch=change))
            self.assertTrue(result.value["ok"], result.value)
            deletion = "--- a/sub/new.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-changed\n"
            denied = await tools.execute(call("apply_patch", patch=deletion))
            self.assertEqual(denied.value["error"]["code"], "permission_denied")
            full = tools.select(["apply_patch"], mode=ApprovalMode.FULL)
            result = await full.execute(call("apply_patch", patch=deletion))
            self.assertEqual(result.value["error"]["code"], "permission_denied")
            full = allowed.select(["apply_patch"], mode=ApprovalMode.FULL)
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

    async def test_patch_reports_hunk_content_beyond_declared_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            tools = bound(workspace, approval=approve)
            target = workspace / "new.txt"
            patch = "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,60 @@\n" + "+line\n" * 73
            patch += "--- /dev/null\n+++ b/other.txt\n@@ -0,0 +1 @@\n+other\n"

            result = await tools.execute(call("apply_patch", patch=patch))
            error = result.value["error"]
            self.assertEqual(error["code"], "invalid_patch")
            self.assertIn("Hunk at patch line 3", error["message"])
            self.assertIn("declared line counts (old: 0, new: 60)", error["message"])
            self.assertIn("patch line 64", error["message"])
            self.assertEqual(list(workspace.iterdir()), [])

            corrected = patch.replace("+1,60 @@", "+1,73 @@", 1)
            result = await tools.execute(call("apply_patch", patch=corrected))
            self.assertTrue(result.value["ok"], result.value)
            self.assertEqual(target.read_text(), "line\n" * 73)
            self.assertEqual((workspace / "other.txt").read_text(), "other\n")

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
            real_rename = os.rename
            calls = []

            def fail_second(*args, **kwargs):
                calls.append(args)
                if len(calls) == 2:
                    raise OSError("injected disk failure")
                return real_rename(*args, **kwargs)

            with patch("daphnis.tools.files.os.rename", side_effect=fail_second):
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

    async def test_patch_detects_moved_parent_and_late_concurrent_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            parent = workspace / "sub"
            parent.mkdir(parents=True)
            moved = root / "moved"
            tools = bound(workspace, approval=approve)
            real_sync = os.fsync

            def move_after_staging(fd):
                real_sync(fd)
                parent.rename(moved)
                parent.mkdir()

            with patch("daphnis.tools.files.os.fsync", side_effect=move_after_staging):
                created = await tools.execute(call(
                    "apply_patch", patch="--- /dev/null\n+++ b/sub/new.txt\n@@ -0,0 +1 @@\n+ours\n"
                ))
            self.assertEqual(created.value.get("error", {}).get("code"), "target_changed")
            self.assertFalse((moved / "new.txt").exists())
            self.assertFalse((parent / "new.txt").exists())

            target = workspace / "data.txt"
            target.write_text("old\n")
            real_replace, real_rename = os.replace, os.rename

            def replace_after_edit(src, dst, **kwargs):
                target.write_text("concurrent\n")
                return real_replace(src, dst, **kwargs)

            def rename_after_edit(src, dst, **kwargs):
                target.write_text("concurrent\n")
                return real_rename(src, dst, **kwargs)

            with (
                patch("daphnis.tools.files.os.replace", side_effect=replace_after_edit),
                patch("daphnis.tools.files.os.rename", side_effect=rename_after_edit),
            ):
                changed = await tools.execute(call(
                    "apply_patch", patch=diff("data.txt", "old\n", "ours\n")
                ))
            self.assertEqual(changed.value.get("error", {}).get("code"), "target_changed")
            self.assertEqual(target.read_text(), "concurrent\n")

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
                "asyncio.base_events.BaseEventLoop.subprocess_exec", new_callable=AsyncMock
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
            # Hosted runners can put system binaries under runner-owned ancestors.
            # Test both policy outcomes without changing production trust rules.
            with patch("daphnis.tools.shell.trusted_read_program", return_value=False):
                result = await tools.execute(call("run_command", argv=["pwd"]))
                self.assertEqual(result.value["error"]["code"], "permission_denied")
            with patch("daphnis.tools.shell.trusted_read_program", return_value=True):
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

    async def test_shell_read_checks_use_the_path_the_command_will_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (outside / "data").write_text("outside")
            (workspace / "data").write_text("inside")
            (workspace / "~").symlink_to(outside, target_is_directory=True)
            (workspace / "-data").write_text("literal")
            tools = bound(workspace)
            with (
                patch.dict(os.environ, {"HOME": str(workspace)}),
                patch("daphnis.tools.shell.trusted_read_program", return_value=True),
            ):
                for mode in ApprovalMode:
                    selected = tools.select(["run_command"], mode=mode)
                    for argv in (["cat", "~/data"], ["cat", "data"], ["ls", "-a"]):
                        result = await selected.execute(call("run_command", argv=argv))
                        self.assertEqual(
                            result.value.get("error", {}).get("code"), "permission_denied"
                        )
                approved = bound(workspace, approval=approve)
                literal = await approved.execute(call("run_command", argv=["cat", "--", "-data"]))
                self.assertTrue(literal.value["ok"], literal.value)
                self.assertEqual(literal.value["stdout"], "literal")
                listing = await approved.execute(call("run_command", argv=["ls", "-a"]))
                self.assertTrue(listing.value["ok"], listing.value)
                self.assertIn("-data", listing.value["stdout"])
                full = tools.select(["run_command"], mode=ApprovalMode.FULL)
                for argv in (["cat", "../outside/data"], ["cat", "data", "-n"]):
                    result = await full.execute(call("run_command", argv=argv))
                    self.assertEqual(result.value["error"]["code"], "permission_denied")
                for argv in (["pwd"], ["uname", "-s"], ["echo", "正常文本"]):
                    result = await full.execute(call("run_command", argv=argv))
                    self.assertTrue(result.value["ok"], result.value)
                for argv in (["pwd", "-P"], ["uname", "--help"], ["echo", "-e", r"\n"]):
                    result = await full.execute(call("run_command", argv=argv))
                    self.assertEqual(result.value["error"]["code"], "permission_denied")
                for arg in ("$(cat /outside/secret)", "`cat /outside/secret`", "|", ">"):
                    result = await full.execute(call("run_command", argv=["echo", arg]))
                    self.assertEqual(result.value["error"]["code"], "command_blocked")

            async def replace_target(request):
                (workspace / "data").unlink()
                (workspace / "data").symlink_to(outside / "data")
                return ApprovalDecision.ALLOW_ONCE

            reader = bound(workspace, approval=replace_target).select(["read_file"], mode=ApprovalMode.ASK)
            result = await reader.execute(call("read_file", path="data"))
            self.assertEqual(result.value["error"]["code"], "target_changed")

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

    async def test_shell_timeout_reports_detached_descendants_as_unverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            pid_file = workspace / "detached.pid"
            child = workspace / "detached.py"
            child.write_text("import time\ntime.sleep(30)\n")
            parent = workspace / "parent.py"
            parent.write_text(
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, sys.argv[1]], start_new_session=True)\n"
                "pathlib.Path(sys.argv[2]).write_text(str(child.pid))\n"
                "time.sleep(30)\n"
            )
            try:
                async with asyncio.timeout(5):
                    result = await bound(workspace, approval=approve).execute(call(
                        "run_command", argv=[sys.executable, str(parent), str(child), str(pid_file)],
                        timeout=1.0,
                    ))
                pid = int(pid_file.read_text())
                os.kill(pid, 0)
                self.assertEqual(result.value["error"]["code"], "command_timeout")
                self.assertEqual(result.value.get("termination_scope"), "original_process_group")
                self.assertTrue(result.value.get("escaped_descendants_may_remain"))
            finally:
                if pid_file.exists():
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)

    async def test_agent_write_read_reject_overwrite_and_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            target = workspace / "result.py"
            content = "def greeting():\r\n    return '你好'\n"
            updated = content.replace("你好", "再见")
            requests, approvals = [], []
            events = EventBus()
            requested_events = []
            events.on("tool.approval_requested", requested_events.append)

            async def approve_change(request):
                approvals.append(request)
                self.assertFalse(request.can_trust)
                if len(approvals) == 1:
                    self.assertEqual(request.tool_name, "write_file")
                    self.assertFalse(target.exists())
                else:
                    self.assertEqual(request.tool_name, "apply_patch")
                    self.assertEqual(target.read_bytes(), content.encode())
                return ApprovalDecision.ALLOW_ONCE

            steps = [
                call("write_file", path=target.name, content=content),
                call("read_file", path=target.name),
                call("write_file", path=target.name, content="overwrite"),
                call("apply_patch", patch=diff(target.name, content, updated)),
            ]

            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                index = len(requests) - 1
                if index == 0:
                    self.assertEqual({tool["function"]["name"] for tool in body["tools"]},
                                     {"read_file", "write_file", "apply_patch"})
                else:
                    output = json.loads(body["messages"][-1]["content"])
                    if index == 1:
                        self.assertEqual(output, {"ok": True, "path": str(target),
                                                  "bytes_written": len(content.encode())})
                    elif index == 2:
                        self.assertEqual(output["content"], content)
                    elif index == 3:
                        self.assertEqual(output["error"]["code"], "file_exists")
                    else:
                        self.assertTrue(output["ok"])
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
                    ["read_file", "write_file", "apply_patch"]
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
                ).run(session=state, prompt="create, read, try to overwrite, then patch")
            self.assertEqual(result.status, TurnStatus.COMPLETED, result.error)
            self.assertEqual(result.tool_calls, 4)
            self.assertEqual(len(approvals), 2)
            self.assertEqual(len(requested_events), 2)
            self.assertEqual(target.read_bytes(), updated.encode())


if __name__ == "__main__":
    unittest.main()
