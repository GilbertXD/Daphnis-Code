import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_tools import bound, call

from daphnis import EventBus, Session
from daphnis.attachments import AttachmentStore
from daphnis.tools import ApprovalDecision, ApprovalMode, create_tools


class CodeSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.tools = bound(self.workspace)

    def write(self, path, text=""):
        target = self.workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    async def execute(self, name, **arguments):
        return (await self.tools.execute(call(name, **arguments))).value

    async def test_read_lines_beyond_the_first_megabyte_and_preserve_newlines(self):
        prefix = (b"a" * 1023 + b"\n") * 1300
        path = self.workspace / "large.txt"
        path.write_bytes(prefix + "一\r\nsecond\rfinal".encode())
        page = await self.execute("read_file", path=path.name, start_line=1301, end_line=1302, max_bytes=16)
        self.assertEqual((page["content"], page["start_line"], page["end_line"]), ("一\r\nsecond\r", 1301, 1302))
        self.assertFalse(page["truncated"])
        self.assertGreater(page["scanned_bytes"], 1024 ** 2)
        last = await self.execute("read_file", path=path.name, start_line=1303)
        self.assertEqual(last["content"], "final")
        self.assertEqual(last["end_line"], 1303)
        self.write("empty.txt")
        for filename, start in ((path.name, 9999), ("empty.txt", 1)):
            empty = await self.execute("read_file", path=filename, start_line=start)
            self.assertEqual((empty["content"], empty["end_line"], empty["truncated"]), ("", None, False))

    async def test_read_cursors_progress_across_unicode_and_long_lines(self):
        long_line = "a" * 65535 + "界🙂\r\n"
        self.write("page.txt", long_line + "尾")
        first = await self.execute("read_file", path="page.txt", start_column=65534, max_bytes=5)
        self.assertEqual(first["content"], "aa界")
        self.assertEqual((first["next_start_line"], first["next_start_column"]), (1, 65537))
        short = await self.execute("read_file", path="page.txt", start_column=65537, max_bytes=3)
        self.assertEqual(short["error"]["code"], "byte_limit_too_small")
        self.assertEqual(short["minimum_bytes"], 4)
        second = await self.execute("read_file", path="page.txt", start_column=65537, max_bytes=6)
        self.assertEqual(second["content"], "🙂\r\n")
        self.assertEqual((second["next_start_line"], second["next_start_column"]), (2, 1))
        third = await self.execute("read_file", path="page.txt", start_line=2, max_bytes=3)
        self.assertEqual(third["content"], "尾")
        self.assertFalse(third["truncated"])
        joined = await self.execute("read_file", path="page.txt", start_column=2, max_bytes=70000)
        self.assertEqual(joined["content"], long_line[1:] + "尾")

    async def test_glob_root_recursive_limits_and_invalid_paths(self):
        for filename in ("src/deep/test_calc.py", "src/core.py", "z.py", "a.py", "notes.md"):
            self.write(filename)
        for pattern, paths in (
            ("*.py", ["a.py", "z.py"]),
            ("**/*.py", ["a.py", "src/core.py", "src/deep/test_calc.py", "z.py"]),
            ("src/**/*.py", ["src/core.py", "src/deep/test_calc.py"]),
            ("**/test_*.py", ["src/deep/test_calc.py"]),
        ):
            with self.subTest(pattern=pattern):
                result = await self.execute("glob", pattern=pattern)
                self.assertEqual(result["matches"], paths)
                self.assertFalse(result["truncated"])
        limited = await self.execute("glob", pattern="**/*.py", max_results=2)
        self.assertEqual(limited["matches"], ["a.py", "src/core.py"])
        self.assertTrue(limited["truncated"])
        exact = await self.execute("glob", pattern="*.py", max_results=2)
        self.assertFalse(exact["truncated"])
        with patch("daphnis.tools.code_search.MAX_SCANNED_ENTRIES", 2):
            capped = await self.execute("glob", pattern="**/*")
        self.assertEqual(capped["scanned_entries"], 2)
        self.assertTrue(capped["truncated"])
        for pattern in ("/**/*.py", "src/../*.py", "[unclosed", "src/**foo.py"):
            self.assertEqual((await self.execute("glob", pattern=pattern))["error"]["code"], "invalid_glob")
        self.assertEqual((await self.execute("glob", pattern="*", path="missing"))["error"]["code"], "file_not_found")
        self.assertEqual((await self.execute("glob", pattern="*", path="a.py"))["error"]["code"], "not_directory")

    async def test_grep_reports_regex_line_context_and_resource_limits(self):
        self.write("src/a.py", "before\n  Value_42 = True\nafter\nvalue_99\n")
        self.write("z.txt", "Value_21\n")
        (self.workspace / "binary.py").write_bytes(b"Value_21\0")
        match = await self.execute("grep", pattern=r"\bvalue_[0-9]{2}\b", include="*.py",
                                   case_sensitive=False, context_lines=1, max_results=1)
        self.assertEqual(match["matches"], [{
            "path": "src/a.py", "line": 2, "column": 3, "text": "  Value_42 = True", "line_truncated": False,
            "before": [{"line": 1, "text": "before"}], "after": [{"line": 3, "text": "after"}],
        }])
        self.assertTrue(match["truncated"])
        empty = await self.execute("grep", pattern="missing", path="src")
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["matches"], [])
        self.assertEqual((await self.execute("grep", pattern="("))["error"]["code"], "invalid_regex")
        with patch("daphnis.tools.code_search.MAX_CONTENT_BYTES", 14):
            capped = await self.execute("grep", pattern=".*", path="src")
        self.assertEqual(capped["scanned_bytes"], 14)
        self.assertTrue(capped["truncated"])
        self.assertEqual([hit["text"] for hit in capped["matches"]], ["before"])
        self.write("src/b.py", "*" * 450 + "needle\n")
        with patch("daphnis.tools.code_search.MAX_SCANNED_FILES", 1):
            limited = await self.execute("grep", pattern="needle", path="src")
        self.assertTrue(limited["truncated"])
        self.assertEqual(limited["scanned_files"], 1)
        result = await self.execute("grep", pattern="needle", path="src")
        self.assertEqual(result["matches"][0]["column"], 451)
        self.assertTrue(result["matches"][0]["line_truncated"])
        self.assertIn("needle", result["matches"][0]["text"])
        self.write("src/long.txt", "a" * 100000 + "!")
        difficult = await asyncio.wait_for(self.execute("grep", pattern="^(a+)+$", include="long.txt"), 3)
        self.assertEqual(difficult["matches"], [])
        self.assertFalse(difficult["truncated"])

    async def test_private_storage_credentials_hard_links_and_symlinks_are_excluded(self):
        history = self.workspace / "data" / "sessions"
        attachments = AttachmentStore(history)
        for hidden in (
            ".env", ".daphnis/skills/off/SKILL.md", "nested/.DAPHNIS/skills/off/SKILL.md",
            "data/skills/off/SKILL.md", "data/skills.json", "data/skills.lock", "data/mcp.json", "data/sessions/saved.txt",
            "data/.skills-pending.json", "src/.daphnis-staged.tmp/updated",
            str((attachments.directory / "snapshot.txt").relative_to(self.workspace)),
        ):
            self.write(hidden, "PRIVATE_FIXTURE")
        self.write("visible.py", "PUBLIC_FIXTURE")
        (self.workspace / "alias").symlink_to(history, target_is_directory=True)
        (self.workspace / "link.txt").symlink_to(self.workspace / ".env")
        (self.workspace / "hard.txt").hardlink_to(self.workspace / ".env")
        self.tools = create_tools(search_excluded_directory=history, attachment_store=attachments,
                                  skill_state_paths=(history.parent / "skills.json", history.parent / "skills.lock")).select(
            ["glob", "grep"], mode=ApprovalMode.FULL,
        ).bind(session=Session(self.workspace, "compatible", "test"), events=EventBus(), turn_id="turn")
        self.assertEqual((await self.execute("glob", pattern="**/*"))["matches"], ["visible.py"])
        self.assertEqual((await self.execute("grep", pattern="PRIVATE_FIXTURE"))["matches"], [])
        for path in ("data/skills/off", "data/sessions", "alias", ".daphnis/skills/off"):
            self.assertEqual((await self.execute("grep", pattern=".", path=path))["error"]["code"], "sensitive_path")
        (self.workspace / "source-link").symlink_to(self.workspace, target_is_directory=True)
        self.assertEqual((await self.execute("glob", pattern="*", path="source-link"))["error"]["code"], "invalid_path")

    async def test_external_roots_obey_approval_and_directory_identity(self):
        outer = self.base / "other"
        outer.mkdir()
        (outer / "source.py").write_text("public")
        denied = await self.execute("glob", pattern="*.py", path="../other")
        self.assertEqual(denied["error"]["code"], "permission_denied")
        approval = AsyncMock(return_value=ApprovalDecision.ALLOW_ONCE)
        self.tools = bound(self.workspace, approval=approval)
        result = await self.execute("glob", pattern="*.py", path="../other")
        self.assertEqual(result["matches"], ["source.py"])
        self.assertEqual(approval.await_args.args[0].details["targets"], [str(outer)])

        async def move_root(_request):
            outer.rename(self.base / "old")
            outer.mkdir()
            (outer / "new.py").write_text("changed")
            return ApprovalDecision.ALLOW_ONCE

        self.tools = bound(self.workspace, approval=move_root)
        changed = await self.execute("grep", pattern=".", path="../other")
        self.assertEqual(changed["error"]["code"], "target_changed")
