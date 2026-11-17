import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from daphnis.skills import (
    SkillChanged,
    SkillError,
    SkillScope,
    SkillState,
    SkillStore,
    _checked_file,
)


def package(parent, name, *, description="Review code", body="Skill instructions"):
    source = parent / name
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nallowed-tools: run_command\n"
        f"metadata:\n  author: test\n---\n{body}\n", encoding="utf-8",
    )
    return source


class SkillStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.other = self.root / "other"
        self.other.mkdir()
        self.home = self.root / "home" / ".daphnis"
        self.store = SkillStore(self.home)

    def test_same_name_in_both_scopes_file_copy_and_disclosure(self):
        source = package(self.workspace, "review", body="GLOBAL BODY")
        (source / "assets").mkdir()
        (source / "assets" / "icon.bin").write_bytes(b"\0\xff")
        (source / "scripts").mkdir()
        (source / "scripts" / "review.sh").write_text("touch ran-script\n")
        (source / "scripts" / "review.sh").chmod(0o700)
        global_skill = self.store.import_skill("review", SkillScope.GLOBAL, self.workspace)
        project_skill = self.store.import_skill(
            package(self.root / "packages", "review", body="PROJECT BODY"), SkillScope.PROJECT, self.workspace,
        )
        self.assertEqual((global_skill.id, project_skill.id), ("global:review", "project:review"))
        self.assertEqual(global_skill.state, SkillState.DISABLED)
        self.assertEqual(self.store.snapshot(self.workspace).instructions, "")
        with self.assertRaisesRegex(SkillError, "already exists"):
            self.store.import_skill(source, SkillScope.GLOBAL, self.workspace)
        self.store.set_state(global_skill.id, SkillState.ON_DEMAND, self.workspace)
        self.store.set_state(project_skill.id, SkillState.ALWAYS, self.workspace)

        snapshot = self.store.snapshot(self.workspace)
        self.assertIn('global:review (on_demand) — "Review code"', snapshot.instructions)
        self.assertIn("project:review (always)", snapshot.instructions)
        self.assertIn("PROJECT BODY", snapshot.instructions)
        self.assertNotIn("GLOBAL BODY", snapshot.instructions)
        self.assertNotIn("allowed-tools:", snapshot.instructions)
        _, directory, text = snapshot.read(global_skill.id)
        self.assertIn("allowed-tools: run_command", text)
        self.assertIn("GLOBAL BODY", text)
        self.assertEqual((Path(directory) / "assets" / "icon.bin").read_bytes(), b"\0\xff")
        self.assertTrue(os.access(Path(directory) / "scripts" / "review.sh", os.X_OK))
        with self.assertRaisesRegex(SkillError, "not readable text"):
            snapshot.read(global_skill.id, "assets/icon.bin")
        self.assertFalse((self.workspace / "ran-script").exists())
        self.assertFalse(self.store.snapshot(self.other).contains(project_skill.id))

    def test_restart_loads_external_catalog_and_preserves_missing_state(self):
        source = package(self.root, "notes")
        imported = self.store.import_skill(source, SkillScope.GLOBAL, self.workspace)
        self.store.set_state(imported.id, SkillState.ALWAYS, self.workspace)
        project_root = self.workspace / ".daphnis" / "skills"
        self.store.load_workspace(self.workspace)
        package(project_root, "testing")
        (project_root / "unfinished").mkdir()
        self.assertEqual([item.id for item in self.store.list(self.workspace)], [imported.id])

        (self.home / "skills" / "notes").rename(self.home / "saved")
        with self.assertLogs("daphnis.skills", level="WARNING") as logs:
            restored = SkillStore(self.home)
            entries = {item.id: item for item in restored.list(self.workspace)}
        self.assertIn("unfinished", logs.output[0])
        self.assertEqual(entries[imported.id].state, SkillState.ALWAYS)
        self.assertIn("missing", entries[imported.id].error)
        self.assertIsNone(entries["project:testing"].error)
        self.assertIsNotNone(entries["project:unfinished"].error)
        restored.remove("project:unfinished", self.workspace)
        self.assertFalse((project_root / "unfinished").exists())
        (self.home / "saved").rename(self.home / "skills" / "notes")
        self.assertIn("Skill instructions", SkillStore(self.home).snapshot(self.workspace).instructions)

    def test_import_rejects_invalid_manifests_links_and_traversal(self):
        source = package(self.root, "testing")
        manifest = source / "SKILL.md"
        valid = manifest.read_bytes()
        for data in (b"name: testing", b"---\nname: testing\n---\n", valid.replace(b"testing", b"other"),
                     valid.replace(b"description: Review code", b"description: [invalid"), valid + b"\xff"):
            with self.subTest(manifest=data):
                manifest.write_bytes(data)
                with self.assertRaises(SkillError):
                    self.store.import_skill(source, SkillScope.PROJECT, self.workspace)
        manifest.write_bytes(valid)
        external = self.root / "external.txt"
        external.write_text("Unrelated user file")
        link = source / "link.txt"
        for create in (lambda: link.symlink_to(external), lambda: os.link(external, link), lambda: os.mkfifo(link)):
            create()
            with self.assertRaisesRegex(SkillError, "regular files without links"):
                self.store.import_skill(source, SkillScope.PROJECT, self.workspace)
            link.unlink()
        alias = self.workspace / "testing"
        alias.symlink_to(source, target_is_directory=True)
        with self.assertRaises(OSError):
            self.store.import_skill(alias, SkillScope.PROJECT, self.workspace)
        with self.assertRaisesRegex(SkillError, "traversal"):
            self.store.import_skill("../testing", SkillScope.PROJECT, self.workspace)
        with (
            patch.object(self.store, "_write_states", side_effect=PermissionError("simulated failure")),
            self.assertRaises(PermissionError),
        ):
            self.store.import_skill(source, SkillScope.PROJECT, self.workspace)
        self.assertEqual(list((self.workspace / ".daphnis" / "skills").iterdir()), [])
        self.assertEqual(self.store.list(self.workspace), ())

    def test_read_checks_the_frozen_files_before_and_after_opening(self):
        source = package(self.root, "notes")
        (source / "references").mkdir()
        (source / "references" / "notes.md").write_text("Original version")
        (source / ".env").write_text("PRIVATE=fake")
        item = self.store.import_skill(source, SkillScope.GLOBAL, self.workspace)
        self.store.set_state(item.id, SkillState.ON_DEMAND, self.workspace)
        snapshot = self.store.snapshot(self.workspace)
        for path in ("../external", "references/../SKILL.md", "/etc/hosts", "references\\notes.md", ".env"):
            with self.subTest(path=path), self.assertRaises(SkillError):
                snapshot.read(item.id, path)
        relative = "references/notes.md"
        target = self.home / "skills" / "notes" / relative

        @contextmanager
        def replace_while_reading(*args):
            with _checked_file(*args) as file:
                yield file
                replacement = target.with_suffix(".new")
                replacement.write_text("New version")
                replacement.replace(target)

        with patch("daphnis.skills._checked_file", replace_while_reading), self.assertRaises(SkillChanged):
            snapshot.read(item.id, relative)
        with self.assertRaises(SkillChanged):
            snapshot.read(item.id, relative)
        self.assertEqual(SkillStore(self.home).snapshot(self.workspace).read(item.id, relative)[2], "New version")
        target.unlink()
        target.symlink_to(source / "references" / "notes.md")
        with self.assertRaises(SkillChanged):
            snapshot.read(item.id, relative)

    def test_concurrent_states_and_removal_preserve_other_workspace_records(self):
        global_skill = self.store.import_skill(package(self.root, "common"), SkillScope.GLOBAL, self.workspace)
        project_skill = self.store.import_skill(package(self.root, "local"), SkillScope.PROJECT, self.workspace)
        another_store = SkillStore(self.home)
        barrier = Barrier(2)

        def update(store, item, state):
            barrier.wait(timeout=3)
            return store.set_state(item.id, state, self.workspace)

        with ThreadPoolExecutor(max_workers=2) as threads:
            first = threads.submit(update, self.store, global_skill, SkillState.ALWAYS)
            second = threads.submit(update, another_store, project_skill, SkillState.ON_DEMAND)
            first.result(timeout=3)
            second.result(timeout=3)
        self.assertEqual({item.state for item in SkillStore(self.home).list(self.workspace)},
                         {SkillState.ALWAYS, SkillState.ON_DEMAND})
        with (
            patch.object(self.store, "_write_states", side_effect=PermissionError("simulated failure")),
            self.assertRaises(PermissionError),
        ):
            self.store.remove(project_skill.id, self.workspace)
        self.assertTrue(self.store.snapshot(self.workspace).contains(project_skill.id))
        self.store.remove(project_skill.id, self.workspace)
        self.assertEqual([item.id for item in SkillStore(self.home).list(self.workspace)], [global_skill.id])

    def test_home_workspace_keeps_project_and_global_packages_separate(self):
        source = package(self.root, "review")
        self.store.import_skill(source, SkillScope.GLOBAL, self.home.parent)
        self.store.import_skill(source, SkillScope.PROJECT, self.home.parent)
        self.assertEqual([item.id for item in SkillStore(self.home).list(self.home.parent)],
                         ["global:review", "project:review"])
        self.assertEqual([item.id for item in self.store.list(self.workspace)], ["global:review"])
