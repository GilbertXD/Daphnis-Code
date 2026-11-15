"""Bounded literal filename/content search without following symlinks."""

import asyncio
import os
from pathlib import Path
from typing import Any

from ..text import decode_text, literal_matches
from .approval import ApprovalMode, Operation, Target, ToolError, path_decision
from .files import MAX_FILE_BYTES, read_bytes

IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)
MAX_SCANNED_ENTRIES = 10000
MAX_CONTENT_BYTES = 8 * MAX_FILE_BYTES


def prepare_search(
    arguments: dict[str, Any], workspace: Path, *, excluded: Path | None = None,
    attachment_directory: Path | None = None,
    approval_mode: ApprovalMode = ApprovalMode.AUTO,
) -> Operation:
    root = Target.capture(arguments.get("path", "."), workspace)
    private = tuple(path for path in (excluded, attachment_directory) if path is not None)
    if any(root.path.is_relative_to(path) for path in private):
        raise ToolError("sensitive_path", "Search cannot read Daphnis Session storage.")

    async def run():
        query = arguments["query"]
        case_sensitive = arguments.get("case_sensitive", True)
        needle = query if case_sensitive else query.casefold()
        mode = arguments.get("mode", "both")
        limit = arguments.get("max_results", 100)
        matches = []
        stack = [root.path]
        scanned = skipped = total_bytes = 0
        truncated = False
        while stack and len(matches) < limit and scanned < MAX_SCANNED_ENTRIES:
            directory = stack.pop()
            target = Target.capture(directory, workspace)
            if not target.path.is_relative_to(root.path):
                skipped += 1
                continue
            try:
                with target.open(directory=True) as fd, os.scandir(fd) as iterator:
                    for entry in iterator:
                        scanned += 1
                        if scanned > MAX_SCANNED_ENTRIES:
                            truncated = True
                            break
                        path = directory / entry.name
                        if entry.is_symlink():
                            skipped += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if path not in private and entry.name not in IGNORED_DIRECTORIES:
                                stack.append(path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        label = str(path.relative_to(root.path))
                        name = entry.name if case_sensitive else entry.name.casefold()
                        if mode in ("filename", "both") and needle in name:
                            matches.append({"path": label, "kind": "filename"})
                        if len(matches) >= limit:
                            truncated = True
                            break
                        if mode in ("content", "both"):
                            budget = min(
                                MAX_FILE_BYTES, MAX_CONTENT_BYTES - total_bytes
                            )
                            if budget <= 0:
                                truncated = True
                            else:
                                try:
                                    file = Target.capture(path, workspace)
                                    if not file.path.is_relative_to(root.path):
                                        skipped += 1
                                        continue
                                    data, shortened = read_bytes(file, budget, mode=approval_mode)
                                    total_bytes += len(data)
                                    truncated |= shortened
                                    text = decode_text(data, shortened)
                                    if "\0" not in text:
                                        for number, position, line in literal_matches(
                                            text.splitlines(), needle, case_sensitive=case_sensitive,
                                        ):
                                            matches.append({
                                                "path": label, "kind": "content", "line": number,
                                                "text": line[max(0, position - 80):position + 220],
                                            })
                                            if len(matches) >= limit:
                                                truncated = True
                                                break
                                except (OSError, UnicodeError, ToolError):
                                    skipped += 1
                        await asyncio.sleep(0)
                        if len(matches) >= limit:
                            break
            except (OSError, ToolError):
                if directory == root.path:
                    raise
                skipped += 1
            await asyncio.sleep(0)
        return {
            "ok": True,
            "path": str(root.path),
            "matches": matches,
            "truncated": truncated or bool(stack),
            "scanned_entries": scanned,
            "scanned_bytes": total_bytes,
            "skipped": skipped,
        }

    return Operation(
        path_decision(workspace, (root,)),
        "Search literal names and UTF-8 content without following symlinks.",
        (root,),
        {"path": str(root.path), **arguments},
        run,
    )
