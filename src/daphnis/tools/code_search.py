"""Bounded glob and regular expression search over approved workspace paths."""

import asyncio
import json
import os
import stat
from fnmatch import fnmatchcase
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any

import re2

from ..text import LINE_BREAKS, decode_text
from .approval import Operation, Target, ToolError, path_decision, signature
from .files import (
    MAX_FILE_BYTES,
    _lock_directory,
    _lock_path,
    is_credential_path,
    read_bytes,
)
from .search import IGNORED_DIRECTORIES, MAX_CONTENT_BYTES, MAX_SCANNED_ENTRIES

MAX_PATTERN_LENGTH = 512
MAX_CONTEXT_LINES = 3
MAX_SCANNED_FILES = 1000
MAX_RESULT_BYTES = 64 * 1024
_SEARCH_SECONDS = 5
_EXCERPT_LENGTH = 320


def _glob_parts(pattern: str) -> tuple[str, ...]:
    parts = tuple(pattern.split("/"))
    if (
        "\\" in pattern or any(ord(char) < 32 for char in pattern)
        or any(part in {"", ".", ".."} or ("**" in part and part != "**") for part in parts)
    ):
        raise ToolError("invalid_glob", "Use a relative glob without traversal, with ** as its own segment.")
    for part in parts:
        cursor = 0
        while (start := part.find("[", cursor)) >= 0:
            start += 1
            if part[start:start + 1] == "!":
                start += 1
            if part[start:start + 1] == "]":
                start += 1
            end = part.find("]", start)
            if end < 0:
                raise ToolError("invalid_glob", "A glob character class is missing its closing bracket.")
            cursor = end + 1
    return parts


def _matches_glob(path: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    positions = {0}
    for part in parts:
        if part == "**":
            positions = set(range(min(positions), len(path) + 1))
        else:
            positions = {
                index + 1 for index in positions if index < len(path) and fnmatchcase(path[index], part)
            }
        if not positions:
            return False
    return len(path) in positions


def _protected(path: Path, private: tuple[Path, ...]) -> bool:
    key = _lock_path(path)
    return (
        ".daphnis" in key.parts or ".git" in key.parts
        or any(part.startswith(".daphnis-") and part.endswith(".tmp") for part in key.parts)
        or (key.name.startswith(".skills-") and key.suffix == ".json")
        or is_credential_path(path) or any(key.is_relative_to(directory) for directory in private)
    )


def _scan_files(root: Target, private: tuple[Path, ...], stopped) -> tuple[list[Target], int, int, bool]:
    stack = [root]
    files = []
    entries = skipped = 0
    truncated = False
    while stack and entries < MAX_SCANNED_ENTRIES and not stopped():
        target = stack.pop()
        children = []
        try:
            with target.open(directory=True) as fd, os.scandir(fd) as iterator:
                for entry in iterator:
                    if entries >= MAX_SCANNED_ENTRIES or stopped():
                        truncated = True
                        break
                    entries += 1
                    path = target.path / entry.name
                    try:
                        entry.name.encode("utf-8")
                        if _protected(path, private) or entry.is_symlink():
                            skipped += 1
                            continue
                        info = entry.stat(follow_symlinks=False)
                    except (UnicodeError, OSError):
                        skipped += 1
                        continue
                    candidate = Target(path, path, signature(info))
                    if stat.S_ISDIR(info.st_mode):
                        if entry.name.casefold() not in IGNORED_DIRECTORIES:
                            children.append(candidate)
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        files.append(candidate)
                    else:
                        skipped += 1
        except (OSError, ToolError):
            if target is root:
                raise
            skipped += 1
        stack.extend(sorted(children, key=lambda child: child.path.name, reverse=True))
        if truncated:
            break
    files.sort(key=lambda file: file.path.as_posix())
    return files, entries, skipped, truncated or bool(stack)


def _excerpt(line: str, column: int = 0) -> str:
    start = max(0, column - 80)
    return line[start:start + _EXCERPT_LENGTH]


def _matching_lines(text: str, relative: str, expression, context: int, shortened: bool, stopped):
    lines = text.splitlines()
    if shortened and text and text[-1] not in LINE_BREAKS:
        lines.pop()  # A partial line cannot reliably match anchors or supply context.
    for index, line in enumerate(lines):
        if stopped():
            break
        match = expression.search(line)
        if match is None:
            continue
        position = match.start()
        excerpt = _excerpt(line, position)
        item = {
            "path": relative, "line": index + 1, "column": position + 1,
            "text": excerpt, "line_truncated": len(excerpt) < len(line),
        }
        if context:
            start = max(index - context, 0)
            end = min(index + context + 1, len(lines))
            item["before"] = [{"line": i + 1, "text": _excerpt(lines[i])} for i in range(start, index)]
            item["after"] = [{"line": i + 1, "text": _excerpt(lines[i])} for i in range(index + 1, end)]
        yield item


def prepare_code_search(
    name: str, arguments: dict[str, Any], workspace: Path, *, private: tuple[Path, ...],
) -> Operation:
    root = Target.capture(arguments.get("path", "."), workspace)
    private = tuple(_lock_path(path) for path in (*private, _lock_directory()))
    if _protected(root.path, private) or _protected(root.requested, private):
        raise ToolError("sensitive_path", "This path is excluded from code search.")
    if root.path != Path(os.path.abspath(root.requested)):
        raise ToolError("invalid_path", "Code search roots cannot traverse symbolic links.")

    expression = None
    if name == "grep":
        options = re2.Options()
        options.case_sensitive = arguments.get("case_sensitive", True)
        options.log_errors = False
        options.max_mem = 8 * 1024 * 1024
        try:
            expression = re2.compile(arguments["pattern"], options=options)
        except (re2.error, UnicodeError):
            raise ToolError("invalid_regex", "Invalid or unsupported RE2 expression. Check syntax and grouping.") from None
        pattern = _glob_parts(arguments["include"]) if "include" in arguments else None
    else:
        pattern = _glob_parts(arguments["pattern"])

    def search(cancelled: Event) -> dict[str, Any]:
        if root.identity is None:
            raise ToolError("file_not_found", "The search path does not exist.")
        single_file = expression is not None and stat.S_ISREG(root.identity[2])
        if not single_file and not stat.S_ISDIR(root.identity[2]):
            raise ToolError("not_directory", "The search path must be a directory.")
        deadline = monotonic() + _SEARCH_SECONDS

        def stopped():
            return cancelled.is_set() or monotonic() >= deadline

        files, entries, skipped, truncated = (
            ([root], 1, 0, False) if single_file else _scan_files(root, private, stopped)
        )
        matches = []
        scanned_files = scanned_bytes = result_bytes = 0
        limit = arguments.get("max_results", 100)
        context = arguments.get("context_lines", 0)

        def add(item):
            nonlocal result_bytes
            size = len(json.dumps(item, ensure_ascii=False).encode("utf-8")) + 2
            if len(matches) >= limit or result_bytes + size > MAX_RESULT_BYTES:
                return False
            matches.append(item)
            result_bytes += size
            return True

        for file in files:
            if stopped():
                truncated = True
                break
            relative = Path(file.path.name) if single_file else file.path.relative_to(root.path)
            parts = relative.parts[-1:] if expression is not None and pattern and len(pattern) == 1 else relative.parts
            if single_file and pattern and len(pattern) > 1 and file.path.is_relative_to(workspace):
                parts = file.path.relative_to(workspace).parts
            if pattern and not _matches_glob(parts, pattern):
                continue
            if expression is None:
                if not add(relative.as_posix()):
                    truncated = True
                    break
                continue
            budget = min(MAX_FILE_BYTES, MAX_CONTENT_BYTES - scanned_bytes)
            if budget <= 0 or scanned_files >= MAX_SCANNED_FILES:
                truncated = True
                break
            scanned_files += 1
            try:
                data, shortened = read_bytes(file, budget)
                scanned_bytes += len(data)
                truncated |= shortened
                if b"\0" in data:
                    skipped += 1
                    continue
                text = decode_text(data, shortened)
            except UnicodeError:
                skipped += 1
                continue
            except (OSError, ToolError):
                if single_file:
                    raise
                skipped += 1
                continue
            if not all(add(item) for item in _matching_lines(text, relative.as_posix(), expression, context, shortened, stopped)):
                truncated = True
                break
        return {
            "ok": True, "path": str(root.path), "matches": matches,
            "truncated": truncated or stopped(), "scanned_entries": entries,
            "scanned_files": scanned_files, "scanned_bytes": scanned_bytes, "skipped": skipped,
        }

    async def run():
        cancelled = Event()
        try:
            return await asyncio.to_thread(search, cancelled)
        finally:
            cancelled.set()

    return Operation(
        path_decision(workspace, (root,)), "Search source files without following symbolic links.",
        (root,), {**arguments, "path": str(root.path)}, run,
    )
