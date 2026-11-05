"""Bounded text reads, directory listing, and exact unified-diff patches."""

import codecs
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .approval import Operation, Target, ToolError, failure, path_decision, signature

MAX_FILE_BYTES = 1_048_576
MAX_PATCH_BYTES = 1_048_576


def read_bytes(target: Target, limit: int) -> tuple[bytes, bool]:
    with target.open() as fd:
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(fd, min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data[:limit]), len(data) > limit


def decode_text(data: bytes, truncated: bool) -> str:
    return codecs.getincrementaldecoder("utf-8")().decode(data, final=not truncated)


def prepare_read(arguments: dict[str, Any], workspace: Path) -> Operation:
    target = Target.capture(arguments["path"], workspace)

    async def run():
        data, truncated = read_bytes(target, arguments.get("max_bytes", 65536))
        lines = decode_text(data, truncated).splitlines(keepends=True)
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line")
        if end is not None and end < start:
            raise ToolError("invalid_range", "end_line must be at least start_line.")
        selected = lines[start - 1 : end]
        return {
            "ok": True,
            "path": str(target.path),
            "content": "".join(selected),
            "start_line": start,
            "end_line": start + len(selected) - 1 if selected else None,
            "truncated": truncated,
            "scanned_bytes": len(data),
        }

    return Operation(
        path_decision(workspace, (target,)),
        "Read a UTF-8 text file.",
        (target,),
        {"path": str(target.path), **arguments},
        run,
    )


def prepare_list(arguments: dict[str, Any], workspace: Path) -> Operation:
    target = Target.capture(arguments.get("path", "."), workspace)

    async def run():
        limit = arguments.get("max_entries", 200)
        entries = []
        with target.open(directory=True) as fd, os.scandir(fd) as iterator:
            for entry in iterator:
                if len(entries) >= limit:
                    return {
                        "ok": True,
                        "path": str(target.path),
                        "entries": sorted(entries, key=lambda item: item["name"]),
                        "truncated": True,
                    }
                kind = (
                    "symlink"
                    if entry.is_symlink()
                    else "directory"
                    if entry.is_dir(follow_symlinks=False)
                    else "file"
                )
                entries.append({"name": entry.name, "type": kind})
        return {
            "ok": True,
            "path": str(target.path),
            "entries": sorted(entries, key=lambda item: item["name"]),
            "truncated": False,
        }

    return Operation(
        path_decision(workspace, (target,)),
        "List directory entries.",
        (target,),
        {"path": str(target.path)},
        run,
    )


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FilePatch:
    path: str
    action: str
    hunks: tuple[Hunk, ...]


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")


def _label(line: str, prefix: str) -> str | None:
    value = line[4:].rstrip("\r\n").split("\t", 1)[0]
    if value == "/dev/null":
        return None
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if not value or value.startswith('"'):
        raise ToolError("invalid_patch", "Patch headers require unquoted paths.")
    return value


def parse_patch(patch: str) -> list[FilePatch]:
    """Plain unified diff only: ---/+++ headers, exact numbered hunks, no renames."""
    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        raise ToolError("patch_too_large", "Patch exceeds the 1 MiB limit.")
    lines = patch.splitlines(keepends=True)
    files = []
    index = 0
    while index < len(lines):
        if (
            not lines[index].startswith("--- ")
            or index + 1 >= len(lines)
            or not lines[index + 1].startswith("+++ ")
        ):
            raise ToolError(
                "invalid_patch", "Expected --- and +++ unified-diff headers."
            )
        old_path = _label(lines[index], "a/")
        new_path = _label(lines[index + 1], "b/")
        if old_path is None and new_path is None:
            raise ToolError("invalid_patch", "Patch has no target path.")
        if old_path is not None and new_path is not None and old_path != new_path:
            raise ToolError("invalid_patch", "Renames are not supported.")
        action = (
            "create" if old_path is None else "delete" if new_path is None else "modify"
        )
        index += 2
        hunks = []
        while index < len(lines) and lines[index].startswith("@@ "):
            match = _HUNK.fullmatch(lines[index].rstrip("\r\n"))
            if match is None:
                raise ToolError("invalid_patch", "Invalid hunk header.")
            old_start, old_count, new_start, new_count = match.groups()
            old_count, new_count = int(old_count or 1), int(new_count or 1)
            index += 1
            content: list[tuple[str, str]] = []
            consumed_old = consumed_new = 0
            while consumed_old < old_count or consumed_new < new_count:
                if (
                    index >= len(lines)
                    or not lines[index]
                    or lines[index][0] not in " +-"
                ):
                    raise ToolError("invalid_patch", "Incomplete hunk body.")
                kind, value = lines[index][0], lines[index][1:]
                consumed_old += kind != "+"
                consumed_new += kind != "-"
                content.append((kind, value))
                index += 1
                if (
                    index < len(lines)
                    and lines[index].rstrip("\r\n") == "\\ No newline at end of file"
                ):
                    content[-1] = (kind, value.removesuffix("\n").removesuffix("\r"))
                    index += 1
                if consumed_old > old_count or consumed_new > new_count:
                    raise ToolError("invalid_patch", "Hunk line counts do not match.")
            hunks.append(
                Hunk(
                    int(old_start), old_count, int(new_start), new_count, tuple(content)
                )
            )
        if not hunks and action == "modify":
            raise ToolError("invalid_patch", "A modification requires a hunk.")
        files.append(
            FilePatch(
                new_path if new_path is not None else old_path, action, tuple(hunks)
            )
        )
        if len(files) > 100:
            raise ToolError("patch_too_large", "A patch may touch at most 100 files.")
    if not files:
        raise ToolError("invalid_patch", "Patch is empty.")
    return files


def _apply_text(original: str, patch: FilePatch) -> str:
    lines = original.splitlines(keepends=True)
    result = []
    cursor = 0
    for hunk in patch.hunks:
        position = hunk.old_start - 1 if hunk.old_count else hunk.old_start
        new_position = hunk.new_start - 1 if hunk.new_count else hunk.new_start
        if position < cursor or position > len(lines):
            raise ToolError(
                "patch_mismatch", "Hunk position is outside the original file."
            )
        result.extend(lines[cursor:position])
        cursor = position
        if new_position != len(result):
            raise ToolError(
                "patch_mismatch", "New hunk position does not match previous changes."
            )
        for kind, text in hunk.lines:
            if kind in (" ", "-"):
                if cursor >= len(lines) or lines[cursor] != text:
                    raise ToolError(
                        "patch_mismatch", "Hunk original text does not match exactly."
                    )
                cursor += 1
            if kind in (" ", "+"):
                result.append(text)
    result.extend(lines[cursor:])
    updated = "".join(result)
    if patch.action == "delete" and updated:
        raise ToolError("patch_mismatch", "A deletion must remove all file content.")
    if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
        raise ToolError("file_too_large", "Patched file exceeds the 1 MiB limit.")
    return updated


def _write(
    target: Target, action: str, content: str, created: list[str], applied: list[str]
) -> None:
    target.verify()
    with target.parent(create=action == "create", created=created) as (fd, name):
        if action == "delete":
            if (
                signature(os.stat(name, dir_fd=fd, follow_symlinks=False))
                != target.identity
            ):
                raise ToolError("target_changed", "File changed before deletion.")
            os.unlink(name, dir_fd=fd)
            applied.append(str(target.path))
            return
        temporary = f".daphnis-{uuid4().hex}.tmp"
        output = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o666,
            dir_fd=fd,
        )
        try:
            with os.fdopen(output, "wb") as stream:
                stream.write(content.encode("utf-8"))
                stream.flush()
                if target.identity is not None:
                    os.fchmod(stream.fileno(), stat.S_IMODE(target.identity[2]))
                os.fsync(stream.fileno())
            if action == "create":
                # Atomic installation never overwrites a concurrently created file.
                os.link(
                    temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False
                )
            else:
                if (
                    signature(os.stat(name, dir_fd=fd, follow_symlinks=False))
                    != target.identity
                ):
                    raise ToolError(
                        "target_changed", "File changed before replacement."
                    )
                os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
            applied.append(str(target.path))
        finally:
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass


def prepare_patch(arguments: dict[str, Any], workspace: Path) -> Operation:
    patches = parse_patch(arguments["patch"])
    targets = tuple(Target.capture(patch.path, workspace) for patch in patches)
    if len({target.path for target in targets}) != len(targets):
        raise ToolError("invalid_patch", "Each normalized target may appear only once.")
    for target, patch in zip(targets, patches, strict=True):
        if patch.action == "create" and target.identity is not None:
            raise ToolError(
                "file_exists", f"Creation will not overwrite: {target.path}"
            )
        if patch.action != "create" and target.identity is None:
            raise ToolError("file_not_found", f"Target does not exist: {target.path}")

    async def run():
        # Read/validate every hunk only after authorization and before any writes.
        updates = []
        for target, patch in zip(targets, patches, strict=True):
            if patch.action == "create":
                original = ""
            else:
                data, truncated = read_bytes(target, MAX_FILE_BYTES)
                if truncated:
                    raise ToolError("file_too_large", "Patch target exceeds 1 MiB.")
                original = decode_text(data, False)
            updates.append(_apply_text(original, patch))
        applied: list[str] = []
        created: list[str] = []
        try:
            for target, patch, updated in zip(targets, patches, updates, strict=True):
                _write(target, patch.action, updated, created, applied)
        except (OSError, ToolError) as exc:
            return failure(
                exc.code
                if isinstance(exc, ToolError)
                else "target_changed"
                if isinstance(exc, FileExistsError)
                else "patch_failed",
                str(exc),
                applied_files=applied,
                created_directories=created,
                partial=bool(applied or created),
            )
        return {
            "ok": True,
            "applied_files": applied,
            "created_directories": created,
            "partial": False,
        }

    return Operation(
        path_decision(
            workspace,
            targets,
            writes_existing=any(p.action != "create" for p in patches),
        ),
        "Apply an exact unified diff; existing content changes require approval.",
        targets,
        {
            "patch": arguments["patch"],
            "changes": [
                {"path": str(target.path), "action": patch.action}
                for target, patch in zip(targets, patches, strict=True)
            ],
        },
        run,
    )
