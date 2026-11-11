"""Bounded text reads, directory listing, and exact unified-diff patches."""

import codecs
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .approval import (
    Decision,
    Operation,
    Target,
    ToolError,
    failure,
    path_decision,
    signature,
    verify_directory,
)

MAX_FILE_BYTES = 1_048_576
MAX_PATCH_BYTES = 1_048_576
_CREDENTIAL_NAMES = frozenset({
    ".netrc", "_netrc", ".git-credentials", ".npmrc", ".pypirc", ".dockercfg", ".envrc",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "credentials.json", "application_default_credentials.json",
})
_CREDENTIAL_PATHS = frozenset({
    (".aws", "credentials"), (".aws", "config"), (".docker", "config.json"),
    (".kube", "config"), ("gh", "hosts.yml"),
})


def is_credential_path(path: Path) -> bool:
    name = path.name.casefold()
    location = tuple(part.casefold() for part in path.parts[-2:])
    return (
        ((name == ".env" or name.startswith(".env.")) and name != ".env.example")
        or name in _CREDENTIAL_NAMES
        or location in _CREDENTIAL_PATHS
        or path.suffix.casefold() in {".pem", ".key", ".p12", ".pfx", ".ppk"}
        or (name.startswith("id_") and not name.endswith(".pub")
            and path.parent.name.casefold() == ".ssh")
    )


def require_readable_path(target: Target) -> None:
    """Protect content, including reads through a resolved symlink."""
    if any(is_credential_path(path) for path in (target.path, target.requested)):
        raise ToolError("sensitive_path", "Reading protected credentials is prohibited.")


def read_bytes(target: Target, limit: int) -> tuple[bytes, bool]:
    require_readable_path(target)
    with target.open() as fd:
        if os.fstat(fd).st_nlink > 1:
            raise ToolError("sensitive_path", "Files with multiple hard links cannot be read.")
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
    require_readable_path(target)

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
    value = value.removeprefix(prefix)
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
            hunk_line = index + 1
            index += 1
            content: list[tuple[str, str]] = []
            consumed_old = consumed_new = 0
            while index < len(lines) and lines[index][0] in " +-":
                if (
                    consumed_old == old_count
                    and consumed_new == new_count
                    and lines[index].startswith("--- ")
                    and index + 1 < len(lines)
                    and lines[index + 1].startswith("+++ ")
                ):
                    break
                kind, value = lines[index][0], lines[index][1:]
                consumed_old += kind != "+"
                consumed_new += kind != "-"
                if consumed_old > old_count or consumed_new > new_count:
                    raise ToolError(
                        "invalid_patch",
                        f"Hunk at patch line {hunk_line} exceeds its declared line counts "
                        f"(old: {old_count}, new: {new_count}) at patch line {index + 1}. "
                        "Correct the counts in the @@ header or adjust the hunk body.",
                    )
                content.append((kind, value))
                index += 1
                if (
                    index < len(lines)
                    and lines[index].rstrip("\r\n") == "\\ No newline at end of file"
                ):
                    content[-1] = (kind, value.removesuffix("\n").removesuffix("\r"))
                    index += 1
            if consumed_old < old_count or consumed_new < new_count:
                raise ToolError("invalid_patch", "Incomplete hunk body.")
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


def _verify_displaced(stage: int, target: Target, original: bytes) -> None:
    try:
        fd = os.open("original", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=stage)
    except OSError as exc:
        raise ToolError("target_changed", "Patch target changed during commit.") from exc
    with os.fdopen(fd, "rb") as stream:
        before = signature(os.fstat(fd))
        expected = target.identity
        # Renaming changes ctime. Compare the other metadata and the actual bytes.
        if (
            expected is None
            or before[:5] != expected[:5]
            or stream.read(len(original) + 1) != original
            or signature(os.fstat(fd)) != before
        ):
            raise ToolError("target_changed", "Patch target changed during commit.")


def _write(
    target: Target, action: str, original: bytes | None, content: str, *,
    workspace: Target, parent: Target, created: list[str], applied: list[str],
) -> None:
    verify_directory(workspace.path, workspace.identity)
    target.verify()
    with target.parent(create=action == "create", created=created) as (fd, name):
        def verify_parent() -> None:
            verify_directory(workspace.path, workspace.identity)
            opened = signature(os.fstat(fd))
            verify_directory(parent.path, opened)
            if parent.identity is not None and parent.identity[:3] != opened[:3]:
                raise ToolError("target_changed", f"Patch parent was replaced: {parent.path}")

        verify_parent()
        staging = f".daphnis-{uuid4().hex}.tmp"
        os.mkdir(staging, mode=0o700, dir_fd=fd)
        stage = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        displaced = installed = keep_original = False
        try:
            if action != "delete":
                verify_parent()
                output = os.open("updated", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o666, dir_fd=stage)
                with os.fdopen(output, "wb") as stream:
                    stream.write(content.encode("utf-8"))
                    stream.flush()
                    if target.identity is not None:
                        os.fchmod(stream.fileno(), stat.S_IMODE(target.identity[2]))
                    os.fsync(stream.fileno())
            verify_parent()
            target.verify()
            if original is not None:
                # Hold the displaced inode until validation. A late writer's
                # version can be restored or recovered even if the name races.
                os.rename(name, "original", src_dir_fd=fd, dst_dir_fd=stage)
                displaced = True
                _verify_displaced(stage, target, original)
                verify_parent()
            if action != "delete":
                os.link(
                    "updated", name, src_dir_fd=stage, dst_dir_fd=fd, follow_symlinks=False
                )
                installed = True
                applied.append(str(target.path))
            if original is not None:
                _verify_displaced(stage, target, original)
            verify_parent()
            if action == "delete":
                applied.append(str(target.path))
        except (OSError, ToolError) as exc:
            if displaced and not installed:
                try:
                    verify_parent()
                    os.link("original", name, src_dir_fd=stage, dst_dir_fd=fd,
                            follow_symlinks=False)
                except (OSError, ToolError):
                    keep_original = True
                    applied.append(str(target.path))
            elif displaced:
                keep_original = True
            if keep_original:
                raise ToolError(
                    "target_changed",
                    "Patch conflict; displaced file retained in the original directory.",
                    recovery_file=str(parent.path / staging / "original"),
                ) from exc
            raise
        finally:
            try:
                for entry in ("updated", "original"):
                    if entry != "original" or not keep_original:
                        try:
                            os.unlink(entry, dir_fd=stage)
                        except FileNotFoundError:
                            continue
            finally:
                os.close(stage)
            if not keep_original:
                os.rmdir(staging, dir_fd=fd)


def prepare_patch(arguments: dict[str, Any], workspace: Path) -> Operation:
    patches = parse_patch(arguments["patch"])
    targets = tuple(Target.capture(patch.path, workspace) for patch in patches)
    workspace_target = Target.capture(workspace, workspace)
    parents = tuple(Target.capture(target.path.parent, workspace) for target in targets)
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
                original, data = "", None
            else:
                data, truncated = read_bytes(target, MAX_FILE_BYTES)
                if truncated:
                    raise ToolError("file_too_large", "Patch target exceeds 1 MiB.")
                original = decode_text(data, False)
            updates.append((data, _apply_text(original, patch)))
        applied: list[str] = []
        created: list[str] = []
        try:
            for target, parent, patch, (original, updated) in zip(
                targets, parents, patches, updates, strict=True
            ):
                _write(target, patch.action, original, updated, workspace=workspace_target,
                       parent=parent, created=created, applied=applied)
        except (OSError, ToolError) as exc:
            return failure(
                exc.code
                if isinstance(exc, ToolError)
                else "target_changed"
                if isinstance(exc, (FileExistsError, FileNotFoundError))
                else "patch_failed",
                str(exc),
                applied_files=applied,
                created_directories=created,
                partial=bool(applied or created),
                **(exc.details if isinstance(exc, ToolError) else {}),
            )
        return {
            "ok": True,
            "applied_files": applied,
            "created_directories": created,
            "partial": False,
        }

    return Operation(
        Decision.ASK,
        "Apply an exact unified diff. Filesystem writes require confirmation.",
        targets,
        {
            "patch": arguments["patch"],
            "changes": [
                {"path": str(target.path), "action": patch.action}
                for target, patch in zip(targets, patches, strict=True)
            ],
        },
        run,
        force_approval=True,
        allow_session_trust=False,
    )
