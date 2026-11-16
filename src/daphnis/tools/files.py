"""Bounded text reads, directory listing, and approved file writes."""

import asyncio
import ctypes
import errno
import os
import re
import stat
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..history import finish_io
from ..text import decode_text, line_range
from .approval import (
    ApprovalMode,
    Decision,
    Operation,
    Target,
    ToolError,
    failure,
    path_decision,
    signature,
    verify_directory,
)

if os.name == "posix":
    import fcntl

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


def _lock_directory() -> Path:
    # A fixed local namespace lets independent processes agree without creating
    # files in the workspace or requiring write access to directories we read.
    return Path("/tmp").resolve() / f".daphnis-file-locks-{os.geteuid()}"


def _lock_path(path: Path) -> Path:
    # Case and Unicode aliases can name the same entry on some filesystems.
    return Path(unicodedata.normalize("NFC", str(path).casefold()))


def _check_lock_entry(info: os.stat_result, *, directory: bool) -> None:
    valid_type = stat.S_ISDIR(info.st_mode) if directory else (
        stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    )
    if not valid_type or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ToolError("permission_denied", "Unsafe file tool lock storage.")


@asynccontextmanager
async def _file_locks(paths: tuple[Path, ...], *, shared: bool = False):
    if os.name != "posix":
        raise ToolError("unsupported_platform", "Hardened file tools require POSIX.")
    directory = _lock_directory()
    lock_paths = {_lock_path(path) for path in paths}
    private_path = _lock_path(directory)
    if any(path.is_relative_to(private_path) for path in lock_paths):
        raise ToolError("sensitive_path", "File tool lock storage is private.")
    directory.mkdir(mode=0o700, exist_ok=True)
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    locks = []
    try:
        _check_lock_entry(os.fstat(root), directory=True)
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        for key in sorted(sha256(os.fsencode(path)).hexdigest() for path in lock_paths):
            fd = os.open(key, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                         0o600, dir_fd=root)
            locks.append(fd)
            _check_lock_entry(os.fstat(fd), directory=False)
            while True:
                try:
                    fcntl.flock(fd, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.01)
        yield
    finally:
        for fd in reversed(locks):
            os.close(fd)
        os.close(root)


def _current_target(target: Target) -> Target:
    current = Target.capture(target.requested, target.path.parent)
    if current.path != target.path:
        raise ToolError("target_changed", f"Target path changed: {target.requested}")
    return current


class _PathOperation(Operation):
    """Authorize a path whose current version will be captured under its lock."""

    def verify(self) -> None:
        for target in self.targets:
            _current_target(target)


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


def require_readable_path(target: Target, *, mode: ApprovalMode = ApprovalMode.AUTO) -> None:
    """Protect content, including reads through a resolved symlink."""
    if mode is not ApprovalMode.FULL and any(
        is_credential_path(path) for path in (target.path, target.requested)
    ):
        raise ToolError("sensitive_path", "Reading protected credentials is prohibited.")


def read_bytes(target: Target, limit: int, *, mode: ApprovalMode = ApprovalMode.AUTO) -> tuple[bytes, bool]:
    require_readable_path(target, mode=mode)
    with target.open() as fd:
        before = os.fstat(fd)
        if before.st_nlink > 1:
            raise ToolError("sensitive_path", "Files with multiple hard links cannot be read.")
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(fd, min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if signature(os.fstat(fd)) != signature(before):
            raise ToolError("target_changed", "File changed while reading.")
        target.verify()
        return bytes(data[:limit]), len(data) > limit


def prepare_read(
    arguments: dict[str, Any], workspace: Path, *, mode: ApprovalMode = ApprovalMode.AUTO,
) -> Operation:
    target = Target.capture(arguments["path"], workspace)
    require_readable_path(target, mode=mode)
    decision = path_decision(workspace, (target,))
    pin_version = mode is ApprovalMode.ASK or (
        mode is not ApprovalMode.FULL and decision is Decision.ASK
    )
    parent = Target.capture(target.path.parent, workspace)

    def read():
        if parent.identity is not None:
            verify_directory(parent.path, parent.identity)
        current = target if pin_version else _current_target(target)
        data, truncated = read_bytes(current, arguments.get("max_bytes", 65536), mode=mode)
        lines = decode_text(data, truncated).splitlines(keepends=True)
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line")
        if end is not None and end < start:
            raise ToolError("invalid_range", "end_line must be at least start_line.")
        selected = line_range(lines, start, end)
        return {
            "ok": True,
            "path": str(target.path),
            "content": "".join(selected),
            "start_line": start,
            "end_line": start + len(selected) - 1 if selected else None,
            "truncated": truncated,
            "scanned_bytes": len(data),
        }

    async def run():
        async with _file_locks((target.path,), shared=True):
            return await finish_io(asyncio.to_thread(read))

    operation = Operation if pin_version else _PathOperation
    return operation(
        decision,
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


def _apply_text(original: str, patch: FilePatch) -> bytes:
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
    encoded = updated.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ToolError("file_too_large", "Patched file exceeds the 1 MiB limit.")
    return encoded


def _publish_new(stage: int, parent: int, name: str) -> None:
    # Use no-replace rename where available so even a killed creator leaves
    # no second link. The shared lock hides the fallback link until cleanup.
    system_call = {"darwin": ("renameatx_np", 4), "linux": ("renameat2", 1)}.get(sys.platform)
    if system_call is not None:
        symbol, flags = system_call
        rename = getattr(ctypes.CDLL(None, use_errno=True), symbol, None)
        if rename is not None:
            rename.argtypes = (
                ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
            )
            rename.restype = ctypes.c_int
            if rename(stage, b"updated", parent, os.fsencode(name), flags) == 0:
                return
            error = ctypes.get_errno()
            if error not in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL):
                raise OSError(error, os.strerror(error), name)
    os.link("updated", name, src_dir_fd=stage, dst_dir_fd=parent, follow_symlinks=False)


def _write(
    target: Target, action: str, original: bytes | None, content: bytes, *,
    workspace: Target, parent: Target, created: list[str] | None, applied: list[str],
) -> None:
    verify_directory(workspace.path, workspace.identity)
    target.verify()
    with target.parent(create=action == "create" and created is not None, created=created) as (fd, name):
        def verify_parent() -> None:
            verify_directory(workspace.path, workspace.identity)
            opened = signature(os.fstat(fd))
            verify_directory(parent.path, opened)
            if parent.identity is not None and parent.identity[:3] != opened[:3]:
                raise ToolError("target_changed", f"Parent directory was replaced: {parent.path}")

        def verify_original() -> None:
            if original is None:
                target.verify()
            else:
                current, truncated = read_bytes(target, len(original))
                if truncated or current != original:
                    raise ToolError("target_changed", "Target content changed before commit.")

        verify_parent()
        if action == "delete":
            verify_original()
            os.unlink(name, dir_fd=fd)
            applied.append(str(target.path))
            verify_parent()
            return

        staging = f".daphnis-{uuid4().hex}.tmp"
        os.mkdir(staging, mode=0o700, dir_fd=fd)
        try:
            stage = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                output = os.open("updated", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o666, dir_fd=stage)
                with os.fdopen(output, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    if target.identity is not None:
                        os.fchmod(stream.fileno(), stat.S_IMODE(target.identity[2]))
                    os.fsync(stream.fileno())

                verify_parent()
                verify_original()
                if action == "create":
                    _publish_new(stage, fd, name)
                else:
                    # The check and replacement are serialized for cooperating
                    # tools. External writers can still act between the two.
                    os.replace("updated", name, src_dir_fd=stage, dst_dir_fd=fd)
                applied.append(str(target.path))
                verify_parent()
            finally:
                try:
                    os.unlink("updated", dir_fd=stage)
                except FileNotFoundError:
                    pass
                finally:
                    os.close(stage)
        finally:
            os.rmdir(staging, dir_fd=fd)


def _workspace_file(value: str, workspace: Path) -> tuple[Target, Target, Target]:
    path = workspace / Path(value).expanduser()
    if path == workspace or ".." in path.parts or not path.is_relative_to(workspace):
        raise ToolError("permission_denied", "A file path inside the workspace without '..' is required.")

    workspace_target = Target.capture(workspace, workspace)
    verify_directory(workspace, workspace_target.identity)
    target = Target(path, path, None)
    try:
        with target.parent() as (fd, name):
            parent = Target(path.parent, path.parent, signature(os.fstat(fd)))
            verify_directory(parent.path, parent.identity)
            try:
                identity = signature(os.stat(name, dir_fd=fd, follow_symlinks=False))
            except FileNotFoundError:
                identity = None
    except FileNotFoundError as exc:
        raise ToolError("file_not_found", "The parent directory must already exist.") from exc
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.ELOOP):
            raise ToolError("invalid_path", "Parent components must be directories, without symlinks.") from exc
        raise
    return Target(path, path, identity), parent, workspace_target


def prepare_write(arguments: dict[str, Any], workspace: Path) -> Operation:
    target, parent, workspace_target = _workspace_file(arguments["path"], workspace)
    path = target.path
    if target.identity is not None:
        raise ToolError("file_exists", f"Creation will not overwrite: {path}")
    content = arguments["content"].encode("utf-8")
    if len(content) > MAX_FILE_BYTES:
        raise ToolError("file_too_large", "New file exceeds the 1 MiB limit.")

    details = {"path": str(path), "bytes_written": len(content)}

    def create():
        applied: list[str] = []
        try:
            if _current_target(target).identity is not None:
                raise FileExistsError(path)
            _write(target, "create", None, content, workspace=workspace_target,
                   parent=parent, created=None, applied=applied)
        except FileExistsError as exc:
            raise ToolError("file_exists", f"Creation will not overwrite: {path}") from exc
        except (OSError, ToolError) as exc:
            if not applied:
                raise
            return failure(
                exc.code if isinstance(exc, ToolError) else "io_error",
                str(exc), partial=True, **details,
            )
        return {"ok": True, **details}

    async def run():
        async with _file_locks((target.path,)):
            return await finish_io(asyncio.to_thread(create))

    return Operation(
        Decision.ASK,
        "Create a new UTF-8 file. Filesystem writes require confirmation.",
        (target,),
        {"path": str(path), "size_bytes": len(content), "content": arguments["content"]},
        run,
        force_approval=True,
        allow_session_trust=False,
    )


def prepare_replace(arguments: dict[str, Any], workspace: Path) -> Operation:
    target, parent, workspace_target = _workspace_file(arguments["path"], workspace)
    require_readable_path(target)
    if target.identity is None:
        raise ToolError("file_not_found", f"Target does not exist: {target.path}")
    if not stat.S_ISREG(target.identity[2]):
        raise ToolError("invalid_file", "replace_text requires a regular file, without symlinks.")
    old, new = arguments["old_text"], arguments["new_text"]

    def replace():
        original, truncated = read_bytes(target, MAX_FILE_BYTES)
        if truncated:
            raise ToolError("file_too_large", "Original file exceeds the 1 MiB limit.")
        text = decode_text(original, False)
        start = text.find(old)
        if start < 0:
            raise ToolError("text_not_found", "old_text was not found. Read the file and use exact text.")
        if text.find(old, start + 1) >= 0:
            raise ToolError("ambiguous_match", "old_text matches more than once. Include surrounding text to make it unique.")
        updated = (text[:start] + new + text[start + len(old):]).encode("utf-8")
        if len(updated) > MAX_FILE_BYTES:
            raise ToolError("file_too_large", "Updated file exceeds the 1 MiB limit.")
        applied: list[str] = []
        try:
            _write(target, "modify", original, updated, workspace=workspace_target,
                   parent=parent, created=None, applied=applied)
        except (OSError, ToolError) as exc:
            if not applied:
                raise
            return failure(
                exc.code if isinstance(exc, ToolError) else "io_error",
                str(exc), partial=True, path=str(target.path),
                **(exc.details if isinstance(exc, ToolError) else {}),
            )
        return {"ok": True, "path": str(target.path), "bytes_written": len(updated)}

    async def run():
        async with _file_locks((target.path,)):
            return await finish_io(asyncio.to_thread(replace))

    return Operation(
        Decision.ASK,
        "Replace one exact text match. Filesystem writes require confirmation.",
        (target,),
        {"path": str(target.path), "old_text": old, "new_text": new},
        run,
        force_approval=True,
        allow_session_trust=False,
    )


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

    def apply():
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

    async def run():
        async with _file_locks(tuple(target.path for target in targets)):
            return await finish_io(asyncio.to_thread(apply))

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
