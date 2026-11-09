"""Path checks, fixed command restrictions, and UI-independent approvals."""

import os
import re
import shutil
import stat
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Decision(StrEnum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


class ApprovalMode(StrEnum):
    ASK = "ask"
    AUTO = "auto"
    FULL = "full"


class ApprovalDecision(StrEnum):
    DENY = "deny"
    ALLOW_ONCE = "allow_once"
    TRUST_SESSION = "trust_session"


class ToolError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class CleanupError(RuntimeError):
    """A process could not be stopped safely."""


def failure(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}, **details}


def signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True)
class Target:
    requested: Path
    path: Path
    identity: tuple[int, ...] | None

    @classmethod
    def capture(cls, value: str | Path, workspace: Path) -> "Target":
        requested = Path(value).expanduser()
        if not requested.is_absolute():
            requested = workspace.absolute() / requested
        try:
            path = requested.resolve()
        except RuntimeError as exc:
            raise ToolError(
                "invalid_path", f"Cannot resolve path: {requested}"
            ) from exc
        try:
            identity = signature(path.stat())
        except FileNotFoundError:
            identity = None
        return cls(requested, path, identity)

    def verify(self) -> None:
        current = Target.capture(self.requested, self.path.parent)
        if current.path != self.path or current.identity != self.identity:
            raise ToolError(
                "target_changed",
                f"Target changed; retry for reassessment: {self.requested}",
            )

    @contextmanager
    def parent(
        self, *, create: bool = False, created: list[str] | None = None
    ) -> Iterator[tuple[int, str]]:
        """Walk the normalized path using directory FDs, refusing new symlinks."""
        if os.name != "posix":
            raise ToolError(
                "unsupported_platform", "Hardened file tools require POSIX."
            )
        fd = os.open(self.path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        current = Path(self.path.anchor)
        try:
            for component in self.path.parts[1:-1]:
                current = current / component
                if create:
                    try:
                        os.mkdir(component, dir_fd=fd)
                        if created is not None:
                            created.append(str(current))
                    except FileExistsError:
                        pass
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
                os.close(fd)
                fd = next_fd
            yield fd, self.path.name
        finally:
            os.close(fd)

    @contextmanager
    def open(self, *, directory: bool = False) -> Iterator[int]:
        self.verify()
        if self.path == Path(self.path.anchor):
            if not directory:
                raise ToolError("invalid_file", "Expected a regular text file.")
            fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
        else:
            with self.parent() as (parent_fd, name):
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if directory:
                    flags |= os.O_DIRECTORY
                fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if signature(info) != self.identity:
                raise ToolError("target_changed", "Target changed before opening.")
            if not (
                stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
            ):
                raise ToolError(
                    "invalid_file", "Expected a directory or regular text file."
                )
            yield fd
        finally:
            os.close(fd)


@dataclass(frozen=True)
class Operation:
    decision: Decision
    reason: str
    targets: tuple[Target, ...]
    details: dict[str, Any]
    run: Callable[[], Awaitable[dict[str, Any]]]
    force_approval: bool = False
    allow_session_trust: bool = True

    def verify(self) -> None:
        for target in self.targets:
            target.verify()


@dataclass(frozen=True)
class ApprovalRequest:
    session_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    workspace: str
    reason: str
    details: dict[str, Any]
    can_trust: bool = False


ApprovalCallback = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


def path_decision(
    workspace: Path, targets: tuple[Target, ...], *, writes_existing: bool = False
) -> Decision:
    if writes_existing or any(
        not target.path.is_relative_to(workspace) for target in targets
    ):
        return Decision.ASK
    return Decision.ALLOW


# No runtime switch disables these restrictions. Unknown execution forms are
# rejected rather than claiming that an argv blacklist is an OS sandbox.
BLOCKED_PROGRAMS = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        "fdisk",
        "sfdisk",
        "cfdisk",
        "parted",
        "wipefs",
        "blkdiscard",
        "dd",
        "sh",
        "bash",
        "zsh",
        "dash",
        "ksh",
        "fish",
        "csh",
        "tcsh",
        "cmd",
        "cmd.exe",
        "powershell",
        "pwsh",
        "osascript",
        "env",
        "xargs",
        "command",
        "exec",
        "nice",
        "nohup",
        "timeout",
        "watch",
        "busybox",
        "chroot",
        "nsenter",
        "setsid",
        "launchctl",
        "systemctl",
        "find",
        "rsync",
        "awk",
        "gawk",
        "mawk",
        "sed",
        "perl",
        "ruby",
        "php",
        "lua",
        "luajit",
        "tclsh",
        "rscript",
        "source",
        "eval",
    }
)
ASK_PROGRAMS = frozenset(
    {
        "git",
        "rm",
        "rmdir",
        "mkdir",
        "touch",
        "cp",
        "mv",
        "ln",
        "chmod",
        "sleep",
        "true",
        "false",
        "make",
        "cmake",
        "ninja",
        "npm",
        "pnpm",
        "yarn",
        "pip",
        "pip3",
        "cargo",
        "rustc",
        "go",
        "gcc",
        "cc",
        "clang",
        "uv",
    }
)
READ_PROGRAMS = frozenset({"pwd", "uname", "echo", "cat", "ls", "head", "tail", "wc"})
CRITICAL_DIRECTORIES = frozenset(
    {
        "/",
        "/bin",
        "/sbin",
        "/usr",
        "/etc",
        "/dev",
        "/proc",
        "/sys",
        "/boot",
        "/var",
        "/private",
        "/System",
        "/Library",
        "/Applications",
    }
)
INTERPRETER = re.compile(r"^(python(?:\d+(?:\.\d+)*)?|node(?:js)?)$")


def _option_words(arguments: list[str]) -> list[str]:
    return arguments[: arguments.index("--")] if "--" in arguments else arguments


def _git_command(arguments: list[str]) -> tuple[str, list[str]]:
    index = 0
    while index < len(arguments) and arguments[index].startswith("-"):
        option = arguments[index]
        if option in ("--version", "--help"):
            return option, arguments[index + 1 :]
        if option in ("-C", "--git-dir", "--work-tree"):
            index += 2
        elif option.startswith(("-C", "--git-dir=", "--work-tree=")):
            index += 1
        elif option in ("--no-pager", "--bare", "--literal-pathspecs"):
            index += 1
        else:
            raise ToolError(
                "command_blocked", f"Unsupported Git global option: {option}"
            )
    if index >= len(arguments):
        raise ToolError("command_blocked", "A Git subcommand is required.")
    return arguments[index], arguments[index + 1 :]


def blocked_reason(program: str, arguments: list[str], workspace: Path) -> str | None:
    name = program.lower()
    options = _option_words(arguments)
    if name in BLOCKED_PROGRAMS or name.startswith("mkfs"):
        return f"Program is blocked: {name}"
    if name == "diskutil":
        return "Disk control commands are not supported."
    if name == "rm":
        recursive = any(
            (
                arg.startswith("-")
                and not arg.startswith("--")
                and any(c in arg[1:] for c in "rR")
            )
            or (arg.startswith("--") and len(arg) > 2 and "--recursive".startswith(arg))
            for arg in options
        )
        force = any(
            (arg.startswith("-") and not arg.startswith("--") and "f" in arg[1:])
            or (arg.startswith("--") and len(arg) > 2 and "--force".startswith(arg))
            for arg in options
        )
        if recursive and force:
            return "Recursive forced deletion is blocked."
    if name in ("rm", "rmdir"):
        for argument in arguments:
            if argument.startswith("-"):
                continue
            target = Target.capture(argument, workspace).path
            critical = {Path(path).resolve() for path in CRITICAL_DIRECTORIES}
            if target in critical or target == Path.home().resolve():
                return "Deletion of a root or critical directory is blocked."
    if name == "git":
        command, rest = _git_command(arguments)
        options = _option_words(rest)
        short_force = any(
            arg.startswith("-") and not arg.startswith("--") and "f" in arg[1:]
            for arg in options
        )
        long_force = any(
            arg.startswith("--force") or (len(arg) > 2 and "--force".startswith(arg))
            for arg in options
        )
        if command == "reset" and any(
            arg.startswith("--hard") or (len(arg) > 2 and "--hard".startswith(arg))
            for arg in options
        ):
            return "Git hard reset is blocked."
        if command == "clean" and (short_force or long_force):
            return "Git forced clean is blocked."
        if command == "push" and (
            short_force
            or long_force
            or "--mirror" in options
            or any(arg.startswith("+") for arg in rest)
        ):
            return "Git force push is blocked."
        known = {
            "--version",
            "--help",
            "status",
            "diff",
            "log",
            "show",
            "rev-parse",
            "ls-files",
            "add",
            "commit",
            "checkout",
            "switch",
            "restore",
            "reset",
            "clean",
            "push",
            "fetch",
            "pull",
            "branch",
            "tag",
            "merge",
            "rebase",
            "init",
            "clone",
            "stash",
        }
        if command not in known:
            return "Unknown Git subcommands and aliases are not supported."
    if INTERPRETER.fullmatch(name):
        for argument in arguments:
            if not argument.startswith("-"):
                continue
            if name.startswith("python"):
                if argument.startswith("--") or "c" in argument[1:]:
                    return "Inline interpreter execution is blocked."
            elif argument.startswith(("--eval", "--print")) or any(
                c in argument[1:] for c in "ep"
            ):
                return "Inline interpreter execution is blocked."
        if not arguments or arguments[0] == "-":
            return "Interactive and stdin interpreter modes are not supported."
    return None


def trusted_read_program(path: Path) -> bool:
    """Automatic reads are restricted to root-owned, non-writable system binaries."""
    if path.parent not in (Path("/bin"), Path("/usr/bin")):
        return False
    for component in (path, *path.parents):
        info = component.stat()
        if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return False
    return True


def read_arguments(program: str, arguments: list[str]) -> list[str] | None:
    """Return explicitly read paths, or None if options need approval."""
    if program in ("echo", "pwd", "uname"):
        return []
    paths = []
    index = 0
    positional = False
    allowed_flags = {"cat": "benstuvAET", "ls": "aAlhndpF1", "wc": "clmwL"}
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--" and not positional:
            positional = True
        elif argument.startswith("-") and not positional:
            if program in ("head", "tail") and argument in ("-n", "-c"):
                index += 1
                if index >= len(arguments) or not arguments[index].isdigit():
                    return None
            elif not (
                program in allowed_flags
                and len(argument) > 1
                and not argument.startswith("--")
                and all(flag in allowed_flags[program] for flag in argument[1:])
            ):
                return None
        else:
            paths.append(argument)
        index += 1
    return paths or (["."] if program == "ls" else [])


def resolve_program(value: str, workspace: Path) -> Target:
    if "/" in value:
        target = Target.capture(value, workspace)
    else:
        located = shutil.which(value)
        if located is None:
            raise ToolError("command_not_found", f"Executable not found: {value}")
        target = Target.capture(located, workspace)
    if (
        target.identity is None
        or not target.path.is_file()
        or not os.access(target.path, os.X_OK)
    ):
        raise ToolError("command_not_found", f"Not an executable file: {value}")
    return target
