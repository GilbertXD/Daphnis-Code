"""Argv-only subprocesses with bounded output and process-group cleanup."""

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any

from .approval import (
    ASK_PROGRAMS,
    INTERPRETER,
    READ_PROGRAMS,
    ApprovalMode,
    CleanupError,
    Decision,
    Operation,
    Target,
    ToolError,
    blocked_reason,
    failure,
    path_decision,
    read_arguments,
    resolve_program,
    trusted_read_program,
)
from .files import require_readable_path

_AUTH_ENV = frozenset({
    "API_KEY", "API_TOKEN", "TOKEN", "PASSWORD",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS", "AZURE_CLIENT_SECRET", "DOCKER_AUTH_CONFIG",
    "KUBECONFIG", "GIT_ASKPASS", "SSH_ASKPASS",
})
_AUTH_SUFFIXES = ("_API_KEY", "_API_TOKEN", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")
logger = logging.getLogger(__name__)


class _CommandOutput(asyncio.SubprocessProtocol):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.buffers = {1: bytearray(), 2: bytearray()}
        self.truncated = {1: False, 2: False}
        self.reaped = asyncio.Event()
        self.done = asyncio.get_running_loop().create_future()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        output = self.buffers[fd]
        remaining = self.limit - len(output)
        output.extend(data[:remaining])
        self.truncated[fd] |= len(data) > remaining

    def process_exited(self) -> None:
        self.reaped.set()

    def connection_lost(self, exc: Exception | None) -> None:
        if not self.done.done():
            if exc is None:
                self.done.set_result(None)
            else:
                self.done.set_exception(exc)


async def _finish(task: asyncio.Task[Any]) -> Any:
    """Finish resource cleanup despite repeated cancellation; caller re-raises it."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _terminate(
    transport: asyncio.SubprocessTransport, output: _CommandOutput
) -> None:
    try:
        if os.name == "posix":
            try:
                os.killpg(transport.get_pid(), signal.SIGTERM)
            except ProcessLookupError:
                pass
            await asyncio.sleep(0.1)
            try:
                os.killpg(transport.get_pid(), signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif transport.get_returncode() is None:
            transport.kill()
        await asyncio.wait_for(output.reaped.wait(), 2)
    except (OSError, TimeoutError) as exc:
        raise CleanupError(
            "Command cleanup failed; processes and external side effects may remain."
        ) from exc
    finally:
        # A detached descendant can hold inherited output pipes open indefinitely.
        transport.close()


async def _run(
    argv: tuple[str, ...], cwd: Path, timeout: float, limit: int
) -> dict[str, Any]:
    output = _CommandOutput(limit)
    spawn = asyncio.create_task(
        asyncio.get_running_loop().subprocess_exec(
            lambda: output,
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                name: value for name, value in os.environ.items()
                if name.upper() not in _AUTH_ENV and not name.upper().endswith(_AUTH_SUFFIXES)
            },
            start_new_session=os.name == "posix",
        )
    )
    try:
        transport, _ = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        output.done.cancel()
        try:
            transport, _ = await _finish(spawn)
        except OSError as exc:
            logger.warning("Command spawn failed during cancellation: %s", type(exc).__name__)
        else:
            await _finish(asyncio.create_task(_terminate(transport, output)))
        raise asyncio.CancelledError(
            "Command cancelled; detached descendants and external side effects may remain."
        ) from None

    timed_out = False
    try:
        async with asyncio.timeout(timeout):
            await output.done
    except TimeoutError:
        timed_out = True
        await _finish(asyncio.create_task(_terminate(transport, output)))
    except asyncio.CancelledError:
        await _finish(asyncio.create_task(_terminate(transport, output)))
        raise asyncio.CancelledError(
            "Command cancelled; detached descendants and external side effects may remain."
        ) from None
    except Exception:
        await _finish(asyncio.create_task(_terminate(transport, output)))
        raise
    finally:
        transport.close()
    result = {
        "stdout": output.buffers[1].decode("utf-8", errors="replace"),
        "stderr": output.buffers[2].decode("utf-8", errors="replace"),
        "exit_code": transport.get_returncode(),
        "timed_out": timed_out,
        "terminated": timed_out,
        "stdout_truncated": output.truncated[1],
        "stderr_truncated": output.truncated[2],
    }
    if timed_out:
        return failure(
            "command_timeout",
            "Command timed out and its leader was reaped. Detached descendants and "
            "external side effects may remain.",
            **result,
            termination_scope="original_process_group" if os.name == "posix" else "leader",
            escaped_descendants_may_remain=True,
        )
    if transport.get_returncode() != 0:
        return failure(
            "command_failed", "Command exited with a nonzero status.", **result
        )
    return {"ok": True, **result}


def _safe_without_file_operands(program: str, arguments: list[str]) -> bool:
    if program == "pwd":
        return not arguments
    if program == "uname":
        return not arguments or (
            len(arguments) == 1 and arguments[0] in {"-a", "-s", "-n", "-r", "-v", "-m"}
        )
    if program == "echo":
        return all(
            not word or (word.isprintable() and not word.startswith("-") and "\\" not in word)
            for word in arguments
        )
    return False


def prepare_command(
    arguments: dict[str, Any], workspace: Path, *, mode: ApprovalMode = ApprovalMode.AUTO,
) -> Operation:
    argv = arguments["argv"]
    if not argv[0].strip():
        raise ToolError("command_blocked", "Executable must be a non-empty string.")
    if any(
        token in {"|", "||", "&&", ";", ">", ">>", "<", "2>", "2>>"}
        or "$(" in token
        or "`" in token
        for token in argv
    ):
        raise ToolError(
            "command_blocked", "Shell expressions are not supported; provide argv only."
        )
    cwd = Target.capture(arguments.get("cwd", "."), workspace)
    if cwd.identity is None or not cwd.path.is_dir():
        raise ToolError(
            "invalid_directory", "Command cwd must be an existing directory."
        )
    reason = blocked_reason(Path(argv[0]).name, argv[1:], workspace, cwd=cwd.path)
    if reason:
        raise ToolError("command_blocked", reason)
    executable = resolve_program(argv[0], cwd.path)
    program = executable.path.name.lower()
    reason = blocked_reason(program, argv[1:], workspace, cwd=cwd.path)
    if reason:
        raise ToolError("command_blocked", reason)
    # Unknown native executables are refused. Recognized scripts go through approval policy.
    script = False
    if program not in ASK_PROGRAMS | READ_PROGRAMS and not INTERPRETER.fullmatch(
        program
    ):
        with executable.open() as fd:
            script = os.read(fd, 2) == b"#!"
        if not script:
            raise ToolError(
                "command_blocked",
                "Executable identity is not supported; manual handling is required.",
            )
    targets = [cwd]
    if INTERPRETER.fullmatch(program):
        index = 1
        while index < len(argv):
            argument = argv[index]
            if argument == "-m":
                break  # Imported code is outside the scope of argv path checks.
            if argument in ("-X", "-W"):
                index += 2
                continue
            if not argument.startswith("-"):
                targets.append(Target.capture(argument, cwd.path))
                break
            index += 1
    inspected = read_arguments(program, argv[1:], mode=mode) if program in READ_PROGRAMS else None
    if mode is ApprovalMode.FULL and program in {"cat", "head", "tail", "wc"} and inspected is None:
        raise ToolError("command_blocked", "File operands must be identifiable for automatic execution.")
    decision = Decision.ASK
    invocation = (str(executable.path), *argv[1:])
    auto_allowed = _safe_without_file_operands(program, argv[1:]) and trusted_read_program(
        executable.path
    )
    if inspected is not None:
        options, paths = inspected
        operands = []
        for path in paths:
            # Subprocesses receive literal argv, so a leading tilde is a directory name.
            target = Target.capture(cwd.path / path, cwd.path)
            if program != "ls":
                require_readable_path(target, mode=mode)
                if target.identity is not None and target.path.stat().st_nlink > 1:
                    raise ToolError("sensitive_path", "Files with multiple hard links cannot be read.")
            targets.append(target)
            operands.append(str(target.path))
        if operands and trusted_read_program(executable.path):
            invocation = (str(executable.path), *options, "--", *operands)
    if auto_allowed:
        # The system executable itself is expected outside the workspace.
        decision = path_decision(workspace, tuple(targets))

    async def run():
        for target in (executable, *targets):
            target.verify()
        return await _run(
            invocation,
            cwd.path,
            arguments.get("timeout", 30.0),
            arguments.get("max_output_bytes", 65536),
        )

    return Operation(
        decision,
        "Execute the displayed argv; file access and unclassified commands require confirmation.",
        # The executable is verified in run(), but isn't a data access target.
        tuple(targets),
        {
            "argv": list(invocation),
            "cwd": str(cwd.path),
            "timeout": arguments.get("timeout", 30.0),
        },
        run,
        force_approval=not auto_allowed,
    )
