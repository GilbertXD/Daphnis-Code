"""Argv-only subprocesses with bounded output and process-group cleanup."""

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

from .approval import (
    ASK_PROGRAMS,
    INTERPRETER,
    READ_PROGRAMS,
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


async def _finish(task: asyncio.Task[Any]) -> Any:
    """Finish resource cleanup despite repeated cancellation; caller re-raises it."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _terminate(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            await asyncio.sleep(0.1)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.wait(), 2)
    except (OSError, TimeoutError) as exc:
        raise CleanupError("Could not terminate and reap the command process.") from exc


async def _run(
    argv: tuple[str, ...], cwd: Path, timeout: float, limit: int
) -> dict[str, Any]:
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    )
    try:
        process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        try:
            process = await _finish(spawn)
        except Exception:
            pass
        else:
            await _finish(asyncio.create_task(_terminate(process)))
        raise

    output = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}

    async def drain(name, reader):
        while chunk := await reader.read(8192):
            remaining = max(0, limit - len(output[name]))
            output[name].extend(chunk[:remaining])
            truncated[name] |= len(chunk) > remaining

    readers = [
        asyncio.create_task(drain("stdout", process.stdout)),
        asyncio.create_task(drain("stderr", process.stderr)),
    ]

    async def cleanup():
        try:
            await _terminate(process)
        finally:
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    timed_out = False
    completion = asyncio.gather(process.wait(), *readers)
    completion.add_done_callback(
        lambda future: future.exception() if not future.cancelled() else None
    )
    try:
        async with asyncio.timeout(timeout):
            await completion
    except TimeoutError:
        timed_out = True
        await _finish(asyncio.create_task(cleanup()))
    except asyncio.CancelledError:
        await _finish(asyncio.create_task(cleanup()))
        raise
    except Exception:
        await _finish(asyncio.create_task(cleanup()))
        raise
    result = {
        "stdout": bytes(output["stdout"]).decode("utf-8", errors="replace"),
        "stderr": bytes(output["stderr"]).decode("utf-8", errors="replace"),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "terminated": timed_out,
        "stdout_truncated": truncated["stdout"],
        "stderr_truncated": truncated["stderr"],
    }
    if timed_out:
        return failure(
            "command_timeout", "Command timed out and was terminated.", **result
        )
    if process.returncode != 0:
        return failure(
            "command_failed", "Command exited with a nonzero status.", **result
        )
    return {"ok": True, **result}


def prepare_command(arguments: dict[str, Any], workspace: Path) -> Operation:
    argv = arguments["argv"]
    if not argv[0].strip():
        raise ToolError("command_blocked", "Executable must be a non-empty string.")
    reason = blocked_reason(Path(argv[0]).name, argv[1:], workspace)
    if reason:
        raise ToolError("command_blocked", reason)
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
    executable = resolve_program(argv[0], cwd.path)
    program = executable.path.name.lower()
    reason = blocked_reason(program, argv[1:], cwd.path)
    if reason:
        raise ToolError("command_blocked", reason)
    # Unknown native executables are refused. Recognized scripts still require ASK.
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
    targets = [cwd, executable]
    if INTERPRETER.fullmatch(program):
        index = 1
        while index < len(argv):
            argument = argv[index]
            if argument == "-m":
                break  # Module execution is always ASK; imported code is not a sandbox.
            if argument in ("-X", "-W"):
                index += 2
                continue
            if not argument.startswith("-"):
                targets.append(Target.capture(argument, cwd.path))
                break
            index += 1
    paths = read_arguments(program, argv[1:]) if program in READ_PROGRAMS else None
    if paths is not None:
        targets.extend(Target.capture(path, cwd.path) for path in paths)
    decision = Decision.ASK
    if paths is not None and trusted_read_program(executable.path):
        # The system executable itself is expected outside the workspace.
        decision = path_decision(workspace, (cwd, *targets[2:]))
    invocation = (str(executable.path), *argv[1:])

    async def run():
        for target in targets:
            target.verify()
        return await _run(
            invocation,
            cwd.path,
            arguments.get("timeout", 30.0),
            arguments.get("max_output_bytes", 65536),
        )

    return Operation(
        decision,
        "Execute inspected argv; scripts/builds/uncertain effects need approval.",
        tuple(targets),
        {
            "argv": list(invocation),
            "cwd": str(cwd.path),
            "timeout": arguments.get("timeout", 30.0),
        },
        run,
    )
