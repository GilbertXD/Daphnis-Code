"""Terminal entry point and non-interactive one-Turn execution."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv
from saturnus import SaturnusError

from .agent import TurnStatus, assistant_text
from .app import Application
from .config import (
    environment_connections,
    environment_context_options,
    load_mcp_config,
)
from .operations import Operations
from .terminal_commands import (
    RECOVERY_COMMANDS,
    recover_turn,
    safe_error,
    sanitize_terminal,
    terminal_command,
)
from .turn_state import RecoveryKind


async def execute(ops: Operations, prompt: str) -> int:
    if action := RECOVERY_COMMANDS.get(prompt.strip()):
        result = await recover_turn(ops, action, stream=False)
    elif prompt.lstrip().startswith("/"):
        print(sanitize_terminal((await terminal_command(ops, prompt)).text))
        return 0
    elif (recovery := ops.recovery) is not None and recovery.kind is RecoveryKind.INTERRUPTED_TURN:
        print("上轮任务未完成。请通过 -p /continue 继续，或 -p /terminate 终止。", file=sys.stderr)
        return 3
    else:
        result = await ops.send(prompt, stream=False)
    if result.message is not None:
        print(sanitize_terminal(assistant_text(result.message)))
    if result.status == TurnStatus.COMPLETED:
        return 0
    if result.status == TurnStatus.PAUSED_LIMIT:
        print(sanitize_terminal(result.notice or result.status))
        return 3
    if result.status == TurnStatus.TERMINATED:
        print("任务已终止。")
        return 3
    print(sanitize_terminal(safe_error(result.error or result.status)), file=sys.stderr)
    return 1


class _TerminalLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return sanitize_terminal(super().format(record))


def main() -> int:
    # Configure the terminal handler here so library log records stay unchanged.
    logger = logging.getLogger("daphnis")
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(_TerminalLogFormatter())
    propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        return _main()
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate


class _TerminalArgumentParser(argparse.ArgumentParser):
    def _print_message(self, message: str | None, file: TextIO | None = None) -> None:
        super()._print_message(sanitize_terminal(message) if message else message, file)


def _main() -> int:
    parser = _TerminalArgumentParser(description="Daphnis terminal Coding Agent")
    parser.add_argument("-p", "--prompt", help="执行一次任务，输出最终答案后退出")
    parser.add_argument("--workspace", type=Path, help="新会话的工作目录，默认当前目录")
    parser.add_argument("--session", help="恢复已有会话 ID，使用其保存的工作区")
    parser.add_argument("--thinking", help="思考档位；default 为服务端默认")
    parser.add_argument("--mcp-config", type=Path, help="加载显式配置的 MCP Server")
    args = parser.parse_args()
    if args.session and args.workspace:
        parser.error("--session 使用已保存工作区，请勿同时指定 --workspace。")
    if args.prompt is None and not sys.stdin.isatty():
        parser.error("交互模式需要终端；非交互执行请使用 -p。")
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        provider, model_id, connections = environment_connections()
        application = Application(
            workspace=args.workspace or Path.cwd(),
            provider=provider,
            model_id=model_id,
            connections=connections,
            **environment_context_options(),
            interactive=args.prompt is None,
        )
    except (OSError, SaturnusError, TypeError, ValueError) as exc:
        print(sanitize_terminal(safe_error(exc)), file=sys.stderr)
        return 2

    ops = Operations(application)

    async def run():
        try:
            if args.session:
                ops.restore_session(args.session)
            if args.mcp_config:
                for name, options in load_mcp_config(args.mcp_config).items():
                    try:
                        await ops.connect_mcp(name, **options)
                    except (ValueError, TypeError, RuntimeError) as exc:
                        print(sanitize_terminal(f"MCP {name}: {safe_error(exc)}"), file=sys.stderr)
            if args.thinking:
                await ops.set_thinking(
                    None if args.thinking == "default" else args.thinking
                )
            if args.prompt is not None:
                return await execute(ops, args.prompt)
            from .tui.app import DaphnisApp

            await DaphnisApp(ops, application.events).run_async()
            return 0
        finally:
            await ops.close()

    try:
        return asyncio.run(run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("已取消。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - Final CLI boundary for unexpected failures.
        print(sanitize_terminal(safe_error(exc)), file=sys.stderr)
        return 1
