"""Terminal entry point and non-interactive one-Turn execution."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import TurnStatus
from .app import Application
from .config import environment_connections, load_mcp_config
from .operations import Operations
from .terminal_commands import safe_error, terminal_command


async def execute(ops: Operations, prompt: str) -> int:
    if prompt.lstrip().startswith("/"):
        print(terminal_command(ops, prompt).text)
        return 0
    result = await ops.send(prompt, stream=False)
    if result.status == TurnStatus.COMPLETED:
        print(result.message.get("content") or "")
        return 0
    print(safe_error(result.error or result.status), file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Daphnis terminal Coding Agent")
    parser.add_argument("-p", "--prompt", help="执行一次任务，输出最终答案后退出")
    parser.add_argument("--workspace", type=Path, help="新会话的工作目录，默认当前目录")
    parser.add_argument("--session", help="恢复已有会话 ID，使用其保存的工作区")
    parser.add_argument("--thinking", help="思考档位；default 为服务端默认")
    parser.add_argument("--mcp-config", type=Path, help="显式加载一个本地 MCP stdio Server")
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
            interactive=args.prompt is None,
        )
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        return 2

    ops = Operations(application)

    async def run():
        try:
            if args.mcp_config:
                await ops.connect_mcp(*load_mcp_config(args.mcp_config))
            if args.session:
                ops.restore_session(args.session)
            if args.thinking:
                ops.set_thinking(
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
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        return 1
