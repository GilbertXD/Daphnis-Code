"""Terminal entry point and non-interactive one-Turn execution."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import TurnStatus
from .app import Application, environment_connections


def safe_error(error: object) -> str:
    text = str(error)
    for name in (
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "MIMO_API_KEY",
        "COMPATIBLE_API_KEY",
    ):
        key = os.getenv(name)
        if key:
            text = text.replace(key, "[REDACTED]")
    return text


async def execute(application: Application, prompt: str | None) -> int:
    try:
        if prompt is not None:
            if prompt.lstrip().startswith("/"):
                print((await application.command(prompt)).text)
                return 0
            result = await application.send(prompt, stream=False)
            if result.status == TurnStatus.COMPLETED:
                print(result.message.get("content") or "")
                return 0
            print(safe_error(result.error or result.status), file=sys.stderr)
            return 1
        from .tui.app import DaphnisApp

        await DaphnisApp(application).run_async()
        return 0
    finally:
        await application.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Daphnis terminal Coding Agent")
    parser.add_argument("-p", "--prompt", help="执行一次任务，输出最终答案后退出")
    parser.add_argument("--workspace", type=Path, help="新会话的工作目录，默认当前目录")
    parser.add_argument("--session", help="恢复已有会话 ID，使用其保存的工作区")
    parser.add_argument("--thinking", help="思考档位；default 为服务端默认")
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

    async def run():
        try:
            if args.session:
                application.restore_session(args.session)
            if args.thinking:
                application.set_thinking(
                    None if args.thinking == "default" else args.thinking
                )
            return await execute(application, args.prompt)
        finally:
            await application.close()

    try:
        return asyncio.run(run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        return 1
