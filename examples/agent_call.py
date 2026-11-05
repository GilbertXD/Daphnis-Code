"""Run one Coding task in a temporary workspace using the configured model."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from daphnis import Agent, EventBus, Session, TurnStatus
from daphnis.models import LLM
from daphnis.tools import ApprovalRequest, create_tools


async def read_approval() -> bool:
    """Cancellable POSIX terminal input without leaving a blocked worker thread."""
    print("批准这次修改？[y/N]：", end="", flush=True)
    loop = asyncio.get_running_loop()
    answer = loop.create_future()
    buffer = bytearray()
    fd = sys.stdin.fileno()

    def readable():
        if answer.done():
            return
        try:
            character = os.read(fd, 1)
            if character in (b"", b"\n") or len(buffer) >= 256:
                answer.set_result(
                    buffer.decode("utf-8", errors="replace").strip().lower()
                    in ("y", "yes")
                )
            else:
                buffer.extend(character)
        except OSError as exc:
            answer.set_exception(exc)

    loop.add_reader(fd, readable)
    try:
        return await answer
    finally:
        loop.remove_reader(fd)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Try one Daphnis Agent task")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--approval", action="store_true", help="试验已有文件修改的批准与拒绝"
    )
    args = parser.parse_args()
    required = ["LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"]
    if os.getenv("LLM_PROVIDER") in ("deepseek", "mimo"):
        required.append("LLM_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error("请在 .env 中填写配置：" + ", ".join(missing))

    events = EventBus()
    events.on(
        "tool.started",
        lambda event: print(f"\n[工具] {event.data['name']}", flush=True),
    )
    events.on(
        "tool.finished",
        lambda event: print(f"[工具状态] {event.data['status']}", flush=True),
    )

    def display(event):
        for choice in event.data["chunk"].get("choices", []):
            content = choice.get("delta", {}).get("content")
            if choice["index"] == 0 and isinstance(content, str):
                print(content, end="", flush=True)

    events.on("model.delta", display)
    with TemporaryDirectory(prefix="daphnis-trial-") as temporary:
        workspace = Path(temporary).resolve()
        (workspace / "input.txt").write_text("hello daphnis\n", encoding="utf-8")
        output = workspace / "result.txt"
        original = "waiting for approval\n"
        decisions: list[bool] = []
        if args.approval:
            output.write_text(original, encoding="utf-8")

        async def approve_change(request: ApprovalRequest) -> bool:
            expected = [{"path": str(output), "action": "modify"}]
            if (
                request.tool_name != "apply_patch"
                or request.details.get("changes") != expected
            ):
                print("\n此演示只审批 result.txt 的修改，其他需审批操作拒绝。")
                return False
            print(f"\n[需要审批] 修改已有文件：{output}")
            print(request.details["patch"])
            approved = await read_approval()
            decisions.append(approved)
            print("[审批结果] " + ("已批准" if approved else "已拒绝"), flush=True)
            return approved

        print(
            f"临时工作目录：{workspace}\n"
            + (
                "任务：将 input.txt 的大写内容写入已有 result.txt，需审批。"
                if args.approval
                else "任务：读取 input.txt，转换为大写并创建 result.txt。"
            )
        )
        async with LLM(
            provider=os.environ["LLM_PROVIDER"],
            model_id=os.environ["LLM_MODEL"],
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.getenv("LLM_API_KEY") or None,
            timeout=60.0,
        ) as llm:
            session = Session(
                workspace=workspace, provider=llm.model.provider, model_id=llm.model.id
            )
            tools = create_tools(
                approval=approve_change if args.approval else None
            ).select(["read_file", "list_directory", "apply_patch"])
            agent = Agent(
                llm=llm,
                tools=tools,
                events=events,
                max_requests=8 if args.approval else 6,
                max_tool_calls=10 if args.approval else 8,
            )
            try:
                result = await agent.run(
                    session=session,
                    prompt=(
                        "请调用工具读取 input.txt，将内容转换为大写，"
                        "通过 apply_patch "
                        + (
                            "修改已有 result.txt，"
                            if args.approval
                            else "创建 result.txt，"
                        )
                        + "随后读取 result.txt 验证内容。"
                        "最后用中文简短说明结果。仅使用工作目录中的相对路径。"
                        + (
                            "注意：result.txt 已经存在，请先读取它，使用修改补丁，"
                            "等待用户审批。若审批被拒绝，请停止修改并如实回复，"
                            "不得删除重建或另建其他文件绕过拒绝。"
                            if args.approval
                            else ""
                        )
                    ),
                    stream=args.stream,
                    thinking_mode=(
                        "off" if "off" in (llm.model.thinking_modes or ()) else None
                    ),
                    output_limit=2048,
                )
            finally:
                await events.drain()
        print(
            f"\n状态：{result.status}；模型请求 {result.model_requests} 次，"
            f"工具调用 {result.tool_calls} 次。"
        )
        if result.status != TurnStatus.COMPLETED:
            print(f"错误：{result.error}")
            return 1
        if not args.stream:
            print(f"助手：{result.message.get('content') or ''}")
        if args.approval:
            if not decisions:
                print("审批验证未通过：模型没有触发 result.txt 的修改审批。")
                return 1
            if not any(decisions):
                unchanged = (
                    output.is_file() and output.read_text(encoding="utf-8") == original
                )
                no_extras = set(workspace.iterdir()) == {
                    workspace / "input.txt",
                    output,
                }
                if not unchanged or not no_extras:
                    print("拒绝验证未通过：文件内容发生变化或出现额外文件。")
                    return 1
                print("拒绝验证通过：result.txt 保留原内容，没有另建文件。")
                return 0
        if (
            not output.is_file()
            or output.read_text(encoding="utf-8").strip() != "HELLO DAPHNIS"
        ):
            print("任务验证未通过：result.txt 未包含预期的大写内容。")
            return 1
        print("任务验证通过：result.txt 内容为 HELLO DAPHNIS。临时目录将在退出时清理。")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n已取消。")
        raise SystemExit(130)
