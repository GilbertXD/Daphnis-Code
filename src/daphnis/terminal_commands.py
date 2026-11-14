"""Translate terminal commands to operations and presentation results."""

import os
import shlex
from dataclasses import dataclass

from .operations import Operations
from .terminal_text import sanitize_terminal
from .tools import ApprovalMode
from .turn_state import RecoveryAction, RecoveryKind

__all__ = [
    "APPROVAL_MODE_TEXT",
    "HELP",
    "CommandResult",
    "safe_error",
    "sanitize_terminal",
    "terminal_command",
]

APPROVAL_MODE_TEXT = {
    ApprovalMode.ASK: ("每次询问", "所有可执行工具调用都需要确认。"),
    ApprovalMode.AUTO: ("安全自动", "自动执行无需额外授权的操作，其余按规则确认。"),
    ApprovalMode.FULL: ("全部自动", "允许读取敏感文件并自动执行工具。"),
}

FULL_APPROVAL_RISK = (
    "模型可能读取 .env 等敏感文件，并将内容发给模型服务。\n"
    "命令和 MCP 工具可能访问工作区外的文件、修改文件或访问网络。"
)

HELP = """/model [list|序号|模型ID|厂商 模型ID]  查看或切换模型（按厂商分列）
/thinking [default|档位]  查看或设置思考；default 为服务端默认
/system [reset]  编辑会话系统提示词，或恢复默认
/tools [all|none|on 工具名...|off 工具名...]  查看或选择工具
/approval [ask|auto|full]  查看或切换工具审批模式
/mcp [reconnect|disconnect|refresh Server名]  查看 MCP 状态或管理连接
/session [会话ID]  列出或恢复会话
/new  新建会话，保留旧会话
/compact  压缩旧上下文，保留原始历史
/retry  重试已保存但未得到回复的用户消息
/continue、/terminate  继续或终止未完成的工具任务
/clear  仅清理屏幕，保留历史
/help  显示帮助
Enter 发送；Shift+Enter 换行；Alt+Enter / F2 为换行备用键。
Esc 中断当前轮；Ctrl+C 运行时取消、空闲时退出；Ctrl+Q 取消并退出。"""


@dataclass(frozen=True)
class CommandResult:
    action: str = "message"
    text: str = ""


RECOVERY_COMMANDS = {
    "/retry": RecoveryAction.RETRY,
    "/continue": RecoveryAction.CONTINUE,
    "/terminate": RecoveryAction.TERMINATE,
}


async def recover_turn(ops: Operations, action: RecoveryAction, *, stream: bool):
    state = ops.recovery
    expected = RecoveryKind.FAILED_USER if action is RecoveryAction.RETRY else RecoveryKind.INTERRUPTED_TURN
    if state is None or state.kind is not expected:
        raise ValueError("当前会话没有适用此操作的未完成任务。")
    if action is RecoveryAction.RETRY:
        return await ops.retry_failed_user(state.key, stream=stream)
    if action is RecoveryAction.CONTINUE:
        return await ops.continue_interrupted_turn(state.key, stream=stream)
    return await ops.terminate_interrupted_turn(state.key)


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


async def terminal_command(
    ops: Operations, text: str, *, full_confirmed: bool = False,
) -> CommandResult:
    words = shlex.split(text)
    name, *args = words or [""]
    if name == "/help" and not args:
        return CommandResult(text=HELP)
    if name == "/clear" and not args:
        return CommandResult("clear", "已清理屏幕；会话历史保留。")
    if name == "/mcp":
        match args:
            case []:
                return CommandResult(
                    text="\n".join(
                        f"{server.name}  {server.state}  {server.tool_count} 个工具"
                        + (f" · {server.error}" if server.error else "")
                        for server in ops.mcp_status()
                    )
                    or "尚未配置 MCP Server。"
                )
            case ["reconnect", server]:
                await ops.reconnect_mcp(server)
                return CommandResult(text=f"MCP {server} 已连接。")
            case ["disconnect", server]:
                await ops.disconnect_mcp(server)
                return CommandResult(text=f"MCP {server} 已断开。")
            case ["refresh", server]:
                await ops.refresh_mcp(server)
                return CommandResult(text=f"MCP {server} 工具目录已刷新。")
            case _:
                raise ValueError("用法：/mcp [reconnect|disconnect|refresh Server名]")
    if ops.running:
        raise RuntimeError("请先等待或取消当前 Turn，再执行该操作。")
    if name == "/system":
        if not args:
            return CommandResult("system", ops.get_system_prompt())
        if args == ["reset"]:
            await ops.reset_system_prompt()
            return CommandResult(text="已恢复默认系统提示词。")
        raise ValueError("用法：/system [reset]")
    if name == "/compact" and not args:
        checkpoint = await ops.compact()
        return CommandResult(text=(
            f"已压缩至第 {checkpoint.through} 条消息，原始历史保留。"
            if checkpoint is not None else "当前没有可有效压缩的旧上下文。"
        ))
    if name == "/approval":
        if args:
            if len(args) != 1 or args[0] not in tuple(ApprovalMode):
                raise ValueError("用法：/approval [ask|auto|full]")
            if args[0] == ApprovalMode.FULL and not full_confirmed and ops.approval_mode is not ApprovalMode.FULL:
                return CommandResult("confirm_full", FULL_APPROVAL_RISK)
            ops.set_approval_mode(args[0])
        mode = ops.approval_mode
        label, description = APPROVAL_MODE_TEXT[mode]
        default = "，默认" if mode is ApprovalMode.AUTO else ""
        message = f"工具审批：{label}（{mode}{default}）\n{description}"
        if mode is ApprovalMode.FULL:
            message += (
                f"\n{FULL_APPROVAL_RISK}\n新建会话或重启后恢复 AUTO。"
            )
        return CommandResult(text=message)
    if name == "/new" and not args:
        session = await ops.new_session()
        return CommandResult("reload", f"新会话：{session.id}")
    if name == "/model":
        groups = ops.list_models()
        choices = tuple(model for models in groups.values() for model in models)
        if not args or args == ["list"]:
            lines = [
                f"{provider}"
                + ("" if ops.configured(provider) else "（未配置）")
                + ": "
                + ", ".join(
                    f"{choices.index(model) + 1}. {model.id}" for model in models
                )
                for provider, models in groups.items()
            ]
            return CommandResult("models", "\n".join(lines))
        if len(args) == 1:
            if args[0].isdigit():
                index = int(args[0]) - 1
                if index < 0 or index >= len(choices):
                    raise ValueError("模型序号超出范围。")
                target = choices[index]
                return CommandResult(text=await ops.set_model(target.provider, target.id))
            matches = [model for model in choices if model.id == args[0]]
            if len(matches) == 1:
                return CommandResult(text=await ops.set_model(matches[0].provider, args[0]))
            if len(matches) > 1:
                raise ValueError("模型名属于多个厂商，请使用 /model 厂商 模型ID。")
            return CommandResult(text=await ops.set_model(ops.session.provider, args[0]))
        if len(args) == 2:
            return CommandResult(text=await ops.set_model(args[0], args[1]))
    if name == "/thinking":
        if not args:
            modes = ops.model.thinking_modes
            if not modes:
                return CommandResult(
                    text="思考选项未知。"
                    if modes is None
                    else "该模型没有可配置的思考选项。"
                )
            return CommandResult(
                "thinking", "可用：default（服务端默认）、" + ", ".join(modes)
            )
        if len(args) == 1:
            await ops.set_thinking(None if args[0] == "default" else args[0])
            return CommandResult(text=f"Thinking：{ops.thinking or '服务端默认'}")
    if name == "/tools":
        if not args:
            available = ops.all_tools
            return CommandResult(
                "tools",
                "\n".join(
                    [
                        ("[启用] " if item in ops.enabled_tools else "[停用] ") + item
                        for item in available
                    ]
                    + [
                        "[离线，保留启用偏好] " + item
                        for item in ops.enabled_tools if item not in available
                    ]
                ),
            )
        if args == ["all"]:
            await ops.set_tools(ops.all_tools, keep_unavailable=True)
        elif args == ["none"]:
            await ops.set_tools(())
        elif len(args) > 1 and args[0] in ("on", "off"):
            known = (
                ops.all_tools
                if args[0] == "on"
                else (*ops.all_tools, *ops.enabled_tools)
            )
            if any(name not in known for name in args[1:]):
                raise ValueError("包含未知工具。")
            if args[0] == "on":
                await ops.set_tools(tuple(dict.fromkeys((*ops.enabled_tools, *args[1:]))))
            else:
                await ops.set_tools(
                    tuple(tool for tool in ops.enabled_tools if tool not in args[1:])
                )
        else:
            raise ValueError("用法：/tools all|none|on 工具名...|off 工具名...")
        return CommandResult(
            text=f"工具：{len(ops.effective_tools)} 个可用，"
            f"已保存 {len(ops.enabled_tools)} 个启用偏好。"
        )
    if name == "/session":
        if not args:
            items = ops.list_sessions()
            return CommandResult(
                "sessions",
                "\n".join(
                    f"{item.id}  {item.provider}/{item.model_id}  "
                    f"{item.message_count} 条  {item.workspace}"
                    for item in items
                )
                or "尚无已保存会话。",
            )
        if len(args) == 1:
            ops.restore_session(args[0])
            return CommandResult("reload", f"恢复会话：{ops.session.id}")
    raise ValueError("未知命令或参数，请输入 /help 查看用法。")
