"""Translate terminal commands to operations and presentation results."""

import os
import re
import shlex
from dataclasses import dataclass

from .operations import Operations
from .tools import ApprovalMode

APPROVAL_MODE_TEXT = {
    ApprovalMode.ASK: ("每次询问", "所有可执行工具调用都需要确认。"),
    ApprovalMode.AUTO: ("安全自动", "自动执行无需额外授权的操作，其余按规则确认。"),
    ApprovalMode.FULL: ("尽可能自动", "跳过常规审批，保留强制审批和硬编码禁止。"),
}

_TERMINAL_CONTROLS = re.compile(
    r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|\x9c|\Z)"  # OSC
    r"|(?:\x1b[PX^_]|[\x90\x98\x9e\x9f]).*?(?:\x1b\\|\x9c|\Z)"  # Control strings
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?"  # CSI, including incomplete sequences
    r"|\x1b[ -/]*[0-~]"
    r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]",
    re.DOTALL,
)


def sanitize_terminal(text: str) -> str:
    """Remove terminal commands while preserving printable text, tabs and LF."""
    return _TERMINAL_CONTROLS.sub("", text)


HELP = """/model [list|序号|模型ID|厂商 模型ID]  查看或切换模型（按厂商分列）
/thinking [default|档位]  查看或设置思考；default 为服务端默认
/tools [all|none|on 工具名...|off 工具名...]  查看或选择工具
/approval [ask|auto|full]  查看或切换工具审批模式
/mcp [reconnect|disconnect|refresh Server名]  查看 MCP 状态或管理连接
/session [会话ID]  列出或恢复会话
/new  新建会话，保留旧会话
/clear  仅清理屏幕，保留历史
/help  显示帮助
Enter 发送；Shift+Enter 换行；Alt+Enter / F2 为换行备用键。
Esc 中断当前轮；Ctrl+C 运行时取消、空闲时退出；Ctrl+Q 取消并退出。"""


@dataclass(frozen=True)
class CommandResult:
    action: str = "message"
    text: str = ""


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


async def terminal_command(ops: Operations, text: str) -> CommandResult:
    words = shlex.split(text)
    name, *args = words or [""]
    if name == "/help" and not args:
        return CommandResult(text=HELP)
    if name == "/clear" and not args:
        return CommandResult("clear", "已清理屏幕；会话历史保留。")
    if name == "/mcp":
        if not args:
            return CommandResult(
                text="\n".join(
                    f"{server.name}  {server.state}  {server.tool_count} 个工具"
                    + (f" · {server.error}" if server.error else "")
                    for server in ops.mcp_status()
                )
                or "尚未配置 MCP Server。"
            )
        if len(args) == 2 and args[0] == "reconnect":
            await ops.reconnect_mcp(args[1])
            return CommandResult(text=f"MCP {args[1]} 已连接。")
        if len(args) == 2 and args[0] == "disconnect":
            await ops.disconnect_mcp(args[1])
            return CommandResult(text=f"MCP {args[1]} 已断开。")
        if len(args) == 2 and args[0] == "refresh":
            await ops.refresh_mcp(args[1])
            return CommandResult(text=f"MCP {args[1]} 工具目录已刷新。")
        raise ValueError("用法：/mcp [reconnect|disconnect|refresh Server名]")
    if ops.running:
        raise RuntimeError("请先等待或取消当前 Turn，再执行该操作。")
    if name == "/approval":
        if args:
            if len(args) != 1 or args[0] not in tuple(ApprovalMode):
                raise ValueError("用法：/approval [ask|auto|full]")
            ops.set_approval_mode(args[0])
        mode = ops.approval_mode
        label, description = APPROVAL_MODE_TEXT[mode]
        default = "，默认" if mode is ApprovalMode.AUTO else ""
        message = f"工具审批：{label}（{mode}{default}）\n{description}"
        if mode is ApprovalMode.FULL:
            message += (
                "\nMCP Server 可能写入文件或访问网络，"
                "Daphnis 无法约束其内部的所有行为。仅本次运行有效。"
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
