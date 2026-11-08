"""Translate terminal commands to operations and presentation results."""

import os
import shlex
from dataclasses import dataclass

from .operations import Operations

HELP = """/model [list|序号|模型ID|厂商 模型ID]  查看或切换模型（按厂商分列）
/thinking [default|档位]  查看或设置思考；default 为服务端默认
/tools [all|none|on 工具名...|off 工具名...]  查看或选择工具
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


def terminal_command(ops: Operations, text: str) -> CommandResult:
    words = shlex.split(text)
    name, *args = words or [""]
    if name == "/help" and not args:
        return CommandResult(text=HELP)
    if name == "/clear" and not args:
        return CommandResult("clear", "已清理屏幕；会话历史保留。")
    if ops.running:
        raise RuntimeError("请先等待或取消当前 Turn，再执行该操作。")
    if name == "/new" and not args:
        session = ops.new_session()
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
                return CommandResult(text=ops.set_model(target.provider, target.id))
            matches = [model for model in choices if model.id == args[0]]
            if len(matches) == 1:
                return CommandResult(text=ops.set_model(matches[0].provider, args[0]))
            if len(matches) > 1:
                raise ValueError("模型名属于多个厂商，请使用 /model 厂商 模型ID。")
            return CommandResult(text=ops.set_model(ops.session.provider, args[0]))
        if len(args) == 2:
            return CommandResult(text=ops.set_model(args[0], args[1]))
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
            ops.set_thinking(None if args[0] == "default" else args[0])
            return CommandResult(text=f"Thinking：{ops.thinking or '服务端默认'}")
    if name == "/tools":
        if not args:
            return CommandResult(
                "tools",
                "\n".join(
                    ("[启用] " if item in ops.enabled_tools else "[停用] ") + item
                    for item in ops.all_tools
                ),
            )
        if args == ["all"]:
            ops.set_tools(ops.all_tools)
        elif args == ["none"]:
            ops.set_tools(())
        elif len(args) > 1 and args[0] in ("on", "off"):
            if any(name not in ops.all_tools for name in args[1:]):
                raise ValueError("包含未知工具。")
            chosen = set(ops.enabled_tools)
            if args[0] == "on":
                chosen.update(args[1:])
            else:
                chosen.difference_update(args[1:])
            ops.set_tools(tuple(name for name in ops.all_tools if name in chosen))
        else:
            raise ValueError("用法：/tools all|none|on 工具名...|off 工具名...")
        return CommandResult(
            text=f"启用工具：{len(ops.enabled_tools)} 个（逐次审批仍生效）。"
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
