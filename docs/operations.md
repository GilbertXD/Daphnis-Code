# 公共 Operations 接口

`Operations` 可从 `daphnis` 导入。多个实例可引用同一个 Application，共享当前会话、配置、任务和审批状态。自行构造 `Application` 时可注入模型连接、目录和 History。交互式客户端也可使用 `daphnis.runtime.open_operations(workspace, *, session_id=None, observe_model_io=False)` 组装运行环境。

## 最小示例

以下示例从进程环境读取配置。终端和 Web 启动入口会加载 `.env`，直接调用 `environment_connections()` 只读取当前环境变量。示例显式加载启动目录的 `.env`。

```python
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from daphnis import Operations, TurnStatus
from daphnis.app import Application
from daphnis.config import environment_connections


async def main():
    load_dotenv(Path.cwd() / ".env", override=False)
    provider, model_id, connections = environment_connections()
    ops = Operations(Application(
        workspace=".",
        provider=provider,
        model_id=model_id,
        connections=connections,
    ))
    unsubscribe = ops.subscribe(
        "model.started", lambda event: print("请求", event.data["request_number"])
    )
    try:
        result = await ops.send("检查当前项目的测试入口", stream=False)
        if result.status == TurnStatus.COMPLETED:
            print(result.message)
        else:
            print(result.error)
    finally:
        unsubscribe()
        await ops.close()


asyncio.run(main())
```

Application 默认以非交互模式运行。需要人工审批的客户端可设置 `interactive=True`，并通过下表中的接口处理请求。

## 接口速查

| 用途 | 接口与行为 |
| --- | --- |
| 当前会话 | `session` 返回 ID、工作区、模型身份和可选标题。`message_snapshot()` 返回已提交消息的独立副本。 |
| 历史 | `list_sessions()` 按更新时间降序列出有效的已保存会话。`scan_sessions()` 同时返回有效会话和无法加载的文件名及错误类型。`await new_session()`、`restore_session(session_id)` 用于新建或恢复。 |
| 会话管理 | `await rename_session(session_id, title)` 保存去除首尾空白后长度为 1–120 的标题。`await delete_session(session_id)` 删除历史，删除当前会话时会选中新建的空白会话，并返回当前会话快照。 |
| 模型 | `model` 返回能力描述。`list_models()` 按厂商分组，`configured(provider)` 查询连接是否已配置。`await set_model(provider, model_id)` 保存选择并返回提示文本，也会重置不适用的思考档位或工具选择。 |
| 思考档位 | `thinking` 返回当前档位。`await set_thinking(mode)` 保存档位，传入 `None` 采用服务端默认。 |
| 工具 | `all_tools` 是可用目录，`enabled_tools` 是已启用偏好（包括已启用的离线 MCP 工具），`effective_tools` 是两者的交集。`await set_tools(names, keep_unavailable=False)` 保存选择，可用 `keep_unavailable=True` 保留未列出的离线偏好。 |
| MCP | `await connect_mcp(name, ...)`、`await reconnect_mcp(name)`、`await disconnect_mcp(name)`、`await refresh_mcp(name)` 管理显式配置的服务。`list_mcp_servers()`、`mcp_status()`、`list_mcp_tools()` 查询连接名、状态和在线工具。 |
| 执行与取消 | `await send(prompt, stream=True)` 执行一轮并返回 `TurnResult`。`running` 查询是否有活跃 Turn 或手动压缩，`await cancel()` 请求取消并等待任务停止。 |
| 上下文 | `await compact()` 在空闲时请求手动压缩，返回已保存的检查点或 `None`。`context_budget` 返回最近一次主请求的输入预算，`compacting` 和 `compacted_through` 表示压缩状态和已提交摘要覆盖到的消息数。 |
| 审批 | `approval_mode` 和 `set_approval_mode(mode)` 查询及设置模式。`pending_approval` 返回当前请求的独立副本或 `None`。`resolve_approval(request_id, decision)` 仅对匹配且仍在等待的请求返回 `True`，用 `can_trust` 判断是否可提交会话信任。 |
| 用量与事件 | `session_usage`、`turn_usage` 返回用量快照。`subscribe(name, callback)` 返回退订函数，事件详见[运行架构与事件](architecture.md#事件订阅)。 |
| 关闭 | `await close()` 关闭共享的 Application。并发关闭共享一次清理，关闭后保留只读查询。 |

运行中的配置修改和会话切换会被拒绝。新建会话沿用工作区、模型、思考档位与工具偏好，消息与用量从零开始。恢复会话使用已保存的工作区和设置，并校验模型与能力。恢复时可保留离线 MCP 工具的启用偏好。持久化格式和并发规则见[本地历史与并发](architecture.md#本地历史与并发)。

`send()` 的普通 Turn 失败由 `TurnResult.status` 与 `error` 表示，锁冲突、状态校验和保存失败等可直接抛出异常。取消时，等待该轮的 `send()` 会收到 `asyncio.CancelledError`。提交前取消会丢弃本轮工作消息和暂存摘要，已开始的磁盘写入会等待完成。

审批使用每次请求独立的 `request_id`，而工具调用 ID 由模型提供。决策为 `deny`、`allow_once` 或 `trust_session`，可从 `daphnis.tools` 导入 `ApprovalDecision` 与 `ApprovalMode`。使用范围见[审批规则](tools.md#审批规则)。
