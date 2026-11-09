# Daphnis

Daphnis 是基于 Saturnus 和 Textual 的终端编程助手。启动入口负责加载
`.env`，`daphnis.config.environment_connections()` 只读取当前环境变量并返回
`(provider, model_id, connections)`。CLI 和 TUI 通过同一套接口操作运行环境：

```text
CLI / TUI ──→ Operations ──→ Application ──→ Agent ──→ Saturnus / Tools
    ↑                              │
    └────────── EventBus ──────────┘
```

Application 持有 Session、历史、按厂商缓存的 ChatClient、工具注册表、审批、
用量和任务。Operations 在这份运行状态上校验并执行用户操作；多个 Operations
可以引用同一个 Application。每轮由 Application 创建一个 Agent，Agent 负责
执行 Turn，并在成功时提交消息。斜杠命令由终端解析，`/clear` 只影响显示。

在已有环境变量的程序中，可以直接用 Python API 发送一轮：

```python
import asyncio

from daphnis import Operations, TurnStatus
from daphnis.app import Application
from daphnis.config import environment_connections


async def main():
    provider, model_id, connections = environment_connections()
    app = Application(
        workspace=".",
        provider=provider,
        model_id=model_id,
        connections=connections,
    )
    ops = Operations(app)
    try:
        if "high" in (ops.model.thinking_modes or ()):
            ops.set_thinking("high")
        result = await ops.send("分析当前项目", stream=False)
        if result.status == TurnStatus.COMPLETED:
            print(result.message)
        else:
            print(result.error)
    finally:
        await ops.close()


asyncio.run(main())
```

| 用途 | Operations 接口 |
| --- | --- |
| 模型与工具 | `list_models()`（按厂商分组）、`configured(provider)`、`set_model(provider, model_id)`、`set_thinking(mode)`、`set_tools(names)` |
| 会话 | `list_sessions()`、`new_session()`、`restore_session(session_id)`、`message_snapshot()` |
| 运行与审批 | `await send(prompt, stream=True)`、`await cancel()`、`set_approval_mode(mode)`、`resolve_approval(turn_id, tool_call_id, decision)`、`await close()` |
| 只读查询 | `session`（ID、工作区和模型身份）、`model`（能力）、`thinking`、`all_tools`、`enabled_tools`、`effective_tools`、`approval_mode`、`running`、`pending_approval`、`session_usage`、`turn_usage` |

`set_thinking(None)` 使用服务端默认设置。消息和待审批详情以独立快照返回。
活跃 Turn 期间不能切换模型、工具、思考档位、权限模式或会话。`await ops.cancel()` 会
停止当前 Turn，等待该轮的 `send()` 会收到 `asyncio.CancelledError`。
`await ops.close()` 会关闭共享的 Application；关闭后仍可读取状态，但不能再
发送或修改设置。

Thinking 和工具选择随会话保存。新会话沿用当前的模型和设置，消息与用量从零开始；
恢复会话会使用它自己的设置。历史读取要求 V2 格式；恢复时 MCP 工具可以暂时离线。

增量输出、工具事件和用量更新由 `app.events` 提供。客户端可以在发送前订阅，
并在离开时取消订阅；Application 关闭时也会清理订阅。

```python
unsubscribe = app.events.on("model.delta", lambda event: print(event.data))
try:
    result = await ops.send("查看项目结构")
finally:
    unsubscribe()
```

### 工具权限

在 TUI 中用 `/approval ask|auto|full` 切换模式，当前模式显示在界面顶部：

- `ask`：逐次审批，包括普通只读工具。
- `auto`（默认）：按固定规则批准普通操作，MCP 调用默认询问。审批时按 Y 仅允许本次，
  按 N 或 Esc 拒绝，按 T 信任当前会话中的这一工具（仅在可授予信任时显示）。
- `full`：跳过一般审批，包括 MCP 调用。MCP Server 可能读写文件或访问网络，
  Daphnis 无法约束其内部的所有行为。

硬编码禁止始终有效；工作区外访问、无法检查访问范围的命令仍需逐次确认。
模式和信任均只存于本次运行的内存。新会话不会继承信任，切到 `ask` 会清除信任；
MCP 服务断线、重连或工具定义变化也会撤销相关信任。

需要审批的调用在非交互模式下会被拒绝。交互客户端可用 `interactive=True`，
通过 `ops.resolve_approval(..., decision)` 返回 `deny`、`allow_once` 或
`trust_session`（可从 `daphnis.tools` 导入 `ApprovalDecision` 和 `ApprovalMode`）。

### MCP 工具

用 `--mcp-config` 显式指定要连接的服务。只配置你信任的 stdio 程序，它们会以
当前用户的权限启动。配置示例中的 `env` 将子进程变量名映射到本地变量名，
相对 `cwd` 以配置文件所在目录为准。

```json
{
  "servers": {
    "demo": {
      "transport": "stdio",
      "command": "python",
      "args": ["server.py"],
      "cwd": "./examples",
      "env": {"SERVICE_API_KEY": "LOCAL_SERVICE_KEY"}
    },
    "remote": {
      "transport": "streamable_http",
      "url": "https://example.com/mcp",
      "auth": {"type": "bearer", "env": "REMOTE_MCP_TOKEN"}
    }
  }
}
```

```sh
uv run daphnis --mcp-config ./mcp.json
```

HTTP 服务使用最终端点地址，远程连接要求 HTTPS，本机回环地址可用 HTTP。
`auth` 可省略，需要认证时使用从指定环境变量读取的 Bearer Token。

`/mcp` 显示服务状态、可用工具数和错误信息。用 `/mcp reconnect <名称>` 手动重连，
`/mcp disconnect <名称>` 主动断开，`/mcp refresh <名称>` 刷新工具目录。
服务支持工具变更通知时，目录会自动刷新，运行中的 Turn 继续使用开始时的快照。

在支持工具的模型下，首次显式连接新服务会默认启用发现的工具；从历史恢复的会话
及由它新建的会话优先使用保存的选择。可用 `/tools` 手动调整，启用后仍按当前权限模式审批。
掉线、主动断开、重连和目录变化都会保留会话偏好。新 Turn 只提供已启用且在线的工具，
目录新增的工具可手动启用。恢复会话不会自动连接服务。工具结果支持文本和结构化数据。

Python 客户端可调用 `ops.connect_mcp(...)`、`ops.mcp_status()`、
`ops.reconnect_mcp(name)`、`ops.disconnect_mcp(name)` 和 `ops.refresh_mcp(name)`。
`ops.all_tools`、`ops.enabled_tools` 和 `ops.effective_tools` 分别表示可用工具、
会话启用偏好，以及当前可提供给下一轮的工具。
