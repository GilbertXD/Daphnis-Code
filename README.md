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
| 运行与审批 | `await send(prompt, stream=True)`、`await cancel()`、`resolve_approval(turn_id, tool_call_id, approved)`、`await close()` |
| 只读查询 | `session`（ID、工作区和模型身份）、`model`（能力）、`thinking`、`all_tools`、`enabled_tools`、`running`、`pending_approval`、`session_usage`、`turn_usage` |

`set_thinking(None)` 使用服务端默认设置。消息和待审批详情以独立快照返回。
活跃 Turn 期间不能切换模型、工具、思考档位或会话。`await ops.cancel()` 会
停止当前 Turn，等待该轮的 `send()` 会收到 `asyncio.CancelledError`。
`await ops.close()` 会关闭共享的 Application；关闭后仍可读取状态，但不能再
发送或修改设置。

Thinking 和工具选择随会话保存。新会话沿用当前的模型和设置，消息与用量从零开始；
恢复会话会使用它自己的设置。历史读取要求 V2 格式。如果模型或工具配置已不受支持，
恢复会报错并保留当前会话。

增量输出、工具事件和用量更新由 `app.events` 提供。客户端可以在发送前订阅，
并在离开时取消订阅；Application 关闭时也会清理订阅。

```python
unsubscribe = app.events.on("model.delta", lambda event: print(event.data))
try:
    result = await ops.send("查看项目结构")
finally:
    unsubscribe()
```

默认情况下，需要审批的工具操作会被拒绝。只有客户端能展示审批请求并调用
`ops.resolve_approval(...)` 时，才应以 `interactive=True` 构造 Application。

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

用 `/tools` 启用工具，每次 MCP 调用仍需审批。掉线和目录变化会保留会话偏好，
新 Turn 只提供已启用且当前可用的工具。主动断开会清除当前会话中该服务的工具选择。
恢复使用 MCP 工具的会话前，先连接对应的服务。工具结果支持文本和结构化数据。

Python 客户端可调用 `ops.connect_mcp(...)`、`ops.mcp_status()`、
`ops.reconnect_mcp(name)`、`ops.disconnect_mcp(name)` 和 `ops.refresh_mcp(name)`。
`ops.all_tools`、`ops.enabled_tools` 和 `ops.effective_tools` 分别表示可用工具、
会话启用偏好，以及当前可提供给下一轮的工具。
