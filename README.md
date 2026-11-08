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
