# Daphnis

## 安装与启动

需要 Python 3.11+ 和 uv。文件工具及历史写入依赖 POSIX 接口，支持 macOS 和 Linux
上的本地文件系统。Windows 与网络文件系统未验证。

```sh
git clone https://github.com/GilbertXD/Daphnis-Code.git
cd Daphnis-Code
uv sync --locked
cp .env.example .env
# 填好 .env 中的模型、服务地址和 API Key
uv run --locked daphnis
```

锁文件使用 [PyPI 上的 Saturnus 0.1.1](https://pypi.org/project/saturnus-llm/0.1.1/)。
同时开发两个仓库时，可在同步后运行 `uv pip install --no-deps -e ../Saturnus`，
再用 `uv run --no-sync daphnis` 使用本地版本。重新运行 `uv sync --locked` 会恢复发布版。

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check --no-cache .
```

## 运行结构

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
历史保存以内容修订检查旧快照，并用每会话的 POSIX 锁协调 Daphnis 进程。
写入冲突时保留磁盘上已提交的历史，重新加载会话后才能继续保存。

增量输出、工具事件和用量更新由 `app.events` 提供。客户端可以在发送前订阅，
并在离开时取消订阅；Application 关闭时也会清理订阅。
并发关闭调用共享同一次资源清理。关闭时给异步订阅者最多 0.5 秒处理已发出的事件，
超时后取消订阅任务。监听器应及时响应取消，避免阻塞事件循环。

```python
unsubscribe = app.events.on("model.delta", lambda event: print(event.data))
try:
    result = await ops.send("查看项目结构")
finally:
    unsubscribe()
```

### 工具权限

在 TUI 中用 `/approval ask|auto|full` 切换模式，当前模式显示在界面顶部：

- `ask`（每次询问）：所有可执行工具调用都需要确认。
- `auto`（安全自动，默认）：自动执行无需额外授权的操作，其余按规则确认。
- `full`（尽可能自动）：跳过常规审批，包括 MCP 调用，仍保留强制审批和硬编码禁止。
  MCP Server 可能读写文件或访问网络，Daphnis 无法约束其内部的所有行为。

审批时按 Y 允许，按 N 或 Esc 拒绝。如果支持会话信任，按 T 切换勾选，再按 Y 确认。
MCP 信任适用于单个工具，后续调用可使用不同参数。`apply_patch` 不接受会话信任。

硬编码禁止始终有效。工作区外访问、无法检查访问范围的命令及所有 `apply_patch`
写入均需逐次确认，包括在 `full` 模式下。
模式和信任均只存于本次运行的内存。新会话不会继承信任，切到 `ask` 会清除信任；
MCP 服务断线、重连或工具定义变化也会撤销相关信任。

需要审批的调用在非交互模式下会被拒绝。交互客户端可用 `interactive=True`，
通过 `ops.resolve_approval(..., decision)` 返回 `deny`、`allow_once` 或
`trust_session`（可从 `daphnis.tools` 导入 `ApprovalDecision` 和 `ApprovalMode`）。

内置文件工具阻止读取常见凭据路径，例如 `.env`、`.env.local`、私钥和
已知的凭据配置，仍可列出文件名或读取 `.env.example` 模板。规则检查解析后的路径，
不会识别放在普通文件中的所有秘密。普通命令进程会过滤已知凭据环境变量。

`run_command` 只对受信任的系统只读命令及可明确解析的参数自动放行。
`cat`、`ls`、`head`、`tail`、`wc` 的已识别路径以检查后的绝对路径执行。
外部命令仍会在启动时解析这些路径，不能防止不受信任的并发进程替换路径。
该工具不提供操作系统沙箱。获准运行的脚本拥有当前用户的文件与进程权限。
超时或取消时会尝试终止原进程组并回收直接子进程。`terminated` 表示该清理步骤完成，
脱离进程组的后代及已产生的外部副作用仍可能存在。

`apply_patch` 检查工作区和父目录身份，通过无覆盖安装保存并发版本。
检测到冲突时会失败，不能原位恢复的版本会保留在工具报告的暂存位置。
替换已有文件时，原路径可能短暂缺失。进程异常退出留下的
`.daphnis-*.tmp/original` 可用于恢复。
POSIX 路径检查和安装之间仍有竞态。对不受信任的同用户进程，目录隔离需要操作系统沙箱。

### MCP 工具

用 `--mcp-config` 显式指定要连接的服务。只配置你信任的 stdio 程序，它们会以
当前用户的权限启动。配置示例中的 `env` 将子进程变量名映射到本地变量名，
相对 `cwd` 以配置文件所在目录为准。
MCP Server 自行控制内部执行，Daphnis 的审批约束的是是否发起调用。

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
掉线、主动断开、重连和目录变化都会保留会话偏好。可在 `/tools` 中取消启用离线工具。
新 Turn 只提供已启用且在线的工具，目录新增的工具可手动启用。
恢复会话不会自动连接服务。文本和结构化结果回传给模型前共用 64 KiB 的 UTF-8
JSON 预算，超限结果会标明被省略的字段。预算在 MCP SDK 解码后生效。

Python 客户端可调用 `ops.connect_mcp(...)`、`ops.mcp_status()`、
`ops.reconnect_mcp(name)`、`ops.disconnect_mcp(name)` 和 `ops.refresh_mcp(name)`。
`ops.all_tools`、`ops.enabled_tools` 和 `ops.effective_tools` 分别表示可用工具、
会话启用偏好，以及当前可提供给下一轮的工具。
