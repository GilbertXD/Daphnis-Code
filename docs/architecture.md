# 运行架构与事件

## 入口与模块职责

```text
操作
  CLI / TUI ─────────────┐
  浏览器 → Web Server / Bridge ─┴─→ Operations → Application → Agent

通知
  Agent / ToolExecutor / Application → EventBus → TUI、UsageTracker
                                               └→ Operations.subscribe → Web Bridge
```

| 模块 | 职责 |
| --- | --- |
| [CLI](../src/daphnis/cli.py)、[TUI](../src/daphnis/tui/app.py) | CLI 装配终端运行环境，按参数执行一轮或启动 Textual TUI，并可加载 MCP 配置。`terminal_commands.py` 解析斜杠命令。TUI 调用 Operations，并把 EventBus 通知转成界面消息。 |
| [Web 入口](../src/daphnis/web/__main__.py)、[运行装配](../src/daphnis/runtime.py) | Web 入口配置回环服务、静态资源、访问令牌和脱敏，委托 `open_operations()` 读取服务端配置、创建交互式 Application 并按需恢复会话。 |
| [Web Server](../src/daphnis/web/server.py)、[Bridge](../src/daphnis/web/bridge.py) | Server 校验 HTTP 命令和 WebSocket 连接。Bridge 通过公共 Operations 查询或执行操作，通过 `Operations.subscribe()` 观察事件，再投影成浏览器快照、流式文本和有界调试记录。Bridge 持有的是 Web 任务、游标和缓存，Core 状态以 Operations 为准。 |
| [Operations](../src/daphnis/operations.py) | 提供面向客户端的命令、只读视图及订阅入口。在共享运行状态上校验操作，协调会话和配置变更。 |
| [Application](../src/daphnis/app.py) | 持有当前 Session、History、按厂商缓存的客户端、工具目录、MCP 连接、审批和用量跟踪器。为每轮装配 Agent，管理任务、持久化、取消和资源清理。 |
| [Agent](../src/daphnis/agent.py)、[Context](../src/daphnis/context.py) | Agent 负责模型和工具循环及消息提交。Context 从工作消息和检查点创建请求快照，并按需加入系统提示。 |
| [Token Budget](../src/daphnis/token_budget.py)、[Compactor](../src/daphnis/compaction.py) | 估算模型请求输入与输出空间。Compactor 选择完整的工具交换边界，在请求副本中修剪输出，并将旧消息摘要为检查点。 |
| [Session](../src/daphnis/session.py)、[History](../src/daphnis/history.py) | Session 持有已提交原始消息、摘要检查点、工作区、模型、思考档位、工具偏好和用量。History 负责本地 V2 JSON、会话列表、版本校验和写入互斥。 |
| [模型适配](../src/daphnis/models)、[模型观测](../src/daphnis/model_io.py) | Daphnis 管理模型能力目录，检查工具和思考选项，并为厂商配置 Saturnus `Compatibility` 映射。Saturnus 提供 Chat Completions 通信、消息与流式响应累积、工具参数校验和注册表执行。可选的模型观测通过 Saturnus 的公开响应接口读取 HTTP 数据。 |
| [工具与审批](../src/daphnis/tools)、[MCP](../src/daphnis/mcp/__init__.py) | ToolExecutor 在调用 Saturnus 注册表执行前检查路径、权限和审批。MCP 模块通过 SDK 建立连接、发现工具、处理目录变化和调用结果，发现的工具进入同一套审批流程。 |
| [EventBus](../src/daphnis/events.py)、[Usage](../src/daphnis/usage.py) | EventBus 分发观察通知。UsageTracker 按请求记录 API 报告的用量并汇总到本轮与会话，流式的重复用量快照以该请求的最新报告为准。 |

压缩、请求额度和回查的细节见[上下文压缩与历史回查](context_compaction.md)。

## 一轮任务的生命周期

1. Application 检查运行状态，选定本轮模型、思考档位、审批模式和在线工具，然后取得会话租约。
2. Agent 创建工作消息副本，分别在加入用户输入后、每组工具调用均有结果后、取得最终回复后提交快照。常见工具失败也作为工具结果保存。
3. Application 在现有会话租约内将快照与检查点一起落盘。保存成功后，Agent 发布内存状态。取消、执行失败或保存失败时保留最近一次成功提交的完整记录。最近一次提交之后执行的工具仍可能产生外部副作用。
4. 请求预算到达收尾边界时，Agent 尽可能使用无工具请求收尾，并返回 `PAUSED_LIMIT` 与继续提示。再次 `send()` 使用该会话已保存的历史和新一轮预算。失败或取消时，晚于最近一次消息保存的用量也可独立保存。

### 上下文与压缩

预算以模型目录登记的窗口为准，按序列化消息、工具定义等估算输入 Token，并预留输出和估算余量。DeepSeek 的默认输出预留随思考档位变化。窗口未知时仍会报告输入估算。已知窗口接近阈值时，Agent 先在请求副本中修剪旧工具输出，再按需摘要旧历史。摘要优先保留当前用户消息和最近一次工具交换。最新工具输出本身过大时也可发送摘录，调用 ID 和结果配对仍完整。

手动 `/compact` 优先保留最近一轮。摘要作为低权限的历史数据发送，生成摘要的请求没有工具。明确的服务端上下文超限可以触发压缩，整个 Turn 最多重试一次，且要求请求确实缩短、流式输出尚未开始。摘要请求和重试计入本轮模型请求上限。

### 本地历史与并发

默认目录是 `~/.daphnis/sessions`，消息、工具结果、思考档位和工具偏好保存在 V2 JSON 中。可选的压缩检查点与原始消息在同一个 JSON 中原子保存。旧会话缺少检查点时使用完整历史。CLI、TUI 与 Web 使用同一目录。History 在同目录写临时文件后替换目标 JSON，并以稳定的 POSIX 锁文件协调 Daphnis 进程。

租约覆盖模型请求、工具调用、人工审批、手动压缩和保存。同会话的锁已被占用时会立即抛出 `SessionBusyError`。旧快照会在请求模型前触发 `SessionConflictError`，需重新加载后再继续。不同会话可以并行。保存和设置写入在线程中执行，取消会等待已开始的写入完成后再释放租约。锁只协调遵守同一机制的进程。

## 事件订阅

通过 `ops.subscribe(name, callback)` 按事件名订阅，返回值用于退订。每个监听器收到独立的 `Event(name, session_id, turn_id, data)` 快照。同步监听器在发布时内联运行，应快速返回。返回可等待对象的监听器由独立任务处理，发布方继续执行。监听器错误会记入日志。关闭 Application 时移除订阅，并给已分发的异步监听器最多 0.5 秒，之后请求取消。

| 事件 | 数据与时机 |
| --- | --- |
| `turn.started`、`turn.finished` | Agent 开始和结束本轮。结束事件包含状态、暂停提示或错误、请求及工具调用计数和可能的外部副作用。消息提交在结束事件前完成。 |
| `model.started`、`model.delta`、`model.usage`、`model.finished` | 以从 1 开始的 `request_number` 标识本轮内的请求，分别提供请求身份、流式分片、API 用量及解析后的模型响应。 |
| `model.request`、`model.response_error` | 启用 `observe_model_io` 时，根据可用的 HTTP 响应观测实际请求 JSON、HTTP 状态及错误响应正文。Web 启动启用此选项。 |
| `context.budget`、`context.overflow` | 主模型请求的输入估算、窗口、输出预留和阈值，以及明确超限后的恢复决定。 |
| `compaction.started`、`compaction.finished` | 手动、压力或超限触发的压缩尝试及结果。摘要模型请求以 `purpose=compaction` 标识。 |
| `compaction.committed`、`compaction.discarded` | 自动检查点随完整的消息快照提交，或在尚未提交时随本轮结束丢弃。 |
| `tool.started`、`tool.approval_requested`、`tool.blocked`、`tool.finished` | 携带请求序号、工具调用 ID 和工具名。审批事件额外提供独立的 `request_id`，工具结束事件包含状态、内容和错误。 |
| `mcp.error`、`mcp.tools_changed` | 通知连接异常或工具目录刷新，`turn_id` 为空字符串。 |

事件用于观察。审批通过 `resolve_approval()` 回传，取消通过 `cancel()` 发起。客户端应以 `send()` 的结果或异常判断整次操作的结局。Web 的 `web.send_finished` 是 Bridge 在该操作结束后生成的传输记录，Core 的订阅事件列表见上表。

Web 通过 `POST /api/command` 接收命令，通过 `/api/events` WebSocket 发送状态帧。Bridge 用会话 ID、状态版本和事件游标区分全量快照与增量记录。服务端调试记录按会话有界缓存，重连时可重新获取仍在缓存中的记录。持续连接的浏览器会在本地累积已收到的同会话事件，直到收到重置帧或切换会话。

Core 事件可能含模型请求正文、工具参数和结果。终端在渲染时过滤控制序列，Web 在浏览器边界做[凭据脱敏](tools.md#web-边界)。本地历史保留消息和工具结果原文。
