# 运行架构与事件

## 入口与模块职责

```text
操作
  浏览器标签页 → Web Server / Bridge → Operations(session_id) → Application
                                                             ├→ SessionRuntime A → Agent（每轮创建）
                                                             └→ SessionRuntime B → Agent（每轮创建）

通知
  Agent / ToolExecutor / Application → EventBus → 各会话的用量和上下文状态
                                               └→ Operations.subscribe → Web Bridge
```

| 模块 | 职责 |
| --- | --- |
| [Web 入口](../src/daphnis/web/__main__.py)、[运行装配](../src/daphnis/runtime.py) | Web 入口配置回环服务、静态资源、访问令牌和脱敏，委托 `open_operations()` 读取服务端配置、创建交互式 Application、按需恢复会话并连接已配置的 MCP 服务。 |
| [Web Server](../src/daphnis/web/server.py)、[Bridge](../src/daphnis/web/bridge.py) | Server 校验 HTTP 命令和 WebSocket 连接。Bridge 按会话维护 Web Run、流式文本、提交确认和有界调试记录，通过 Operations 查询当前状态。每条 WebSocket 独立选择要显示的会话。 |
| [Operations](../src/daphnis/operations.py) | 提供命令、只读快照及订阅入口。会话操作通过显式会话 ID 定位 Runtime；MCP 连接和目录管理面向整个 Application。 |
| [Application](../src/daphnis/app.py) | 按 ID 管理已加载的 Runtime。共享 History、EventBus、模型目录、ChatClient 和 DocumentAdapter 缓存、工具目录、MCP 连接与配置。负责装配 Agent、提交事务、取消和释放资源。 |
| [SessionRuntime](../src/daphnis/session_runtime.py) | 保存一个 Session 的活动任务、审批请求与模式、用量跟踪器、上下文预算和压缩状态。空闲 Runtime 保留在内存中；Runtime 本身不落盘。 |
| [Agent](../src/daphnis/agent.py)、[Context](../src/daphnis/context.py) | Agent 负责模型和工具循环及消息提交。Context 从工作消息和检查点创建请求快照，并按需加入系统提示。 |
| [Token Budget](../src/daphnis/token_budget.py)、[Compactor](../src/daphnis/compaction.py) | 估算模型请求输入与输出空间。Compactor 选择完整的工具交换边界，在请求副本中修剪输出，并将旧消息摘要为检查点。 |
| [Session](../src/daphnis/session.py)、[History](../src/daphnis/history.py) | Session 持有已提交原始消息、摘要检查点、工作区、模型、思考档位、工具偏好和用量。History 负责本地 V2 JSON、会话列表、版本校验和写入互斥。 |
| [模型适配](../src/daphnis/models)、[模型观测](../src/daphnis/model_io.py) | Daphnis 管理模型能力目录，检查工具和思考选项，并为厂商配置 Saturnus `Compatibility` 映射。Saturnus 提供 Chat Completions 通信、消息与流式响应累积、工具参数校验和注册表执行。可选的模型观测通过 Saturnus 的公开响应接口读取 HTTP 数据。 |
| [工具与审批](../src/daphnis/tools)、[MCP](../src/daphnis/mcp/__init__.py) | ToolExecutor 在调用 Saturnus 注册表执行前检查路径、权限和审批。MCP 模块通过 SDK 建立连接、发现工具、处理目录变化和调用结果，发现的工具进入同一套审批流程。 |
| [EventBus](../src/daphnis/events.py)、[Usage](../src/daphnis/usage.py) | EventBus 分发观察通知。UsageTracker 按请求记录 API 报告的用量并汇总到本轮与会话，流式的重复用量快照以该请求的最新报告为准。 |

压缩、请求额度和回查的细节见[上下文压缩与历史回查](context_compaction.md)。

## 一轮任务的生命周期

1. Application 检查目标 Runtime，选定本轮模型、思考档位、审批模式和在线工具。在第一次等待前登记该任务，然后取得目标会话的 History 租约。
2. Agent 提交用户输入。收到模型的工具调用后，Agent 先校验响应和调用数量，再将 Assistant 消息与整组占位结果一次提交。每次工具开始执行和返回结果时，Agent 分别提交对应状态。最终回复也单独提交。
3. Application 在会话租约内将快照、检查点与用量一起落盘。保存成功后，Agent 发布内存状态。取消或保存失败时，最近一次成功提交的状态仍可恢复。
4. 最终 Assistant 回复经校验并成功持久化后，Turn 才返回 `COMPLETED`。流式正文与思考在这之前都是临时显示。取消与最终保存重合时，以落盘结果为准。请求额度耗尽后仍会处理最后一次响应。若尚无最终回复，返回 `PAUSED_LIMIT` 并保留恢复入口。失败或取消时，晚于最近一次消息保存的用量也可独立保存。

### 工具状态与中断恢复

History 允许最后一个工具组含占位。每条占位是带 `tool_call_id` 的工具消息，通过 ID 关联同组 Assistant 消息中的调用。扩展字段 `daphnis_tool_state` 存放 `PENDING` 或 `RUNNING`，占位的 `content` 为 null。保存真实结果时，用同一位置的标准工具消息替换占位。标准结果消息隐式表示 `FINISHED`，旧 V2 会话可直接读取。

ToolExecutor 先校验参数、准备操作、处理审批并核验目标，然后等待 Agent 的执行前回调。回调将该调用保存为 `RUNNING`，之后 ToolExecutor 才能进入实际操作。内置与 MCP 工具共用屏障，工具入口继续核验工作区和目标。审批拒绝、参数错误等可直接从 `PENDING` 保存为错误结果。关键保存失败会结束当前轮，已持久化的状态保留在原位置。

重载会校验末尾工具组，并从最后一条已提交消息推导恢复状态。只有原 User 已保存时，可重试原消息。发送新消息会在同一次保存中把旧 User 标记为 `ABANDONED`。原文和索引保留在 History，Working、摘要与模型可用的历史回查会排除这条消息。

末尾存在工具事实时，可以显式选择继续或终止。Agent 在会话租约内把 `CONTINUE` 或 `TERMINATED` 决策写入原 User 的元数据，并将遗留的 `RUNNING` 转成 `execution_result_unknown`。继续时，组内没有结果未知的调用才会按当前规则处理 `PENDING`。组内出现未知结果或用户选择终止时，剩余 `PENDING` 以 `execution_skipped` 补齐。已保存的结果始终复用。继续从完整的工具交换发起新模型请求，终止只保存决策与工具事实。恢复操作使用关联到原 User 和当前尝试的键，避免重复或过期操作生效。

Working 由当前 History 和检查点投影生成。模型请求保留合法工具交换的顺序，并移除 Daphnis 内部元数据。系统提示词覆盖值保存在 Session 中，默认值由内置配置提供。工具审批与硬编码限制由执行层独立实施。

从保存 `RUNNING` 到保存结果期间，进程可能在实际调用之前、调用期间或返回之后中断。这三个窗口都按结果未知处理。已产生的外部副作用需要核对，History 原子替换只保护会话文件。

### 上下文与压缩

预算以模型目录登记的窗口为准，按序列化消息、工具定义等估算输入 Token，并预留输出和估算余量。DeepSeek 的默认输出预留随思考档位变化。窗口未知时仍会报告输入估算。已知窗口接近阈值时，Agent 先在请求副本中修剪旧工具输出，再按需摘要旧历史。摘要优先保留当前用户消息和最近一次工具交换。最新工具输出本身过大时也可发送摘录，调用 ID 和结果配对仍完整。

通过 `Operations.compact(session_id)` 手动压缩时，优先保留最近一轮。摘要作为低权限的历史数据发送，生成摘要的请求没有工具。明确的服务端上下文超限可以触发压缩，整个 Turn 最多重试一次，且要求请求确实缩短、流式输出尚未开始。摘要请求和重试计入本轮模型请求上限。

### 本地历史与并发

默认目录是 `~/.daphnis/sessions`，消息、工具结果、思考档位和工具偏好保存在 V2 JSON 中。可选的压缩检查点与原始消息在同一个 JSON 中原子保存。旧会话缺少检查点时使用完整历史。History 在同目录写临时文件后替换目标 JSON，并以稳定的 POSIX 锁文件协调 Daphnis 进程。

文本附件正文独立保存在 `~/.daphnis/attachments/<session-id>/<attachment-id>.txt`，对应的 `attachment_refs` 按原始 User 消息索引保存文件名、大小、行数与 SHA-256。自定义 History 目录会派生相邻的专用附件目录。Application 持有会话租约时，先原子发布并校验所有文件快照，再提交引用它们的 User 消息。Web 收到提交确认后才清理对应的待发送内容；相同发送标识与相同内容的重试复用已有确认。删除会话时在同一租约内清理快照。中断时尚未关联的快照也随对应会话清理。

Agent 只在模型请求投影中加入附件。当前轮的附件按合计约 4,000 Token、已知可用空间的 5% 以及输出预留判断能否内联，超过时使用元数据和工具引用。后续轮次及历史摘要只自动带入旧附件的元数据，附件工具已实际读取的内容仍作为普通工具结果进入压缩。失败重试和中断继续读取已保存的快照、重新核算预算，不创建新副本。Web 历史只展示附件标签，对有关会话默认隐藏模型请求详情与附件工具正文。

每个会话同时只能执行一个 Turn、恢复任务或手动压缩。会话设置锁也只保护目标 Runtime。保存绑定启动任务时的会话，切换浏览器页面不会转移写入、用量或审批。关闭 Application 时，先取消并等待所有会话任务和已开始的写入，再关闭共享资源。

History 租约覆盖模型请求、工具调用、人工审批、手动压缩和保存。同会话的租约已被其他进程占用时会抛出 `SessionBusyError`。旧快照会在请求模型前触发 `SessionConflictError`；在新的 Application 实例中加载该会话即可获取新快照。取消会等待已开始的写入完成。会话租约不协调不同会话对同一工作区文件的写入。

MCP 目录与连接由会话共享。每轮根据已启用且在线的工具选取定义快照；执行时仍检查连接。连接或发现工具不修改任何会话的启用偏好。活动任务期间拒绝主动修改全局 MCP 配置。目录变化立即撤销相关的自动信任，等活动任务结束后更新目录；刷新失败时保留旧目录，重新验证前不会记住该连接的新信任决定。

## 事件订阅

通过 `ops.subscribe(name, callback)` 按事件名订阅，返回值用于退订。每个监听器收到独立的 `Event(name, session_id, turn_id, data)` 快照。同步监听器在发布时内联运行，应快速返回。返回可等待对象的监听器由独立任务处理，发布方继续执行。监听器错误会记入日志。关闭 Application 时移除订阅，并给已分发的异步监听器最多 0.5 秒，之后请求取消。

| 事件 | 数据与时机 |
| --- | --- |
| `turn.started`、`turn.finished` | Agent 开始和结束本轮。结束事件包含状态、暂停提示或错误、请求及工具调用计数和可能的外部副作用。消息提交在结束事件前完成。 |
| `model.started`、`model.delta`、`model.usage`、`model.finished` | 以从 1 开始的 `request_number` 标识本轮内的请求，分别提供请求身份、流式分片、API 用量及解析后的模型响应。 |
| `model.request`、`model.response_error` | 保留实际请求 JSON、HTTP 状态及错误响应正文的订阅、传输和展示协议。当前暂不产生这两类事件，`observe_model_io` 参数仅用于调用兼容。 |
| `context.budget`、`context.overflow` | 主模型请求的输入估算、窗口、输出预留和阈值，以及明确超限后的恢复决定。 |
| `compaction.started`、`compaction.finished` | 手动、压力或超限触发的压缩尝试及结果。摘要模型请求以 `purpose=compaction` 标识。 |
| `compaction.committed`、`compaction.discarded` | 自动检查点随下一次消息快照保存，或在尚未提交时随本轮结束丢弃。 |
| `tool.started` | 占位已经保存，开始处理调用。携带工具名、调用 ID、请求序号及参数。恢复已有调用时请求序号为 0。 |
| `tool.approval_requested`、`tool.blocked` | 通知审批请求及策略拦截。审批事件包含独立的 `request_id`。 |
| `tool.executing` | `RUNNING` 已保存，即将执行工具。 |
| `tool.finished` | 真实结果或可报告的工具错误已保存。事件携带 `COMPLETED` 或 `ERROR`、结果正文和错误详情。 |
| `mcp.error`、`mcp.tools_changed` | Application 级别的连接异常或目录刷新；`session_id`、`turn_id` 均为空字符串。 |

事件用于观察，工具状态由 Agent 通过 Session 保存接口直接提交。异步订阅器可能以不同顺序完成。审批通过 `resolve_approval(request_id, decision, session_id=...)` 回传，取消通过 `cancel(session_id)` 发起。客户端应以 `send()` 的结果或异常判断操作结局。Web 的 `web.send_finished` 是 Bridge 在操作结束后生成的传输记录。

Web 通过 `POST /api/command` 接收带会话 ID 的命令。停止还要匹配 Run ID，审批还要匹配 Request ID。标签页通过 `/api/events` WebSocket 选择会话；初次连接和重连可在查询参数提供会话 ID，也可发送 `{"session_id": "..."}` 切换。Bridge 按会话记录版本和事件游标，共享目录变化可单独发送。切换或重连时重新获取所选会话的快照、活动 Run 和仍在缓存中的调试记录。

Core 事件可能含模型请求正文、工具参数和结果。Web 在浏览器边界做[凭据脱敏](tools.md#web-边界)。本地历史保留消息和工具结果原文。
