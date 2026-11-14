# 上下文压缩与历史回查

## 设计与职责

Daphnis 将模型请求控制在可用上下文内，同时保留会话原始记录。Agent 负责触发压缩、管理请求额度、处理上下文超限并提交本轮结果。Context 构造有效请求，Compactor 选择合法切分点、修剪请求副本并生成摘要。Session 持有已提交的原始消息和检查点，History 负责本地持久化与重载。模型通信、消息容器和工具注册表由 Saturnus 提供。

## 从原始历史到模型请求

每轮从 Session 的已提交历史创建工作副本，加入本轮用户消息、模型响应和工具结果。Context 使用工作历史与当前检查点构造本次模型请求。检查点记录被压缩前缀的边界 `through`，以及含 `goals`、`progress`、`files`、`next_steps` 的四字段 JSON 摘要。

摘要请求要求返回单个 JSON 对象。MiMo 和 DeepSeek 的默认接入使用 JSON 模式。解析时允许前后空白和可选的 Markdown 围栏。四个字段必须齐全且唯一，列表和对象转成 JSON 文本，额外字段归入 `progress`。整次摘要最多为 JSON 无效、字段校验失败或输出截断重试一次。重试沿用同一批历史和上一份有效摘要，并计入维护额度。

有效上下文包含原有系统和开发者消息、低权限的历史摘要，以及边界后的消息。当切分点位于用户轮次内部时，投影会按需带回这轮原始用户消息。请求视图中的工具结果可以显示为预览，原始历史仍保留完整内容。

## Token 预算

`TokenCounter` 负责本地文本计数。请求预算计入消息结构、聊天封装余量、工具定义和响应格式。窗口已知时，发送前检查估算输入与输出预留之和是否超过窗口的 95%。输出预留优先使用实际发送的显式输出上限，其次使用已知的模型或厂商默认值，最后用通用估算。

窗口可按部署覆盖模型目录值。窗口未知时，固定 Token 阈值和服务端超限恢复仍可用。通过相同请求编号关联 `context.budget.input_tokens` 与 `model.usage.usage.prompt_tokens` 可以观察本地估算与服务端报告的偏差。

## 配置阈值与部署限制

在 `.env` 中设置 `DAPHNIS_CONFIG=daphnis.toml`，CLI 和 Web 启动时会读取这份 TOML。默认开启自动摘要，按窗口 95% 触发，优先保留近期 4,096 Token。

```toml
[compaction]
threshold_mode = "percent"
threshold_value = 0.75
recent_tokens = 4096

[models."mimo:mimo-v2.6-flash"]
output_limit = 8192

[models."mimo:mimo-v2.6-flash".compaction]
threshold_mode = "tokens"
threshold_value = 131072

[models."compatible:proxy-model-id"]
context_window = 64000
```

阈值针对输入与输出预留之和，最多为已知窗口的 95%。模型覆盖继承全局设置，切换阈值模式时要同时设置 `threshold_value`。示例中的 MiMo 输入达到 122,880 Token 时触发。若阈值不高于输出预留，Agent 只按安全上限检查。`auto_enabled = false` 关闭自动生成摘要，安全修剪和手动压缩仍可用。`output_limit` 随主请求发送，摘要使用独立的输出上限和窗口预算。

## 自动压缩与手动压缩

- **自动压缩**在主请求达到阈值或超过安全上限时先修剪旧工具结果，再按需生成摘要。压力持续时，同一轮可以多次推进检查点。若修剪和摘要后仍超过安全上限，Agent 在发送前停止。
- **手动压缩**在空闲时使用 `/compact`，或调用 `await ops.compact()`。手动和自动压缩都会估算候选摘要的可用空间。只有候选缩短请求时才接受，额度允许时可按更短的目标再试一次。
- **切分**优先按 Token 目标保留近期完整用户轮次。有效空间不足时降低保留量，最少留下最新独立消息或完整工具交换。并行工具调用及对应结果作为完整交换处理。
- **摘要分块**先尝试装入完整来源字段。字段过长时按行边界和字符范围分块，每段带原消息索引、角色、字段、范围和工具关联 ID。结构化字段以 JSON 文本编码。摘要请求依次承接上一个摘要，所有必要分块完成后才生成检查点。
- **超限恢复**只处理明确的上下文超限。Agent 尝试缩小请求，且仅在仍有额度、请求确实缩小、流式响应尚未开始输出可见内容时重试一次。已有工具结果随工作历史继续使用。

## 请求额度

| `Agent` 和 `Application` 参数 | 默认值 | 范围 |
| --- | ---: | --- |
| `max_primary_requests` | 20 | 每轮主模型请求，包括超限重试 |
| `max_maintenance_requests` | 12 | 每轮摘要请求，手动压缩单独计数 |
| `max_total_requests` | 32 | 每轮或单次手动压缩内的全部模型请求 |
| `recent_tokens` | 4096 | 近期上下文的保留目标，可按可用空间下调 |

主请求和维护请求分别计数。请求通过同一个总额度，并在本轮内共用从 1 开始的 `request_number`。自动维护会为下一次主请求留出至少一个总额度。维护额度不足时仍可执行确定性修剪。`max_maintenance_requests=0` 可禁用摘要请求。直接构造 Agent 的旧 `max_requests` 参数仍表示共享总上限，同时限制主请求数；请与新的主请求、总请求参数择一使用。

主请求或总额度耗尽后，Agent 仍处理最后一次响应。最终回复成功保存时返回 `COMPLETED`。尚未形成最终回复时返回 `PAUSED_LIMIT` 并显示恢复操作。继续工具任务会从已有结果直接发起请求，使用新一轮额度。仅因可用请求额度不足而无法继续压缩时也会暂停。

## 历史回查

在支持工具的会话中可启用 `history_search`、`history_read` 和 `history_operations`。需要为已有会话增加工具时，可用终端命令 `/tools on history_search history_read history_operations`。主模型请求存在压缩检查点，且同时启用了搜索和读取工具时，上下文说明会提示可回查的范围 `[0, through)`。该提示只描述消息范围。

```text
history_search({"query":"发布标记","limit":5})
history_read({"message_id":12,"start":0})
history_read({"message_id":12,"start":4096,"end":8192})
history_operations({"start":0,"limit":10})
```

`history_search` 对可见文本做不区分大小写的字面匹配，按消息从新到旧返回，每条最多 240 个字符。默认 10 条，上限 20 条。`history_read` 只读当前 Session 已提交的原始记录，每页最多 4096 个字符。使用返回的 `next_start` 翻页，`end` 为可选的排他上界。`message_id` 是从 0 开始的原始消息位置，压缩和会话重载不会改变它。范围按脱敏后的可见文本计算。

`history_operations` 按结果消息的原始顺序返回可核对的内置工具调用。每项包含调用消息 ID、工具名、结果消息 ID 和可公开的工具调用 ID。无法公开的调用 ID 返回 null。使用 `next_start` 继续，默认 10 条，上限 20 条。索引省略参数和结果正文，历史操作不能证明当前工作区仍处于相同状态。

可回查用户和助手的可见文本，以及来源可核对的内置工具结果。系统和开发者消息、私有推理、工具参数、受保护凭据文件的结果、外部工具及历史回查工具自身的结果在可见范围外。回查不访问其他会话，所有返回内容只作历史数据使用。工具执行继续遵循当前审批模式和硬编码权限规则。

## 提交、取消与持久化

自动压缩在轮内暂存检查点，并在下一次消息提交时与原始消息一起保存。暂停、失败或取消后，最近一次成功保存的历史和检查点继续生效。未完成的工具组可以保存在原始历史末尾。主请求和手动压缩都必须等该工具组补齐，历史回查只索引有可见结果的调用。恢复方式见[工具状态与中断恢复](architecture.md#工具状态与中断恢复)。

Application 将原始消息、检查点和用量一起写入会话文件，保存成功后发布内存状态。手动压缩同样先保存再发布。重载时校验边界并重新构造有效上下文。

## 可选 Tokenizer 与观测

`deepseek-v4-pro` 可使用匹配版本的本地 tokenizer。安装可选的 Python 包 `tokenizers`，下载发布文件，将 `DEEPSEEK_V4_PRO_TOKENIZER` 指向该文件。加载会校验 SHA-256，同一文件在进程内缓存。缺少配置、依赖或文件，或校验失败时自动回退到估算。通过代码装配时，也可为 `ModelInfo.token_counter` 选择计数实现。 [DeepSeek V4 Pro tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/0e1a0e5/tokenizer.json)

| EventBus 事件 | 观测内容 |
| --- | --- |
| `context.budget` | 主请求输入、窗口、输出预留、阈值及 `count_source` |
| `model.started`、`model.usage` | 请求编号与服务端报告用量；摘要请求带 `purpose=compaction` |
| `turn.finished` | 总模型请求、主请求、维护请求和工具调用计数 |
| `compaction.started`、`compaction.finished` | 触发原因、尝试结果；摘要失败时提供 JSON、字段、响应内容、结束状态或额度等分类，自动压缩另有前后输入计数 |
| `compaction.committed`、`compaction.discarded` | 自动检查点随下一次消息快照提交，或在尚未提交时丢弃 |
| `context.overflow` | 超限后是否重试以及停止原因 |

摘要是有损的。继承要求的提示可以减少遗漏，原始用户消息和可定位的工具结果仍是核查依据。摘要、工具输出和回查摘录均不能授予权限。脱敏覆盖已知连接凭据和可识别的凭据格式；未标记的秘密不适合写入会话或工具日志。
