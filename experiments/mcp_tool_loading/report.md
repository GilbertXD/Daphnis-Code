# MCP 工具加载 A/B 对照

## 实验设置

- 基线是 `feat/cache-usage` 的 `1bf9f62`。Native 保持默认行为。Gateway 只通过 Application 的 `mcp_tool_loading="gateway"` 启用。
- MiMo 使用 `mimo-v2.6-flash`，DeepSeek 使用 `deepseek-flash`。各模式使用相同模型参数，非流式、思考关闭、主请求输出上限 768 Token、每条轨迹最多 6 个主请求。关闭自动 Compaction。
- 每条轨迹用新 Session，工作区为空。模式与任务共用同一实验工作区、工具选择和 Microsoft Learn 服务配置。启用 `glob`、`grep`、`list_directory` 和 `read_file`，ASK 审批只对预先列出的 Microsoft Learn 只读工具按次放行。[Microsoft Learn MCP 文档](https://learn.microsoft.com/en-us/training/support/mcp-developer-reference)
- T1 查询 Azure Blob 四个访问层及哪个层离线，要求引用访问层文档。T2 对照 ZRS、GRS 和 RA-GRS，并核对离线访问层，要求引用冗余和访问层两篇文档。脚本固定提示词及判分规则，检查预设事实关键词、官方引用是否出现在成功工具结果里，以及最少的独立查询数。判分无需模型自评，也未对答案做逐句人工审核。[Azure Blob 访问层](https://learn.microsoft.com/en-us/azure/storage/blobs/access-tiers-overview) · [Azure Storage 冗余](https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy)
- 冒烟阶段每个模型各跑一对 T1。正式阶段每个模型、每个任务各跑三对，采用 Native → Gateway、Gateway → Native、Native → Gateway 的顺序。

## 请求与执行路径

Native 在 Microsoft Learn 激活后把三个原生工具定义加到下一次 API 请求。Gateway 通过激活结果返回三个工具的名称、说明和参数 Schema。后续的 `call_mcp_tool` 使用当前请求的会话工具快照路由到原工具。它保留模型给出的外层 call ID，并用真实工具名校验、审批和记录会话信任。工具执行前的 `RUNNING` 落盘和恢复沿用原路径。

| 主请求 API 工具列表 | 激活前 | 激活后 |
|---|---:|---:|
| Native | 5 个，3,907 B，`541b3a06` | 8 个，7,838 B，`a479ad5f` |
| Gateway | 6 个，4,336 B，`d1bd2948` | 6 个，4,336 B，`d1bd2948` |

字节数与 SHA-256 前缀基于实际请求中的 `tools`，以紧凑 UTF-8 JSON 序列化，保留顺序。完整哈希、按顺序排列的工具名及逐项定义哈希见原始数据。同一模式下的系统提示在正式阶段保持稳定，Native 和 Gateway 的系统提示哈希不同。

## 正式结果

仅统计主请求，未命中输入列为有效报告缓存数据范围内的 Token。

| 模型 | 模式 | 加权命中率 | 缓存覆盖 | 未命中输入 | 总输入 | 总输出 | 主请求总数 / 均值 | 工具调用 / 参数错误 | 任务完成 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiMo | Native | 31.14% | 16 / 18 | 89,422 | 133,989 | 1,383 | 18 / 3.00 | 15 / 0 | 6 / 6 |
| MiMo | Gateway | 36.27% | 23 / 24 | 102,239 | 162,131 | 1,950 | 24 / 4.00 | 23 / 8 | 6 / 6 |
| DeepSeek | Native | 34.03% | 18 / 18 | 91,077 | 138,053 | 2,315 | 18 / 3.00 | 16 / 0 | 6 / 6 |
| DeepSeek | Gateway | 41.39% | 18 / 18 | 82,479 | 140,719 | 2,524 | 18 / 3.00 | 15 / 0 | 6 / 6 |

### MiMo

Gateway 的加权命中率高 5.13 个百分点。总输入增加 28,142，已知未命中输入增加 12,817，平均每条轨迹增加一个主请求。即使 Native 的 缓存情况未知的 4,119 个输入 Token 全部未命中，其未命中总量上界 93,541 仍低于 Gateway 已知的 102,239。Gateway 的八次参数校验失败都发生在实际审批和执行前，模型随后完成了任务。另有两次参数完全相同的重复调用，重复本身无法判断是否有必要。

### DeepSeek

Gateway 的加权命中率高 7.36 个百分点，已知未命中输入减少 8,598（9.44%），总输入增加 2,666，输出增加 209。两个模式的主请求数和完成率相同，均无参数错误。Native 有一次额外调用被实验审批规则拒绝，任务随后达标。两种模式各激活 6 次。

### 激活前后

下表每格仅比较两种模式均有效报告的同一任务和同一重复序号。每条轨迹在各阶段只取首个请求。

| 模型 | 阶段 | 有效配对 | Native 命中 / 输入 | Gateway 命中 / 输入 |
|---|---|---:|---:|---:|
| MiMo | 激活前 | 4 | 4,672 / 6,322 | 4,992 / 6,898 |
| MiMo | 激活后首次请求 | 5 | 10,432 / 12,264 | 10,880 / 12,784 |
| MiMo | 工具结果后首次请求 | 6 | 25,344 / 109,729 | 15,680 / 99,754 |
| DeepSeek | 激活前 | 6 | 7,552 / 10,011 | 8,192 / 10,791 |
| DeepSeek | 激活后首次请求 | 6 | 12,928 / 15,390 | 12,032 / 16,063 |
| DeepSeek | 工具结果后首次请求 | 6 | 26,496 / 112,652 | 38,016 / 113,865 |

Native 切换工具定义后，两家仍报告了较高的命中量。激活后的首个请求中，MiMo 两种方案接近，DeepSeek 的 Gateway 命中率较低。DeepSeek 的端到端收益主要出现在纳入首个工具结果之后。由于系统提示、请求前缀、工具结果、服务端缓存预热和请求时间各有差异，本实验无法把任何差值单独归因于工具定义变更。

### 试验规模与维护请求

四条冒烟轨迹均通过，记录 13 个主请求及 4 个标题请求，合计报告 74,305 输入和 1,001 输出 Token。正式 24 条轨迹共记录 78 个主请求及 24 个标题请求，合计报告 578,555 输入和 8,536 输出 Token。正式主请求全部完成且没有 HTTP 重试。标题中有一次 MiMo Native 请求缺少完成记录和 Usage，已单列在原始数据中。其余 23 个标题请求合计报告 3,663 输入和 364 输出 Token。没有 Compaction 请求。

### 决定

保留 Native 默认实现。DeepSeek 在这组任务中降低了未命中输入且保持相同的任务完成率，MiMo 则出现更多未命中输入、请求和参数错误。现有证据不足以支持跨模型迁移，也没有按厂商实际账单测量费用。实验只覆盖一个包含三个工具的公共 MCP Server 和两类英文任务，缓存预热无法完全控制。

## 本地验证

21 项定向 Python 测试通过。覆盖 Native 基线、Gateway 请求定义稳定性、会话隔离、禁用保护、原始 Schema 校验、审批和真实工具信任、共享连接、取消恢复、Usage 快照及脱敏记录。Ruff 与 `git diff --check` 通过。已逐条核对请求级缓存字段和会话累计，确认所有 Gateway 主请求的工具定义哈希在各轨迹内保持不变。

## 复现与数据

从项目根目录运行以下命令。脚本加载本地已配置的提供商。输出文件以追加方式写入，复跑时请指定新文件。两条命令分开运行，先检查冒烟数据，再启动正式对照。

```sh
uv run --locked python experiments/mcp_tool_loading/run.py --stage smoke --output /tmp/mcp-smoke.jsonl
uv run --locked python experiments/mcp_tool_loading/run.py --stage formal --request-budget 180 --output /tmp/mcp-formal.jsonl
```

[原始 JSONL](results.jsonl) 中的 `request` 行记录请求目的、阶段、模型参数、工具定义摘要、HTTP 状态和 API Usage 缓存计数。`trajectory` 行记录判分、工具调用结果和审批、失败与重试，以及会话用量。完整消息、工具参数、凭据和 MCP 返回正文均未写入。

命中率以有效报告缓存字段的请求为分母。未命中输入等于这些请求中输入 Token 减去命中 Token。覆盖不足时，未命中输入只能视作已知下界。正式汇总只计算主请求，标题与 Compaction 另列。缓存命中率与未命中 Token 均不直接换算成费用。模型收到工具参数错误后再次请求，计入主请求数。HTTP 或服务异常后的重试单独记录。

一次脚本调试在等待审批时取消，未产生完整轨迹。其请求用量未能核对，服务端可能保留了预热缓存。正式阶段的统计没有混入调试或冒烟样本。
