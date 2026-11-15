# MCP 路线图

Web 支持从 MCP Prompt 生成、编辑并直接运行有序 User / Assistant 文本消息组，沿用 Session 历史、整轮恢复和压缩。

## A. Prompts 完整能力

- **A1｜Assistant 续写**：评估模型适配层的 Assistant 前缀能力。目前以 Assistant 结尾的 Prompt 需要用户补充一条 User 消息才能运行。
- **A2｜非文本内容**：支持图片、嵌入资源等内容，并与 Daphnis 的多模态模型能力和消息表示协调。
- **A3｜目录变化通知**：接收服务端的 Prompts 目录变化通知，按需刷新列表。
- **A4｜参数自动补全**：接入 MCP Completions，为模板参数提供候选值。
- **A5｜Web 交互**：考虑通过输入框的 `/` 命令唤出模板、收藏模板和按常用程度排序。

## B. 模型自主使用 Prompts

### B1｜模型发现模板

可评估为主 Agent 选择性提供 Daphnis 自己的 `list_prompts` 和 `get_prompt` 工具，让模型发现和选择模板。需要界定可访问的 MCP 服务、可向外部发送的参数、用户审批条件，以及如何将返回内容标为不可信数据。还需限制重复查询及 Token 消耗。

### B2｜与 Subagent 集成

未来可按「主 Agent 选择 Prompt → 获取模板 → 构造子任务 → 申请必要权限 → 创建 Subagent → 汇总结果」推进。模板提供任务描述，Daphnis 负责创建 Subagent 并为其设置基础 System Prompt。外部模板不能提升权限。Daphnis 控制工作区、工具权限、模型额度与请求上限，Application 统一管理创建、取消、失败恢复和并发限制。相关实现归入后续 Subagent 开发。

## C. MCP 其他能力

- **C1｜Resources**：单独设计资源浏览、读取和上下文集成，明确资源引用、动态加载、Token 预算、更新通知及压缩策略。
- **C2｜更多客户端能力**：根据具体使用场景评估 Sampling、Elicitation 等能力。
