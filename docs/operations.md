# 公共 Operations 接口

`Operations` 可从 `daphnis` 导入。多个实例可引用同一个 Application，共享当前会话、配置、任务和审批状态。自行构造 `Application` 时可注入模型连接、目录和 History。Web 使用 `daphnis.runtime.open_operations(workspace, *, session_id=None, observe_model_io=False)` 组装交互式运行环境，并恢复已启用的 MCP 服务。

## 最小示例

Web 运行时会加载 `.env`。直接调用 `environment_connections()` 只读取当前环境变量，因此示例显式加载启动目录的 `.env`。

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
| 系统提示词 | `get_system_prompt()` 返回当前生效文本。`await set_system_prompt(prompt)` 保存会话覆盖值，支持多行。`await reset_system_prompt()` 恢复内置默认值。 |
| 工具 | `all_tools` 是可用目录，`enabled_tools` 是已启用偏好（包括已启用的离线 MCP 工具），`effective_tools` 是两者的交集。`await set_tools(names, keep_unavailable=False)` 保存选择，可用 `keep_unavailable=True` 保留未列出的离线偏好。 |
| MCP | `managed_mcp()` 查看已保存配置。空闲时可用 `await save_mcp(name, enabled=..., options=..., create=...)` 创建或编辑配置，省略 `options` 可只切换启用状态。`await restart_mcp(name)` 重连已启用的服务，`await delete_mcp(name)` 删除配置。`connect_mcp()`、`reconnect_mcp()`、`disconnect_mcp()` 和 `refresh_mcp()` 仍用于直接管理运行时连接。`mcp_status()` 和 `list_mcp_tools()` 查询实时状态和在线工具。 |
| MCP Prompts | `list_mcp_prompt_servers()` 列出已连接且支持模板的服务。`await list_mcp_prompts(server)` 返回模板定义，`await get_mcp_prompt(server, name, arguments)` 校验字符串参数并返回 SDK 的 `GetPromptResult`，保留消息角色、顺序与内容类型。通过 `await run_mcp_prompt(PromptInput(...))`（从 `daphnis.prompt_groups` 导入 `PromptInput`）可将已验证的有序 User / Assistant 文本作为一轮任务运行。Session 保存实际文本和来源，重试沿用已保存的快照。 |
| MCP Resources | `list_mcp_resource_servers()` 列出已连接且声明资源能力的服务。`await list_mcp_resources(server)` 和 `await list_mcp_resource_templates(server)` 读取目录。`await read_mcp_resource(server, uri, arguments=None)` 读取目录中的固定 URI 或展开模板，返回可随 `send(..., attachments=...)` 提交的文本资源。 |
| 执行与取消 | `await send(prompt, stream=True)` 执行一轮并返回 `TurnResult`。`attachments` 接受 `TextAttachmentInput` 和 `ImageAttachmentInput` 的混合序列，两者均从 `daphnis.attachments` 导入。`running` 查询是否有活跃 Turn 或手动压缩，`await cancel()` 请求取消并等待任务停止。 |
| 恢复 | `recovery` 返回当前待处理状态的 `kind`、原 User 的 `message_index` 和 `key`。将该键传给 `retry_failed_user(key)`、`continue_interrupted_turn(key)` 或 `terminate_interrupted_turn(key)`。 |
| 上下文 | `await compact()` 在空闲时请求手动压缩，返回已保存的检查点或 `None`。`context_budget` 返回最近一次主请求的输入预算，`compacting` 和 `compacted_through` 表示压缩状态和已提交摘要覆盖到的消息数。 |
| 审批 | `approval_mode` 和 `set_approval_mode(mode)` 查询及设置模式。`pending_approval` 返回当前请求的独立副本或 `None`。`resolve_approval(request_id, decision)` 仅对匹配且仍在等待的请求返回 `True`，用 `can_trust` 判断是否可提交会话信任。 |
| 用量与事件 | `session_usage`、`turn_usage` 返回用量快照。`subscribe(name, callback)` 返回退订函数，事件详见[运行架构与事件](architecture.md#事件订阅)。 |
| 关闭 | `await close()` 关闭共享的 Application。并发关闭共享一次清理，关闭后保留只读查询。 |

运行中的配置修改和会话切换会被拒绝。新建会话沿用工作区、模型、思考档位与工具偏好，采用默认系统提示词，消息与用量从零开始。恢复会话使用已保存的工作区和设置，并校验模型与能力。恢复时可保留离线 MCP 工具的启用偏好。持久化格式和并发规则见[本地历史与并发](architecture.md#本地历史与并发)。

`send()` 的普通 Turn 失败由 `TurnResult.status` 与 `error` 表示，锁冲突、状态校验和保存失败等可直接抛出异常。取消时会等待已开始的磁盘写入。若最终回复已保存，返回 `COMPLETED`，否则等待该轮的调用会收到 `asyncio.CancelledError`。

审批使用每次请求独立的 `request_id`，而工具调用 ID 由模型提供。决策为 `deny`、`allow_once` 或 `trust_session`，可从 `daphnis.tools` 导入 `ApprovalDecision` 与 `ApprovalMode`。使用范围见[审批规则](tools.md#审批规则)。

## 发送图片附件

```python
from pathlib import Path

from daphnis.attachments import ImageAttachmentInput

path = Path("screenshot.png")
result = await ops.send(
    "这张截图有什么问题？",
    attachments=[ImageAttachmentInput(path.name, path.read_bytes(), "image/png")],
)
```

默认模型目录中的 `mimo-v2.6-flash`、`mimo-v2.6-pro` 和 `deepseek-flash` 已确认支持图片。自定义模型目录可通过 `ModelInfo.supports_image` 声明能力。支持静态 PNG、JPEG 和 WebP，单张最多 5 MiB、2000 万像素，每轮最多 4 张和 20 MiB。恢复会话后，未被压缩覆盖且仍在上下文中的图片会随对应 User 消息发出。

MiMo 的图片 Token 预算按尺寸估算，DeepSeek 每张预留 1024 Token。一次模型请求的原始图片总量上限为 32 MiB，开启自动压缩时可先压缩较早的历史以释放空间。

含图片的模型请求仍可使用 HTTP 流式响应。Daphnis 在收到完整响应并清理可能被分段回显的图片数据后，发布一次合并的 `model.delta`。

## 恢复未完成的任务

先加载原会话。Web 可从会话列表选中原会话，代码可调用 `restore_session(session_id)`。加载会话只读取历史。

如果 `recovery.kind` 为 `FAILED_USER`，Web 在原消息旁显示可点击的「!」。重试直接复用原消息。发送新消息则作废旧消息，作废记录仍可在界面查看。

如果为 `INTERRUPTED_TURN`，用 Web 的「继续 / 终止」。请求额度暂停也适用。继续从已保存的工具事实恢复，终止保存 `TERMINATED` 并允许发送新消息。Web 可处理人工批准，直接使用 Application 的代码需启用交互模式并接入审批。结果未知时应检查实际环境。
