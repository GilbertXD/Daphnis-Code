# 公共 Operations 接口

`Operations` 可从 `daphnis` 导入。每次会话操作都指定 `session_id`；多个 Operations 实例可以访问同一个 Application 中的会话和共享资源。自行构造 `Application` 时可注入模型连接、目录和 History。Web 使用 `daphnis.runtime.open_operations(workspace, *, session_id=None, observe_model_io=False)` 组装交互式运行环境并连接已启用的 MCP 服务。用 `ops.initial_session_id` 取得启动时创建或恢复的会话 ID。

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
        "model.started", lambda event: print(event.session_id, "请求", event.data["request_number"])
    )
    try:
        first = ops.initial_session_id
        second = (await ops.new_session(session_id=first)).id
        results = await asyncio.gather(
            ops.send(session_id=first, prompt="检查项目的测试入口", stream=False),
            ops.send(session_id=second, prompt="总结项目的依赖", stream=False),
        )
        for result in results:
            print(result.message if result.status == TurnStatus.COMPLETED else result.error)
    finally:
        unsubscribe()
        await ops.close()


asyncio.run(main())
```

Application 默认以非交互模式运行。需要人工审批的客户端可设置 `interactive=True`，并通过下表中的接口处理请求。

## 接口速查

| 用途 | 接口与行为 |
| --- | --- |
| 用途 | 接口与行为 |
| --- | --- |
| 会话与消息 | `session(session_id)` 返回 ID、工作区、模型身份和标题。`message_snapshot(session_id)` 返回已提交消息的独立副本。`attachment_snapshot(session_id)` 返回附件引用。 |
| 历史 | `list_sessions()` 按更新时间降序列出有效的已保存会话。`scan_sessions()` 额外返回无法加载的文件名及错误类型。`await new_session(session_id=source_id)` 按来源设置新建会话；`restore_session(session_id)` 加载或复用已有会话。两者返回会话快照。 |
| 会话管理 | `await rename_session(session_id, title)` 保存 1–120 字符的标题。`await move_session(session_id, project_id, expected_workspace=...)` 移动到目标项目。`await delete_session(session_id)` 删除历史和附件；界面可用 `replacement=True` 请求返回一个空白会话。 |
| 模型 | `model(session_id)` 返回能力描述，`list_models(session_id)` 按厂商分组。`configured(provider)` 查询连接。`await set_model(provider, model_id, session_id=...)` 保存选择，并清除不适用的思考档位或工具设置。 |
| 思考档位 | `thinking(session_id)` 查询档位，`await set_thinking(mode, session_id=...)` 保存档位。传入 `None` 使用厂商默认。 |
| 系统提示词 | `get_system_prompt(session_id)` 返回生效文本。`await set_system_prompt(prompt, session_id=...)` 保存覆盖值，`await reset_system_prompt(session_id)` 恢复内置值。 |
| 工具 | `all_tools` 是共享的可用目录。`enabled_tools(session_id)` 查询启用偏好，`effective_tools(session_id)` 查询实际可用项。`await set_tools(names, session_id=..., keep_unavailable=True)` 可以保留未列出的离线偏好。 |
| MCP | `managed_mcp()` 查看配置。`await save_mcp(name, enabled=..., options=..., create=...)` 创建或编辑；省略 `options` 可只切换启用状态。`restart_mcp()`、`delete_mcp()` 管理已保存的连接。直接管理运行时连接可用 `connect_mcp()`、`reconnect_mcp()`、`disconnect_mcp()` 和 `refresh_mcp()`。配置更新要求 Application 没有活动任务。 |
| MCP Prompts | `list_mcp_prompt_servers()`、`await list_mcp_prompts(server)` 和 `await get_mcp_prompt(server, name, arguments)` 读取模板与预览。`await run_mcp_prompt(PromptInput(...), session_id=...)` 在指定会话运行已验证的 User / Assistant 消息。重试复用已保存的快照。 |
| MCP Resources | `list_mcp_resource_servers()`、`await list_mcp_resources(server)` 和 `await list_mcp_resource_templates(server)` 读取资源目录。`await read_mcp_resource(server, uri, arguments=None)` 返回可作为 `send()` 附件的文本资源。 |
| 执行与取消 | `await send(prompt, session_id=..., stream=True, attachments=...)` 返回 `TurnResult`。`running(session_id)` 查询活动任务；`await cancel(session_id)` 取消并等候该任务结束。 |
| 恢复 | `recovery(session_id)` 返回恢复类型、原 User 的索引及键。把键和 `session_id` 传给 `retry_failed_user()`、`continue_interrupted_turn()` 或 `terminate_interrupted_turn()`。 |
| 上下文 | `await compact(session_id)` 在会话空闲时请求手动压缩，返回检查点或 `None`。`context_budget(session_id)`、`compacting(session_id)` 和 `compacted_through(session_id)` 查询预算、压缩状态和已覆盖消息数。 |
| 审批 | `approval_mode(session_id)` 和 `set_approval_mode(mode, session_id=...)` 查询及设置模式。`pending_approval(session_id)` 查询待处理请求。`resolve_approval(request_id, decision, session_id=...)` 只处理会话和请求 ID 均匹配、仍在等待的审批。 |
| 用量与事件 | `session_usage(session_id)` 和 `turn_usage(session_id)` 返回快照。`subscribe(name, callback)` 订阅共享 EventBus，监听方按 `event.session_id` 和 `event.turn_id` 区分事件。 |
| 关闭 | `await close()` 关闭 Application，停止所有会话的任务后释放共享资源。并发关闭共享一次清理。 |

同一会话的 Turn、恢复、手动压缩和设置写入互斥。该会话忙碌时，仍可查看它或操作其他会话。新建会话沿用来源的工作区、模型、思考档位与工具偏好，使用默认系统提示词和 AUTO 审批，消息与用量从零开始。首次恢复使用磁盘上的工作区和设置，校验模型与工具能力，并保留离线 MCP 工具偏好。已加载的会话复用其 Runtime。持久化规则见[本地历史与并发](architecture.md#本地历史与并发)。

`send()` 的普通 Turn 失败由 `TurnResult.status` 与 `error` 表示，锁冲突、状态校验和保存失败等可直接抛出异常。取消时会等待已开始的磁盘写入。若最终回复已保存，返回 `COMPLETED`，否则等待该轮的调用会收到 `asyncio.CancelledError`。

审批使用每次请求独立的 `request_id`，而工具调用 ID 由模型提供。决策为 `deny`、`allow_once` 或 `trust_session`，可从 `daphnis.tools` 导入 `ApprovalDecision` 与 `ApprovalMode`。使用范围见[审批规则](tools.md#审批规则)。

## 发送图片附件

```python
from pathlib import Path

from daphnis.attachments import ImageAttachmentInput

path = Path("screenshot.png")
result = await ops.send(
    "这张截图有什么问题？",
    session_id=ops.initial_session_id,
    attachments=[ImageAttachmentInput(path.name, path.read_bytes(), "image/png")],
)
```

默认模型目录中的 `mimo-v2.6-flash`、`mimo-v2.6-pro` 和 `deepseek-flash` 已确认支持图片。自定义模型目录可通过 `ModelInfo.supports_image` 声明能力。支持静态 PNG、JPEG 和 WebP，单张最多 5 MiB、2000 万像素，每轮最多 4 张和 20 MiB。恢复会话后，未被压缩覆盖且仍在上下文中的图片会随对应 User 消息发出。

MiMo 的图片 Token 预算按尺寸估算，DeepSeek 每张预留 1024 Token。一次模型请求的原始图片总量上限为 32 MiB，开启自动压缩时可先压缩较早的历史以释放空间。

含图片的模型请求仍可使用 HTTP 流式响应。Daphnis 在收到完整响应并清理可能被分段回显的图片数据后，发布一次合并的 `model.delta`。

## 恢复未完成的任务

先选中或通过 `restore_session(session_id)` 加载原会话，再通过 `recovery(session_id)` 查询可用操作。加载本身不会启动任务。

如果恢复状态的 `kind` 为 `FAILED_USER`，Web 在原消息旁显示可点击的「!」。重试直接复用原消息。发送新消息则作废旧消息，作废记录仍可在界面查看。

如果为 `INTERRUPTED_TURN`，用 Web 的「继续 / 终止」。请求额度暂停也适用。继续从已保存的工具事实恢复，终止保存 `TERMINATED` 并允许发送新消息。Web 可处理人工批准，直接使用 Application 的代码需启用交互模式并接入审批。结果未知时应检查实际环境。
