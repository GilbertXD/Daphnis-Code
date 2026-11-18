# Daphnis

Daphnis 是通过浏览器使用的本地 Coding Agent，内置 Chat Completions 模型通信与流式响应处理。它通过模型 API 理解任务，在指定工作区内读取和搜索代码，并按工具权限规则执行命令、应用补丁或调用 MCP 工具。会话历史存储在本地。

## 安装与配置

需要 Python 3.11+、uv 和 Node.js 20.19+（20.x）或 22.12+。文件工具和历史写入依赖 POSIX 接口，适用于 macOS 和 Linux 的本地文件系统。Windows 和网络文件系统未验证。

```sh
git clone https://github.com/GilbertXD/Daphnis-Code.git
cd Daphnis-Code
uv sync --locked --extra web
cp .env.example .env
```

编辑 `.env`，填入模型服务的连接信息。Web 从**启动命令的当前目录**读取 `.env`，已有环境变量优先。工作区由 `--workspace` 或恢复的会话决定。

| 变量 | 用途 |
| --- | --- |
| `LLM_PROVIDER` | 启动时使用的厂商，取值为 `deepseek`、`mimo` 或 `compatible` |
| `LLM_MODEL` | 启动时使用的模型 ID。DeepSeek 和 MiMo 使用[仓库中的模型目录](src/daphnis/models/models.py) |
| `LLM_BASE_URL`、`LLM_API_KEY` | 启动厂商的服务地址和 API Key |
| `DAPHNIS_MCP_CONFIG` | 可选的旧版 MCP JSON 导入文件，见[配置示例](docs/tools.md#mcp-配置与生命周期) |

需要在运行中切换厂商时，分别设置 `DEEPSEEK_BASE_URL` / `DEEPSEEK_API_KEY`、`MIMO_BASE_URL` / `MIMO_API_KEY` 或 `COMPATIBLE_BASE_URL` / `COMPATIBLE_API_KEY`。同一厂商的专用地址和密钥配置优先于通用配置。各厂商使用独立的连接和密钥。`compatible` 支持填写任意兼容 Chat Completions 的模型 ID，未登记模型的工具和思考能力默认为未知。

## Web 使用

在仓库根目录构建前端并启动服务。

```sh
npm --prefix web ci
npm --prefix web run build
uv run --locked --extra web daphnis-web --workspace .
```

打开终端打印的地址，默认是 `http://127.0.0.1:8765`，并输入同一终端显示的访问令牌。可用 `--port` 更换端口。`--workspace` 默认取启动目录，恢复历史会话时可改用 `--session`，工作区由已保存的会话决定。

- 在输入框发送任务，或按 Ctrl/⌘+Enter。运行中可停止当前轮，并在页面上处理工具审批。
- 一个会话运行、等待审批或压缩上下文时，可以切换或新建另一个会话并发送任务。切回即可继续查看进度、审批或停止该轮。每个会话同时只能执行一个任务；使用同一工作区的会话需要自行协调文件写入。
- 未命名的会话会先用第一条用户消息的截取文本作为备用标题。正文回复完成后，再请求模型生成最多 40 个字符的正式标题。模型支持关闭思考时，标题请求会关闭思考。若首轮未完成或标题生成失败，保留备用标题，可在侧栏手动重命名。
- 在输入框旁选择 ASK、AUTO 或 FULL，运行期间无法切换。确认风险后启用 FULL，可自动执行受支持的命令和 MCP 工具，并读取 `.env` 等敏感文件。直接调用的高危命令仍会被拦截，脚本和外部工具可按其进程权限访问文件或网络。新建会话或重启后使用 AUTO。详见[工具与审批规则](docs/tools.md)。
- 在左侧「新建项目」登记已有工作区目录，留空时使用当前工作区。点击项目旁的「＋」在其工作区新建会话。点击「无项目」旁的「＋」会沿用当前工作区，新会话归入「无项目」。原有会话也显示在「无项目」中。
- 点击项目名称展开或折叠分组。首次发送前，模型、思考档位、工具和系统提示词的改动只保存在内存中。发送首条消息时，会话设置随消息一起保存。
- 点击项目旁的「×」可移除项目。该项目的会话归入「无项目」，并保留各自的工作目录和历史消息。
- 恢复会话时使用会话保存的工作目录。项目记录缺失、无效或目录不匹配时，会话显示在「无项目」。
- 通过会话菜单的「移动」可更改归属。移到其他项目后，后续工具操作使用目标项目的工作目录。移到「无项目」会保留原工作目录。
- 在侧栏新建、恢复、重命名或删除会话，在输入框下方选择模型、思考档位和工具，或编辑、重置会话系统提示词。
- 点击输入框旁的「Skills」导入本地 `SKILL.md` 目录并设置加载方式。项目 Skills 随会话工作区切换。[Skills 使用说明](docs/skills.md)
- 点击顶部「MCP」手动添加服务，或通过「发现」添加 Registry 中的远程服务。选择适用于所有工作区的 Global 或只适用于当前工作区的 Project，再选择禁用、始终加载或按需加载。按需服务在模型需要时才激活，工具调用仍按当前会话审批。工具勾选和配置迁移见 [MCP 配置与生命周期](docs/tools.md#mcp-配置与生命周期)。
- 点击输入框旁的「Prompts」，选择 MCP Prompt、填写参数并编辑预览，点击「运行」直接开始任务。[MCP Prompts 使用说明](docs/tools.md#mcp-prompts)
- 从「MCP → Resources」浏览和读取资源，预览后点击「添加到消息」，再像普通附件一样发送。[MCP Resources 使用说明](docs/tools.md#mcp-resources)
- 在输入框左下角点「+」或拖入文件。支持图片的模型还可以接收粘贴的截图。图片支持 PNG、JPEG 和 WebP，每轮最多 4 张、单张 5 MiB。UTF-8 文本附件每轮最多 10 个、单个 128 KiB、总计 256 KiB。[文本附件](docs/tools.md#文本附件)
- 草稿中的图片可以预览和移除。发送后图片与附件快照保存在本地，附件内容可能发送给所选模型服务。对话中的图片可点击放大，压缩历史后仍可查看。
- 使用 DeepSeek Flash 时，可以选择或拖入 PDF、DOCX、PPTX 或 XLSX，每轮一份，单份最多 10 MiB。PDF、DOCX 和 PPTX 可勾选「保留图表」。文档原件保存在本地，可以从历史中下载。发送时会上传给 DeepSeek，远端解析结果最多保留 30 天，期间可跨轮次和重启复用。解析费用与模型 Token 用量由服务端分别计费。
- 点击失败消息旁的「!」可重试。发送新消息会作废旧失败消息，原文仍可查看。工具任务未完成时，输入区上方提供「继续 / 终止」。
- 展开右侧调试面板查看事件、工具调用、已提交上下文和模型响应，并在运行状态中查看 Token 预算与摘要检查点。HTTP 请求正文和错误响应正文暂不可用。刷新页面或关闭标签页后，服务端仍继续执行。重连可读取当前状态、待审批请求和仍在缓存中的调试记录。
- 每个标签页独立选择会话；各会话保留自己的审批模式。重启服务后使用新令牌，已保存的会话仍可从历史打开并继续或终止未完成的任务。调试记录仅保留在当前进程。

## 开发与参考

- [运行架构与事件](docs/architecture.md)
- [公共 Operations 接口](docs/operations.md)
- [本地 Skills](docs/skills.md)
- [上下文压缩与历史回查](docs/context_compaction.md)
- [工具、审批与 MCP](docs/tools.md)

```sh
uv run --locked --extra web python -m unittest discover -s tests -v
uv run --locked --extra web ruff check --no-cache .
npm --prefix web ci
npm --prefix web run build  # 先运行 TypeScript 类型检查，再构建前端
```
