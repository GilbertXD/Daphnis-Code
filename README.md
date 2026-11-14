# Daphnis

Daphnis 是基于 Saturnus 的本地 Coding Agent。它通过模型 API 理解任务，在指定工作区内读取和搜索代码，并按工具权限规则执行命令、应用补丁或调用 MCP 工具。终端和浏览器客户端共用本地会话历史。

## 安装与配置

需要 Python 3.11+ 和 uv。文件工具和历史写入依赖 POSIX 接口，适用于 macOS 和 Linux 的本地文件系统。Windows 和网络文件系统未验证。

```sh
git clone https://github.com/GilbertXD/Daphnis-Code.git
cd Daphnis-Code
uv sync --locked
cp .env.example .env
```

编辑 `.env`，填入模型服务的连接信息。终端和 Web 都从**启动命令的当前目录**读取 `.env`，已有环境变量优先。工作区由 `--workspace` 或恢复的会话决定。

| 变量 | 用途 |
| --- | --- |
| `LLM_PROVIDER` | 启动时使用的厂商，取值为 `deepseek`、`mimo` 或 `compatible` |
| `LLM_MODEL` | 启动时使用的模型 ID。DeepSeek 和 MiMo 使用[仓库中的模型目录](src/daphnis/models/models.py) |
| `LLM_BASE_URL`、`LLM_API_KEY` | 启动厂商的服务地址和 API Key |

需要在运行中切换厂商时，分别设置 `DEEPSEEK_BASE_URL` / `DEEPSEEK_API_KEY`、`MIMO_BASE_URL` / `MIMO_API_KEY` 或 `COMPATIBLE_BASE_URL` / `COMPATIBLE_API_KEY`。同一厂商的专用地址和密钥配置优先于通用配置。各厂商使用独立的连接和密钥。`compatible` 支持填写任意兼容 Chat Completions 的模型 ID，未登记模型的工具和思考能力默认为未知。

## 终端使用

```sh
# 打开交互式 TUI
uv run --locked daphnis --workspace .

# 执行一轮任务并输出最终答案
uv run --locked daphnis --workspace . -p "检查当前项目的测试入口"

# 用保存的工作区和设置恢复会话
uv run --locked daphnis --session '会话ID'
```

`--workspace` 默认取当前目录，与 `--session` 二选一。可用 `--thinking` 指定已支持的思考档位，或用 `default` 采用服务端默认设置。单次执行会拒绝需要人工确认的工具调用。可用 `--help` 查看参数。

在 TUI 中按 Enter 发送，Shift+Enter 换行，Esc 中断当前轮。Ctrl+C 在运行时取消、空闲时退出，Ctrl+Q 取消并退出。

| 命令 | 用途 |
| --- | --- |
| `/help` | 查看命令和快捷键 |
| `/model`、`/thinking`、`/tools` | 查看或调整模型、思考档位和启用的工具 |
| `/system`、`/system reset` | 多行编辑会话系统提示词，或恢复内置默认值 |
| `/session [会话ID]`、`/new` | 列出或恢复会话，或新建会话并保留旧历史 |
| `/retry`、`/continue`、`/terminate` | 重试未得到回复的用户消息，或继续、终止未完成的工具任务 |
| `/compact` | 为旧上下文生成摘要，保留原始会话历史 |
| `/clear` | 清理屏幕，保留会话历史 |
| `/approval [ask\|auto\|full]` | 查看或调整工具审批模式，默认 `auto` |
| `/mcp` | 查看已配置的 MCP 服务，或管理连接 |

TUI 底部显示最近一次模型请求的输入 Token 估算、已知窗口和摘要检查点。已知窗口接近阈值时，Agent 会自动修剪工具输出并按需摘要。

单轮 CLI 可用 `--session '会话ID' -p /continue` 显式恢复，`-p /retry` 和 `-p /terminate` 分别用于重试和终止。只有最终回复已保存时，本轮才算完成。

工具需要确认时，先查看路径、命令参数或补丁，再按 Y 允许、N 或 Esc 拒绝。支持会话信任时，按 T 切换勾选后再按 Y 确认。可用 `--mcp-config` 加载显式配置的 MCP 服务。配置示例和权限规则见[工具、审批与 MCP](docs/tools.md)。

## Web 使用

在仓库根目录使用同一份 `.env`。前端构建需要 Node.js 20.19+（20.x）或 22.12+。

```sh
npm --prefix web ci
npm --prefix web run build
uv run --locked --extra web daphnis-web --workspace .
```

打开终端打印的地址，默认是 `http://127.0.0.1:8765`，并输入同一终端显示的访问令牌。可用 `--port` 更换端口，或用 `--session` 恢复会话。

- 在输入框发送任务，或按 Ctrl/⌘+Enter。运行中可停止当前轮，并在页面上处理工具审批。
- 在输入框旁选择 ASK、AUTO 或 FULL，运行期间无法切换。新建会话或重启后使用 AUTO。
- 在侧栏新建、恢复、重命名或删除会话，在输入框下方选择模型、思考档位和工具，或编辑、重置会话系统提示词。
- 点击失败消息旁的「!」可重试。发送新消息会作废旧失败消息，原文仍可查看。工具任务未完成时，输入区上方提供「继续 / 终止」。
- 展开右侧调试面板查看事件、工具调用、上下文和请求/响应，并在运行状态中查看 Token 预算与摘要检查点。刷新页面或关闭标签页后，服务端仍继续执行。重连可读取当前状态、待审批请求和仍在缓存中的调试记录。
- 每个 Web 进程运行一个本地 Agent，同一进程的多个标签页共享当前会话和审批模式。服务重启后使用新令牌，进程内调试记录随进程结束而清空。MCP 配置入口在终端客户端。

## 开发与参考

- [运行架构与事件](docs/architecture.md)
- [公共 Operations 接口](docs/operations.md)
- [上下文压缩与历史回查](docs/context_compaction.md)
- [工具、审批与 MCP](docs/tools.md)

锁文件使用已发布的 `saturnus-llm==0.1.1`。同时开发两个仓库时，可在同步后执行 `uv pip install --no-deps -e ../Saturnus`，再用 `uv run --no-sync daphnis` 使用本地 Saturnus。重新执行 `uv sync --locked` 会恢复发布版。

```sh
uv run --locked --extra web python -m unittest discover -s tests -v
uv run --locked --extra web ruff check --no-cache .
npm --prefix web ci
npm --prefix web run build  # 先运行 TypeScript 类型检查，再构建前端
```
