# 本地 Skills

打开 **Options → Skills**，点击「导入 Skill」并输入 Daphnis 服务所在机器上的本地目录。可使用绝对路径，也可输入相对于当前会话工作区的路径。选择「全局」或「当前工作区」，点击「导入」后可在列表中设置加载方式。导入会复制目录，新副本默认停用。同一作用域已存在同名目录时会报错。移除操作删除受管理的副本和状态记录，原始导入来源会保留。

## 目录与格式

全局 Skill 存在 `~/.daphnis/skills/<name>/`，项目 Skill 存在 `<workspace>/.daphnis/skills/<name>/`。项目作用域取决于会话的工作区路径，无项目会话也能使用该工作区的 Skills。当工作区为用户主目录、两个存储路径重合时，项目包存入同一根目录下的 `.project/<name>/`。模型通过 `global:<name>` 和 `project:<name>` 区分两个作用域的同名包。

`SKILL.md` 以 `---` 围起的 YAML frontmatter 开头，包含与目录同名的 `name` 和 `description`，后面是 Markdown 正文。名称使用 1–64 个小写字母、数字或单个连字符，连字符只出现在名称中间。描述最多 1024 个字符。合法附加字段随文件复制，权限设置继续由 Daphnis 的工具审批决定。[Agent Skills 格式规范](https://agentskills.io/specification)

```markdown
---
name: code-review
description: Review changes for behavior regressions and missing tests.
---

Read the diff and its callers. Report findings with file locations.
```

合法包可以包含 `scripts/`、`references/` 和 `assets/`。导入接受普通文件和目录，拒绝符号链接、硬链接和特殊文件。附属脚本由现有 `run_command` 执行，沿用 ASK / AUTO / FULL、命令校验、超时和取消规则。

## 加载方式

| 状态 | 主模型请求 |
| --- | --- |
| `disabled` | 省略该 Skill 的目录和正文，禁止通过 `read_skill` 读取。 |
| `always` | 每次加入完整 Markdown 正文，计入现有 Token 预算。固定内容超过安全预算时报告错误。 |
| `on_demand` | 提供作用域、名称、描述和 Skill ID。有工具能力的模型可选择使用 `read_skill`。 |

`read_skill(skill_id, path=None)` 默认读取 `SKILL.md`。可指定包内的 UTF-8 文本路径，例如 `references/checklist.md`。单个文本文件限 128 KiB，每个工具结果限 8 KiB。若返回 `next_line` 和 `next_column`，用它们作为下一次调用的 `start_line` 和 `start_column`。二进制文件可以保存在包内。工具结果作为不可信指导材料处理，并遵循当前工具审批。

固定正文独立于 Session 消息，只用于主模型请求。自动标题和摘要请求省略该正文。`read_skill` 的工具结果按现有机制提交到历史，后续新 Turn 或摘要使用占位说明。模型需要资料时，可以读取该轮仍启用的包。

## 本地状态

Daphnis 启动时扫描全局目录，首次打开工作区时扫描项目目录。导入和移除会即时更新目录。手动修改文件后重启服务以扫描新版。问题包会显示错误，其他包可继续使用。

用户的加载选择保存于 `~/.daphnis/skills.json`，文件暂时缺失时保留状态。每轮开始时固定有效目录、状态和必要正文。管理操作改变后续轮次的选择。本轮按需读取时若文件已被替换、修改或删除，工具报告文件变化。[Operations 接口](operations.md#管理-skills)
