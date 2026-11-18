import { Fragment, memo, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AttachmentLabels } from './attachments'
import type { AttachmentHistory } from './attachments'
import { readableJson } from './protocol'
import type { AgentEvent, AttachmentInfo, LoadAttachment, LoadedSkill, Message, View } from './protocol'

export function Copy({ text }: { text: string }) {
  const [label, setLabel] = useState('复制')
  return <button className="copy" type="button" onClick={async () => {
    try {
      await navigator.clipboard.writeText(text)
      setLabel('已复制')
    } catch { setLabel('复制失败') }
  }} title={label} aria-label={label}>
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" aria-hidden="true">
      <rect x="5" y="5" width="8" height="8" rx="1"/><path d="M10 3H4a1 1 0 0 0-1 1v6"/>
    </svg><span className="copy-feedback" aria-live="polite">{label !== '复制' && label}</span>
  </button>
}

export function DataBlock({ value }: { value: unknown }) {
  const text = readableJson(value)
  return <div className="data-block"><Copy text={text} /><pre>{text}</pre></div>
}

function nodeText(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(nodeText).join('')
  if (node && typeof node === 'object' && 'props' in node) {
    return nodeText((node.props as { children?: ReactNode }).children)
  }
  return ''
}

const RichText = memo(function RichText({ text }: { text: string }) {
  return <Markdown remarkPlugins={[remarkGfm]} skipHtml disallowedElements={['img', 'input']}
    components={{
      a: ({ children, href }) => href
        ? <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>
        : <span>{children}</span>,
      pre: ({ children }) => <div className="data-block">
        <Copy text={nodeText(children)} /><pre>{children}</pre>
      </div>,
      table: ({ children }) => <div className="table-scroll"><table>{children}</table></div>,
    }}>{text}</Markdown>
})

export type TraceTool = {
  key: string
  id: string
  name: string
  turn: string
  number: number
  status: string
  arguments?: string
  content?: string | null
  error?: unknown
  start?: number
  end?: number
  events: AgentEvent[]
  omitted?: string
  loadedSkill?: LoadedSkill
}

export function collectTools(events: AgentEvent[]) {
  const tools = new Map<string, TraceTool>()
  for (const event of events) {
    const d = event.data
    if (!event.name.startsWith('tool.') || !d.tool_call_id) continue
    const key = `${event.turn_id}:${d.request_number ?? 0}:${d.tool_call_id}`
    const tool = tools.get(key) ?? {
      key, id: d.tool_call_id, name: d.name ?? '工具', turn: event.turn_id,
      number: d.request_number ?? 0, status: '准备中', events: [],
    }
    tool.events.push(event)
    if (event.name === 'tool.started') Object.assign(tool, { start: event.time, arguments: d.arguments })
    if (event.name === 'tool.approval_requested') tool.status = '等待审批'
    if (event.name === 'tool.executing') tool.status = '运行中'
    if (event.name === 'tool.blocked') Object.assign(tool, { status: '已拦截', error: d.reason })
    if (event.name === 'tool.finished') Object.assign(tool, {
      end: event.time, status: d.status ?? '已结束', content: d.content, error: d.error,
      loadedSkill: d.loaded_skill,
    })
    if (d.omitted) tool.omitted = d.omitted
    tools.set(key, tool)
  }
  return [...tools.values()]
}

const states: Record<string, string> = {
  COMPLETED: '完成', ERROR: '出错', FAILED: '失败', CANCELLED: '已停止', PAUSED_LIMIT: '预算暂停', TERMINATED: '已终止',
}

export function ToolRow({ tool, locate, inspect }: {
  tool: TraceTool
  locate?: boolean
  inspect?: () => void
}) {
  return <details className="tool-row" id={locate ? `tool-${tool.key}` : undefined}>
    <summary><strong>工具调用</strong><span className="tool-name">{tool.name}</span>
      <span className="tool-status">{tool.start != null && tool.end != null &&
        <span title="从工具开始到结束的时间，包含等待审批">
          {Math.max(0, tool.end - tool.start).toFixed(2)}s
        </span>}
        <span>{states[tool.status] ?? tool.status}</span>
      </span>
    </summary>
    <div className="tool-body">
      {tool.arguments != null && <><h4>参数</h4><DataBlock value={tool.arguments} /></>}
      {tool.content != null && <><h4>结果</h4><DataBlock value={tool.content} /></>}
      {tool.error != null && <><h4>错误</h4><DataBlock value={tool.error} /></>}
      {tool.omitted && <p className="muted">{tool.omitted}</p>}
      {tool.events.length > 0 && <details className="minor">
        <summary>关联事件 · {tool.events.length}</summary>
        <ul>{tool.events.map((event) => <li key={event.id}>#{event.id} · {event.name}</li>)}</ul>
      </details>}
      {inspect && <button className="small" onClick={inspect}>定位到对话</button>}
    </div>
  </details>
}

function messageText(message: Message) {
  if (typeof message.content === 'string' && message.content.trim()) return message.content
  if (Array.isArray(message.content)) return message.content
    .filter((part) => part.type === 'text').map((part) => part.text ?? '').join('')
  return message.refusal ?? ''
}

function AssistantMessage({ text, thinking }: { text: string; thinking?: string | null }) {
  if (!text && !thinking) return null
  return <article className="assistant-message">
    {thinking && <details className="thinking"><summary>Thinking</summary>
      <div className="reasoning">{thinking}</div>
    </details>}
    {text && <RichText text={text} />}
  </article>
}

type Retry = { index: number; disabled: boolean; run: () => void }

function UserMessage({ text, state, retry, attachments = [], history }: {
  text: string; state?: Message['daphnis_turn_state']; retry?: Retry; attachments?: AttachmentInfo[]
  history?: AttachmentHistory
}) {
  return <article className={`user-message${state === 'ABANDONED' ? ' abandoned' : ''}`}>
    {retry && <button type="button" className="retry-mark" disabled={retry.disabled}
      aria-label="重试此消息" title="尚无已保存回复，点击重试" onClick={retry.run}>!</button>}
    <RichText text={text} />
    <AttachmentLabels files={attachments} history={history} />
    {state === 'ABANDONED' && <small>已作废</small>}
    {state === 'TERMINATED' && <small>任务已终止</small>}
  </article>
}

function savedResult(message: Message, name?: string): { status: string; error?: unknown; loadedSkill?: LoadedSkill } {
  if (message.daphnis_tool_state === 'PENDING') return { status: '等待恢复' }
  if (message.daphnis_tool_state === 'RUNNING') return { status: '结果未知' }
  try {
    const value = JSON.parse(typeof message.content === 'string' ? message.content : '')
    if (value?.source === 'daphnis_recovery') {
      return { status: value.error?.code === 'execution_skipped' ? '未执行' : '结果未知', error: value.error }
    }
    if (value?.ok === false || value?.error) return { status: 'ERROR', error: value.error }
    if (value?.ok === true) {
      const skill = name === 'read_skill' && value.path === 'SKILL.md' && typeof value.skill_id === 'string'
        ? /^(?:global|project):([a-z0-9]+(?:-[a-z0-9]+)*)$/.exec(value.skill_id) : null
      return { status: 'COMPLETED', loadedSkill: skill ? { id: skill[0], name: skill[1] } : undefined }
    }
  } catch { /* Plain-text results are supported. */ }
  return { status: '已保存' }
}

function skillNotice(skill: LoadedSkill, shown: Set<string>, anchor?: string) {
  if (shown.has(skill.id)) return null
  shown.add(skill.id)
  return <p key={anchor ?? skill.id} id={anchor} className="muted skill-loaded">已加载 {skill.name} Skill</p>
}

function transcriptTool(tool: TraceTool, shown: Set<string>) {
  return tool.loadedSkill
    ? skillNotice(tool.loadedSkill, shown, `tool-${tool.key}`)
    : <ToolRow key={tool.key} tool={tool} locate />
}

function committedRows({
  messages, tools, turns, retry, recoveryTurn, attachmentRefs, promptGroups, sessionId, loadImage, loadDocument,
}: {
  messages: Message[]
  tools: TraceTool[]
  turns: { id: string; start: number; end: number }[]
  retry?: Retry
  recoveryTurn?: string
  attachmentRefs: View['state']['attachment_refs']
  promptGroups: View['state']['prompt_groups']
  sessionId: string
  loadImage: LoadAttachment
  loadDocument: LoadAttachment
}) {
  let calls: Message['tool_calls'] = null
  let request = 0
  const groups = new Map((promptGroups ?? []).map((group) => [group.start, group]))
  let promptEnd = 0
  const shownSkills = new Set<string>()
  const rows = messages.map((message, index) => {
    const group = groups.get(index)
    if (group) {
      promptEnd = group.end
      request = 0
      calls = null
      shownSkills.clear()
      const state = message.daphnis_turn_state
      const taskRetry = retry?.index === index ? retry : undefined
      return <Fragment key={group.id}><section className="prompt-transcript" aria-label="MCP Prompt">
        <div className="prompt-source muted">
          <span>{group.server} / {group.prompt}</span>
          {state === 'ABANDONED' && <small>已作废</small>}
          {state === 'TERMINATED' && <small>任务已终止</small>}
          {taskRetry && (messages[group.end - 1]?.role === 'user'
            ? <button type="button" className="mcp-text-button" disabled={taskRetry.disabled}
              onClick={taskRetry.run}>重试 Prompt</button>
            : <small>当前模型无法续写 Assistant 结尾的 Prompt，请发送新消息。</small>)}
        </div>
        {messages.slice(index, group.end).map((item, offset) => item.role === 'user'
          ? <UserMessage key={offset} text={messageText(item)} attachments={attachmentRefs?.[index + offset]} />
          : <div key={offset}>
            <AssistantMessage text={messageText(item)} />
            {attachmentRefs?.[index + offset] && <div className="assistant-message">
              <small className="muted">Assistant · Resource</small>
              <AttachmentLabels files={attachmentRefs[index + offset]} />
            </div>}
          </div>)}
      </section>
        {message.daphnis_loaded_skills?.map((skill) => skillNotice(skill, shownSkills))}
      </Fragment>
    }
    if (index < promptEnd) return null
    if (message.role === 'user') {
      request = 0
      calls = null
      shownSkills.clear()
      return <Fragment key={index}>
        <UserMessage text={messageText(message)} attachments={attachmentRefs?.[index]}
          history={{ sessionId, messageIndex: index, loadImage, loadDocument }}
          state={message.daphnis_turn_state} retry={retry?.index === index ? retry : undefined} />
        {message.daphnis_loaded_skills?.map((skill) => skillNotice(skill, shownSkills))}
      </Fragment>
    }
    if (message.role === 'assistant') {
      request += 1
      calls = message.tool_calls
      return <AssistantMessage key={index} text={messageText(message)} thinking={message.reasoning_content} />
    }
    if (message.role !== 'tool') return null
    const call = calls?.find(({ id }) => id === message.tool_call_id)
    const turn = turns.find(({ start, end }) => index >= start && index < end)
    const resumed = message.daphnis_tool_state && recoveryTurn
      ? tools.find((tool) => tool.turn === recoveryTurn && tool.number === 0 && tool.id === message.tool_call_id)
      : undefined
    const observed = resumed ?? tools.find((tool) => tool.turn === turn?.id && tool.number === request && tool.id === message.tool_call_id)
    const id = message.tool_call_id ?? String(index)
    const saved = savedResult(message, call?.function.name)
    const tool: TraceTool = {
      key: observed?.key ?? `saved-${index}`, id, turn: turn?.id ?? '', number: request,
      name: call?.function.name ?? '工具结果', ...saved,
      events: [], ...observed, status: resumed?.status ?? (observed?.end != null ? observed.status : saved.status),
      arguments: call?.function.arguments ?? observed?.arguments,
      content: typeof message.content === 'string' ? message.content : observed?.content,
      loadedSkill: saved.loadedSkill ?? resumed?.loadedSkill,
    }
    return <Fragment key={index}>{transcriptTool(tool, shownSkills)}</Fragment>
  })
  return { rows, shownSkills }
}

export function Transcript({ view, tools, retry, loadImage, loadDocument }: {
  view: View; tools: TraceTool[]; retry?: Retry; loadImage: LoadAttachment; loadDocument: LoadAttachment
}) {
  const scroll = useRef<HTMLDivElement>(null)
  const follow = useRef(true)
  useEffect(() => {
    if (follow.current && scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight
  }, [view])
  const turns = view.events.flatMap((event) => {
    if (event.name !== 'turn.finished' || event.data.committed_count == null) return []
    const start = view.events.find((start) => start.name === 'turn.started' && start.turn_id === event.turn_id)
    return start?.data.base_count == null ? [] : [
      { id: event.turn_id, start: start.data.base_count, end: event.data.committed_count },
    ]
  })
  const { live, events } = view
  const run = view.run?.kind === 'compaction' ? null : view.run
  const promptGroups = view.state.prompt_groups
  const activeGroup = run && promptGroups?.find((group) => group.start === run.base_count)
  const committedInput = run && run.prompt !== null && view.state.messages[run.base_count]?.role === 'user'
  const committed = run
    ? view.state.messages.slice(0, activeGroup?.end ?? run.base_count + Number(!!committedInput))
    : view.state.messages
  const { rows, shownSkills } = committedRows({
    messages: committed, tools, turns, retry, attachmentRefs: view.state.attachment_refs, promptGroups,
    sessionId: view.session_id, loadImage, loadDocument,
    recoveryTurn: run?.prompt === null && !activeGroup ? run.turn_id : undefined,
  })
  if (run && run.prompt !== null && !committedInput) shownSkills.clear()
  const activeTools = run ? tools.filter((tool) => tool.turn === run.turn_id && tool.number > 0) : []
  const finished = run?.turn_id
    ? events.filter((event) => event.turn_id === run.turn_id && event.name === 'model.finished' && event.data.purpose !== 'compaction')
    : []
  const requests = [...new Set([
    ...Object.keys(live).map(Number), ...activeTools.map((tool) => tool.number),
    ...finished.map((event) => event.data.request_number ?? 0),
  ])].sort((a, b) => a - b)
  const outcome = events.findLast((event) => event.name === 'web.send_finished')
  return <div className="transcript" ref={scroll} onScroll={(event) => {
    const el = event.currentTarget
    follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }}>
    <div className="transcript-inner">
      {view.state.messages.length === 0 && !run && <div className="empty">
        <p className="eyebrow">DAPHNIS / CODING AGENT</p>
        <h2>今天要处理什么？</h2>
        <p>描述任务，或让 Agent 检查当前工作区。</p>
      </div>}
      {rows}
      {run && <>
        {run.prompt !== null && !committedInput && <UserMessage text={run.prompt}
          attachments={run.attachments?.map((name, index) => ({ id: String(index), name }))} />}
        {requests.map((number) => {
          const message = finished.find((event) => event.data.request_number === number)?.data.response?.choices?.[0]?.message
          return <div key={number}>
            <AssistantMessage text={message ? messageText(message) : live[number]?.content ?? ''}
              thinking={message?.reasoning_content ?? live[number]?.reasoning} />
            {activeTools.filter((tool) => tool.number === number).map((tool) =>
              transcriptTool(tool, shownSkills))}
          </div>
        })}
        <p className="muted live-status" role="status">
          {view.status.question ? '等待回答' : view.status.approval ? '等待工具审批' : 'Agent 运行中…'}
        </p>
      </>}
      {!run && outcome && outcome.data.status !== 'COMPLETED' && <div className="outcome" role="status">
        <strong>{states[outcome.data.status ?? ''] ?? '未完成'}</strong>
        {outcome.data.notice && <p>{outcome.data.notice}</p>}
        {typeof outcome.data.error === 'string' && <p>{outcome.data.error}</p>}
      </div>}
    </div>
  </div>
}
