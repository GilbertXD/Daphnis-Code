import { useCallback, useEffect, useRef, useState } from 'react'

const selectionKey = 'daphnis.selectedSession'

export type Model = {
  id: string
  provider: string
  context_window: number | null
  supports_tools: boolean | null
  supports_image: boolean | null
  supports_document: boolean | null
  thinking_modes: string[] | null
  configured?: boolean
}

export type Session = {
  id: string
  workspace: string
  provider: string
  model_id: string
  title: string | null
  project_id: string | null
  updated_at?: number
  message_count?: number
  recovery_kind?: Recovery['kind'] | null
}

export type Project = { id: string; name: string; workspace: string }

export type MCPOptions = {
  transport: 'stdio' | 'streamable_http'
  command?: string
  args?: string[]
  cwd?: string
  env?: Record<string, string>
  url?: string
  auth?: { type: 'bearer'; env: string }
  public_only?: boolean
}

export type MCPServer = {
  name: string
  options: MCPOptions
  enabled: boolean
  state: string
  tool_count: number
  error: string | null
}

export type MCPPrompt = {
  name: string
  title: string
  description: string
  arguments: { name: string; title: string; description: string; required: boolean }[]
}

export type ResourceText = {
  name: string
  uri: string
  mime_type: string
  text: string
  size_bytes: number
  redacted: boolean
}

export type ResourceDraft = ResourceText & { id: string }
export type ResourceEntry = {
  uri: string
  name: string
  title: string
  description: string
  mime_type: string
  variables: string[] | null
  error: string | null
}
export type ResourceCatalog = { entries: ResourceEntry[]; errors: string[] }

export type MCPPromptResult = {
  description: string
  redacted: boolean
  preview_id: string | null
  messages: { role: 'user' | 'assistant'; content: { type: string; text?: string; resource?: ResourceText; error?: string } }[]
}

export type PromptGroup = {
  id: string
  server: string
  prompt: string
  start: number
  end: number
  edited: boolean
}

export type RegistryRemote = {
  transport: string
  url: string
  auth: 'none' | 'bearer' | 'unsupported'
  reason: string | null
}

export type RegistryServer = {
  name: string
  version: string
  publisher: string
  title: string
  description: string
  repository: string
  source: string
  status: string
  remotes: RegistryRemote[]
  packages: { type: string; identifier: string; version: string }[]
}

export type RegistryPage = { servers: RegistryServer[]; cursor: string | null }

type HistoryIssue = { filename: string; error: string }

export type ToolCall = { id: string; function: { name: string; arguments: string } }
export type Message = {
  role: string
  content?: string | { type: string; text?: string }[] | null
  refusal?: string | null
  reasoning_content?: string | null
  tool_calls?: ToolCall[] | null
  tool_call_id?: string
  daphnis_tool_state?: 'PENDING' | 'RUNNING'
  daphnis_turn_state?: 'ABANDONED' | 'CONTINUE' | 'TERMINATED'
}

export type Recovery = { kind: 'FAILED_USER' | 'INTERRUPTED_TURN'; key: string; message_index: number }

export type ApprovalMode = 'ask' | 'auto' | 'full'

export type Approval = {
  request_id: string
  tool_call_id: string
  tool_name: string
  workspace: string
  reason: string
  details: Record<string, unknown>
  can_trust: boolean
}

export type Usage = {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  requests: number
  reported_requests: number
  unrecorded: boolean
}

export type ContextBudget = {
  input_tokens: number
  context_window: number | null
  output_reserve: number
  compaction_threshold: number | null
  needs_compaction: boolean
  auto_enabled: boolean
  trigger_tokens: number | null
  hard_limit: number | null
  threshold_blocked_by_reserve: boolean
  count_source: 'estimate' | 'local_tokenizer'
  unknown_documents?: number
}

export type Status = {
  running: boolean
  approval: Approval | null
  session_usage: Usage
  turn_usage: Usage | null
  context_budget: ContextBudget | null
  compacting: boolean
  compacted_through: number
}

export type EventData = {
  name?: string
  request_number?: number
  purpose?: string
  trigger?: string
  tool_call_id?: string
  arguments?: string
  content?: string | null
  status?: string
  error?: unknown
  notice?: string | null
  reason?: string
  request?: Approval
  body?: Record<string, unknown> | null
  response?: { choices?: { message?: Message }[] }
  http_status?: number
  base_count?: number
  committed_count?: number
  chunks?: number
  omitted?: string
}

export type AgentEvent = { id: number; name: string; turn_id: string; time: number; data: EventData }
export type LiveText = { content: string; reasoning: string; count: number }

export type ReadingMode = 'text' | 'visual'

export type AttachmentInfo = {
  id: string
  name: string
  type?: 'text' | 'image' | 'document'
  mime_type?: string
  width?: number
  height?: number
  size_bytes?: number
  reading_mode?: ReadingMode
}

export type LoadAttachment = (
  sessionId: string, messageIndex: number, attachmentId: string, signal: AbortSignal,
) => Promise<Blob>

export type SkillScope = 'global' | 'project'
export type SkillState = 'disabled' | 'always' | 'on_demand'
export type Skill = {
  id: string
  scope: SkillScope
  name: string
  description: string
  state: SkillState
  error: string | null
}

type State = {
  session: Session
  sessions: Session[]
  projects: Project[]
  projects_error: string | null
  mcp_servers: MCPServer[]
  mcp_error: string | null
  mcp_prompt_servers: string[]
  mcp_resource_servers: string[]
  skills: Skill[]
  history_issues: HistoryIssue[]
  history_error: string | null
  messages: Message[]
  prompt_groups: PromptGroup[]
  attachment_refs?: Record<number, AttachmentInfo[]>
  recovery: Recovery | null
  model: Model
  models: Model[]
  thinking: string | null
  system_prompt: string
  all_tools: string[]
  enabled_tools: string[]
  effective_tools: string[]
  approval_mode: ApprovalMode
}

type SharedState = Pick<State,
  'session' | 'sessions' | 'projects' | 'projects_error' | 'mcp_servers' | 'mcp_error' | 'mcp_prompt_servers'
  | 'mcp_resource_servers' | 'history_issues' | 'history_error' | 'all_tools' | 'effective_tools' | 'skills'
>

type Frame = {
  session_id: string
  selected_for?: string
  revision: number
  cursor: number
  reset: boolean
  state?: State
  shared?: SharedState
  status: Status
  run: ({ id: string; turn_id: string; session_id: string } & (
    { kind: 'compaction' } | { kind?: 'turn'; prompt: string | null; base_count: number; attachments?: string[] }
  )) | null
  live: Record<string, LiveText>
  events: AgentEvent[]
}

export type View = Omit<Frame, 'state' | 'shared' | 'selected_for'> & { state: State }
export type Command = (
  action: string,
  fields?: Record<string, unknown> | FormData,
) => Promise<{
  notice: string | null; revision: number; session_id?: string; registry_page?: RegistryPage; registry_server?: RegistryServer
  mcp_prompts?: MCPPrompt[]; mcp_prompt_result?: MCPPromptResult
  mcp_resource_catalog?: ResourceCatalog; mcp_resource_preview?: ResourceDraft[]
}>

export function useConnection(token: string) {
  const [view, setView] = useState<View | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  const selected = useRef(sessionStorage.getItem(selectionKey) ?? '')
  const socket = useRef<WebSocket | null>(null)

  function rememberSelection(sessionId: string) {
    selected.current = sessionId
    sessionStorage.setItem(selectionKey, sessionId)
  }

  useEffect(() => {
    let disposed = false
    let retry: number | undefined
    let delay = 300
    setView(null)
    setConnected(false)
    setError('')
    if (!token) return
    function connect() {
      const url = new URL('/api/events', location.href)
      url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      const initialSelection = selected.current
      if (initialSelection) url.searchParams.set('session_id', initialSelection)
      const ws = new WebSocket(url)
      socket.current = ws
      ws.onopen = () => {
        ws.send(token)
        if (selected.current !== initialSelection) ws.send(JSON.stringify({ session_id: selected.current }))
      }
      ws.onmessage = ({ data }) => {
        if (disposed) return
        const frame = JSON.parse(data) as Frame & { error?: string }
        if (frame.error) {
          setError(frame.error)
          return
        }
        if (frame.session_id !== selected.current && frame.selected_for !== selected.current) return
        if (frame.session_id !== selected.current) rememberSelection(frame.session_id)
        delay = 300
        setError('')
        setConnected(true)
        setView((previous) => {
          const previousState = previous?.session_id === frame.session_id ? previous.state : undefined
          const state = frame.state ?? (previousState && { ...previousState, ...frame.shared })
          if (!state) return null
          const reset = frame.reset || previous?.session_id !== frame.session_id
          const events = reset
            ? frame.events
            : frame.events.length
              ? [...(previous?.events ?? []), ...frame.events.filter((event) => event.id > (previous?.cursor ?? 0))]
              : previous?.events ?? []
          return { ...frame, state, events }
        })
      }
      ws.onclose = ({ code }) => {
        if (disposed) return
        setConnected(false)
        if (code === 4401) {
          setView(null)
          setError('访问令牌无效。请使用当前 Daphnis 服务打印的令牌。')
          return
        }
        retry = window.setTimeout(connect, delay)
        delay = Math.min(delay * 2, 5000)
      }
    }
    connect()
    return () => {
      disposed = true
      window.clearTimeout(retry)
      socket.current?.close()
      socket.current = null
    }
  }, [token])

  const command: Command = async (action, fields = {}) => {
    if (!connected || !view || view.session_id !== selected.current) throw new Error('连接中断，请等待重新连接。')
    const multipart = fields instanceof FormData
    if (multipart) fields.set('session_id', view.session_id)
    let response: Response
    try {
      response = await fetch(action === 'send_attachments' ? '/api/attachments/send' : '/api/command', {
        method: 'POST',
        headers: {
          Authorization: 'Bearer ' + token,
          ...(multipart ? {} : { 'Content-Type': 'application/json' }),
        },
        body: multipart ? fields : JSON.stringify({ action, session_id: view.session_id, ...fields }),
      })
    } catch {
      throw new Error('请求结果未知。请在连接恢复后查看当前状态。')
    }
    const result = await response.json()
    if (!response.ok) throw new Error(result.error || '操作失败。')
    if (action === 'send_attachments' && response.status !== 202) throw new Error('附件提交结果未知。请重试。')
    if (result.session_id && result.session_id !== selected.current) {
      rememberSelection(result.session_id)
      setConnected(false)
      if (socket.current?.readyState === WebSocket.OPEN) {
        socket.current.send(JSON.stringify({ session_id: result.session_id }))
      }
    }
    return result
  }
  const loadAttachment = useCallback(async (
    kind: 'image' | 'document', sessionId: string, messageIndex: number, attachmentId: string, signal: AbortSignal,
  ) => {
    const query = new URLSearchParams({
      session_id: sessionId, message_index: String(messageIndex), attachment_id: attachmentId,
    })
    const response = await fetch(`/api/attachments/${kind}?` + query, {
      headers: { Authorization: 'Bearer ' + token },
      mode: 'same-origin', redirect: 'error', cache: 'no-store', signal,
    })
    if (!response.ok) throw new Error(kind === 'image' ? '图片不可用' : '文档不可用')
    return response.blob()
  }, [token])
  const loadImage: LoadAttachment = useCallback((...args) => loadAttachment('image', ...args), [loadAttachment])
  const loadDocument: LoadAttachment = useCallback((...args) => loadAttachment('document', ...args), [loadAttachment])
  return { view, connected, error, command, loadImage, loadDocument }
}

export function sessionTitle(session: Session) {
  return session.title ?? (session.updated_at ? `会话 ${session.id.slice(0, 8)}` : '新会话')
}

export function readableJson(value: unknown): string {
  if (typeof value === 'string') {
    try { return JSON.stringify(JSON.parse(value), null, 2) }
    catch { return value }
  }
  return JSON.stringify(value, null, 2) ?? ''
}

export const clock = (seconds: number) =>
  new Date(seconds * 1000).toLocaleTimeString('zh-CN', { hour12: false })
