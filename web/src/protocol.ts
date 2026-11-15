import { useEffect, useState } from 'react'

export type Model = {
  id: string
  provider: string
  context_window: number | null
  supports_tools: boolean | null
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
}

export type MCPServer = {
  name: string
  options: MCPOptions
  enabled: boolean
  state: string
  tool_count: number
  error: string | null
}

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

type State = {
  session: Session
  sessions: Session[]
  projects: Project[]
  projects_error: string | null
  mcp_servers: MCPServer[]
  mcp_error: string | null
  history_issues: HistoryIssue[]
  history_error: string | null
  messages: Message[]
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

type Frame = {
  session_id: string
  revision: number
  cursor: number
  reset: boolean
  state?: State
  status: Status
  run: { id: string; turn_id: string; session_id: string; prompt: string | null; base_count: number } | null
  live: Record<string, LiveText>
  events: AgentEvent[]
}

export type View = Omit<Frame, 'state'> & { state: State }
export type Command = (
  action: string,
  fields?: Record<string, unknown>,
) => Promise<{ notice: string | null; revision: number }>

export function useConnection(token: string) {
  const [view, setView] = useState<View | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    let disposed = false
    let socket: WebSocket | undefined
    let retry: number | undefined
    let delay = 300
    setView(null)
    setConnected(false)
    setError('')
    if (!token) return
    function connect() {
      const url = new URL('/api/events', location.href)
      url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(url)
      socket = ws
      ws.onopen = () => ws.send(token)
      ws.onmessage = ({ data }) => {
        if (disposed) return
        const frame = JSON.parse(data) as Frame & { error?: string }
        if (frame.error) {
          setError(frame.error)
          return
        }
        delay = 300
        setError('')
        setConnected(true)
        setView((previous) => {
          const state = frame.state ?? previous?.state
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
      socket?.close()
    }
  }, [token])

  const command: Command = async (action, fields = {}) => {
    if (!connected || !view) throw new Error('连接中断，请等待重新连接。')
    let response: Response
    try {
      response = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + token },
        body: JSON.stringify({ action, session_id: view.session_id, ...fields }),
      })
    } catch {
      throw new Error('请求结果未知。请在连接恢复后查看当前状态。')
    }
    const result = await response.json()
    if (!response.ok) throw new Error(result.error || '操作失败。')
    return result
  }
  return { view, connected, error, command }
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
