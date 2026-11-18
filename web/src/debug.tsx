import { memo } from 'react'
import { clock } from './protocol'
import type { AgentEvent, Usage, View } from './protocol'
import { DataBlock, ToolRow } from './transcript'
import type { TraceTool } from './transcript'

export type Tab = 'events' | 'tools' | 'context' | 'requests'

const tabs: { id: Tab; title: string }[] = [
  { id: 'events', title: '事件' }, { id: 'tools', title: '工具' },
  { id: 'context', title: '上下文' }, { id: 'requests', title: '请求 / 响应' },
]

const names: Record<string, string> = {
  'turn.started': '新一轮对话开始',
  'turn.finished': 'Agent 结束本轮',
  'model.started': '开始模型请求',
  'model.request': '捕获模型请求',
  'model.delta': '收到流式分片（合并展示）',
  'model.usage': '模型报告 Token 用量',
  'model.finished': '收到完整模型响应',
  'model.response_error': '模型 HTTP 响应出错',
  'tool.started': '开始调用工具',
  'tool.executing': '执行工具',
  'tool.approval_requested': '需要确认工具权限',
  'tool.question_requested': '等待用户回答',
  'tool.blocked': '工具调用已拦截',
  'tool.finished': '工具调用结束',
  'web.send_finished': 'Operation 返回',
  'context.budget': '估算模型请求的上下文占用',
  'context.overflow': '模型确认上下文超限',
  'compaction.started': '开始压缩上下文',
  'compaction.finished': '压缩尝试结束',
  'compaction.committed': '随 Turn 提交压缩检查点',
  'compaction.discarded': '丢弃暂存的压缩检查点',
  'mcp.error': 'MCP 连接错误',
  'mcp.tools_changed': 'MCP 工具目录更新',
}

function Tokens({ title, usage }: { title: string; usage: Usage | null }) {
  const cachedTokens = usage?.cached_prompt_tokens ?? 0
  const cacheInput = usage?.cache_reported_prompt_tokens ?? 0
  const cacheRequests = usage?.cache_reported_requests ?? 0
  const rate = cacheInput > 0 ? cachedTokens / cacheInput : null
  return <div><strong>{title}</strong>{usage ? <dl className="facts">
    <div><dt>输入</dt><dd>{usage.prompt_tokens.toLocaleString()}</dd></div>
    <div><dt>输出</dt><dd>{usage.completion_tokens.toLocaleString()}</dd></div>
    <div><dt>合计</dt><dd>{usage.total_tokens.toLocaleString()}</dd></div>
    <div><dt>报告用量的请求</dt><dd>{usage.reported_requests} / {usage.requests}</dd></div>
    <div><dt>缓存命中</dt><dd>{cacheRequests > 0 ? `${cachedTokens.toLocaleString()} Tokens` : '未知'}</dd></div>
    <div><dt>缓存命中率</dt><dd>{rate === null ? '未知' : `${(rate * 100).toFixed(1)}%`}
      {rate !== null && (cacheRequests < usage.requests || usage.unrecorded) && <small className="muted">（仅基于已报告缓存信息的请求）</small>}
    </dd></div>
    <div><dt>缓存统计覆盖</dt><dd>{cacheRequests} / {usage.requests} 次请求</dd></div>
    {usage.unrecorded && <div><dt>历史用量</dt><dd>记录不完整</dd></div>}
  </dl> : <p className="muted">暂无</p>}</div>
}

function StatusPanel({ view }: { view: View }) {
  const { status, state } = view
  return <details className="debug-status">
    <summary>运行状态 <span className="muted">
      {status.approval ? '等待审批' : status.compacting ? '压缩上下文' : status.running ? '运行中' : '空闲'} · {status.session_usage.total_tokens.toLocaleString()} Tokens
    </span></summary>
    <dl className="facts">
      <div><dt>会话</dt><dd>{state.session.id}</dd></div>
      <div><dt>工作区</dt><dd>{state.session.workspace}</dd></div>
      <div><dt>模型</dt><dd>{state.model.provider} / {state.model.id}</dd></div>
      <div><dt>窗口</dt><dd>{state.model.context_window?.toLocaleString() ?? '未知'}</dd></div>
      {status.context_budget && <>
        <div><dt>最近请求输入估算</dt><dd>≈ {status.context_budget.input_tokens.toLocaleString()}（{status.context_budget.count_source === 'local_tokenizer' ? '本地分词' : '长度估算'}）</dd></div>
        {!!status.context_budget.unknown_documents && <div><dt>文档 Token</dt>
          <dd>{status.context_budget.unknown_documents} 份实际成本未知，输入估算含安全预留</dd></div>}
        <div><dt>输出预留</dt><dd>{status.context_budget.output_reserve.toLocaleString()}</dd></div>
        <div><dt>自动压缩阈值（输入 + 预留）</dt><dd>{!status.context_budget.auto_enabled ? '已关闭'
          : status.context_budget.threshold_blocked_by_reserve ? '不高于输出预留，未生效'
          : status.context_budget.trigger_tokens?.toLocaleString() ?? '未知'}</dd></div>
        <div><dt>安全上限（输入 + 预留）</dt><dd>{status.context_budget.hard_limit?.toLocaleString() ?? '未知'}</dd></div>
      </>}
      <div><dt>摘要检查点</dt><dd>{status.compacted_through ? `第 ${status.compacted_through} 条消息` : '无'}</dd></div>
      <div><dt>审批模式</dt><dd>{state.approval_mode}</dd></div>
    </dl>
    <Tokens title="本轮 Token" usage={status.turn_usage} />
    <Tokens title="会话 Token" usage={status.session_usage} />
  </details>
}

const Timeline = memo(function Timeline({ events }: { events: AgentEvent[] }) {
  if (!events.length) return <p className="muted">此会话暂无已捕获的事件。</p>
  return <ol className="timeline">{events.map((event) => (
    <li key={event.id}>
      <time>{clock(event.time)}</time>
      <details>
        <summary><strong>{event.name}</strong><span className="muted">
          {names[event.name] ?? event.name}{event.data.name ? ' · ' + event.data.name : ''}
          {event.data.chunks != null ? ` · ${event.data.chunks} 个流式分片` : ''}
        </span></summary>
        <DataBlock value={{ turn_id: event.turn_id, ...event.data }} />
      </details>
    </li>
  ))}</ol>
})

const Requests = memo(function Requests({ events }: { events: AgentEvent[] }) {
  const requests = new Map<string, AgentEvent[]>()
  for (const event of events) {
    if (!['model.started', 'model.request', 'model.finished', 'model.response_error'].includes(event.name)) continue
    const key = `${event.turn_id}:${event.data.request_number}`
    const items = requests.get(key) ?? []
    items.push(event)
    requests.set(key, items)
  }
  if (!requests.size) return <p className="muted">运行一轮后可查看模型请求和响应。</p>
  return <div className="request-list">{[...requests].reverse().map(([key, items]) => {
    const request = items.find((event) => event.name === 'model.request')
    const response = items.find((event) => event.name === 'model.finished' || event.name === 'model.response_error')
    const first = items[0]
    return <details className="request-row" key={key}>
      <summary>请求 {first.data.request_number} <span className="muted">
        {clock(first.time)}{first.data.purpose === 'compaction' ? ' · 历史摘要' : first.data.purpose === 'title' ? ' · 会话标题' : ''}{request?.data.http_status ? ` · HTTP ${request.data.http_status}` : ''}
      </span></summary>
      <h4>{request?.data.body && request.data.omitted ? '请求安全预览' : '请求 JSON'}</h4>
      {request?.data.omitted && <p className="muted">{request.data.omitted}</p>}
      {request?.data.body
        ? <DataBlock value={request.data.body} />
        : !request?.data.omitted && <p className="muted">暂无可用的 HTTP 请求正文。</p>}
      <h4>响应 JSON</h4>
      {response?.data.response || response?.data.body
        ? <DataBlock value={response.data.response ?? response.data.body} />
        : <p className="muted">{response?.data.omitted ?? '暂无完整 JSON 响应。'}</p>}
    </details>
  })}</div>
})

export function Debug({ view, tools, tab, select, close }: {
  view: View
  tools: TraceTool[]
  tab: Tab
  select: (tab: Tab) => void
  close: () => void
}) {
  const lastRequest = view.events.findLast((event) => event.name === 'model.request' && !event.data.purpose)
  return <aside className="debug" aria-label="调试详情">
    <div className="debug-heading">
      <button className="icon" aria-label="关闭调试面板" onClick={close}>›</button>
      <h2>调试详情</h2>
    </div>
    <div className="tabs" role="group" aria-label="调试页签">{tabs.map(({ id, title }) =>
      <button key={id} type="button" aria-pressed={tab === id} onClick={() => select(id)}>{title}</button>)}
    </div>
    <div className="debug-body" key={tab}>
      {tab === 'events' && <Timeline events={view.events} />}
      {tab === 'tools' && (tools.length ? tools.map((tool) => <ToolRow key={tool.key} tool={tool}
        inspect={document.getElementById(`tool-${tool.key}`) ? () => {
          const row = document.getElementById(`tool-${tool.key}`) as HTMLDetailsElement | null
          if (row) {
            row.open = true
            row.scrollIntoView({ block: 'center' })
            row.querySelector('summary')?.focus({ preventScroll: true })
          }
        } : undefined} />) : <p className="muted">此会话暂无已捕获的工具调用。</p>)}
      {tab === 'context' && <>
        <details className="context-group" open><summary>最近模型请求的上下文</summary>
          {lastRequest ? <>
            {lastRequest.data.omitted && <p className="muted">{lastRequest.data.omitted}</p>}
            {lastRequest.data.body?.messages
              ? <DataBlock value={lastRequest.data.omitted ? lastRequest.data.body : lastRequest.data.body.messages} />
              : !lastRequest.data.omitted && <p className="muted">已捕获请求，暂无可展示的消息。</p>}
          </> : <p className="muted">暂无已捕获的请求上下文。</p>}
        </details>
        <details className="context-group" open={!lastRequest}><summary>当前已提交消息</summary>
          <DataBlock value={view.state.messages} />
        </details>
      </>}
      {tab === 'requests' && <Requests events={view.events} />}
      <StatusPanel view={view} />
      <p className="debug-note">调试数据保留在当前服务进程中，按会话有限缓存。浏览器收到的内容已执行凭据脱敏。</p>
    </div>
  </aside>
}
