import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { Debug } from './debug'
import type { Tab } from './debug'
import { sessionTitle, useConnection } from './protocol'
import type { Approval, Command, Model, Session, View } from './protocol'
import { collectTools, DataBlock, Transcript } from './transcript'
import './style.css'

const tokenKey = 'daphnis.access'
const modelKey = (model: Model) => JSON.stringify([model.provider, model.id])

function Dialog({ title, close, children }: { title: string; close: () => void; children: ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog ref={dialog} onCancel={(event) => { event.preventDefault(); close() }} aria-label={title}>
    <div className="dialog-heading"><h2>{title}</h2><button className="icon" aria-label="关闭" onClick={close}>×</button></div>
    {children}
  </dialog>
}

function Sidebar({ sessions, selected, disabled, visible, historyError, onNew, onSelect, onEdit }: {
  sessions: Session[]
  selected: string
  disabled: boolean
  visible: boolean
  historyError: string | null
  onNew: () => void
  onSelect: (id: string) => void
  onEdit: (action: 'rename' | 'delete', session: Session) => void
}) {
  return <aside className={`sidebar ${visible ? 'sidebar-visible' : ''}`} aria-label="会话历史">
    <div className="section-heading"><h2>历史会话</h2>
      <button className="icon" onClick={onNew} disabled={disabled} aria-label="新建会话" title="新建会话">＋</button>
    </div>
    <nav>{historyError && <p className="history-error" role="alert">{historyError}</p>}
      {sessions.map((session) => <div className={`session-row ${session.id === selected ? 'selected' : ''}`} key={session.id}>
      <button className="session-select" aria-current={session.id === selected ? 'page' : undefined}
        disabled={disabled && session.id !== selected} onClick={() => onSelect(session.id)}>
        <strong>{sessionTitle(session)}</strong>
        <span className="muted">{session.updated_at
          ? new Date(session.updated_at * 1000).toLocaleString('zh-CN', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
          }) : '当前工作区'}</span>
      </button>
      <details className="session-menu">
        <summary aria-label={`管理 ${sessionTitle(session)}`} aria-disabled={disabled}
          onClick={(event) => { if (disabled) event.preventDefault() }}>⋯</summary>
        <div>{(['rename', 'delete'] as const).map((action) =>
          <button key={action} disabled={disabled} onClick={(event) => {
            event.currentTarget.closest('details')?.removeAttribute('open')
            onEdit(action, session)
          }}>{action === 'rename' ? '重命名' : '删除'}</button>)}</div>
      </details>
    </div>)}</nav>
    <div className="sidebar-footer"><span className="mono">DAPHNIS</span><span>本地工作区</span></div>
  </aside>
}

function ApprovalCard({ request, disabled, choose }: {
  request: Approval
  disabled: boolean
  choose: (decision: string) => void
}) {
  return <section className="approval" aria-label="工具审批">
    <div className="approval-heading"><strong>工具审批</strong><span className="mono">{request.tool_name}</span></div>
    <p>{request.reason}</p>
    <details open><summary>检查操作详情</summary><DataBlock value={request.details} /></details>
    <div className="actions">
      <button disabled={disabled} onClick={() => choose('deny')}>拒绝</button>
      {request.can_trust && <button disabled={disabled} onClick={() => choose('trust_session')}>本会话信任此工具</button>}
      <button className="primary" disabled={disabled} onClick={() => choose('allow_once')}>允许一次</button>
    </div>
  </section>
}

function ToolsDialog({ view, close, apply, disabled, notice }: {
  view: View
  close: () => void
  apply: (names: string[]) => Promise<boolean>
  disabled: boolean
  notice: string
}) {
  const [selected, setSelected] = useState(() => new Set(view.state.enabled_tools))
  const tools = [...new Set([...view.state.all_tools, ...view.state.enabled_tools])]
  return <Dialog title="工具配置" close={close}>
    <p className="muted">启用的工具仍按 Daphnis 的权限规则执行。</p>
    <div className="tool-options">{tools.map((name) =>
      <label key={name}><input type="checkbox" checked={selected.has(name)} onChange={(event) => {
        const checked = event.currentTarget.checked
        setSelected((current) => {
          const next = new Set(current)
          if (checked) next.add(name)
          else next.delete(name)
          return next
        })
      }} />
        <span className="mono">{name}</span>
        {!view.state.all_tools.includes(name) && <small className="muted">当前不可用</small>}
      </label>)}</div>
    {notice && <p role="status">{notice}</p>}
    <div className="actions">
      <button type="button" onClick={() => setSelected(new Set())}>停用全部</button>
      <button className="primary" disabled={disabled} onClick={async () => {
        if (await apply(tools.filter((name) => selected.has(name)))) close()
      }}>应用</button>
    </div>
  </Dialog>
}

type Modal = { action: 'tools' } | { action: 'rename' | 'delete'; session: Session }

function Workspace({ view, connected, command }: {
  view: View
  connected: boolean
  command: Command
}) {
  const [open, setOpen] = useState(false)
  const [sidebar, setSidebar] = useState(false)
  const [tab, setTab] = useState<Tab>('events')
  const [modal, setModal] = useState<Modal | null>(null)
  const [title, setTitle] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [acknowledgedRevision, setAcknowledgedRevision] = useState(0)
  const inFlight = useRef(false)
  const [notice, setNotice] = useState('')
  const { state, status, session_id: id } = view
  const disabled = busy || !connected || status.running || view.revision < acknowledgedRevision
  const draft = drafts[id] ?? ''
  const tools = useMemo(() => collectTools(view.events), [view.events])
  const sessions = state.sessions.some((session) => session.id === id)
    ? state.sessions : [state.session, ...state.sessions]

  async function perform(action: string, fields?: Record<string, unknown>) {
    if (inFlight.current) return false
    inFlight.current = true
    setBusy(true)
    setNotice('')
    try {
      const response = await command(action, fields)
      setAcknowledgedRevision(response.revision)
      setNotice(response.notice ?? '')
      return true
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '操作失败。')
      return false
    } finally {
      inFlight.current = false
      setBusy(false)
    }
  }

  function updateDraft(text: string) {
    setDrafts((current) => ({ ...current, [id]: text }))
  }

  async function send() {
    if (!disabled && draft.trim() && await perform('send', { prompt: draft })) {
      setDrafts((current) => current[id] === draft ? { ...current, [id]: '' } : current)
    }
  }

  return <div className={`shell ${open ? 'debug-open' : ''}`}>
    <header className="brand"><button className="icon menu-toggle" aria-label="显示会话历史"
      aria-expanded={sidebar} onClick={() => setSidebar(!sidebar)}>☰</button><strong>Daphnis</strong></header>
    <header className="topbar">
      <h1>{sessionTitle(state.session)}</h1>
      <span className={`connection ${connected ? '' : 'offline'}`} role="status">
        <span aria-hidden="true" />{connected ? status.compacting ? '压缩上下文' : status.running ? '运行中' : '已连接' : '重新连接中'}
      </span>
    </header>
    <Sidebar sessions={sessions} selected={id} disabled={disabled} visible={sidebar} historyError={state.history_error}
      onNew={async () => { if (await perform('new')) setSidebar(false) }}
      onSelect={async (target) => { if (target === id || await perform('restore', { id: target })) setSidebar(false) }}
      onEdit={(action, session) => {
        setNotice('')
        setTitle(session.title ?? sessionTitle(session))
        setModal({ action, session })
      }} />
    <main className="conversation" aria-label="Agent 对话">
      <Transcript key={id} view={view} tools={tools} />
      <div className="composer-zone">
        {status.approval && <ApprovalCard key={status.approval.request_id} request={status.approval}
          disabled={busy || !connected} choose={(decision) => void perform('approve', {
            request_id: status.approval?.request_id, decision,
          })} />}
        {!modal && notice && <p className="notice" role="status">{notice}</p>}
        <form className="composer" onSubmit={(event) => { event.preventDefault(); void send() }}>
          <textarea aria-label="消息" placeholder="请输入你的需求…" value={draft} rows={2}
            onChange={(event) => updateDraft(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing) {
                event.preventDefault()
                void send()
              }
            }} />
          <div className="composer-controls">
            <div className="selectors">
              <select aria-label="模型" title={`${state.model.provider} / ${state.model.id}`}
                value={modelKey(state.model)} disabled={disabled} onChange={(event) => {
                  const model = state.models.find((candidate) => modelKey(candidate) === event.target.value)
                  if (model) void perform('model', { provider: model.provider, model_id: model.id })
                }}>
                {[...new Set(state.models.map((model) => model.provider))].map((provider) =>
                  <optgroup label={provider} key={provider}>
                    {state.models.filter((model) => model.provider === provider).map((model) =>
                      <option key={modelKey(model)} value={modelKey(model)} disabled={!model.configured}>{model.id}</option>)}
                  </optgroup>)}
              </select>
              <select aria-label="Thinking" value={state.thinking ?? ''} disabled={disabled || !state.model.thinking_modes?.length}
                onChange={(event) => void perform('thinking', { mode: event.target.value || null })}>
                <option value="">Thinking · 默认</option>
                {state.model.thinking_modes?.map((mode) => <option key={mode} value={mode}>
                  Thinking · {mode === 'off' ? '关闭' : mode === 'on' ? '开启' : mode}
                </option>)}
              </select>
              <button type="button" className="tool-control" title="选择启用的工具"
                disabled={disabled || state.model.supports_tools !== true} onClick={() => {
                  setNotice('')
                  setModal({ action: 'tools' })
                }}>
                工具 {state.effective_tools.length ? `· ${state.effective_tools.length} 已启用` : '· 停用'} <span aria-hidden="true">⌄</span>
              </button>
            </div>
            {status.running
              ? <button className="primary send" type="button" disabled={busy || !connected || !view.run}
                onClick={() => void perform('cancel', { run_id: view.run?.id })} aria-label="停止" title="停止">■</button>
              : <button className="primary send" type="submit" disabled={disabled || !draft.trim()}
                aria-label="发送" title="发送 · Ctrl / ⌘ + Enter">↑</button>}
          </div>
        </form>
      </div>
    </main>
    {open
      ? <Debug key={id} view={view} tools={tools} tab={tab} select={setTab} close={() => setOpen(false)} />
      : <button className="debug-rail" type="button" aria-label="展开调试面板" title="展开调试面板"
        onClick={() => setOpen(true)}><span aria-hidden="true">‹</span></button>}
    {modal?.action === 'tools' && <ToolsDialog view={view} close={() => setModal(null)} disabled={disabled}
      notice={notice} apply={(names) => perform('tools', { names })} />}
    {modal && modal.action !== 'tools' && <Dialog title={modal.action === 'rename' ? '重命名会话' : '删除会话'}
      close={() => setModal(null)}>
      <form onSubmit={async (event) => {
        event.preventDefault()
        if (await perform(modal.action, modal.action === 'rename'
          ? { id: modal.session.id, title }
          : { id: modal.session.id })) {
          if (modal.action === 'delete') setDrafts(({ [modal.session.id]: _deleted, ...rest }) => rest)
          setModal(null)
        }
      }}>
        {modal.action === 'rename'
          ? <label className="field">会话名称<input autoFocus value={title} maxLength={120} required
            onChange={(event) => setTitle(event.currentTarget.value)} /></label>
          : <p>永久删除「{sessionTitle(modal.session)}」及其已保存的消息？</p>}
        {notice && <p role="status">{notice}</p>}
        <div className="actions"><button type="button" onClick={() => setModal(null)}>取消</button>
          <button className="primary" disabled={disabled} type="submit">{modal.action === 'rename' ? '保存' : '删除'}</button></div>
      </form>
    </Dialog>}
  </div>
}

function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem(tokenKey) ?? '')
  const [entry, setEntry] = useState('')
  const { view, connected, error, command } = useConnection(token)
  function login(event: FormEvent) {
    event.preventDefault()
    const value = entry.trim()
    if (!value) return
    sessionStorage.setItem(tokenKey, value)
    setToken(value)
    setEntry('')
  }

  if (view) return <Workspace view={view} connected={connected} command={command} />
  return <main className="access">
    <section className="access-card">
      <p className="eyebrow">DAPHNIS / LOCAL WORKSPACE</p>
      <h1>Daphnis</h1>
      {token && !error ? <><p role="status">正在连接本地 Agent…</p>
        <button onClick={() => { sessionStorage.removeItem(tokenKey); setToken('') }}>更换令牌</button>
      </> : <form onSubmit={login}>
        <p>输入启动终端中显示的访问令牌。</p>
        {error && <p role="alert">{error}</p>}
        <label className="field">访问令牌<input type="password" autoFocus autoComplete="off"
          value={entry} onChange={(event) => setEntry(event.currentTarget.value)} required /></label>
        <button className="primary" type="submit">连接 Agent</button>
      </form>}
    </section>
  </main>
}

createRoot(document.getElementById('root')!).render(<App />)
