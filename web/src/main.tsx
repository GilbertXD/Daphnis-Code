import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { Debug } from './debug'
import type { Tab } from './debug'
import { MCPManager } from './mcp'
import { PromptPicker } from './prompts'
import { sessionTitle, useConnection } from './protocol'
import type { Approval, Command, Model, Project, Session, View } from './protocol'
import { collectTools, DataBlock, Transcript } from './transcript'
import './style.css'

const tokenKey = 'daphnis.access'
const modelKey = (model: Model) => JSON.stringify([model.provider, model.id])
const sessionActions = [
  ['rename', '重命名'], ['move', '移动'], ['delete', '删除'],
] as const
type SessionAction = (typeof sessionActions)[number][0]

function Dialog({ title, close, children, className }: {
  title: string; close: () => void; children: ReactNode; className?: string
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog ref={dialog} className={className} onCancel={(event) => { event.preventDefault(); close() }} aria-label={title}>
    <div className="dialog-heading"><h2>{title}</h2><button className="icon" aria-label="关闭" onClick={close}>×</button></div>
    {children}
  </dialog>
}

function Sidebar({ projects, sessions, selected, disabled, visible, projectsError, historyError, historyIssueCount,
  onCreateProject, onRemoveProject, onNew, onSelect, onEdit }: {
  projects: Project[]
  sessions: Session[]
  selected: string
  disabled: boolean
  visible: boolean
  projectsError: string | null
  historyError: string | null
  historyIssueCount: number
  onCreateProject: () => void
  onRemoveProject: (id: string, name: string) => void
  onNew: (projectId: string | null) => void
  onSelect: (id: string) => void
  onEdit: (action: SessionAction, session: Session) => void
}) {
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => new Set())
  const sessionList = useRef<HTMLElement>(null)
  const selectedRow = useRef<HTMLDivElement>(null)
  useEffect(() => {
    selectedRow.current?.scrollIntoView({ block: 'nearest' })
  }, [selected, visible])
  useEffect(() => {
    function closeMenusOnOutsideClick(event: MouseEvent) {
      const target = event.target
      if (!(target instanceof Node)) return
      sessionList.current?.querySelectorAll<HTMLDetailsElement>('.session-menu[open]').forEach((menu) => {
        if (!menu.contains(target)) menu.open = false
      })
    }

    document.addEventListener('click', closeMenusOnOutsideClick, true)
    return () => document.removeEventListener('click', closeMenusOnOutsideClick, true)
  }, [])
  const groups: { id?: string | null; name: string; workspace?: string; sessions: Session[] }[] = [
    { id: null, name: '无项目', sessions: sessions.filter((session) => session.project_id === null) },
    ...projects.map((project) => ({
      ...project, sessions: sessions.filter((session) => session.project_id === project.id),
    })),
  ]
  const projectIds = new Set(projects.map((project) => project.id))
  const unlisted = sessions.filter((session) => session.project_id !== null && !projectIds.has(session.project_id))
  if (unlisted.length) groups.push({ name: '未登记项目', sessions: unlisted })

  function toggleGroup(key: string) {
    setCollapsedGroups((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return <aside className={`sidebar ${visible ? 'sidebar-visible' : ''}`} aria-label="会话历史">
    <div className="section-heading"><h2>历史会话</h2>
      <button className="project-create" onClick={onCreateProject} disabled={disabled}>新建项目</button>
    </div>
    {projectsError && <p className="history-notice" role="alert">{projectsError}</p>}
    {historyError && <p className="history-notice" role="alert">{historyError}</p>}
    {historyIssueCount > 0 && <p className="history-notice muted" role="status">{historyIssueCount} 个历史会话无法加载。</p>}
    <nav ref={sessionList}>
      {groups.map(({ id, name, workspace, sessions }) => {
        const key = id === null ? 'personal' : id ?? 'unlisted'
        const expanded = !collapsedGroups.has(key)
        return <section className="session-group" key={key} aria-label={name}>
          <div className="session-group-heading">
            <h3><button className="session-group-toggle" type="button" title={workspace}
              aria-expanded={expanded} onClick={() => toggleGroup(key)}>
              <span aria-hidden="true">{expanded ? '▾' : '▸'}</span><span>{name}</span>
            </button></h3>
            {id !== undefined && <button className="icon" type="button" onClick={() => {
              if (!expanded) toggleGroup(key)
              onNew(id)
            }} disabled={disabled}
              aria-label={`在${name}中新建会话`} title={`在${name}中新建会话`}>＋</button>}
            {typeof id === 'string' && <button className="icon" type="button" disabled={disabled}
              onClick={() => onRemoveProject(id, name)} aria-label={`移除项目 ${name}`} title="移除项目">×</button>}
          </div>
          {expanded && sessions.map((session) => <div className={`session-row ${session.id === selected ? 'selected' : ''}`}
            key={session.id} ref={session.id === selected ? selectedRow : undefined}>
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
              <div>{sessionActions.map(([action, label]) =>
                <button key={action} disabled={disabled} onClick={(event) => {
                  event.currentTarget.closest('details')?.removeAttribute('open')
                  onEdit(action, session)
                }}>{label}</button>)}</div>
            </details>
          </div>)}
        </section>
      })}
    </nav>
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
  const available = new Set(view.state.all_tools)
  const tools = [...new Set([...available, ...view.state.enabled_tools])]
  const builtIn: string[] = []
  const otherMcp: string[] = []
  const offlineMcp: string[] = []
  const groups = view.state.mcp_servers.map(({ name }) => ({
    name, prefix: `mcp__${name}__`, tools: [] as string[],
  }))
  for (const tool of tools) {
    if (!tool.startsWith('mcp__')) {
      builtIn.push(tool)
      continue
    }
    const group = groups.find(({ prefix }) => tool.startsWith(prefix))
    if (group) group.tools.push(tool)
    else if (available.has(tool)) otherMcp.push(tool)
    else offlineMcp.push(tool)
  }

  function changeSelection(names: Iterable<string>, checked: boolean) {
    setSelected((current) => {
      const next = new Set(current)
      for (const name of names) {
        if (checked) next.add(name)
        else next.delete(name)
      }
      return next
    })
  }

  function toolRows(names: string[], prefix = '') {
    return names.map((name) => <label key={name}>
      <input type="checkbox" checked={selected.has(name)} disabled={disabled}
        onChange={(event) => changeSelection([name], event.currentTarget.checked)} />
      <span className="mono">{name.slice(prefix.length)}</span>
      {!available.has(name) && <small className="muted">当前不可用</small>}
    </label>)
  }

  return <Dialog title="工具配置" close={close}>
    <p className="muted">选择仅应用于当前会话，工具调用仍遵循审批规则。</p>
    <div className="tool-options">
      <section>
        <h3>内置工具</h3>
        {builtIn.length ? toolRows(builtIn) : <p className="muted">暂无内置工具。</p>}
      </section>
      {!!(groups.length || otherMcp.length || offlineMcp.length) && <section>
        <h3>MCP 工具</h3>
        {groups.map((group) => {
          const online = group.tools.filter((name) => available.has(name))
          const checked = online.filter((name) => selected.has(name)).length
          const allChecked = online.length > 0 && checked === online.length
          const action = allChecked ? '全部停用' : '全部启用'
          return <div className="tool-group" key={group.name}>
            <details>
              <summary aria-label={`${group.name}，已选 ${checked} 个可用工具，共 ${online.length} 个`}>
                {group.name} <small className="muted">{checked} / {online.length}</small>
              </summary>
              <div className="tool-group-rows">
                {group.tools.length ? toolRows(group.tools, group.prefix) : <p className="muted">暂无工具。</p>}
              </div>
            </details>
            <button className="tool-group-action" type="button" aria-label={`${group.name} ${action}`}
              disabled={disabled || online.length === 0} onClick={() => changeSelection(online, !allChecked)}>
              {action}
            </button>
          </div>
        })}
        {!!otherMcp.length && <details className="tool-group">
          <summary>其他 MCP 工具</summary>
          <div className="tool-group-rows">{toolRows(otherMcp)}</div>
        </details>}
        {!!offlineMcp.length && <details className="tool-group">
          <summary>其他不可用 MCP 工具</summary>
          <div className="tool-group-rows">{toolRows(offlineMcp)}</div>
        </details>}
      </section>}
    </div>
    {notice && <p role="status">{notice}</p>}
    <div className="actions">
      <button type="button" disabled={disabled || !tools.some((name) => available.has(name) && selected.has(name))}
        onClick={() => changeSelection(available, false)}>停用全部可用工具</button>
      <button className="primary" disabled={disabled} onClick={async () => {
        if (await apply(tools.filter((name) => selected.has(name)))) close()
      }}>应用</button>
    </div>
  </Dialog>
}

function SystemPromptDialog({ initial, close, apply, disabled, notice }: {
  initial: string
  close: () => void
  apply: (prompt: string | null) => Promise<boolean>
  disabled: boolean
  notice: string
}) {
  const [prompt, setPrompt] = useState(initial)
  return <Dialog title="会话系统提示词" close={close}>
    <form onSubmit={async (event) => {
      event.preventDefault()
      if (await apply(prompt)) close()
    }}>
      <label className="field">系统提示词<textarea rows={8} value={prompt}
        onChange={(event) => setPrompt(event.currentTarget.value)} /></label>
      {notice && <p role="status">{notice}</p>}
      <div className="actions">
        <button type="button" disabled={disabled} onClick={async () => {
          if (await apply(null)) close()
        }}>恢复默认</button>
        <button className="primary" disabled={disabled || !prompt.trim()} type="submit">保存</button>
      </div>
    </form>
  </Dialog>
}

function ProjectDialog({ close, create, disabled, notice }: {
  close: () => void
  create: (name: string, workspace: string | null) => Promise<boolean>
  disabled: boolean
  notice: string
}) {
  const [name, setName] = useState('')
  const [workspace, setWorkspace] = useState('')
  return <Dialog title="新建项目" close={close}>
    <form onSubmit={async (event) => {
      event.preventDefault()
      if (await create(name.trim(), workspace.trim() || null)) close()
    }}>
      <label className="field">项目名称<input autoFocus value={name} maxLength={120} required
        onChange={(event) => setName(event.currentTarget.value)} /></label>
      <label className="field">工作区路径<input value={workspace} placeholder="留空使用当前工作区"
        onChange={(event) => setWorkspace(event.currentTarget.value)} /></label>
      {notice && <p role="status">{notice}</p>}
      <div className="actions">
        <button type="button" onClick={close}>取消</button>
        <button className="primary" disabled={disabled || !name.trim()} type="submit">创建</button>
      </div>
    </form>
  </Dialog>
}

function MoveSessionDialog({ session, projects, close, move, disabled, notice }: {
  session: Session
  projects: Project[]
  close: () => void
  move: (projectId: string | null, workspace: string | null) => Promise<boolean>
  disabled: boolean
  notice: string
}) {
  const [targetId, setTargetId] = useState(session.project_id ?? '')
  const target = projects.find((project) => project.id === targetId)
  const workspace = target?.workspace ?? session.workspace
  const changesWorkspace = workspace !== session.workspace
  const blocked = changesWorkspace && session.recovery_kind === 'INTERRUPTED_TURN'
  const canMove = !disabled && !blocked && targetId !== (session.project_id ?? '') && (!targetId || !!target)
  return <Dialog title="移动到项目" close={close}>
    <form onSubmit={async (event) => {
      event.preventDefault()
      if (canMove && await move(target?.id ?? null, target?.workspace ?? null)) close()
    }}>
      <p>移动「{sessionTitle(session)}」</p>
      <label className="field">目标项目<select value={targetId} onChange={(event) => setTargetId(event.currentTarget.value)}>
        <option value="">无项目</option>
        {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
      </select></label>
      <p>{changesWorkspace ? '后续工具操作将使用工作目录：' : '工作目录：'}<code>{workspace}</code></p>
      {blocked && <p role="status">请先打开该会话，继续或终止未完成的工具任务，再更换工作目录。</p>}
      {notice && <p role="status">{notice}</p>}
      <div className="actions">
        <button type="button" onClick={close}>取消</button>
        <button className="primary" disabled={!canMove} type="submit">确认移动</button>
      </div>
    </form>
  </Dialog>
}

type Modal = { action: 'tools' | 'project' | 'mcp' } | { action: 'system'; sessionId: string }
  | { action: 'prompts'; sessionId: string }
  | { action: 'confirm_full'; sessionId: string }
  | { action: 'rename' | 'delete'; session: Session }
  | { action: 'move'; session: Session }
  | { action: 'remove_project'; projectId: string; name: string }

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
  const composerInput = useRef<HTMLTextAreaElement>(null)
  const { state, status, session_id: id } = view
  useEffect(() => {
    setModal((current) => current?.action === 'prompts' && current.sessionId !== id ? null : current)
  }, [id])
  const disabled = busy || !connected || status.running || view.revision < acknowledgedRevision
  const recovery = state.recovery
  const interrupted = recovery?.kind === 'INTERRUPTED_TURN'
  const draft = drafts[id] ?? ''
  const tools = useMemo(() => collectTools(view.events), [view.events])
  const sessions = state.sessions.some((session) => session.id === id)
    ? state.sessions : [state.session, ...state.sessions]
  const sessionToMove = modal?.action === 'move' && (
    modal.session.id === id
      ? { ...state.session, recovery_kind: recovery?.kind }
      : state.sessions.find((session) => session.id === modal.session.id)
  )

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

  function updateDraft(text: string | ((draft: string) => string)) {
    setDrafts((current) => ({ ...current, [id]: typeof text === 'function' ? text(current[id] ?? '') : text }))
  }

  async function send() {
    if (!disabled && !interrupted && draft.trim() && await perform('send', { prompt: draft })) {
      setDrafts((current) => current[id] === draft ? { ...current, [id]: '' } : current)
    }
  }

  return <div className={`shell ${open ? 'debug-open' : ''}`}>
    <header className="brand"><button className="icon menu-toggle" aria-label="显示会话历史"
      aria-expanded={sidebar} onClick={() => setSidebar(!sidebar)}>☰</button><strong>Daphnis</strong></header>
    <header className="topbar">
      <h1>{sessionTitle(state.session)}</h1>
      <div className="topbar-actions">
        <button type="button" aria-haspopup="dialog" onClick={() => {
          setNotice('')
          setModal({ action: 'mcp' })
        }}>MCP</button>
        <span className={`connection ${connected ? '' : 'offline'}`} role="status">
          <span aria-hidden="true" />{connected ? status.compacting ? '压缩上下文' : status.running ? '运行中' : '已连接' : '重新连接中'}
        </span>
      </div>
    </header>
    <Sidebar projects={state.projects} sessions={sessions} selected={id} disabled={disabled} visible={sidebar}
      projectsError={state.projects_error} historyError={state.history_error}
      historyIssueCount={state.history_issues.length}
      onCreateProject={() => {
        setNotice('')
        setModal({ action: 'project' })
      }}
      onRemoveProject={(projectId, name) => {
        setNotice('')
        setModal({ action: 'remove_project', projectId, name })
      }}
      onNew={(projectId) => void perform('project_open', { project_id: projectId })}
      onSelect={async (target) => { if (target === id || await perform('restore', { id: target })) setSidebar(false) }}
      onEdit={(action, session) => {
        setNotice('')
        setTitle(session.title ?? sessionTitle(session))
        setModal({ action, session })
      }} />
    <main className="conversation" aria-label="Agent 对话">
      <Transcript key={id} view={view} tools={tools} retry={recovery?.kind === 'FAILED_USER'
        ? { index: recovery.message_index, disabled, run: () => void perform('retry', { key: recovery.key }) }
        : undefined} />
      <div className="composer-zone">
        {status.approval && <ApprovalCard key={status.approval.request_id} request={status.approval}
          disabled={busy || !connected} choose={(decision) => void perform('approve', {
            request_id: status.approval?.request_id, decision,
          })} />}
        {interrupted && !status.running && <div className="recovery" role="group" aria-label="未完成任务">
          <span>未完成任务</span>
          <div className="actions">
            <button type="button" disabled={disabled} onClick={() => void perform('continue', { key: recovery.key })}>继续</button>
            <button type="button" disabled={disabled} onClick={() => void perform('terminate', { key: recovery.key })}>终止</button>
          </div>
        </div>}
        {!modal && notice && <p className="notice" role="status">{notice}</p>}
        <form className="composer" onSubmit={(event) => { event.preventDefault(); void send() }}>
          <textarea ref={composerInput} aria-label="消息" placeholder="请输入你的需求…" value={draft} rows={2}
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
              <button type="button" disabled={!connected} onClick={() => {
                setNotice('')
                setModal({ action: 'prompts', sessionId: id })
              }}>模板</button>
              <button type="button" disabled={disabled} onClick={() => {
                setNotice('')
                setModal({ action: 'system', sessionId: id })
              }}>系统提示词</button>
              <select aria-label="工具审批模式"
                value={state.approval_mode} disabled={disabled}
                onChange={(event) => {
                  const mode = event.currentTarget.value
                  if (mode === 'full') {
                    setNotice('')
                    setModal({ action: 'confirm_full', sessionId: id })
                  } else {
                    void perform('set_approval_mode', { mode })
                  }
                }}>
                <option value="ask">ASK</option>
                <option value="auto">AUTO</option>
                <option value="full">FULL</option>
              </select>
            </div>
            {status.running
              ? <button className="primary send" type="button" disabled={busy || !connected || !view.run}
                onClick={() => void perform('cancel', { run_id: view.run?.id })} aria-label="停止" title="停止">■</button>
              : <button className="primary send" type="submit" disabled={disabled || interrupted || !draft.trim()}
                aria-label="发送" title="发送 · Ctrl / ⌘ + Enter">↑</button>}
          </div>
        </form>
      </div>
    </main>
    {open
      ? <Debug key={id} view={view} tools={tools} tab={tab} select={setTab} close={() => setOpen(false)} />
      : <button className="debug-rail" type="button" aria-label="展开调试面板" title="展开调试面板"
        onClick={() => setOpen(true)}><span aria-hidden="true">‹</span></button>}
    {modal?.action === 'tools' && <ToolsDialog key={id} view={view} close={() => setModal(null)}
      disabled={disabled || state.model.supports_tools !== true}
      notice={notice} apply={(names) => perform('tools', { names })} />}
    {modal?.action === 'mcp' && <Dialog title="MCP 服务" close={() => setModal(null)} className="mcp-dialog">
      <MCPManager servers={state.mcp_servers} disabled={disabled} error={state.mcp_error}
        notice={notice} perform={perform} command={command} />
    </Dialog>}
    {modal?.action === 'prompts' && modal.sessionId === id &&
      <Dialog title="MCP 模板" close={() => setModal(null)} className="mcp-dialog">
        <PromptPicker servers={state.mcp_prompt_servers} connected={connected} command={command} insert={(text) => {
          updateDraft((current) => current ? `${current}\n\n${text}` : text)
          setModal(null)
          requestAnimationFrame(() => composerInput.current?.focus())
        }} />
      </Dialog>}
    {modal?.action === 'system' && modal.sessionId === id && <SystemPromptDialog initial={state.system_prompt} close={() => setModal(null)}
      disabled={disabled} notice={notice} apply={(prompt) => perform('system', { prompt })} />}
    {modal?.action === 'confirm_full' && modal.sessionId === id && <Dialog title="启用 FULL" close={() => setModal(null)}>
      <form onSubmit={async (event) => {
        event.preventDefault()
        if (!disabled && await perform('confirm_full')) setModal(null)
      }}>
        <p>工具调用将自动执行。模型可能读取 .env 等敏感文件，并将内容发给模型服务。命令和 MCP 工具可能访问工作区外的文件、修改文件或访问网络。</p>
        {notice && <p role="status">{notice}</p>}
        <div className="actions">
          <button autoFocus type="button" onClick={() => setModal(null)}>取消</button>
          <button className="primary" disabled={disabled} type="submit">确认启用 FULL</button>
        </div>
      </form>
    </Dialog>}
    {modal?.action === 'project' && <ProjectDialog close={() => setModal(null)} disabled={disabled}
      notice={notice} create={(name, workspace) => perform('project_create', { name, workspace })} />}
    {modal?.action === 'remove_project' && <Dialog title="移除项目" close={() => setModal(null)}>
      <form onSubmit={async (event) => {
        event.preventDefault()
        if (await perform('project_remove', { project_id: modal.projectId })) setModal(null)
      }}>
        <p>移除「{modal.name}」？该项目的会话会移到「无项目」，工作目录和文件会保留。</p>
        {notice && <p role="status">{notice}</p>}
        <div className="actions">
          <button type="button" onClick={() => setModal(null)}>取消</button>
          <button className="primary" disabled={disabled} type="submit">移除</button>
        </div>
      </form>
    </Dialog>}
    {sessionToMove && <MoveSessionDialog session={sessionToMove} projects={state.projects}
      close={() => setModal(null)} disabled={disabled} notice={notice}
      move={(projectId, workspace) => perform('move', { id: sessionToMove.id, project_id: projectId, workspace })} />}
    {(modal?.action === 'rename' || modal?.action === 'delete') && <Dialog title={modal.action === 'rename' ? '重命名会话' : '删除会话'}
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
