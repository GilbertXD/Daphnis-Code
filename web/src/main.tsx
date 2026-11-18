import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { AttachmentLabels, MAX_FILES, MAX_TOTAL, isImageFile, readFiles } from './attachments'
import type { DraftAttachment } from './attachments'
import { Debug } from './debug'
import type { Tab } from './debug'
import { MCPManager } from './mcp'
import { PromptPicker } from './prompts'
import { sessionTitle, useConnection } from './protocol'
import type { Approval, Command, LoadAttachment, Model, Project, ResourceDraft, Session, View } from './protocol'
import { QuestionForm } from './questions'
import { SkillsManager } from './skills'
import { collectTools, DataBlock, Transcript } from './transcript'
import './style.css'

const tokenKey = 'daphnis.access'
const modelKey = (model: Model) => JSON.stringify([model.provider, model.id])
const thinkingLabel = (mode: string | null) => mode === 'off' ? '关闭' : mode === 'on' ? '开启' : mode || '默认'
const approvalModes = [
  { value: 'ask', label: 'ASK' },
  { value: 'auto', label: 'AUTO' },
  { value: 'full', label: 'FULL' },
]
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
  const groups = view.state.mcp_servers.map(({ id, tool_prefix }) => ({
    name: id, prefix: tool_prefix, tools: [] as string[],
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

type SettingOption = { value: string; label: string; unavailable?: boolean }

function SettingsDialog({ title, value, groups, close, apply, disabled, notice }: {
  title: string
  value: string
  groups: { title?: string; options: SettingOption[] }[]
  close: () => void
  apply: (value: string) => void
  disabled: boolean
  notice: string
}) {
  const [selected, setSelected] = useState(value)
  const canApply = !disabled && selected !== value
    && groups.some(({ options }) => options.some((option) => option.value === selected && !option.unavailable))
  return <Dialog title={title} close={close}>
    <form onSubmit={(event) => {
      event.preventDefault()
      if (canApply) apply(selected)
    }}>
      <div className="tool-options" role="radiogroup" aria-label={title}>
        {groups.map((group) => <section key={group.title ?? title}>
          {group.title && <h3>{group.title}</h3>}
          {group.options.map(({ value, label, unavailable }) => <label key={value}>
            <input type="radio" name="setting" value={value} checked={selected === value}
              disabled={disabled || unavailable} onChange={() => setSelected(value)} />
            <span>{label}</span>
            {unavailable && <small className="muted">未配置</small>}
          </label>)}
        </section>)}
      </div>
      {notice && <p role="status">{notice}</p>}
      <div className="actions">
        <button type="button" onClick={close}>取消</button>
        <button className="primary" type="submit" disabled={!canApply}>应用</button>
      </div>
    </form>
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

type SettingAction = 'model' | 'thinking' | 'approval'
type Modal = { action: 'tools' | 'project' | 'mcp' } | { action: 'system' | 'skills' | SettingAction; sessionId: string }
  | { action: 'prompts'; sessionId: string }
  | { action: 'confirm_full'; sessionId: string }
  | { action: 'rename' | 'delete'; session: Session }
  | { action: 'move'; session: Session }
  | { action: 'remove_project'; projectId: string; name: string }

const attachmentKey = (file: DraftAttachment) => file.kind === 'document' ? `${file.id}:${file.readingMode}` : file.id

function Workspace({ view, connected, command, loadImage, loadDocument }: {
  view: View
  connected: boolean
  command: Command
  loadImage: LoadAttachment
  loadDocument: LoadAttachment
}) {
  const [open, setOpen] = useState(false)
  const [sidebar, setSidebar] = useState(false)
  const [tab, setTab] = useState<Tab>('events')
  const [modal, setModal] = useState<Modal | null>(null)
  const [title, setTitle] = useState('')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [draftFiles, setDraftFiles] = useState<Record<string, DraftAttachment[]>>({})
  const previewUrls = useRef(new Set<string>())
  const mounted = useRef(false)
  const deletedSessions = useRef(new Set<string>())
  const [uploadErrors, setUploadErrors] = useState<Record<string, string>>({})
  const [readingFiles, setReadingFiles] = useState<Set<string>>(() => new Set())
  const fileReads = useRef(new Set<string>())
  const submissions = useRef(new Map<string, { id: string; text: string; ids: string[] }>())
  const [dragging, setDragging] = useState(false)
  const dragDepth = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const [pendingCommands, setPendingCommands] = useState<Set<string>>(() => new Set())
  const [acknowledged, setAcknowledged] = useState<Record<string, number>>({})
  const inFlight = useRef(new Set<string>())
  const [notices, setNotices] = useState<Record<string, string>>({})
  const { state, status, session_id: id } = view
  const busy = pendingCommands.has(id)
  const navigationDisabled = !connected || pendingCommands.has('')
  const notice = notices[id] ?? ''
  const setNotice = (text: string) => setNotices((current) => ({ ...current, [id]: text }))
  useEffect(() => {
    if (view.reset) {
      // A new connection can come from a restarted server with fresh revisions.
      setAcknowledged((current) => current[id] > view.revision ? { ...current, [id]: view.revision } : current)
    }
  }, [id, view.reset, view.revision])
  useEffect(() => {
    setModal((current) => current && 'sessionId' in current && current.sessionId !== id ? null : current)
    dragDepth.current = 0
    setDragging(false)
  }, [id])
  useEffect(() => {
    mounted.current = true
    const urls = previewUrls.current
    return () => {
      mounted.current = false
      urls.forEach((url) => URL.revokeObjectURL(url))
      urls.clear()
    }
  }, [])
  useEffect(() => {
    const retained = new Set(Object.values(draftFiles).flat()
      .flatMap((file) => file.kind === 'image' ? [file.previewUrl] : []))
    for (const url of previewUrls.current) {
      if (retained.has(url)) continue
      URL.revokeObjectURL(url)
      previewUrls.current.delete(url)
    }
  }, [draftFiles])
  const disabled = busy || navigationDisabled || status.running || view.revision < (acknowledged[id] ?? 0)
  const recovery = state.recovery
  const interrupted = recovery?.kind === 'INTERRUPTED_TURN'
  const compacting = view.run?.kind === 'compaction'
  const lastOperation = view.events.findLast((event) => event.name === 'turn.started'
    || (event.name === 'compaction.finished' && event.data.trigger === 'manual'))
  const compactResult = lastOperation?.name === 'compaction.finished' ? lastOperation.data : null
  const compactNotice = compactResult?.status === 'committed' ? '上下文压缩完成'
    : compactResult?.status === 'skipped' ? '暂无可压缩的历史消息'
    : compactResult?.status === 'cancelled' ? '已停止上下文压缩'
    : compactResult?.status === 'failed'
      ? `压缩失败：${compactResult.reason || (typeof compactResult.error === 'string' ? compactResult.error : '未知错误')}`
      : ''
  const draft = drafts[id] ?? ''
  const files = draftFiles[id] ?? []
  const reading = readingFiles.has(id)
  const keepDraftSession = !!(draft || files.length || reading)
  const allowsImages = state.model.supports_image === true
  const imagesBlocked = !allowsImages && files.some((file) => file.kind === 'image')
  const imageNotice = state.model.supports_image === false ? '当前模型不支持图片。' : '当前模型的图片能力未确认。'
  const allowsDocuments = state.model.supports_document === true
  const documentsBlocked = !allowsDocuments && files.some((file) => file.kind === 'document')
  const tools = useMemo(() => collectTools(view.events), [view.events])
  const sessions = state.sessions.some((session) => session.id === id)
    ? state.sessions : [state.session, ...state.sessions]
  const sessionToMove = modal?.action === 'move' && (
    modal.session.id === id
      ? { ...state.session, recovery_kind: recovery?.kind }
      : state.sessions.find((session) => session.id === modal.session.id)
  )

  async function perform(action: string, fields?: Record<string, unknown> | FormData) {
    const navigation = ['restore', 'project_open', 'new'].includes(action)
      || (action === 'delete' && fields && !(fields instanceof FormData) && fields.id === id)
    const scope = navigation ? '' : id
    if (inFlight.current.has(scope)) return false
    inFlight.current.add(scope)
    setPendingCommands(new Set(inFlight.current))
    setNotice('')
    try {
      const response = await command(action, fields)
      if (!response.session_id) {
        setAcknowledged((current) => ({ ...current, [id]: response.revision }))
      }
      setNotice(response.notice ?? '')
      return true
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '操作失败。')
      return false
    } finally {
      inFlight.current.delete(scope)
      setPendingCommands(new Set(inFlight.current))
    }
  }

  function openSetting(action: SettingAction) {
    setNotice('')
    setModal({ action, sessionId: id })
  }

  async function applySetting(action: string, fields: Record<string, unknown>) {
    if (await perform(action, fields)) setModal((current) => current === modal ? null : current)
  }

  function updateDraft(text: string | ((draft: string) => string)) {
    setDrafts((current) => ({ ...current, [id]: typeof text === 'function' ? text(current[id] ?? '') : text }))
  }

  async function addFiles(added: File[]) {
    if (!added.length || fileReads.current.has(id)) return
    fileReads.current.add(id)
    setReadingFiles(new Set(fileReads.current))
    setUploadErrors((current) => ({ ...current, [id]: '' }))
    try {
      const result = await readFiles(added, files, allowsImages, allowsDocuments)
      if (!mounted.current || deletedSessions.current.has(id)) return
      const additions = result.added.map((file): DraftAttachment => {
        if (file.kind !== 'image') return file
        const previewUrl = URL.createObjectURL(file.file)
        previewUrls.current.add(previewUrl)
        return { ...file, previewUrl }
      })
      if (additions.length) setDraftFiles((current) => ({ ...current, [id]: [...(current[id] ?? []), ...additions] }))
      setUploadErrors((current) => ({ ...current, [id]: result.error }))
    } finally {
      fileReads.current.delete(id)
      if (mounted.current) setReadingFiles(new Set(fileReads.current))
    }
  }

  function addResources(resources: ResourceDraft[]) {
    if (disabled || fileReads.current.has(id)) return '请等待当前操作结束。'
    const added = resources.filter((resource) => !files.some((file) => file.id === resource.id))
      .map((resource): DraftAttachment => ({
        kind: 'resource', id: resource.id, name: resource.name, size: resource.size_bytes, resourceId: resource.id,
      }))
    const next = [...files, ...added]
    if (!added.length) return '所选资源已在待发送附件中。'
    const texts = next.filter((file) => file.kind === 'text' || file.kind === 'resource')
    if (texts.length > MAX_FILES || texts.reduce((sum, file) => sum + file.size, 0) > MAX_TOTAL) {
      return '每次最多发送 10 份附件或资源，总计最多 256 KiB。'
    }
    setDraftFiles((current) => ({ ...current, [id]: next }))
    setUploadErrors((current) => ({ ...current, [id]: '' }))
    setModal(null)
    return null
  }

  async function send() {
    if (disabled || interrupted || imagesBlocked || documentsBlocked || fileReads.current.has(id) || (!draft.trim() && !files.length)) return
    if (files.length) {
      const pending = submissions.current.get(id)
      const ids = files.map(attachmentKey)
      const submission = pending?.text === draft && pending.ids.length === ids.length && pending.ids.every((value, index) => value === ids[index])
        ? pending : { id: crypto.randomUUID().replaceAll('-', ''), text: draft, ids }
      submissions.current.set(id, submission)
      let accepted: boolean
      if (files.some((file) => file.kind === 'image' || file.kind === 'document')) {
        const form = new FormData()
        form.set('submission_id', submission.id)
        form.set('prompt', draft)
        const manifest = files.map((file, index) => {
          if (file.kind === 'resource') return { kind: file.kind, resource_id: file.resourceId }
          if (file.kind === 'text') return { kind: file.kind, name: file.name, text: file.text }
          const field = `${file.kind}_${index}`
          const content = file.file.type === file.mimeType ? file.file : file.file.slice(0, file.size, file.mimeType)
          form.set(field, content, file.name)
          return file.kind === 'document'
            ? { kind: file.kind, name: file.name, field, reading_mode: file.readingMode }
            : { kind: file.kind, name: file.name, field }
        })
        form.set('manifest', JSON.stringify(manifest))
        accepted = await perform('send_attachments', form)
      } else {
        accepted = await perform('send_attachments', {
          submission_id: submission.id, prompt: draft,
          attachments: files.filter((file) => file.kind === 'text' || file.kind === 'resource').map((file) => file.kind === 'resource'
            ? { resource_id: file.resourceId } : { name: file.name, text: file.text }),
        })
      }
      if (!accepted) return
      submissions.current.delete(id)
      const sent = new Set(ids)
      setDraftFiles((current) => ({ ...current, [id]: (current[id] ?? []).filter((file) => !sent.has(attachmentKey(file))) }))
    } else if (!await perform('send', { prompt: draft })) {
      return
    }
    setDrafts((current) => current[id] === draft ? { ...current, [id]: '' } : current)
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
    <Sidebar projects={state.projects} sessions={sessions} selected={id} disabled={navigationDisabled} visible={sidebar}
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
      onNew={(projectId) => void perform('project_open', { project_id: projectId, keep_draft_session: keepDraftSession })}
      onSelect={async (target) => {
        if (target === id || await perform('restore', { id: target, keep_draft_session: keepDraftSession })) setSidebar(false)
      }}
      onEdit={(action, session) => {
        setNotice('')
        setTitle(session.title ?? sessionTitle(session))
        setModal({ action, session })
      }} />
    <main className="conversation" aria-label="Agent 对话">
      <Transcript key={id} view={view} tools={tools} loadImage={loadImage} loadDocument={loadDocument} retry={recovery?.kind === 'FAILED_USER'
        ? { index: recovery.message_index, disabled, run: () => void perform('retry', { key: recovery.key }) }
        : undefined} />
      <div className="composer-zone">
        {status.approval && <ApprovalCard key={status.approval.request_id} request={status.approval}
          disabled={busy || !connected} choose={(decision) => void perform('approve', {
            request_id: status.approval?.request_id, decision,
          })} />}
        {status.question && <QuestionForm key={status.question.request_id} request={status.question}
          disabled={busy || !connected} submit={(answers) => perform('answer_question', {
            request_id: status.question?.request_id, turn_id: status.question?.turn_id, answers,
          })} />}
        {interrupted && !status.running && <div className="recovery" role="group" aria-label="未完成任务">
          <span>未完成任务</span>
          <div className="actions">
            <button type="button" disabled={disabled} onClick={() => void perform('continue', { key: recovery.key })}>继续</button>
            <button type="button" disabled={disabled} onClick={() => void perform('terminate', { key: recovery.key })}>终止</button>
          </div>
        </div>}
        {!modal && (compacting || notice || (!status.running && compactNotice)) &&
          <p className="notice" role="status">{compacting ? '正在压缩上下文' : notice || compactNotice}</p>}
        <form className={`composer${dragging ? ' attachment-drop' : ''}`} onSubmit={(event) => { event.preventDefault(); void send() }}
          onDragEnter={(event) => {
            if (!event.dataTransfer.types.includes('Files')) return
            event.preventDefault()
            dragDepth.current += 1
            setDragging(true)
          }}
          onDragOver={(event) => {
            if (!event.dataTransfer.types.includes('Files')) return
            event.preventDefault()
            event.dataTransfer.dropEffect = 'copy'
          }}
          onDragLeave={() => {
            dragDepth.current = Math.max(0, dragDepth.current - 1)
            if (!dragDepth.current) setDragging(false)
          }}
          onDrop={(event) => {
            if (!event.dataTransfer.types.includes('Files')) return
            event.preventDefault()
            dragDepth.current = 0
            setDragging(false)
            void addFiles(Array.from(event.dataTransfer.files))
          }}>
          <AttachmentLabels files={files} changeReadingMode={busy ? undefined : (fileId, readingMode) => setDraftFiles((current) => ({
            ...current, [id]: (current[id] ?? []).map((file) =>
              file.kind === 'document' && file.id === fileId ? { ...file, readingMode } : file),
          }))} remove={(fileId) => setDraftFiles((current) => ({
            ...current, [id]: (current[id] ?? []).filter((file) => file.id !== fileId),
          }))} />
          {reading && <p className="muted" role="status">正在读取附件…</p>}
          {imagesBlocked && <p className="notice" role="status">{imageNotice}请切换模型或移除图片后再发送。</p>}
          {documentsBlocked && <p className="notice" role="status">当前模型不支持文档。请切换模型或移除文档后再发送。</p>}
          {uploadErrors[id] && <p className="notice" role="alert">{uploadErrors[id]}</p>}
          <input ref={fileInput} type="file" hidden multiple onChange={(event) => {
            const selected = Array.from(event.currentTarget.files ?? [])
            event.currentTarget.value = ''
            void addFiles(selected)
          }} />
          <textarea aria-label="消息" placeholder="请输入你的需求…" value={draft} rows={2}
            onChange={(event) => updateDraft(event.currentTarget.value)}
            onPaste={(event) => {
              const images = Array.from(event.clipboardData.files).filter(isImageFile)
              if (!images.length) return
              if (!event.clipboardData.getData('text/plain')) event.preventDefault()
              void addFiles(images)
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing) {
                event.preventDefault()
                void send()
              }
            }} />
          <div className="composer-controls">
            <div className="selectors">
              <button className="attachment-picker" type="button" aria-label="添加附件" title="添加附件" disabled={reading}
                onClick={() => fileInput.current?.click()}>+</button>
              <button type="button" className="model-control" aria-haspopup="dialog" aria-label={`模型 · ${state.model.id}`}
                disabled={disabled} onClick={() => openSetting('model')}>{state.model.id}</button>
              <button type="button" aria-haspopup="dialog" disabled={disabled || !state.model.thinking_modes?.length}
                onClick={() => openSetting('thinking')}>Thinking · {thinkingLabel(state.thinking)}</button>
              <button type="button" className="tool-control" title="选择启用的工具"
                disabled={disabled || state.model.supports_tools !== true} onClick={() => {
                  setNotice('')
                  setModal({ action: 'tools' })
                }}>
                工具 {state.effective_tools.length ? `· ${state.effective_tools.length} 已启用` : '· 停用'}
              </button>
              <button type="button" aria-haspopup="dialog" title="管理全局和当前工作区 Skills" onClick={() => {
                setNotice('')
                setModal({ action: 'skills', sessionId: id })
              }}>Skills</button>
              <button type="button" aria-haspopup="dialog" title="管理 MCP 服务和资源" onClick={() => {
                setNotice('')
                setModal({ action: 'mcp' })
              }}>MCP</button>
              <button type="button" title="选择 MCP 提示词" disabled={!connected} onClick={() => {
                setNotice('')
                setModal({ action: 'prompts', sessionId: id })
              }}>Prompts</button>
              <button type="button" title="编辑当前会话的系统提示词" disabled={disabled} onClick={() => {
                setNotice('')
                setModal({ action: 'system', sessionId: id })
              }}>系统提示词</button>
              <button type="button" title="将较早的会话内容压缩为摘要"
                disabled={disabled || interrupted} onClick={() => void perform('compact')}>
                {compacting ? '压缩中' : '压缩上下文'}
              </button>
              <button type="button" aria-haspopup="dialog" aria-label={`工具审批模式 · ${state.approval_mode.toUpperCase()}`}
                disabled={disabled} onClick={() => openSetting('approval')}>{state.approval_mode.toUpperCase()}</button>
            </div>
            {status.running
              ? <button className="primary send" type="button" disabled={busy || !connected || !view.run}
                onClick={() => void perform('cancel', { run_id: view.run?.id })} aria-label="停止" title="停止">■</button>
              : <button className="primary send" type="submit"
                disabled={disabled || interrupted || reading || imagesBlocked || documentsBlocked || (!draft.trim() && !files.length)}
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
    {modal?.action === 'skills' && modal.sessionId === id &&
      <Dialog title="Skills" close={() => setModal(null)} className="mcp-dialog">
        <SkillsManager skills={state.skills} notice={notice} perform={perform}
          disabled={!connected || busy || navigationDisabled || view.revision < (acknowledged[id] ?? 0)} />
      </Dialog>}
    {modal?.action === 'mcp' && <Dialog title="MCP 服务" close={() => setModal(null)} className="mcp-dialog">
      <MCPManager key={id} servers={state.mcp_servers} disabled={disabled} connected={connected} error={state.mcp_error}
        notice={notice} perform={perform} command={command}
        resourceServers={state.mcp_resource_servers} addResources={addResources} />
    </Dialog>}
    {modal?.action === 'prompts' && modal.sessionId === id &&
      <Dialog title="MCP Prompts" close={() => setModal(null)} className="mcp-dialog">
        <PromptPicker servers={state.mcp_prompt_servers} connected={connected} command={command}
          runDisabled={disabled || interrupted} run={async (previewId, texts) => {
            if (await perform('mcp_run_prompt', { session_id: modal.sessionId, preview_id: previewId, texts })) {
              setModal((current) => current === modal ? null : current)
            }
          }} />
        {notice && <p role="alert">{notice}</p>}
      </Dialog>}
    {modal?.action === 'system' && modal.sessionId === id && <SystemPromptDialog initial={state.system_prompt} close={() => setModal(null)}
      disabled={disabled} notice={notice} apply={(prompt) => perform('system', { prompt })} />}
    {modal?.action === 'model' && modal.sessionId === id && <SettingsDialog title="选择模型" value={modelKey(state.model)}
      close={() => setModal(null)} disabled={disabled} notice={notice}
      groups={[...new Set(state.models.map((model) => model.provider))].map((provider) => ({
        title: provider,
        options: state.models.filter((model) => model.provider === provider).map((model) => ({
          value: modelKey(model), label: model.id, unavailable: !model.configured,
        })),
      }))}
      apply={(key) => {
        const model = state.models.find((candidate) => modelKey(candidate) === key && candidate.configured)
        if (model) void applySetting('model', { provider: model.provider, model_id: model.id })
      }} />}
    {modal?.action === 'thinking' && modal.sessionId === id && <SettingsDialog title="思考模式" value={state.thinking ?? ''}
      close={() => setModal(null)} disabled={disabled} notice={notice}
      groups={[{ options: ['', ...(state.model.thinking_modes ?? [])].map((value) => ({ value, label: thinkingLabel(value) })) }]}
      apply={(mode) => void applySetting('thinking', { mode: mode || null })} />}
    {modal?.action === 'approval' && modal.sessionId === id && <SettingsDialog title="工具审批模式" value={state.approval_mode}
      close={() => setModal(null)} disabled={disabled} notice={notice} groups={[{ options: approvalModes }]}
      apply={(mode) => {
        if (mode === 'full') {
          setModal({ action: 'confirm_full', sessionId: id })
        } else {
          void applySetting('set_approval_mode', { mode })
        }
      }} />}
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
      close={() => setModal(null)} disabled={busy || navigationDisabled || (sessionToMove.id === id && status.running)} notice={notice}
      move={(projectId, workspace) => perform('move', { id: sessionToMove.id, project_id: projectId, workspace })} />}
    {(modal?.action === 'rename' || modal?.action === 'delete') && <Dialog title={modal.action === 'rename' ? '重命名会话' : '删除会话'}
      close={() => setModal(null)}>
      <form onSubmit={async (event) => {
        event.preventDefault()
        if (await perform(modal.action, modal.action === 'rename'
          ? { id: modal.session.id, title }
          : { id: modal.session.id })) {
          if (modal.action === 'delete') {
            const target = modal.session.id
            deletedSessions.current.add(target)
            submissions.current.delete(target)
            setDrafts(({ [target]: _deleted, ...rest }) => rest)
            setDraftFiles(({ [target]: _deleted, ...rest }) => rest)
            setUploadErrors(({ [target]: _deleted, ...rest }) => rest)
          }
          setModal(null)
        }
      }}>
        {modal.action === 'rename'
          ? <label className="field">会话名称<input autoFocus value={title} maxLength={120} required
            onChange={(event) => setTitle(event.currentTarget.value)} /></label>
          : <p>永久删除「{sessionTitle(modal.session)}」及其已保存的消息？</p>}
        {notice && <p role="status">{notice}</p>}
        <div className="actions"><button type="button" onClick={() => setModal(null)}>取消</button>
          <button className="primary" disabled={busy || navigationDisabled || (modal.session.id === id && status.running)}
            type="submit">{modal.action === 'rename' ? '保存' : '删除'}</button></div>
      </form>
    </Dialog>}
  </div>
}

function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem(tokenKey) ?? '')
  const [entry, setEntry] = useState('')
  const { view, connected, error, command, loadImage, loadDocument } = useConnection(token)
  function login(event: FormEvent) {
    event.preventDefault()
    const value = entry.trim()
    if (!value) return
    sessionStorage.setItem(tokenKey, value)
    setToken(value)
    setEntry('')
  }

  if (view) return <Workspace view={view} connected={connected} command={command}
    loadImage={loadImage} loadDocument={loadDocument} />
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
