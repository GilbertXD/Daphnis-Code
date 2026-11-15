import { useState } from 'react'
import type { FormEvent } from 'react'
import type { Command, MCPOptions, MCPServer, RegistryPage, RegistryServer } from './protocol'

type Perform = (action: string, fields: Record<string, unknown>) => Promise<boolean>

const statusText: Record<string, string> = {
  Connected: '已连接', Disconnected: '未连接', Connecting: '连接中', Error: '连接失败',
}

export function MCPEditor({ initial, disabled, save, cancel }: {
  initial?: MCPServer
  disabled: boolean
  save: (name: string, enabled: boolean, options: MCPOptions) => Promise<boolean>
  cancel: () => void
}) {
  const [name, setName] = useState(initial?.name ?? '')
  const [enabled, setEnabled] = useState(initial?.enabled ?? true)
  const [transport, setTransport] = useState(initial?.options.transport ?? 'stdio')
  const [command, setCommand] = useState(initial?.options.command ?? '')
  const [args, setArgs] = useState(JSON.stringify(initial?.options.args ?? []))
  const [cwd, setCwd] = useState(initial?.options.cwd ?? '')
  const [env, setEnv] = useState(JSON.stringify(initial?.options.env ?? {}))
  const [url, setUrl] = useState(initial?.options.url ?? '')
  const [auth, setAuth] = useState(initial?.options.auth?.env ?? '')
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    let options: MCPOptions
    if (transport === 'stdio') {
      try {
        const parsedArgs: unknown = JSON.parse(args)
        const parsedEnv: unknown = JSON.parse(env)
        if (!Array.isArray(parsedArgs) || parsedArgs.some((arg) => typeof arg !== 'string')
          || !parsedEnv || typeof parsedEnv !== 'object' || Array.isArray(parsedEnv)
          || Object.values(parsedEnv).some((value) => typeof value !== 'string')) {
          throw new Error()
        }
        options = { transport, command: command.trim(), args: parsedArgs, env: parsedEnv as Record<string, string> }
        if (cwd.trim()) options.cwd = cwd.trim()
      } catch {
        setError('参数须为 JSON 字符串数组，环境变量引用须为 JSON 对象。')
        return
      }
    } else {
      options = { transport, url: url.trim() }
      if (auth.trim()) options.auth = { type: 'bearer', env: auth.trim() }
    }
    if (await save(name.trim(), enabled, options)) cancel()
  }

  return <form onSubmit={(event) => void submit(event)}>
    <fieldset disabled={disabled}>
      <label className="field">服务标识<input value={name} readOnly={!!initial} required
        maxLength={20} pattern="[A-Za-z0-9\-]{1,20}" placeholder="字母、数字或连字符"
        onChange={(event) => setName(event.target.value)} /></label>
      <label className="field">连接方式<select value={transport} disabled={initial?.options.public_only}
        onChange={(event) => setTransport(event.target.value as MCPOptions['transport'])}>
        <option value="stdio">stdio</option>
        <option value="streamable_http">Streamable HTTP</option>
      </select></label>
      {transport === 'stdio' ? <>
        <label className="field">启动程序<input required value={command} onChange={(event) => setCommand(event.target.value)} /></label>
        <label className="field">参数（JSON 数组）<textarea rows={2} required value={args}
          onChange={(event) => setArgs(event.target.value)} /></label>
        <label className="field">工作目录（可选）<input value={cwd} onChange={(event) => setCwd(event.target.value)} /></label>
        <label className="field">环境变量引用（JSON 对象）<textarea rows={2} required value={env}
          onChange={(event) => setEnv(event.target.value)} /></label>
        <p className="muted">启用后会以当前用户权限启动程序。凭据请通过环境变量名引用。</p>
      </> : <>
        <label className="field">服务地址<input type="url" required value={url} onChange={(event) => setUrl(event.target.value)} /></label>
        <label className="field">Bearer Token 环境变量名（可选）<input value={auth}
          onChange={(event) => setAuth(event.target.value)} /></label>
      </>}
      <label><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> 启用服务</label>
      {error && <p role="alert">{error}</p>}
      <div className="actions">
        <button type="button" onClick={cancel}>取消</button>
        <button className="primary" type="submit">{enabled ? '保存并连接' : '保存'}</button>
      </div>
    </fieldset>
  </form>
}

export function MCPManager({ servers, disabled, error, notice, perform, command }: {
  servers: MCPServer[]
  disabled: boolean
  error: string | null
  notice: string
  perform: Perform
  command: Command
}) {
  const [editing, setEditing] = useState<MCPServer | 'new' | null>(null)
  const [discover, setDiscover] = useState(false)
  const blocked = disabled || !!error
  return <>
    <div className="mcp-tabs" aria-label="MCP 页面">
      <button type="button" aria-pressed={!discover} onClick={() => setDiscover(false)}>已配置</button>
      <button type="button" aria-pressed={discover} onClick={() => { setEditing(null); setDiscover(true) }}>发现</button>
    </div>
    {error && <p role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    {discover ? <RegistryPanel command={command} disabled={blocked} add={async (fields) => {
      if (await perform('mcp_add_registry', fields)) setDiscover(false)
    }} /> : editing ? <MCPEditor initial={editing === 'new' ? undefined : editing} disabled={blocked}
      cancel={() => setEditing(null)}
      save={(name, enabled, options) => perform('mcp_save', { name, enabled, options, create: editing === 'new' })} /> : <>
      {!servers.length && !error && <p className="muted">尚未配置 MCP 服务。</p>}
      <ul className="mcp-list">{servers.map((server) => <li key={server.name}>
        <div className="mcp-summary">
          <strong>{server.name}</strong>
          <span className="muted">{server.enabled ? statusText[server.state] ?? server.state : '已禁用'} · {server.tool_count} 个工具</span>
          <label><input type="checkbox" checked={server.enabled} disabled={blocked}
            onChange={(event) => {
              const enabled = event.target.checked
              if (enabled && server.options.transport === 'stdio' && !window.confirm(`允许启动本地程序 ${server.options.command}？`)) return
              void perform('mcp_enable', { name: server.name, enabled })
            }} /> 启用</label>
        </div>
        {server.error && <p role="status">{server.error}</p>}
        <div className="actions">
          <button disabled={blocked || !server.enabled} onClick={() => void perform('mcp_restart', { name: server.name })}>重连</button>
          <button disabled={blocked} onClick={() => setEditing(server)}>编辑</button>
          <button disabled={blocked} onClick={() => {
            if (window.confirm(`删除 MCP 服务「${server.name}」？`)) void perform('mcp_delete', { name: server.name })
          }}>删除</button>
        </div>
      </li>)}</ul>
      <div className="actions"><button className="primary" disabled={blocked} onClick={() => setEditing('new')}>添加 MCP 服务</button></div>
    </>}
  </>
}

function RegistryPanel({ command, disabled, add }: {
  command: Command
  disabled: boolean
  add: (fields: Record<string, unknown>) => Promise<void>
}) {
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [page, setPage] = useState<RegistryPage | null>(null)
  const [detail, setDetail] = useState<RegistryServer | null>(null)
  const [remoteIndex, setRemoteIndex] = useState('')
  const [name, setName] = useState('')
  const [auth, setAuth] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const remote = remoteIndex !== '' ? detail?.remotes[Number(remoteIndex)] : undefined

  async function load(task: () => Promise<void>) {
    if (busy) return
    setBusy(true)
    setError('')
    try { await task() }
    catch (cause) { setError(cause instanceof Error ? cause.message : '无法读取官方 MCP Registry。') }
    finally { setBusy(false) }
  }

  async function search(text: string, cursor: string | null) {
    await load(async () => {
      const result = (await command('registry_search', { query: text, cursor })).registry_page
      if (!result) throw new Error('搜索结果无效。')
      setSubmitted(text)
      setPage(cursor && page ? { servers: [...page.servers, ...result.servers], cursor: result.cursor } : result)
    })
  }

  async function inspect(server: RegistryServer) {
    await load(async () => {
      const result = (await command('registry_detail', { registry_name: server.name, version: server.version })).registry_server
      if (!result) throw new Error('服务详情无效。')
      setDetail(result)
      setName((result.name.split('/').at(-1) ?? '').replace(/[^A-Za-z0-9-]/g, '-').slice(0, 20))
      setRemoteIndex('')
      setAuth('')
    })
  }

  return <>
    {error && <p role="alert">{error}</p>}
    {busy && <p role="status">正在查询官方 MCP Registry…</p>}
    {detail ? <>
      <button type="button" onClick={() => setDetail(null)}>返回搜索结果</button>
      <h3>{detail.title || detail.name}</h3>
      <p className="mcp-address">{detail.name} · {detail.version} · {detail.publisher}</p>
      {detail.description && <p>{detail.description}</p>}
      {detail.repository && <p className="mcp-address">来源 {detail.source} <code>{detail.repository}</code></p>}
      {!!detail.remotes.length && <ul className="mcp-list">{detail.remotes.map((item, index) => <li key={index}>
        <code>{item.url}</code>
        <div className="muted">{item.transport} · {item.reason ?? (item.auth === 'bearer' ? '需要 Bearer Token' : '未声明认证要求')}</div>
      </li>)}</ul>}
      {!!detail.packages.length && <><p>本地安装信息</p><ul className="mcp-list">{detail.packages.map((item, index) =>
        <li key={index}>{item.type} <code>{item.identifier}</code> {item.version}</li>)}</ul></>}
      {detail.status !== 'active' ? <p>该版本当前不可添加。</p> : detail.remotes.some((item) => item.auth !== 'unsupported') ?
        <form onSubmit={(event) => {
          event.preventDefault()
          if (remote && !disabled) void add({
            name: name.trim(), registry_name: detail.name, version: detail.version, url: remote.url,
            auth_env: auth.trim() || null,
          })
        }}>
          <fieldset disabled={disabled}>
            <label className="field">选择远程地址<select required value={remoteIndex}
              onChange={(event) => { setRemoteIndex(event.target.value); setAuth('') }}>
              <option value="">请选择地址</option>
              {detail.remotes.map((item, index) => item.auth !== 'unsupported' &&
                <option key={index} value={index}>{item.url}</option>)}
            </select></label>
            <label className="field">服务标识<input required maxLength={20} pattern="(?:[A-Za-z0-9]|-){1,20}"
              value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label className="field">Bearer Token 环境变量名<input required={remote?.auth === 'bearer'}
              placeholder={remote?.auth === 'bearer' ? '必填' : '可选'} value={auth}
              onChange={(event) => setAuth(event.target.value)} /></label>
            <p className="muted">支持无认证或 Bearer Token。连接后，工具仍按当前会话的选择和审批规则执行。</p>
            <div className="actions"><button className="primary" type="submit" disabled={!remote}>确认添加并连接</button></div>
          </fieldset>
        </form> : <p>没有可添加的远程地址。本地软件包需自行安装。</p>}
    </> : <>
      <form onSubmit={(event) => {
        event.preventDefault()
        if (!busy && query.trim().length >= 3) void search(query.trim(), null)
      }}>
        <label className="field">搜索官方 MCP Registry<input required value={query} minLength={3} maxLength={160}
          onChange={(event) => setQuery(event.target.value)} /></label>
        <div className="actions"><button className="primary" disabled={busy || query.trim().length < 3}>搜索</button></div>
      </form>
      {page && !page.servers.length && <p className="muted">未找到服务。</p>}
      <ul className="mcp-list">{page?.servers.map((item) => <li key={item.name}>
        <button type="button" disabled={busy} onClick={() => void inspect(item)}>{item.title || item.name}</button>
        <div className="muted">{item.name} · {item.version} · {item.publisher}</div>
        {item.description && <p>{item.description}</p>}
        <div className="muted">{[...new Set([...item.remotes.map((remote) => remote.transport), ...item.packages.map((pkg) => pkg.type)])].filter(Boolean).join(' · ')}</div>
      </li>)}</ul>
      {page?.cursor && <div className="actions"><button disabled={busy} onClick={() => void search(submitted, page.cursor)}>加载更多</button></div>}
    </>}
  </>
}
