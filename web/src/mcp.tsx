import { useState } from 'react'
import type { FormEvent } from 'react'
import type { Command, MCPOptions, MCPServer, RegistryPage, RegistryRemote, RegistryServer } from './protocol'

type Perform = (action: string, fields: Record<string, unknown>) => Promise<boolean>

const statusText: Record<string, string> = {
  Connected: '已连接', Disconnected: '未连接', Connecting: '连接中', Error: '连接失败',
}
const serviceName = /^[A-Za-z0-9-]{1,20}$/
const environmentName = /^[A-Za-z_][A-Za-z0-9_]*$/

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
        const parsedArgs: unknown = JSON.parse(args.trim() || '[]')
        const parsedEnv: unknown = JSON.parse(env.trim() || '{}')
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
        maxLength={20} pattern="(?:[A-Za-z0-9]|-){1,20}" placeholder="字母、数字或连字符"
        onChange={(event) => setName(event.target.value)} /></label>
      <label className="field">连接方式<select value={transport} disabled={initial?.options.public_only}
        onChange={(event) => setTransport(event.target.value as MCPOptions['transport'])}>
        <option value="stdio">本地程序（stdio）</option>
        <option value="streamable_http">远程服务（Streamable HTTP）</option>
      </select></label>
      {transport === 'stdio' ? <>
        <label className="field">启动程序<input required value={command} onChange={(event) => setCommand(event.target.value)} /></label>
        <p className="muted">启用后会以当前用户权限运行已安装的程序。</p>
        <details className="mcp-details" open={!!error || undefined}>
          <summary>高级设置</summary>
          <label className="field">参数（JSON 数组）<textarea rows={2} value={args}
            onChange={(event) => setArgs(event.target.value)} /></label>
          <label className="field">工作目录（可选）<input value={cwd} onChange={(event) => setCwd(event.target.value)} /></label>
          <label className="field">环境变量引用（JSON 对象）<textarea rows={2} value={env}
            onChange={(event) => setEnv(event.target.value)} /></label>
          <p className="muted">把程序读取的变量名映射到服务端已有的变量名。</p>
        </details>
      </> : <>
        <label className="field">服务地址<input type="url" required value={url} onChange={(event) => setUrl(event.target.value)} /></label>
        <details className="mcp-details">
          <summary>高级设置</summary>
          <label className="field">Bearer Token 环境变量名（可选）<input value={auth}
            onChange={(event) => setAuth(event.target.value)} /></label>
          <p className="muted">在 Daphnis 服务端设置凭据，这里只填写环境变量名。</p>
        </details>
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
    {notice && (discover || editing || !servers.some((server) => server.error === notice)) && <p role="status">{notice}</p>}
    {discover ? <RegistryPanel command={command} servers={servers} disabled={blocked} add={async (fields) => {
      const saved = await perform('mcp_add_registry', fields)
      if (saved) setDiscover(false)
      return saved
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
        <div className="mcp-row-actions">
          {server.enabled && server.state === 'Error' && <button className="mcp-text-button" disabled={blocked}
            onClick={() => void perform('mcp_restart', { name: server.name })}>重试连接</button>}
          <details className="mcp-operations">
            <summary>更多操作<span className="mcp-sr-only"> · {server.name}</span></summary>
            <div className="mcp-row-actions">
              {server.state !== 'Error' && <button className="mcp-text-button" disabled={blocked || !server.enabled}
                onClick={() => void perform('mcp_restart', { name: server.name })}>重连</button>}
              <button className="mcp-text-button" disabled={blocked} onClick={() => setEditing(server)}>编辑</button>
              <button className="mcp-text-button" disabled={blocked} onClick={() => {
                if (window.confirm(`删除 MCP 服务「${server.name}」？`)) void perform('mcp_delete', { name: server.name })
              }}>删除</button>
            </div>
          </details>
        </div>
      </li>)}</ul>
      <div className="actions"><button className="primary" disabled={blocked} onClick={() => setEditing('new')}>添加 MCP 服务</button></div>
    </>}
  </>
}

function registryTitle(server: RegistryServer) {
  return server.title || server.name.split('/').at(-1) || server.name
}

function addressHost(address: string) {
  try { return new URL(address).host }
  catch { return '' }
}

function targetLabel(address: string) {
  try {
    const { host, pathname } = new URL(address)
    const path = pathname.length > 48 ? `${pathname.slice(0, 28)}…${pathname.slice(-16)}` : pathname
    return host + (path === '/' ? '' : path)
  } catch {
    return address.length > 72 ? `${address.slice(0, 71)}…` : address
  }
}

function compatibleRemotes(server: RegistryServer) {
  return server.status === 'active' ? server.remotes.filter((remote) => remote.auth !== 'unsupported') : []
}

function suggestedName(registryName: string, servers: MCPServer[]) {
  const base = (registryName.split('/').at(-1) ?? '').replace(/[^A-Za-z0-9-]+/g, '-').replace(/^-+|-+$/g, '') || 'mcp'
  const used = new Set(servers.map((server) => server.name))
  let name = base.slice(0, 20)
  for (let index = 2; used.has(name); index++) {
    const suffix = `-${index}`
    name = base.slice(0, 20 - suffix.length) + suffix
  }
  return name
}

function CopyableAddress({ address }: { address: string }) {
  const [status, setStatus] = useState('')
  return <div className="mcp-address">
    <code className={status === '请手动复制地址' ? 'mcp-url mcp-url-full' : 'mcp-url'} title={address}>{address}</code>
    <button className="mcp-text-button" type="button" aria-label="复制地址" onClick={async () => {
      try { await navigator.clipboard.writeText(address); setStatus('已复制') }
      catch { setStatus('请手动复制地址') }
    }}>复制</button>
    {status && <span className="muted" role="status">{status}</span>}
  </div>
}

function RegistryPanel({ command, servers, disabled, add }: {
  command: Command
  servers: MCPServer[]
  disabled: boolean
  add: (fields: Record<string, unknown>) => Promise<boolean>
}) {
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [page, setPage] = useState<RegistryPage | null>(null)
  const [detail, setDetail] = useState<RegistryServer | null>(null)
  const [remoteIndex, setRemoteIndex] = useState('')
  const [customName, setCustomName] = useState('')
  const [auth, setAuth] = useState('')
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const remotes = detail ? compatibleRemotes(detail) : []
  const remote = remotes.length === 1 ? remotes[0] : remoteIndex !== '' ? remotes[Number(remoteIndex)] : undefined
  const autoName = detail ? suggestedName(detail.name, servers) : ''
  const name = customName.trim() || autoName
  const invalidName = !!detail && (!serviceName.test(name) || servers.some((server) => server.name === name))
  const needsToken = remote?.auth === 'bearer'
  const canAdd = !!remote && !invalidName && (!needsToken || environmentName.test(auth.trim())) && !disabled

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
      setRemoteIndex('')
      setCustomName('')
      setAuth('')
      setConfirming(false)
    })
  }

  function unavailableMessage(server: RegistryServer) {
    if (server.status !== 'active') return '该服务当前不可添加。'
    if (server.remotes.length) return '没有可直接添加的地址。'
    return server.packages.length ? '该服务需要自行安装，目前无法直接添加。' : '该服务未提供可直接添加的地址。'
  }

  if (detail && remote && confirming) return <section className="mcp-panel" aria-labelledby="mcp-confirm-title">
    <h3 id="mcp-confirm-title">确认添加 {registryTitle(detail)}</h3>
    <p>连接目标 <strong>{addressHost(remote.url)}</strong></p>
    <CopyableAddress key={remote.url} address={remote.url} />
    <p className="muted">调用工具时，参数会发送到此地址，并遵循当前会话的审批设置。</p>
    {invalidName && <p role="alert">服务标识已被占用，请返回修改。</p>}
    <div className="actions">
      <button type="button" onClick={() => setConfirming(false)} disabled={disabled}>返回详情</button>
      <button className="primary" type="button" disabled={!canAdd} onClick={() => void add({
        name, registry_name: detail.name, version: detail.version, url: remote.url,
        auth_env: needsToken ? auth.trim() : null,
      })}>添加并连接</button>
    </div>
  </section>

  if (detail) return <section className="mcp-panel" aria-labelledby="mcp-detail-title">
    <button className="mcp-text-button mcp-back" type="button" onClick={() => setDetail(null)}>← 返回搜索结果</button>
    <h3 id="mcp-detail-title">{registryTitle(detail)}</h3>
    {detail.description && <p>{detail.description}</p>}
    <p className="muted">登记的发布者 {detail.publisher}</p>
    {detail.repository ? <div>
      <p className="muted">来源仓库</p>
      <CopyableAddress key={detail.repository} address={detail.repository} />
    </div> : <p className="muted">{detail.source ? `登记的来源 ${detail.source}` : '未提供来源仓库'}</p>}
    <p className="muted">发布者和来源由 Registry 条目提供，添加前请核对。</p>
    <form onSubmit={(event) => {
      event.preventDefault()
      if (canAdd) setConfirming(true)
    }}>
      <fieldset disabled={disabled}>
        {remotes.length > 1 && <label className="field">连接目标<select aria-label="连接目标" required value={remoteIndex}
          onChange={(event) => { setRemoteIndex(event.target.value); setAuth('') }}>
          <option value="">请选择远程地址</option>
          {remotes.map((item, index) => <option key={index} value={index}>{targetLabel(item.url)}</option>)}
        </select></label>}
        {remote && <p className="mcp-auth">
          {remotes.length === 1 && <span>连接到 {addressHost(remote.url) || remote.url} · </span>}
          {needsToken ? '需要 Bearer Token' : '无需配置凭据'}
        </p>}
        {needsToken && <div>
          <label className="field">Bearer Token 环境变量名<input required value={auth} placeholder="例如 MY_MCP_TOKEN"
            pattern="[A-Za-z_][A-Za-z0-9_]*" onChange={(event) => setAuth(event.target.value)} /></label>
          <p className="muted">先在 Daphnis 服务端设置 Token，再填写对应的变量名。</p>
        </div>}
        {!remotes.length && <>
          <p>{unavailableMessage(detail)}</p>
          {detail.status === 'active' && detail.remotes.length > 0 && <p className="muted">
            {[...new Set(detail.remotes.map(remoteAuth))].join(' ')}
          </p>}
        </>}
        <details className="mcp-details">
          <summary>高级信息</summary>
          <dl className="mcp-metadata">
            <dt>Registry 标识</dt><dd>{detail.name}</dd>
            <dt>版本</dt><dd>{detail.version}</dd>
          </dl>
          {detail.remotes.map((item, index) => <div key={index} className="mcp-technical-item">
            <CopyableAddress address={item.url} />
            <p className="muted">{item.transport} · {remoteAuth(item)}</p>
          </div>)}
          {detail.packages.map((item, index) => <p key={index} className="mcp-technical-item">
            {item.type} <code>{item.identifier}</code> {item.version}
          </p>)}
          {!!remotes.length && <label className="field">服务标识（自动生成，可修改）<input maxLength={20} value={customName}
            placeholder={autoName} aria-invalid={invalidName} onChange={(event) => setCustomName(event.target.value)} /></label>}
        </details>
        {invalidName && <p role="alert">服务标识须为未使用的 1–20 位字母、数字或连字符。</p>}
        {!!remotes.length && <div className="actions"><button className="primary" type="submit" disabled={!canAdd}>添加并连接</button></div>}
      </fieldset>
    </form>
  </section>

  return <section className="mcp-panel" aria-label="发现 MCP 服务">
    <form className="mcp-search" onSubmit={(event) => {
      event.preventDefault()
      if (!busy && query.trim().length >= 3) void search(query.trim(), null)
    }}>
      <label className="field">搜索官方 MCP Registry<input required value={query} minLength={3} maxLength={160}
        placeholder="输入至少 3 个字符" onChange={(event) => setQuery(event.target.value)} /></label>
      <button className="primary" disabled={busy || query.trim().length < 3}>搜索</button>
    </form>
    {error && <p role="alert">{error}</p>}
    {busy && <p role="status">正在查询官方 MCP Registry…</p>}
    {page && !page.servers.length && <p className="muted">未找到服务。</p>}
    <ul className="mcp-list">{page?.servers.map((item) => <li key={item.name} className="mcp-result">
      <h3><button className="mcp-text-button" type="button" disabled={busy} onClick={() => void inspect(item)}>
        {registryTitle(item)}
      </button></h3>
      {item.description && <p className="mcp-result-description">{item.description}</p>}
      <p className="muted">{item.publisher}{` · ${addressHost(item.repository) || item.source || '未提供来源'}`}</p>
      <p className="muted">
        {[item.remotes.length ? '远程服务' : '', item.packages.length ? (item.remotes.length ? '提供本地安装包' : '需要本地安装') : '']
          .filter(Boolean).join(' · ') || '未提供连接方式'}
        {compatibleRemotes(item).length ? ' · 可直接添加' : ' · 暂不能直接添加'}
      </p>
    </li>)}</ul>
    {page?.cursor && <div className="actions"><button disabled={busy} onClick={() => void search(submitted, page.cursor)}>加载更多</button></div>}
  </section>
}

function remoteAuth(remote: RegistryRemote) {
  if (remote.auth === 'unsupported') return remote.reason ?? '暂不支持此连接方式或认证要求。'
  return remote.reason ?? (remote.auth === 'bearer' ? '需要 Bearer Token' : '无需配置凭据')
}
