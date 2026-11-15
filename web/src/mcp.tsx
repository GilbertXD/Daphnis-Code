import { useState } from 'react'
import type { FormEvent } from 'react'
import type { MCPOptions, MCPServer } from './protocol'

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
      <label className="field">连接方式<select value={transport}
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

export function MCPManager({ servers, disabled, error, notice, perform }: {
  servers: MCPServer[]
  disabled: boolean
  error: string | null
  notice: string
  perform: Perform
}) {
  const [editing, setEditing] = useState<MCPServer | 'new' | null>(null)
  const blocked = disabled || !!error
  return <>
    {error && <p role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    {editing ? <MCPEditor initial={editing === 'new' ? undefined : editing} disabled={blocked}
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
