import { useRef, useState } from 'react'
import type { FormEvent } from 'react'
import type { Command, ResourceCatalog, ResourceDraft, ResourceEntry, ResourceText } from './protocol'

export function ResourcePreview({ resource }: { resource: ResourceText }) {
  return <div className="resource-preview">
    <strong>{resource.name}</strong>
    {resource.name !== resource.uri && <p className="muted">{resource.uri}</p>}
    <p className="muted">{resource.mime_type} · {resource.size_bytes.toLocaleString()} 字节</p>
    {resource.redacted && <p className="muted">预览已隐藏检测到的凭据，将使用此处显示的正文。</p>}
    <details>
      <summary>查看全文</summary>
      <pre>{resource.text}</pre>
    </details>
  </div>
}


type AddResources = (resources: ResourceDraft[]) => string | null

function ResourceDetail({ server, entry, command, connected, disabled, add, back }: {
  server: string
  entry: ResourceEntry
  command: Command
  connected: boolean
  disabled: boolean
  add: AddResources
  back: () => void
}) {
  const [fields, setFields] = useState(new Map<string, string>())
  const [preview, setPreview] = useState<ResourceDraft[] | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const inFlight = useRef(false)

  async function read(event: FormEvent) {
    event.preventDefault()
    if (!connected || inFlight.current || entry.error) return
    inFlight.current = true
    setLoading(true)
    setError('')
    setPreview(null)
    try {
      const arguments_ = entry.variables ? Object.fromEntries(fields) : null
      const result = await command('mcp_read_resource', { server, uri: entry.uri, arguments: arguments_ })
      if (!result.mcp_resource_preview?.length) throw new Error('MCP 资源响应为空。')
      setSelected(new Set(result.mcp_resource_preview.map(({ id }) => id)))
      setPreview(result.mcp_resource_preview)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法读取 MCP 资源。')
    } finally {
      inFlight.current = false
      setLoading(false)
    }
  }

  return <div className="prompt-panel">
    <button type="button" className="mcp-text-button" onClick={back}>← 返回资源目录</button>
    <h3>{entry.title || entry.name}</h3>
    <p className="muted resource-uri">{entry.uri}</p>
    {entry.description && <p>{entry.description}</p>}
    {entry.mime_type && <p className="muted">{entry.mime_type}</p>}
    <form onSubmit={(event) => void read(event)}>
      <fieldset disabled={!connected || loading || !!entry.error}>
        {entry.variables?.map((name) => <label className="field" key={name}>
          <span>{name}</span>
          <input required maxLength={512} value={fields.get(name) ?? ''} onChange={(event) => {
            const value = event.target.value
            setFields((current) => new Map(current).set(name, value))
            setPreview(null)
          }} />
        </label>)}
        <div className="actions"><button type="submit" className="primary"
          disabled={!connected || loading || !!entry.error || entry.variables?.some((name) => !fields.get(name)?.trim())}>
          {loading ? '正在读取…' : '读取资源'}
        </button></div>
      </fieldset>
    </form>
    {(entry.error || error) && <p role="alert">{entry.error || error}</p>}
    {preview && <div className="prompt-preview">
      {preview.map((resource) => <div key={resource.id} className="resource-result">
        {preview.length > 1 && <label><input type="checkbox" checked={selected.has(resource.id)}
          onChange={(event) => {
            const checked = event.target.checked
            setSelected((current) => {
              const next = new Set(current)
              if (checked) next.add(resource.id)
              else next.delete(resource.id)
              return next
            })
          }} /> 选择 {resource.name}</label>}
        <ResourcePreview resource={resource} />
      </div>)}
      <div className="actions"><button type="button" className="primary" disabled={disabled || !selected.size}
        onClick={() => setError(add(preview.filter(({ id }) => selected.has(id))) ?? '')}>添加到消息</button></div>
    </div>}
  </div>
}

function ResourceGroup({ server, command, connected, disabled, add }: {
  server: string
  command: Command
  connected: boolean
  disabled: boolean
  add: AddResources
}) {
  const [catalog, setCatalog] = useState<ResourceCatalog | null>(null)
  const [entry, setEntry] = useState<ResourceEntry | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const inFlight = useRef(false)
  async function load() {
    if (inFlight.current || !connected) return
    inFlight.current = true
    setLoading(true)
    setError('')
    try {
      const result = await command('mcp_list_resources', { server })
      if (!result.mcp_resource_catalog) throw new Error('MCP 资源目录响应无效。')
      setCatalog(result.mcp_resource_catalog)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法读取 MCP 资源目录。')
    } finally {
      inFlight.current = false
      setLoading(false)
    }
  }
  return <details className="prompt-group" onToggle={(event) => {
    if (event.currentTarget.open && !catalog && !error) void load()
  }}>
    <summary>{server}</summary>
    {entry ? <ResourceDetail server={server} entry={entry} command={command} connected={connected}
      disabled={disabled} add={add} back={() => setEntry(null)} /> : <>
      {loading && <p className="muted" role="status">正在读取资源目录…</p>}
      {error && <p role="alert">{error}</p>}
      {catalog?.errors.map((message, index) => <p key={index} role="alert">{message}</p>)}
      {catalog && !catalog.entries.length && !catalog.errors.length && <p className="muted">此服务没有提供资源。</p>}
      <ul className="prompt-list">{catalog?.entries.map((item) => <li key={item.uri}>
        <button className="mcp-text-button" type="button" onClick={() => setEntry(item)}>
          {item.title || item.name}
        </button>
        <p className="muted resource-uri">{item.uri}{(item.variables || item.error) && ' · URI 模板'}</p>
      </li>)}</ul>
      <button className="mcp-text-button" type="button" disabled={!connected || loading}
        onClick={() => void load()}>刷新目录</button>
    </>}
  </details>
}

export function ResourceBrowser({ servers, connected, disabled, command, add }: {
  servers: string[]
  connected: boolean
  disabled: boolean
  command: Command
  add: AddResources
}) {
  return <>
    {!servers?.length && <p className="muted">当前没有已连接且提供资源的 MCP 服务。</p>}
    {servers?.map((server) => <ResourceGroup key={server} server={server} command={command}
      connected={connected} disabled={disabled} add={add} />)}
  </>
}
