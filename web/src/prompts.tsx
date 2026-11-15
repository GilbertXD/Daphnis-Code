import { useRef, useState } from 'react'
import type { FormEvent } from 'react'
import type { Command, MCPPrompt, MCPPromptResult } from './protocol'

const MAX_INSERT_BYTES = 16 * 1024

function PromptGroup({ server, connected, command, select }: {
  server: string
  connected: boolean
  command: Command
  select: (prompt: MCPPrompt) => void
}) {
  const [prompts, setPrompts] = useState<MCPPrompt[] | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const inFlight = useRef(false)

  async function load() {
    if (inFlight.current) return
    inFlight.current = true
    setLoading(true)
    setError('')
    try {
      const response = await command('mcp_list_prompts', { server })
      if (!response.mcp_prompts) throw new Error('模板列表无效。')
      setPrompts(response.mcp_prompts)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法读取模板列表。')
    } finally {
      setLoading(false)
      inFlight.current = false
    }
  }

  return <details className="prompt-group" onToggle={(event) => {
    if (event.currentTarget.open && prompts === null && !error) void load()
  }}>
    <summary>{server}{prompts !== null && <small className="muted"> · {prompts.length} 个模板</small>}</summary>
    {loading && <p className="muted" role="status">正在读取模板…</p>}
    {error && <div className="prompt-panel">
      <p role="alert">{error}</p>
      <button className="mcp-text-button" type="button" disabled={!connected} onClick={() => void load()}>重试</button>
    </div>}
    {prompts?.length === 0 && <p className="muted">此服务没有提供模板。</p>}
    {!!prompts?.length && <ul className="prompt-list">{prompts.map((prompt) => <li key={prompt.name}>
      <button className="mcp-text-button" type="button" onClick={() => select(prompt)}>{prompt.title || prompt.name}</button>
      {prompt.description && <p className="muted">{prompt.description}</p>}
    </li>)}</ul>}
  </details>
}

function PromptForm({ server, prompt, connected, command, back, insert }: {
  server: string
  prompt: MCPPrompt
  connected: boolean
  command: Command
  back: () => void
  insert: (text: string) => void
}) {
  const [values, setValues] = useState(new Map<string, string>())
  const [result, setResult] = useState<MCPPromptResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const disabled = !connected || busy
  const incomplete = prompt.arguments.some((field) => field.required && !values.get(field.name)?.trim())

  async function generate(event: FormEvent) {
    event.preventDefault()
    if (incomplete || disabled) return
    setError('')
    setResult(null)
    setBusy(true)
    try {
      const response = await command('mcp_get_prompt', {
        server, name: prompt.name, arguments: Object.fromEntries([...values].filter(([, value]) => value !== '')),
      })
      if (!response.mcp_prompt_result) throw new Error('模板结果无效。')
      setResult(response.mcp_prompt_result)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法生成模板。')
    } finally {
      setBusy(false)
    }
  }

  return <div className="prompt-panel">
    <button className="mcp-text-button" type="button" onClick={back}>← 返回模板列表</button>
    <h3>{prompt.title || prompt.name}</h3>
    <p className="muted">来源 {server}{prompt.title ? ` · ${prompt.name}` : ''}</p>
    {prompt.description && <p>{prompt.description}</p>}
    <form onSubmit={(event) => void generate(event)}>
      <fieldset disabled={disabled}>
        {prompt.arguments.map((field) => <label className="field" key={field.name}>
          <span>{field.title || field.name} <small className="muted">{field.required ? '必填' : '选填'}</small></span>
          {field.description && <small className="muted">{field.description}</small>}
          <textarea rows={2} required={field.required} value={values.get(field.name) ?? ''}
            onChange={(event) => {
              const value = event.target.value
              setValues((current) => new Map(current).set(field.name, value))
              setResult(null)
            }} />
        </label>)}
        {!!prompt.arguments.length && <p className="muted">生成时会将这些参数发给 {server}。</p>}
        <div className="actions"><button className="primary" type="submit" disabled={disabled || incomplete}>
          {busy ? '正在生成…' : '生成预览'}
        </button></div>
      </fieldset>
    </form>
    {error && <p role="alert">{error}</p>}
    {result && <PromptPreview result={result} insert={insert} disabled={disabled} />}
  </div>
}

function PromptPreview({ result, insert, disabled }: {
  result: MCPPromptResult
  insert: (text: string) => void
  disabled: boolean
}) {
  const message = result.messages.length === 1 ? result.messages[0] : null
  const text = message?.content.type === 'text' ? message.content.text ?? '' : ''
  const unsupported = !message ? '当前仅支持插入恰好一条消息的模板。'
    : message.role !== 'user' ? '当前仅支持插入用户角色的消息。'
    : message.content.type !== 'text' ? '当前仅支持插入纯文本消息。'
    : !text.trim() ? '模板返回了空文本。'
    : new TextEncoder().encode(text).length > MAX_INSERT_BYTES ? '文本超过 16 KiB，无法插入。' : ''

  return <section className="prompt-preview" aria-label="模板预览">
    <h3>生成结果</h3>
    {result.description && <p>{result.description}</p>}
    {result.redacted && <p className="muted">预览和插入内容中的凭据已隐藏。</p>}
    {result.messages.map(({ role, content }, index) => <div className="prompt-message" key={index}>
      <p className="muted">{role === 'user' ? '用户' : '助手'} · {content.type}</p>
      {content.type === 'text' ? <pre>{content.text}</pre> : <p className="muted">非文本内容仅显示类型。</p>}
    </div>)}
    {unsupported && <p role="status">{unsupported}</p>}
    <div className="actions"><button type="button" className="primary" disabled={disabled || !!unsupported}
      onClick={() => insert(text)}>插入输入框</button></div>
  </section>
}

export function PromptPicker({ servers, connected, command, insert }: {
  servers: string[]
  connected: boolean
  command: Command
  insert: (text: string) => void
}) {
  const [selected, setSelected] = useState<{ server: string; prompt: MCPPrompt } | null>(null)
  return <>
    <div className="prompt-catalog" hidden={selected !== null}>
      {!servers.length && <p className="muted">当前没有已连接且支持模板的 MCP 服务。</p>}
      {servers.map((server) => <PromptGroup key={server} server={server} connected={connected} command={command}
        select={(prompt) => setSelected({ server, prompt })} />)}
    </div>
    {selected && <PromptForm {...selected}
      connected={connected && servers.includes(selected.server)} command={command}
      back={() => setSelected(null)} insert={insert} />}
  </>
}
