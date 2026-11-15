import { useRef, useState } from 'react'
import type { FormEvent } from 'react'
import type { Command, MCPPrompt, MCPPromptResult } from './protocol'
import { ResourcePreview } from './resources'

const MAX_PROMPT_BYTES = 64 * 1024
const MAX_PROMPT_MESSAGES = 64
type RunPrompt = (previewId: string, texts: string[]) => Promise<void>

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

function PromptForm({ server, prompt, connected, command, back, run, runDisabled }: {
  server: string
  prompt: MCPPrompt
  connected: boolean
  command: Command
  back: () => void
  run: RunPrompt
  runDisabled: boolean
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
    {result && <PromptPreview result={result} run={run} disabled={runDisabled || disabled} />}
  </div>
}

function PromptPreview({ result, run, disabled }: {
  result: MCPPromptResult
  run: RunPrompt
  disabled: boolean
}) {
  const [texts, setTexts] = useState(() => result.messages.map(({ content }) => content.text ?? ''))
  const appended = texts.length > result.messages.length
  const roles = result.messages.map(({ role }) => role)
  if (appended) roles.push('user')
  const hasNonText = result.messages.some(({ content }) => content.type !== 'text' && !content.resource)
  const nonEmpty = texts.every((text, index) => result.messages[index]?.content.resource || text.trim())
  const bytes = texts.reduce((size, text, index) =>
    size + new TextEncoder().encode(JSON.stringify({ role: roles[index], content: text })).length + 3, 0)
  const needsUser = roles.at(-1) === 'assistant'
  const unavailable = hasNonText ? '此 Prompt 包含暂不支持的内容类型，无法完整运行。'
    : !result.preview_id ? '此 Prompt 没有可运行的消息，或消息数量超出限制。'
    : needsUser ? '当前模型接口需要以 User 消息结束。请补充一条 User 消息。'
    : !nonEmpty ? '请填写每条消息的正文。'
    : bytes > MAX_PROMPT_BYTES ? '消息组超过 64 KiB。请缩短正文。' : ''

  function update(index: number, text: string) {
    setTexts((current) => current.map((value, i) => i === index ? text : value))
  }

  return <section className="prompt-preview" aria-label="Prompt 预览">
    <h3>预览与编辑</h3>
    {result.description && <p>{result.description}</p>}
    {result.redacted && <p className="muted">预览已隐藏检测到的凭据。运行时只会提交下方可见的正文。</p>}
    {result.messages.map(({ role, content }, index) => <div className="prompt-message" key={index}>
      {content.type === 'text' ? <label className="field">
        <span>{role === 'user' ? 'User' : 'Assistant'}</span>
        <textarea rows={3} readOnly={!result.preview_id} value={texts[index]}
          onChange={(event) => update(index, event.target.value)} />
      </label> : <>
        <p className="muted">{role === 'user' ? 'User' : 'Assistant'} · {content.type}</p>
        {content.resource && <ResourcePreview resource={content.resource} />}
        {content.error && <p role="status">{content.error}</p>}
      </>}
    </div>)}
    {appended && <div className="prompt-message">
      <label className="field">
        <span>User</span>
        <textarea rows={3} value={texts.at(-1)} onChange={(event) => update(texts.length - 1, event.target.value)} />
      </label>
      <button type="button" className="mcp-text-button" onClick={() => setTexts((current) => current.slice(0, -1))}
        disabled={disabled}>移除补充消息</button>
    </div>}
    {result.preview_id && !appended && texts.length < MAX_PROMPT_MESSAGES &&
      <button type="button" className="mcp-text-button" disabled={disabled}
        onClick={() => setTexts((current) => [...current, ''])}>补充 User 消息</button>}
    {unavailable && <p role="status">{unavailable}</p>}
    <div className="actions"><button type="button" className="primary" disabled={disabled || !!unavailable}
      onClick={() => result.preview_id && void run(result.preview_id, texts)}>运行</button></div>
  </section>
}

export function PromptPicker({ servers, connected, command, run, runDisabled }: {
  servers: string[]
  connected: boolean
  command: Command
  run: RunPrompt
  runDisabled: boolean
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
      back={() => setSelected(null)} run={run} runDisabled={runDisabled} />}
  </>
}
