import { useState } from 'react'
import type { FormEvent } from 'react'
import type { MemoryItem, MemoryScope } from './protocol'

const scopeLabel: Record<MemoryScope, string> = { project: '当前工作区', global: '全局' }
const filterTabs = [
  ['all', '全部'],
  ['project', '当前工作区'],
  ['global', '全局'],
] as const
type FilterScope = (typeof filterTabs)[number][0]

function formatTime(timestamp: number) {
  return new Date(timestamp * 1000).toLocaleString('zh-CN', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

function previewText(text: string) {
  const lines = text.split('\n')
  if (text.length <= 220 && lines.length <= 4) return null
  const head = lines.slice(0, 3).join('\n')
  return (head.length > 220 ? head.slice(0, 220) : head) + '…'
}

export function MemoryManager({ memories, error, disabled, notice, perform, clearNotice }: {
  memories: MemoryItem[]
  error: string | null
  disabled: boolean
  notice: string
  perform: (action: string, fields: Record<string, unknown>) => Promise<boolean>
  clearNotice: () => void
}) {
  const [filter, setFilter] = useState<FilterScope>('all')
  const [query, setQuery] = useState('')
  const [editing, setEditing] = useState<MemoryItem | 'new' | null>(null)
  const [scope, setScope] = useState<MemoryScope>('project')
  const [title, setTitle] = useState('')
  const [tags, setTags] = useState('')
  const [content, setContent] = useState('')
  const [formError, setFormError] = useState('')
  const blocked = disabled || !!error

  function openEditor(target: MemoryItem | 'new') {
    clearNotice()
    setFormError('')
    setEditing(target)
    if (target === 'new') {
      setScope(filter === 'global' ? 'global' : 'project')
      setTitle('')
      setTags('')
      setContent('')
    } else {
      setScope(target.scope)
      setTitle(target.title)
      setTags(target.tags.join(', '))
      setContent(target.content)
    }
  }

  function closeEditor() {
    clearNotice()
    setFormError('')
    setEditing(null)
  }

  async function saveMemory(event: FormEvent) {
    event.preventDefault()
    if (blocked || !title.trim() || !content.trim()) return
    setFormError('')
    const parsedTags = [...new Set(
      tags.split(/[,，]/).map((item) => item.trim()).filter(Boolean),
    )]
    if (parsedTags.length > 10 || parsedTags.some((tag) => tag.length > 40)) {
      setFormError('最多可填写 10 个标签，单个标签不超过 40 个字符。')
      return
    }
    if (await perform('memory_save', {
      id: editing === 'new' || editing === null ? null : editing.id,
      scope,
      title: title.trim(),
      content: content.trim(),
      tags: parsedTags,
    })) {
      closeEditor()
    }
  }

  if (editing) return <form onSubmit={(event) => void saveMemory(event)}>
    <fieldset disabled={blocked}>
      <label className="field">作用域
        <select value={scope} disabled={editing !== 'new'}
          onChange={(event) => setScope(event.target.value as MemoryScope)}>
          <option value="project">当前工作区</option>
          <option value="global">全局</option>
        </select>
      </label>
      <label className="field">标题
        <input required maxLength={120} value={title} onChange={(event) => setTitle(event.target.value)}
          placeholder="简短概括偏好、项目约定或排错结论" />
      </label>
      <label className="field">标签（可选）
        <input value={tags} onChange={(event) => setTags(event.target.value)}
          placeholder="以逗号分隔，最多 10 个" />
      </label>
      <label className="field">内容
        <textarea required rows={6} maxLength={4000} value={content}
          onChange={(event) => setContent(event.target.value)} />
      </label>
      {(formError || notice) && <p role="alert">{formError || notice}</p>}
      <div className="actions">
        <button type="button" onClick={closeEditor}>取消</button>
        <button className="primary" type="submit" disabled={!title.trim() || !content.trim()}>保存</button>
      </div>
    </fieldset>
  </form>

  const needle = query.trim().toLowerCase()
  const visible = memories.filter((item) => {
    if (filter !== 'all' && item.scope !== filter) return false
    if (!needle) return true
    return item.title.toLowerCase().includes(needle)
      || item.content.toLowerCase().includes(needle)
      || item.tags.some((tag) => tag.toLowerCase().includes(needle))
  })

  return <>
    <div className="mcp-tabs" aria-label="记忆作用域">
      {filterTabs.map(([id, label]) => <button key={id} type="button" aria-pressed={filter === id}
        onClick={() => setFilter(id)}>{label}</button>)}
    </div>
    {error && <p role="alert">{error}</p>}
    {(memories.length > 0 || query) && <label className="field">
      <span className="mcp-sr-only">筛选记忆</span>
      <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="按标题、标签或内容筛选" />
    </label>}
    {!error && memories.length === 0 && <p className="muted">全局和当前工作区均无记忆。</p>}
    {!error && memories.length > 0 && visible.length === 0 && <p className="muted">没有匹配的记忆。</p>}
    <ul className="mcp-list">{visible.map((item) => {
      const preview = previewText(item.content)
      return <li key={`${item.scope}:${item.id}`}>
        <div className="mcp-summary">
          <strong>{item.title}</strong>
          <span className="muted">
            {scopeLabel[item.scope]}{item.tags.length ? ` · ${item.tags.join(' / ')}` : ''} · {formatTime(item.updated_at)}
          </span>
          <div className="skill-controls">
            <button className="mcp-text-button" type="button" disabled={blocked}
              onClick={() => openEditor(item)}>编辑</button>
            <button className="icon" type="button" disabled={blocked} aria-label={`删除记忆 ${item.title}`}
              title="删除记忆" onClick={() => {
                if (window.confirm(`从${scopeLabel[item.scope]}删除记忆「${item.title}」？`)) {
                  void perform('memory_remove', { id: item.id, scope: item.scope })
                }
              }}>×</button>
          </div>
        </div>
        {preview ? <>
          <p className="memory-body">{preview}</p>
          <details className="mcp-operations">
            <summary>查看全文<span className="mcp-sr-only"> · {item.title}</span></summary>
            <p className="memory-body">{item.content}</p>
          </details>
        </> : <p className="memory-body">{item.content}</p>}
      </li>
    })}</ul>
    <div className="actions"><button className="primary" type="button" disabled={blocked}
      onClick={() => openEditor('new')}>新建记忆</button></div>
    {notice && <p role="alert">{notice}</p>}
  </>
}
