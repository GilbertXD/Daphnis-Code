import { useState } from 'react'
import type { FormEvent } from 'react'
import type { Skill, SkillScope } from './protocol'

const scopeLabel = { global: '全局', project: '当前工作区' }

export function SkillsManager({ skills, disabled, notice, perform, clearNotice }: {
  skills: Skill[]
  disabled: boolean
  notice: string
  perform: (action: string, fields: Record<string, unknown>) => Promise<boolean>
  clearNotice: () => void
}) {
  const [importing, setImporting] = useState(false)
  const [source, setSource] = useState('')
  const [scope, setScope] = useState<SkillScope>('project')

  function closeImport() {
    setImporting(false)
    setSource('')
    setScope('project')
  }

  async function importSkill(event: FormEvent) {
    event.preventDefault()
    if (!disabled && source.trim() && await perform('skill_import', { source: source.trim(), scope })) {
      closeImport()
    }
  }

  if (importing) return <form onSubmit={(event) => void importSkill(event)}>
    <fieldset disabled={disabled}>
      <label className="field">本地 Skill 目录
        <input required value={source} onChange={(event) => setSource(event.target.value)}
          placeholder="绝对路径或工作区相对路径" />
      </label>
      <label className="field">导入到
        <select value={scope} onChange={(event) => setScope(event.target.value as SkillScope)}>
          <option value="project">当前工作区</option>
          <option value="global">全局</option>
        </select>
      </label>
      {notice && <p role="alert">{notice}</p>}
      <div className="actions">
        <button type="button" onClick={() => { clearNotice(); closeImport() }}>取消</button>
        <button className="primary" type="submit" disabled={!source.trim()}>导入</button>
      </div>
    </fieldset>
  </form>

  return <>
    {skills.length === 0 && <p className="muted">全局和当前工作区均无 Skill。</p>}
    <ul className="mcp-list">{skills.map((skill) => <li key={skill.id}>
      <div className="mcp-summary">
        <strong>{skill.name}</strong>
        <span className="muted">{scopeLabel[skill.scope]}</span>
        <div className="skill-controls">
          <select aria-label={`${scopeLabel[skill.scope]} ${skill.name} 加载方式`} value={skill.state}
            disabled={disabled || !!skill.error} onChange={(event) => void perform('skill_set_state', {
              skill_id: skill.id, state: event.target.value,
            })}>
            <option value="disabled">停用</option>
            <option value="always">固定加载</option>
            <option value="on_demand">按需加载</option>
          </select>
          <button className="icon" type="button" disabled={disabled} aria-label={`移除 Skill ${skill.name}`}
            title="移除 Skill" onClick={() => {
              if (window.confirm(`从${scopeLabel[skill.scope]}移除 ${skill.name}？`)) {
                void perform('skill_remove', { skill_id: skill.id })
              }
            }}>×</button>
        </div>
      </div>
      {skill.description && <p>{skill.description}</p>}
      {skill.error && <p role="alert">{skill.error}</p>}
    </li>)}</ul>
    <div className="actions"><button className="primary" type="button" disabled={disabled} onClick={() => {
      clearNotice()
      setImporting(true)
    }}>导入 Skill</button></div>
    {notice && <p role="alert">{notice}</p>}
  </>
}
