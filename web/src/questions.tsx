import { useState } from 'react'
import type { Question, QuestionAnswers, QuestionRequest } from './protocol'

type Draft = { selected: string[]; custom: boolean; text: string }

function answerFor(question: Question, draft: Draft) {
  const text = draft.text.trim()
  if (question.type === 'text') return text
  if (draft.custom && !text) return ''
  if (question.type === 'single_select') return draft.custom ? text : draft.selected[0] ?? ''
  return Array.from(new Set(draft.custom ? [...draft.selected, text] : draft.selected))
}

export function QuestionForm({ request, disabled, submit }: {
  request: QuestionRequest
  disabled: boolean
  submit: (answers: QuestionAnswers) => Promise<boolean>
}) {
  const [step, setStep] = useState(0)
  const [submitted, setSubmitted] = useState(false)
  const [drafts, setDrafts] = useState<Draft[]>(() => request.questions.map(() => ({ selected: [], custom: false, text: '' })))
  const question = request.questions[step]
  const draft = drafts[step]
  const last = step + 1 === request.questions.length
  const ready = answerFor(question, draft).length > 0
  const multi = question.type === 'multi_select'

  function update(changes: Partial<Draft>) {
    setDrafts((current) => current.map((item, i) => i === step ? { ...item, ...changes } : item))
  }

  function select(option: string) {
    if (!multi) {
      update({ selected: [option], custom: false })
    } else {
      update({ selected: draft.selected.includes(option)
        ? draft.selected.filter((value) => value !== option) : [...draft.selected, option] })
    }
  }

  return <form className="question-form" aria-label="Agent 提问" onSubmit={async (event) => {
    event.preventDefault()
    if (!ready || disabled || submitted) return
    if (!last) {
      setStep(step + 1)
      return
    }
    const answers = request.questions.map((item, i) => [item.id, answerFor(item, drafts[i])] as const)
    if (answers.every(([, answer]) => answer.length > 0)) setSubmitted(await submit(Object.fromEntries(answers)))
  }}>
    <div className="approval-heading">
      <strong>Agent 提问</strong>
      {request.questions.length > 1 && <span className="muted mono">{step + 1} / {request.questions.length}</span>}
    </div>
    <fieldset key={question.id} disabled={disabled || submitted}>
      <legend>{question.question}</legend>
      {question.type !== 'text' && <>
        {question.options.map((option) => <label className="question-option" key={option}>
          <input type={multi ? 'checkbox' : 'radio'} name={request.request_id} checked={draft.selected.includes(option)}
            onChange={() => select(option)} />
          <span>{option}</span>
        </label>)}
        <label className="question-option">
          <input type={multi ? 'checkbox' : 'radio'} name={request.request_id} checked={draft.custom}
            onChange={() => update({ custom: !draft.custom, ...(multi ? {} : { selected: [] }) })} />
          <span>其他</span>
        </label>
      </>}
      {(question.type === 'text' || draft.custom) && <label className="field">
        {question.type === 'text' ? '回答' : '自定义回答'}
        <textarea rows={2} maxLength={2000} value={draft.text} onChange={(event) => update({ text: event.target.value })} />
      </label>}
    </fieldset>
    <div className="actions">
      {step > 0 && <button type="button" disabled={disabled || submitted} onClick={() => setStep(step - 1)}>上一步</button>}
      <button type="submit" className="primary" disabled={!ready || disabled || submitted}>{last ? '提交回答' : '下一步'}</button>
    </div>
  </form>
}
