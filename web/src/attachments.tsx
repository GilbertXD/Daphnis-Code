const MAX_FILE = 128 * 1024
const MAX_TOTAL = 256 * 1024
const MAX_FILES = 10
export const textFileTypes = '.txt,.md,.markdown,.rst,.py,.go,.json,.jsonl,.yaml,.yml,.toml,.ini,.cfg,.conf,.js,.jsx,.ts,.tsx,.css,.html,.xml,.csv,.tsv,.log,.sql,.sh,.bash,.zsh,.rs,.java,.kt,.c,.h,.cpp,.hpp,.cc,.cs,.rb,.php,.swift,.vue,.svelte,.tex,.diff,.patch'
const extensions = new Set(textFileTypes.split(','))
const blocked = /^(?:\.env|id_(?:rsa|ed25519|ecdsa|dsa)\.|(?:application_default_credentials|credentials|secrets)\.json$)/i
const binary = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/

export type DraftAttachment = { id: string; name: string; text: string; size: number }

export async function readTextFiles(files: File[], existing: DraftAttachment[]) {
  const added: DraftAttachment[] = []
  const errors = new Set<string>()
  let total = existing.reduce((sum, file) => sum + file.size, 0)
  for (const file of files) {
    if (existing.length + added.length === MAX_FILES) {
      errors.add('每次最多可添加 10 个附件。')
      break
    }
    const extension = file.name.match(/\.[^.]+$/)?.[0].toLowerCase()
    if (!extension || !extensions.has(extension) || blocked.test(file.name)
      || file.name.length > 255 || file.name !== file.name.trim() || /[/\\]/.test(file.name) || binary.test(file.name)) {
      errors.add('请选择受支持的文本文件，凭据文件无法添加。')
      continue
    }
    if (file.size > MAX_FILE || total + file.size > MAX_TOTAL) {
      errors.add('单个附件最多 128 KiB，总计最多 256 KiB。')
      continue
    }
    try {
      const bytes = await file.arrayBuffer()
      if (bytes.byteLength > MAX_FILE || total + bytes.byteLength > MAX_TOTAL) {
        errors.add('单个附件最多 128 KiB，总计最多 256 KiB。')
        continue
      }
      const text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes)
      if (binary.test(text) || /-----BEGIN [A-Z ]*PRIVATE KEY-----/.test(text)) {
        errors.add('不能添加二进制内容或私钥。')
        continue
      }
      added.push({ id: crypto.randomUUID(), name: file.name, text, size: bytes.byteLength })
      total += bytes.byteLength
    } catch {
      errors.add('部分文件无法读取为有效的 UTF-8 文本。')
    }
  }
  return { added, error: [...errors].join(' ') }
}

export function AttachmentLabels({ files, remove }: {
  files: { id: string; name: string }[]
  remove?: (id: string) => void
}) {
  if (!files.length) return null
  return <ul className="attachment-labels" aria-label={remove ? '待发送附件' : '消息附件'}>
    {files.map(({ id, name }) => <li key={id}>
      <span>{name}</span>
      {remove && <button type="button" aria-label={`移除 ${name}`} onClick={() => remove(id)}>×</button>}
    </li>)}
  </ul>
}
