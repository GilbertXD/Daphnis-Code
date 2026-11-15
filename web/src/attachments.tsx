import { useEffect, useRef, useState } from 'react'
import type { AttachmentInfo, LoadAttachmentImage } from './protocol'

const MAX_FILE = 128 * 1024
const MAX_IMAGE_FILE = 5 * 1024 * 1024
const MAX_IMAGE_TOTAL = 20 * 1024 * 1024
const MAX_IMAGES = 4
export const MAX_TOTAL = 256 * 1024
export const MAX_FILES = 10
export const textFileTypes = '.txt,.md,.markdown,.rst,.py,.go,.json,.jsonl,.yaml,.yml,.toml,.ini,.cfg,.conf,.js,.jsx,.ts,.tsx,.css,.html,.xml,.csv,.tsv,.log,.sql,.sh,.bash,.zsh,.rs,.java,.kt,.c,.h,.cpp,.hpp,.cc,.cs,.rb,.php,.swift,.vue,.svelte,.tex,.diff,.patch'
const extensions = new Set(textFileTypes.split(','))
const imageExtensions = new Map([
  ['.png', 'image/png'], ['.jpg', 'image/jpeg'], ['.jpeg', 'image/jpeg'], ['.webp', 'image/webp'],
])
const imageTypes = new Set(imageExtensions.values())
export const imageFileTypes = [...imageTypes, ...imageExtensions.keys()].join(',')
const blocked = /^(?:\.env|id_(?:rsa|ed25519|ecdsa|dsa)\.|(?:application_default_credentials|credentials|secrets)\.json$)/i
const binary = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/

type DraftFile = { id: string; name: string; size: number }
type ImageFile = DraftFile & { kind: 'image'; file: File; mimeType: string }
type TextDraft = DraftFile & ({ kind: 'text'; text: string } | { kind: 'resource'; resourceId: string })
export type ImageDraft = ImageFile & { previewUrl: string }
export type DraftAttachment = TextDraft | ImageDraft

function extension(name: string) {
  return name.match(/\.[^.]+$/)?.[0].toLowerCase() ?? ''
}

export function isImageFile(file: File) {
  return file.type.startsWith('image/') || imageExtensions.has(extension(file.name))
}

export async function readFiles(files: File[], existing: DraftAttachment[], allowImages: boolean) {
  const added: (TextDraft | ImageFile)[] = []
  const errors = new Set<string>()
  const images: ImageFile[] = existing.filter((file) => file.kind === 'image')
  let imageTotal = images.reduce((sum, file) => sum + file.size, 0)
  let textTotal = existing.reduce((sum, file) => sum + (file.kind === 'image' ? 0 : file.size), 0)
  let textCount = existing.length - images.length
  for (const file of files) {
    const suffix = extension(file.name)
    const safeName = !!file.name && file.name !== '.' && file.name !== '..' && !blocked.test(file.name)
      && file.name.length <= 255 && file.name === file.name.trim() && !/[/\\]/.test(file.name) && !binary.test(file.name)
    if (isImageFile(file)) {
      if (!allowImages) {
        errors.add('当前模型无法发送图片。')
        continue
      }
      const mimeType = !file.type || file.type === 'application/octet-stream' ? imageExtensions.get(suffix) : file.type
      if (!safeName || !mimeType || !imageTypes.has(mimeType)) {
        errors.add('请选择有效的 PNG、JPEG 或 WebP 图片。')
        continue
      }
      if (images.some((image) => image.name === file.name && image.size === file.size
        && image.mimeType === mimeType && image.file.lastModified === file.lastModified)) continue
      if (images.length >= MAX_IMAGES || !file.size || file.size > MAX_IMAGE_FILE || imageTotal + file.size > MAX_IMAGE_TOTAL) {
        errors.add('每次最多 4 张图片，单张最多 5 MiB，总计最多 20 MiB。')
        continue
      }
      const image: ImageFile = { kind: 'image', id: crypto.randomUUID(), name: file.name, size: file.size, file, mimeType }
      added.push(image)
      images.push(image)
      imageTotal += file.size
      continue
    }
    if (!safeName || !extensions.has(suffix)) {
      errors.add('部分文件格式不受支持，凭据文件无法添加。')
      continue
    }
    if (textCount >= MAX_FILES) {
      errors.add('每次最多可添加 10 个文本附件或资源。')
      continue
    }
    if (file.size > MAX_FILE || textTotal + file.size > MAX_TOTAL) {
      errors.add('单个文本附件最多 128 KiB，总计最多 256 KiB。')
      continue
    }
    try {
      const bytes = await file.arrayBuffer()
      if (bytes.byteLength > MAX_FILE || textTotal + bytes.byteLength > MAX_TOTAL) {
        errors.add('单个文本附件最多 128 KiB，总计最多 256 KiB。')
        continue
      }
      const text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes)
      if (binary.test(text) || /-----BEGIN [A-Z ]*PRIVATE KEY-----/.test(text)) {
        errors.add('不能添加二进制内容或私钥。')
        continue
      }
      added.push({ kind: 'text', id: crypto.randomUUID(), name: file.name, text, size: bytes.byteLength })
      textTotal += bytes.byteLength
      textCount++
    } catch {
      errors.add('部分文件无法读取为有效的 UTF-8 文本。')
    }
  }
  return { added, error: [...errors].join(' ') }
}

type ImageHistory = { sessionId: string; messageIndex: number; loadImage: LoadAttachmentImage }

function HistoryThumbnail({ info, sessionId, messageIndex, loadImage }: ImageHistory & { info: AttachmentInfo }) {
  const root = useRef<HTMLDivElement>(null)
  const dialog = useRef<HTMLDialogElement>(null)
  const [url, setUrl] = useState<string>()
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    const container = root.current
    if (!container) return
    const controller = new AbortController()
    let objectUrl: string | undefined
    let observer: IntersectionObserver | undefined
    const load = () => {
      observer?.disconnect()
      void loadImage(sessionId, messageIndex, info.id, controller.signal).then((blob) => {
        if (controller.signal.aborted) return
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
      }).catch(() => {
        if (!controller.signal.aborted) setFailed(true)
      })
    }
    if (typeof IntersectionObserver === 'undefined') {
      load()
    } else {
      observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) load()
      }, { rootMargin: '120px' })
      observer.observe(container)
    }
    return () => {
      observer?.disconnect()
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [sessionId, messageIndex, info.id, loadImage])

  return <div className="history-thumbnail" ref={root}>
    {url && !failed
      ? <button type="button" className="image-thumb" aria-label={`查看 ${info.name}`}
        onClick={() => dialog.current?.showModal()}>
        <img src={url} alt="" onError={() => setFailed(true)} />
      </button>
      : <small className="image-thumb image-status muted" role="status">{failed ? '图片不可用' : '加载图片…'}</small>}
    <dialog className="image-viewer" ref={dialog} aria-label={info.name}
      onClick={(event) => { if (event.target === event.currentTarget) dialog.current?.close() }}>
      <header><span title={info.name}>{info.name}</span>
        <button type="button" aria-label="关闭大图" onClick={() => dialog.current?.close()}>×</button></header>
      {url && <img src={url} alt={info.name} />}
    </dialog>
  </div>
}

export function AttachmentLabels({ files, remove, history }: {
  files: (AttachmentInfo | DraftAttachment)[]
  remove?: (id: string) => void
  history?: ImageHistory
}) {
  if (!files.length) return null
  return <ul className={remove ? 'attachment-labels attachment-labels-pending' : 'attachment-labels'}
    aria-label={remove ? '待发送附件' : '消息附件'}>
    {files.map((file) => {
      const { id, name } = file
      const draftImage = 'kind' in file && file.kind === 'image' ? file : null
      const savedImage = 'type' in file && file.type === 'image' ? file : null
      return <li key={id} className={draftImage || savedImage ? 'attachment-entry-image' : undefined}>
        {draftImage && <img className="image-thumb" src={draftImage.previewUrl} alt={name} />}
        {savedImage && history && <HistoryThumbnail info={savedImage} {...history} />}
        <span title={name}>{name}</span>
        {remove && <button type="button" aria-label={`移除 ${name}`} onClick={() => remove(id)}>×</button>}
      </li>
    })}
  </ul>
}
