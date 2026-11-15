import type { ResourceText } from './protocol'

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
