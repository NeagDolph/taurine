import { useMemo, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'

export interface CitationTarget {
  title: string
  noteId: string
}

interface MessageMarkdownProps {
  markdown: string
  citations: Map<string, CitationTarget>
  onOpenNote: (noteId: string) => void
  streaming?: boolean
}

const CITATION_RE = /\[((?:N|E|F|P|T)\d+)\]/g

// [N3]/[F7] tokens become markdown links with a citation: scheme so the
// standard link renderer can turn them into chips without a custom parser.
function citationsToLinks(md: string): string {
  return md.replace(CITATION_RE, (_match, id: string) => `[${id}](citation:${id})`)
}

export function MessageMarkdown({
  markdown,
  citations,
  onOpenNote,
  streaming = false
}: MessageMarkdownProps): React.JSX.Element {
  const processed = useMemo(() => citationsToLinks(markdown), [markdown])

  return (
    <div className={`markdown-body message-markdown${streaming ? ' streaming' : ''}`}>
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith('citation:')) {
              const id = href.slice('citation:'.length)
              const target = citations.get(id)
              const kind = id.startsWith('F') ? 'fact' : id.startsWith('E') ? 'passage' : 'note'
              return (
                <button
                  className={`citation-chip citation-${kind}`}
                  type="button"
                  title={target?.title ?? id}
                  onClick={() => target?.noteId && onOpenNote(target.noteId)}
                >
                  {id}
                </button>
              )
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            )
          },
          pre: ({ children }) => <CodeBlock>{children}</CodeBlock>
        }}
      >
        {processed}
      </ReactMarkdown>
      {streaming ? <span className="stream-cursor" aria-hidden="true" /> : null}
    </div>
  )
}

function CodeBlock({ children }: { children?: ReactNode }): React.JSX.Element {
  const [copied, setCopied] = useState(false)

  const copy = (event: React.MouseEvent<HTMLButtonElement>): void => {
    const pre = event.currentTarget.parentElement?.querySelector('pre')
    const text = pre?.textContent ?? ''
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div className="code-block">
      <button className="code-copy" type="button" onClick={copy}>
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre>{children}</pre>
    </div>
  )
}
