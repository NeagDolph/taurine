import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import type { SummaryEvent } from '../lib/sidecar'

interface SummaryPanelProps {
  summary: SummaryEvent | null
  onOpenNote: (noteId: string) => void
}

const CITATION_RE = /\[((?:N|F)\d+)\]/g

// [N3]/[F7] tokens become markdown links with a citation: scheme so the
// standard link renderer can turn them into chips without a custom parser.
function citationsToLinks(md: string): string {
  return md.replace(CITATION_RE, (_match, id: string) => `[${id}](citation:${id})`)
}

export function SummaryPanel({ summary, onOpenNote }: SummaryPanelProps): React.JSX.Element {
  const byShortId = useMemo(() => {
    const map = new Map<string, { title: string; noteId: string }>()
    if (!summary) {
      return map
    }
    for (const item of summary.grounding) {
      map.set(item.short_id, { title: item.title, noteId: item.note_id })
    }
    for (const fact of summary.facts) {
      map.set(fact.fact_id, { title: fact.text, noteId: fact.src })
    }
    return map
  }, [summary])

  const markdown = useMemo(() => (summary ? citationsToLinks(summary.summary_md) : ''), [summary])

  return (
    <section className="summary-region" aria-label="Meeting summary">
      <div className="summary-heading">
        <span className="section-label">Meeting so far</span>
        {summary ? <span className="summary-rev">rev {summary.rev}</span> : null}
      </div>

      {!summary || !summary.summary_md ? (
        <div className="empty-surface summary-empty">
          The summary appears here once the conversation touches your corpus
        </div>
      ) : (
        <div className="summary-body markdown-body">
          <ReactMarkdown
            components={{
              a: ({ href, children }) => {
                if (href?.startsWith('citation:')) {
                  const id = href.slice('citation:'.length)
                  const target = byShortId.get(id)
                  const kind = id.startsWith('F') ? 'fact' : 'note'
                  return (
                    <button
                      className={`citation-chip citation-${kind}`}
                      type="button"
                      title={target?.title ?? id}
                      onClick={() => target && onOpenNote(target.noteId)}
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
              }
            }}
          >
            {markdown}
          </ReactMarkdown>
        </div>
      )}

      {summary && summary.highlights.length > 0 ? (
        <ul className="summary-highlights">
          {summary.highlights.map((highlight) => {
            const fact = summary.facts.find((item) => item.fact_id === highlight.fact_id)
            return (
              <li key={highlight.fact_id}>
                <span className="citation-chip citation-fact">{highlight.fact_id}</span>
                <span className="highlight-fact">{fact?.text ?? ''}</span>
                <span className="highlight-why">{highlight.why}</span>
              </li>
            )
          })}
        </ul>
      ) : null}
    </section>
  )
}
