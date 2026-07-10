import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import type { SummaryEvent } from '../lib/sidecar'

interface SummaryPanelProps {
  summary: SummaryEvent | null
  onOpenNote: (noteId: string) => void
  onShowInMap?: (noteId: string) => void
}

const CITATION_RE = /\[((?:N|E|F|P|T)\d+)\]/g

// [N3]/[F7] tokens become markdown links with a citation: scheme so the
// standard link renderer can turn them into chips without a custom parser.
function citationsToLinks(md: string): string {
  return md.replace(CITATION_RE, (_match, id: string) => `[${id}](citation:${id})`)
}

export function SummaryPanel({
  summary,
  onOpenNote,
  onShowInMap
}: SummaryPanelProps): React.JSX.Element {
  const byShortId = useMemo(() => {
    const map = new Map<string, { title: string; noteId: string }>()
    if (!summary) {
      return map
    }
    for (const item of summary.evidence?.notes ?? []) {
      map.set(item.note_ref, { title: item.title, noteId: item.note_id })
    }
    for (const passage of summary.evidence?.passages ?? []) {
      map.set(passage.evidence_id, { title: passage.text, noteId: passage.note_id })
    }
    for (const fact of summary.evidence?.facts ?? []) {
      const source = summary.evidence?.passages.find((row) =>
        fact.source_evidence_ids.includes(row.evidence_id)
      )
      map.set(fact.fact_ref, {
        title: `${fact.subject} ${fact.predicate} ${fact.object}`,
        noteId: source?.note_id ?? ''
      })
    }
    for (const path of summary.evidence?.paths ?? []) {
      map.set(path.path_id, { title: path.rendered, noteId: '' })
    }
    return map
  }, [summary])

  const markdown = useMemo(() => (summary ? citationsToLinks(summary.summary_md) : ''), [summary])

  return (
    <section
      className="flex min-h-24 flex-col gap-2 overflow-y-auto rounded-[10px] border border-white/10 border-l-[3px] border-l-accent bg-surface-2/70 px-3.5 py-3 shadow-[0_16px_40px_rgba(0,0,0,0.4)] backdrop-blur-md"
      aria-label="Meeting summary"
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="section-label">Meeting so far</span>
        {summary ? <span className="text-[11px] text-fg/60">rev {summary.rev}</span> : null}
      </div>

      {!summary || !summary.summary_md ? (
        <div className="empty-surface min-h-16">
          The summary appears here once the conversation touches your corpus
        </div>
      ) : (
        <div className="markdown-body text-[13px] leading-normal text-fg">
          <ReactMarkdown
            components={{
              a: ({ href, children }) => {
                if (href?.startsWith('citation:')) {
                  const id = href.slice('citation:'.length)
                  const target = byShortId.get(id)
                  const kind = id.startsWith('F') ? 'fact' : id.startsWith('E') ? 'passage' : 'note'
                  const noteId = target?.noteId ?? ''
                  return (
                    <span className="map-cite-wrap">
                      <button
                        className={`citation-chip citation-${kind}`}
                        type="button"
                        title={target?.title ?? id}
                        onClick={() => noteId && onOpenNote(noteId)}
                      >
                        {id}
                      </button>
                      {onShowInMap && noteId ? (
                        <button
                          className="map-cite-btn"
                          type="button"
                          title="Show in map"
                          aria-label={`Show ${target?.title ?? id} in map`}
                          onClick={() => onShowInMap(noteId)}
                        >
                          ⌖
                        </button>
                      ) : null}
                    </span>
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
        <ul className="m-0 flex list-none flex-col gap-1.5 border-t border-white/10 p-0 pt-2">
          {summary.highlights.map((highlight) => {
            const fact = summary.evidence?.facts.find((item) => item.fact_ref === highlight.fact_id)
            return (
              <li className="flex items-baseline gap-1.5 text-xs" key={highlight.fact_id}>
                <span className="citation-chip citation-fact">{highlight.fact_id}</span>
                <span className="text-fg">
                  {fact ? `${fact.subject} ${fact.predicate} ${fact.object}` : ''}
                </span>
                <span className="italic text-fg/65">{highlight.why}</span>
              </li>
            )
          })}
        </ul>
      ) : null}
    </section>
  )
}
