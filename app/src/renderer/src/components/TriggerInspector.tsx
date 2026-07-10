import { useEffect } from 'react'
import type { LiveGroundingEvent } from '../lib/sidecar'

interface TriggerInspectorProps {
  grounding: LiveGroundingEvent | null
  onClose: () => void
  onOpenNote: (noteId: string) => void
}

export function TriggerInspector({
  grounding,
  onClose,
  onOpenNote
}: TriggerInspectorProps): React.JSX.Element | null {
  useEffect(() => {
    if (!grounding) return undefined
    const close = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [grounding, onClose])

  if (!grounding) return null
  const trace = grounding.trace
  const headHits = arrayRows(trace.head_hits)
  const rejected = arrayRows(trace.rejected_hits)
  const triples = arrayRows(trace.triple_decisions)
  const paths = arrayRows(trace.paths)
  const rerank = arrayRows(trace.rerank)
  const latency = objectRows(trace.latency_ms)

  return (
    <aside className="trigger-inspector" role="dialog" aria-label="Retrieval inspector">
      <div className="trigger-inspector-header">
        <div>
          <span className="section-label">Retrieval trace</span>
          <h2>{grounding.decision.accepted ? 'Evidence accepted' : 'Retrieval abstained'}</h2>
        </div>
        <button
          className="icon-button"
          type="button"
          aria-label="Close retrieval inspector"
          onClick={onClose}
        >
          &times;
        </button>
      </div>

      <div className="trigger-inspector-badges">
        <span
          className={`trigger-action-badge trigger-action-${grounding.decision.accepted ? 'grounded' : 'none'}`}
        >
          {grounding.decision.reason}
        </span>
        <span className="meta-chip confidence">
          {Math.round(grounding.decision.confidence * 100)}%
        </span>
        {trace.debug &&
        typeof trace.debug === 'object' &&
        (trace.debug as Record<string, unknown>).rerank_degraded ? (
          <span className="meta-chip">degraded</span>
        ) : null}
      </div>

      <TraceSection title="Accepted heads" rows={headHits} />
      <TraceSection title="Rejected heads" rows={rejected} />
      <TraceSection title="Triple recognition" rows={triples} />
      <TraceSection title="Explanatory paths" rows={paths} />
      <TraceSection
        title="Reranking and MMR"
        rows={[...rerank, ...arrayRows(trace.mmr_exclusions)]}
      />
      <TraceSection title="Stage latency" rows={latency} />

      <div className="trigger-match-table-wrap">
        <table className="trigger-match-table">
          <thead>
            <tr>
              <th>Evidence</th>
              <th>Source note</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {grounding.passages.map((passage) => (
              <tr key={passage.evidence_id}>
                <td>{passage.evidence_id}</td>
                <td>{passage.title}</td>
                <td>
                  <button
                    className="secondary-button compact"
                    type="button"
                    onClick={() => onOpenNote(passage.note_id)}
                  >
                    Open in Notes
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <footer className="trigger-inspector-footer">
        <code>{String(trace.active_build_id ?? '')}</code>
        <code>{String(trace.calibration_hash ?? '')}</code>
      </footer>
    </aside>
  )
}

function TraceSection({
  title,
  rows
}: {
  title: string
  rows: string[]
}): React.JSX.Element | null {
  if (rows.length === 0) return null
  return (
    <section className="retrieval-trace-section">
      <h3>{title}</h3>
      <ul>
        {rows.slice(0, 20).map((row, index) => (
          <li key={`${title}-${index}`}>
            <code>{row}</code>
          </li>
        ))}
      </ul>
    </section>
  )
}

function arrayRows(value: unknown): string[] {
  return Array.isArray(value) ? value.map((row) => compact(row)) : []
}

function objectRows(value: unknown): string[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return []
  return Object.entries(value as Record<string, unknown>).map(
    ([key, row]) => `${key}: ${compact(row)}`
  )
}

function compact(value: unknown): string {
  const rendered = typeof value === 'string' ? value : JSON.stringify(value)
  return rendered.length > 240 ? `${rendered.slice(0, 237)}…` : rendered
}
