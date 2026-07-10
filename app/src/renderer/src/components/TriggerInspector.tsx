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
    <aside
      className="fixed bottom-[68px] right-3 top-24 z-30 grid w-[min(440px,calc(100vw-28px))] grid-rows-[auto_auto_auto_minmax(0,1fr)_auto] gap-3 rounded-[10px] border border-white/15 bg-surface-1/90 p-4 text-fg shadow-[0_28px_80px_rgba(0,0,0,0.6)] backdrop-blur-xl"
      role="dialog"
      aria-label="Retrieval inspector"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="section-label">Retrieval trace</span>
          <h2 className="mt-1 text-[15px] font-bold leading-snug text-fg">
            {grounding.decision.accepted ? 'Evidence accepted' : 'Retrieval abstained'}
          </h2>
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

      <div className="flex flex-wrap gap-1.5">
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

      <div className="min-h-0 overflow-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className={MATCH_TH}>Evidence</th>
              <th className={MATCH_TH}>Source note</th>
              <th className={MATCH_TH} />
            </tr>
          </thead>
          <tbody>
            {grounding.passages.map((passage) => (
              <tr key={passage.evidence_id}>
                <td className={MATCH_TD}>{passage.evidence_id}</td>
                <td className={MATCH_TD}>{passage.title}</td>
                <td className={MATCH_TD}>
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

      <footer className="grid gap-1 border-t border-white/[0.08] pt-2.5">
        <code className="truncate font-mono text-[11px] text-fg/45">
          {String(trace.active_build_id ?? '')}
        </code>
        <code className="truncate font-mono text-[11px] text-fg/45">
          {String(trace.calibration_hash ?? '')}
        </code>
      </footer>
    </aside>
  )
}

const MATCH_TH =
  'border-b border-white/[0.08] px-1.5 py-2 text-left align-top text-[11px] font-bold uppercase tracking-wide text-fg/45'
const MATCH_TD = 'border-b border-white/[0.08] px-1.5 py-2 text-left align-top text-xs'

function TraceSection({
  title,
  rows
}: {
  title: string
  rows: string[]
}): React.JSX.Element | null {
  if (rows.length === 0) return null
  return (
    <section className="border-t border-white/[0.08] pt-2.5">
      <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-fg/65">
        {title}
      </h3>
      <ul className="m-0 grid list-none gap-1 p-0">
        {rows.slice(0, 20).map((row, index) => (
          <li key={`${title}-${index}`}>
            <code className="font-mono text-[10px] text-fg/45 [overflow-wrap:anywhere]">{row}</code>
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
