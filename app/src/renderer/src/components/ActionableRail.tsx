import { useState } from 'react'
import { sourceLabel } from '../lib/labels'
import type { Actionable, EvidencePack } from '../lib/sidecar'

interface ActionableRailProps {
  actionables: Actionable[]
  evidence: EvidencePack | null
  onStatus: (id: string, status: 'confirmed' | 'dismissed') => void
  onOpenNote: (noteId: string) => void
}

export function ActionableRail({
  actionables,
  evidence,
  onStatus,
  onOpenNote
}: ActionableRailProps): React.JSX.Element | null {
  const open = actionables.filter((item) => item.status !== 'dismissed')
  if (open.length === 0) {
    return null
  }

  return (
    <section className="flex min-h-0 flex-col gap-2 overflow-y-auto" aria-label="Actionables">
      <div className="section-label">Actionables</div>
      <div className="flex flex-col gap-2.5">
        {open.map((item) => (
          <ActionableCard
            key={item.id}
            item={item}
            evidence={evidence}
            onStatus={onStatus}
            onOpenNote={onOpenNote}
          />
        ))}
      </div>
    </section>
  )
}

interface ActionableCardProps {
  item: Actionable
  evidence: EvidencePack | null
  onStatus: (id: string, status: 'confirmed' | 'dismissed') => void
  onOpenNote: (noteId: string) => void
}

function ActionableCard({
  item,
  evidence,
  onStatus,
  onOpenNote
}: ActionableCardProps): React.JSX.Element {
  const [expanded, setExpanded] = useState(false)

  const groundingRefs = item.grounding.map((id) => ({
    id,
    fact: evidence?.facts.find((row) => row.fact_ref === id),
    passage: evidence?.passages.find((row) => row.evidence_id === id),
    note: evidence?.notes.find((row) => row.note_ref === id)
  }))

  return (
    <article
      className={`flex flex-col gap-2 rounded-[10px] border border-white/10 border-l-[3px] bg-surface-2/70 px-3.5 py-3 shadow-[0_16px_40px_rgba(0,0,0,0.4)] ${
        item.status === 'confirmed' ? 'border-l-[#58c98d]' : 'border-l-[#e5b84a]'
      }`}
    >
      <div className="flex items-start justify-between gap-2.5">
        <div className="min-w-0">
          <h2 className="text-[13px] font-bold leading-snug text-fg">{item.title}</h2>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {item.assignee ? <span className="meta-chip">{item.assignee}</span> : null}
            {item.due ? <span className="meta-chip due">{item.due}</span> : null}
            <span className="meta-chip confidence">{Math.round(item.confidence * 100)}%</span>
            {item.status === 'confirmed' ? <span className="meta-chip ok">confirmed</span> : null}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {item.status !== 'confirmed' ? (
            <button
              className="secondary-button"
              type="button"
              onClick={() => onStatus(item.id, 'confirmed')}
            >
              Confirm
            </button>
          ) : null}
          <button
            className="icon-button"
            type="button"
            aria-label={`Dismiss ${item.title}`}
            onClick={() => onStatus(item.id, 'dismissed')}
          >
            &times;
          </button>
        </div>
      </div>

      <blockquote className="m-0 border-l-2 border-white/15 pl-2 text-xs italic leading-snug text-fg/65">
        &ldquo;{item.quote}&rdquo;
      </blockquote>

      <button
        className="link-button"
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        {expanded ? 'Hide grounding' : `Why this actionable · ${groundingRefs.length} sources`}
      </button>

      {expanded ? (
        <div className="border-t border-white/10 pt-2 text-xs">
          {item.why_new ? <p className="mb-1.5 italic text-fg/65">{item.why_new}</p> : null}
          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
            {groundingRefs.map((ref) =>
              ref.fact ? (
                <li className="flex items-baseline gap-1.5" key={ref.id}>
                  <span className="citation-chip citation-fact">{ref.id}</span>
                  <span className="inline-flex min-w-0 flex-wrap gap-1 text-fg/65">
                    {ref.fact.subject} {ref.fact.predicate} {ref.fact.object}
                  </span>
                </li>
              ) : ref.passage ? (
                <li className="flex items-baseline gap-1.5" key={ref.id}>
                  <button
                    className="citation-chip citation-passage"
                    type="button"
                    onClick={() => onOpenNote(ref.passage!.note_id)}
                  >
                    {ref.id}
                  </button>
                  <span className="inline-flex min-w-0 flex-wrap gap-1 text-fg/65">{ref.passage.text}</span>
                </li>
              ) : ref.note ? (
                <li className="flex items-baseline gap-1.5" key={ref.id}>
                  <button
                    className="citation-chip citation-note"
                    type="button"
                    onClick={() => onOpenNote(ref.note!.note_id)}
                  >
                    {ref.id}
                  </button>
                  <span className="inline-flex min-w-0 flex-wrap gap-1 text-fg/65">
                    <strong className="font-semibold text-fg">{ref.note.title}</strong>
                    <span className={`source-chip source-${ref.note.source}`}>
                      {sourceLabel(ref.note.source)}
                    </span>
                  </span>
                </li>
              ) : ref.id.startsWith('T') ? (
                <li className="flex items-baseline gap-1.5" key={ref.id}>
                  <span className="citation-chip citation-transcript">{ref.id}</span>
                  <span className="inline-flex min-w-0 flex-wrap gap-1 text-fg/65">Transcript evidence</span>
                </li>
              ) : null
            )}
          </ul>
        </div>
      ) : null}
    </article>
  )
}
