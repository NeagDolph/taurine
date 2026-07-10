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
    <section className="actionable-rail" aria-label="Actionables">
      <div className="section-label">Actionables</div>
      <div className="actionable-cards">
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
    <article className={`actionable-card${item.status === 'confirmed' ? ' confirmed' : ''}`}>
      <div className="card-header">
        <div>
          <h2>{item.title}</h2>
          <div className="actionable-chips">
            {item.assignee ? <span className="meta-chip">{item.assignee}</span> : null}
            {item.due ? <span className="meta-chip due">{item.due}</span> : null}
            <span className="meta-chip confidence">{Math.round(item.confidence * 100)}%</span>
            {item.status === 'confirmed' ? <span className="meta-chip ok">confirmed</span> : null}
          </div>
        </div>
        <div className="actionable-actions">
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

      <blockquote className="actionable-quote">&ldquo;{item.quote}&rdquo;</blockquote>

      <button
        className="link-button"
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        {expanded ? 'Hide grounding' : `Why this actionable · ${groundingRefs.length} sources`}
      </button>

      {expanded ? (
        <div className="grounding-drawer">
          {item.why_new ? <p className="grounding-why-new">{item.why_new}</p> : null}
          <ul>
            {groundingRefs.map((ref) =>
              ref.fact ? (
                <li key={ref.id}>
                  <span className="citation-chip citation-fact">{ref.id}</span>
                  <span className="grounding-text">
                    {ref.fact.subject} {ref.fact.predicate} {ref.fact.object}
                  </span>
                </li>
              ) : ref.passage ? (
                <li key={ref.id}>
                  <button
                    className="citation-chip citation-passage"
                    type="button"
                    onClick={() => onOpenNote(ref.passage!.note_id)}
                  >
                    {ref.id}
                  </button>
                  <span className="grounding-text">{ref.passage.text}</span>
                </li>
              ) : ref.note ? (
                <li key={ref.id}>
                  <button
                    className="citation-chip citation-note"
                    type="button"
                    onClick={() => onOpenNote(ref.note!.note_id)}
                  >
                    {ref.id}
                  </button>
                  <span className="grounding-text">
                    <strong>{ref.note.title}</strong>
                    <span className={`source-chip source-${ref.note.source}`}>
                      {sourceLabel(ref.note.source)}
                    </span>
                  </span>
                </li>
              ) : ref.id.startsWith('T') ? (
                <li key={ref.id}>
                  <span className="citation-chip citation-transcript">{ref.id}</span>
                  <span className="grounding-text">Transcript evidence</span>
                </li>
              ) : null
            )}
          </ul>
        </div>
      ) : null}
    </article>
  )
}
