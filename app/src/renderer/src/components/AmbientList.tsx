import type { CSSProperties } from 'react'
import { sourceLabel } from '../lib/labels'
import type { EvidenceNote, EvidencePassage } from '../lib/sidecar'

interface AmbientListProps {
  notes: EvidenceNote[]
  passages: EvidencePassage[]
  noNewEvidence?: boolean
  onInspect?: () => void
  onOpenNote?: (noteId: string) => void
}

export function AmbientList({
  notes,
  passages,
  noNewEvidence = false,
  onInspect,
  onOpenNote
}: AmbientListProps): React.JSX.Element {
  const passageByNote = new Map(passages.map((row) => [row.note_id, row]))
  const maxScore = Math.max(1e-6, ...notes.map((row) => row.score))

  return (
    <aside className="ambient-sidebar" aria-label="Retrieved evidence">
      <div className="sidebar-heading">
        <span className="section-label">Evidence</span>
        <span className="count-pill">{notes.length}</span>
        {onInspect ? (
          <button className="turn-action" type="button" onClick={onInspect}>
            Inspect
          </button>
        ) : null}
      </div>
      {noNewEvidence && notes.length > 0 ? (
        <div className="ambient-prior-context">No new evidence · showing prior context</div>
      ) : null}

      {notes.length === 0 ? (
        <div className="empty-surface ambient-empty">No knowledge-base evidence yet</div>
      ) : (
        <ol className="ambient-list">
          {notes.map((note) => {
            const passage = passageByNote.get(note.note_id)
            const fill = Math.round(Math.max(0, Math.min(1, note.score / maxScore)) * 100)
            const style = { '--score-fill': `${fill}%` } as CSSProperties
            return (
              <li className="ambient-item" key={note.note_ref}>
                <div className="ambient-title-row">
                  <h3
                    className={onOpenNote ? 'clickable' : undefined}
                    onClick={onOpenNote ? () => onOpenNote(note.note_id) : undefined}
                  >
                    <span className="citation-chip citation-note">{note.note_ref}</span>
                    {note.title}
                  </h3>
                  <span>{note.score.toFixed(2)}</span>
                </div>
                <div className="ambient-meta-row">
                  <span className={`source-chip source-${note.source}`}>
                    {sourceLabel(note.source)}
                  </span>
                  {passage ? (
                    <span className="citation-chip citation-passage">{passage.evidence_id}</span>
                  ) : null}
                </div>
                <p>{passage?.text ?? 'Source metadata only'}</p>
                <div className="score-track" style={style} aria-hidden="true">
                  <div className="score-fill" />
                </div>
              </li>
            )
          })}
        </ol>
      )}
    </aside>
  )
}
