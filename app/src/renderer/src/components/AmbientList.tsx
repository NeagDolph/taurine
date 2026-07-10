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
    <aside
      className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] rounded-[10px] border border-white/[0.08] bg-surface-1/60 p-3"
      aria-label="Retrieved evidence"
    >
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <span className="section-label">Evidence</span>
        <span className="count-pill">{notes.length}</span>
        {onInspect ? (
          <button className="turn-action" type="button" onClick={onInspect}>
            Inspect
          </button>
        ) : null}
      </div>
      {noNewEvidence && notes.length > 0 ? (
        <div className="px-0.5 pb-2 text-[11px] text-fg/45">
          No new evidence · showing prior context
        </div>
      ) : null}

      {notes.length === 0 ? (
        <div className="empty-surface self-start">No knowledge-base evidence yet</div>
      ) : (
        <ol className="m-0 flex min-h-0 list-none flex-col gap-1.5 overflow-y-auto p-0">
          {notes.map((note) => {
            const passage = passageByNote.get(note.note_id)
            const fill = Math.round(Math.max(0, Math.min(1, note.score / maxScore)) * 100)
            const style = { '--score-fill': `${fill}%` } as CSSProperties
            return (
              <li
                className="rounded-lg border border-white/[0.08] bg-white/[0.03] px-2.5 py-2"
                key={note.note_ref}
              >
                <div className="flex items-start justify-between gap-2.5">
                  <h3
                    className={`text-[12.5px] font-semibold leading-snug text-fg${
                      onOpenNote ? ' cursor-pointer hover:underline' : ''
                    }`}
                    onClick={onOpenNote ? () => onOpenNote(note.note_id) : undefined}
                  >
                    <span className="citation-chip citation-note">{note.note_ref}</span>
                    {note.title}
                  </h3>
                  <span className="text-[11px] font-semibold text-fg/45">
                    {note.score.toFixed(2)}
                  </span>
                </div>
                <div className="my-1 flex flex-wrap items-center gap-1">
                  <span className={`source-chip source-${note.source}`}>
                    {sourceLabel(note.source)}
                  </span>
                  {passage ? (
                    <span className="citation-chip citation-passage">{passage.evidence_id}</span>
                  ) : null}
                </div>
                <p className="mt-1 text-xs leading-snug text-fg/65">
                  {passage?.text ?? 'Source metadata only'}
                </p>
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
