import type { CSSProperties } from 'react'
import { ageLabel, sourceLabel } from '../lib/labels'
import type { AmbientItem } from '../lib/sidecar'

interface AmbientListProps {
  items: AmbientItem[]
  onOpenNote?: (noteId: string) => void
}

export function AmbientList({ items, onOpenNote }: AmbientListProps): React.JSX.Element {
  const maxScore = Math.max(1e-6, ...items.map((item) => item.score))

  return (
    <aside className="ambient-sidebar" aria-label="Grounding sources">
      <div className="sidebar-heading">
        <span className="section-label">Grounding</span>
        <span className="count-pill">{items.length}</span>
      </div>

      {items.length === 0 ? (
        <div className="empty-surface ambient-empty">No grounding sources yet</div>
      ) : (
        <ol className="ambient-list">
          {items.map((item) => {
            const fill = Math.round(Math.max(0, Math.min(1, item.score / maxScore)) * 100)
            const style = { '--score-fill': `${fill}%` } as CSSProperties

            return (
              <li className="ambient-item" key={item.note_id}>
                <div className="ambient-title-row">
                  <h3
                    className={onOpenNote ? 'clickable' : undefined}
                    onClick={onOpenNote ? () => onOpenNote(item.note_id) : undefined}
                  >
                    {item.short_id ? (
                      <span className="citation-chip citation-note">{item.short_id}</span>
                    ) : null}
                    {item.title}
                  </h3>
                  <span>{formatScore(item.score)}</span>
                </div>
                <div className="ambient-meta-row">
                  <span className={`source-chip source-${item.source}`}>
                    {sourceLabel(item.source)}
                  </span>
                  <span className="age-chip">{ageLabel(item.age_days)}</span>
                  {item.why?.exact_alias ? (
                    <span className="meta-chip exact" title={`Spoken: “${item.why.exact_alias}”`}>
                      exact
                    </span>
                  ) : null}
                  {item.why && item.why.ppr_mass > 0 ? (
                    <span className="meta-chip ppr" title="Reached via graph traversal">
                      graph
                    </span>
                  ) : null}
                  {item.why?.heads.includes('fact') ? (
                    <span className="meta-chip fact" title="Matched an edge fact">
                      fact
                    </span>
                  ) : null}
                </div>
                <p>{item.gist}</p>
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

function formatScore(score: number): string {
  if (score >= 10) {
    return score.toFixed(0)
  }

  return score.toFixed(2)
}
