import type { CSSProperties } from 'react'
import type { AmbientItem } from '../lib/sidecar'

interface AmbientListProps {
  items: AmbientItem[]
}

export function AmbientList({ items }: AmbientListProps): React.JSX.Element {
  const maxScore = Math.max(1, ...items.map((item) => item.score))

  return (
    <aside className="ambient-sidebar" aria-label="Ambient notes">
      <div className="sidebar-heading">
        <span className="section-label">Ambient</span>
        <span className="count-pill">{items.length}</span>
      </div>

      {items.length === 0 ? (
        <div className="empty-surface ambient-empty">No matches yet</div>
      ) : (
        <ol className="ambient-list">
          {items.map((item) => {
            const fill = Math.round(Math.max(0, Math.min(1, item.score / maxScore)) * 100)
            const style = { '--score-fill': `${fill}%` } as CSSProperties

            return (
              <li className="ambient-item" key={item.note_id}>
                <div className="ambient-title-row">
                  <h3>{item.title}</h3>
                  <span>{formatScore(item.score)}</span>
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
