import type { CSSProperties } from 'react'
import { sourceColor, sourceLabel, type IdeaLensResult } from '../../lib/graph'

interface IdeaPanelProps {
  result: IdeaLensResult
  onSelect: (noteId: string) => void
  onClose: () => void
}

/**
 * Idea lens results: matched notes sorted by strength, rendered in the same
 * right-side panel slot as the note selection panel.
 */
export function IdeaPanel({ result, onSelect, onClose }: IdeaPanelProps): React.JSX.Element {
  return (
    <aside className="map-side-panel map-idea-panel" aria-label="Idea lens results">
      <header className="map-side-header">
        <div className="map-side-kicker">
          <span className="map-idea-kicker">
            <span aria-hidden="true">✦</span> Idea lens
          </span>
          <button
            className="map-icon-btn map-side-close"
            type="button"
            aria-label="Clear idea lens"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <h3 className="map-side-title">{result.idea}</h3>
        <div className="map-side-meta">
          <span>
            {result.matches.length} match{result.matches.length === 1 ? '' : 'es'}
          </span>
          {result.latency_ms > 0 ? <span>{(result.latency_ms / 1000).toFixed(1)}s</span> : null}
        </div>
      </header>

      <div className="map-side-body">
        {result.matches.length === 0 ? (
          <div className="map-side-state">No related notes found</div>
        ) : (
          <ol className="map-idea-list" aria-label="Matched notes">
            {result.matches.map((match) => (
              <li key={match.note_id}>
                <button
                  className="map-idea-match"
                  type="button"
                  onClick={() => onSelect(match.note_id)}
                >
                  <span className="map-idea-match-head">
                    <strong>{match.title}</strong>
                    <span className="map-idea-match-pct">{Math.round(match.strength * 100)}%</span>
                  </span>
                  <span className="map-idea-match-sub">
                    <span
                      className="map-idea-match-source"
                      style={{ '--map-source-color': sourceColor(match.source) } as CSSProperties}
                    >
                      <span aria-hidden="true" />
                      {sourceLabel(match.source)}
                    </span>
                    <span className="map-idea-strength" aria-hidden="true">
                      <span style={{ width: `${Math.round(match.strength * 100)}%` }} />
                    </span>
                  </span>
                  {match.reason ? <span className="map-idea-reason">{match.reason}</span> : null}
                </button>
              </li>
            ))}
          </ol>
        )}
      </div>

      <footer className="map-idea-foot">
        judged {result.considered} note{result.considered === 1 ? '' : 's'}, expanded{' '}
        {result.expanded}
      </footer>
    </aside>
  )
}
