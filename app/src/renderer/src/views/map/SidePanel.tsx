import { useEffect, useState, type CSSProperties } from 'react'
import ReactMarkdown from 'react-markdown'
import { communityColor, sourceColor, sourceLabel, type MapNode } from '../../lib/graph'
import { fetchNotePreview, type NotePreview } from '../../lib/sidecar'
import { formatTimestamp } from './render'

interface SidePanelProps {
  noteId: string
  /** graph node when present in the currently displayed graph (extra metadata) */
  node: MapNode | null
  onClose: () => void
  onFocus: (noteId: string) => void
  onAskAbout: (noteId: string, title: string) => void
  onOpenNote: (noteId: string) => void
  onSelectNeighbor: (noteId: string) => void
}

type PanelState =
  | { status: 'loading'; note: null; error: '' }
  | { status: 'ready'; note: NotePreview; error: '' }
  | { status: 'error'; note: null; error: string }

// NOTE: the parent renders <SidePanel key={noteId}>, so a selection change
// remounts this component — state can start at "loading" without effects.
export function SidePanel({
  noteId,
  node,
  onClose,
  onFocus,
  onAskAbout,
  onOpenNote,
  onSelectNeighbor
}: SidePanelProps): React.JSX.Element {
  const [state, setState] = useState<PanelState>({ status: 'loading', note: null, error: '' })

  useEffect(() => {
    const controller = new AbortController()

    fetchNotePreview(noteId, { signal: controller.signal })
      .then((note) => setState({ status: 'ready', note, error: '' }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return
        }
        setState({
          status: 'error',
          note: null,
          error: error instanceof Error ? readableNoteError(error.message) : 'Note fetch failed'
        })
      })

    return () => controller.abort()
  }, [noteId])

  const note = state.status === 'ready' ? state.note : null
  const title = note?.title ?? node?.title ?? '…'
  const source = note?.source ?? node?.source ?? 'apple_notes'
  const gist = note?.gist ?? node?.gist ?? ''
  const sourceStyle = { '--map-source-color': sourceColor(source) } as CSSProperties

  return (
    <aside className="map-side-panel" aria-label="Selected note">
      <header className="map-side-header">
        <div className="map-side-kicker">
          <span className="map-side-source" style={sourceStyle}>
            <span aria-hidden="true" />
            {sourceLabel(source)}
          </span>
          {node && node.community >= 0 ? (
            <span
              className="map-side-community"
              style={{ '--map-community-color': communityColor(node.community) } as CSSProperties}
              title={`Community ${node.community}`}
            >
              <span aria-hidden="true" />C{node.community}
            </span>
          ) : null}
          <button
            className="map-icon-btn map-side-close"
            type="button"
            aria-label="Close panel"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <h3 className="map-side-title">{title}</h3>
        {gist ? <p className="map-side-gist">{gist}</p> : null}
        <div className="map-side-meta">
          {node && node.created > 0 ? <span>Created {formatTimestamp(node.created)}</span> : null}
          {node && node.modified > 0 ? <span>Edited {formatTimestamp(node.modified)}</span> : null}
          {node ? (
            <span>
              {node.degree} connection{node.degree === 1 ? '' : 's'}
            </span>
          ) : null}
        </div>
        {node && node.entities.length > 0 ? (
          <div className="map-side-entities" aria-label="Entities">
            {node.entities.map((entity) => (
              <span className="map-entity-chip" key={entity}>
                {entity}
              </span>
            ))}
          </div>
        ) : null}
      </header>

      <div className="map-side-actions">
        <button className="map-btn map-btn-accent" type="button" onClick={() => onFocus(noteId)}>
          Focus
        </button>
        <button className="map-btn" type="button" onClick={() => onAskAbout(noteId, title)}>
          Ask about this
        </button>
        {source === 'apple_notes' ? (
          <button className="map-btn" type="button" onClick={() => onOpenNote(noteId)}>
            Open note
          </button>
        ) : null}
      </div>

      <div className="map-side-body">
        {state.status === 'loading' ? (
          <div className="map-side-state">Loading note…</div>
        ) : state.status === 'error' ? (
          <div className="map-side-state map-side-error" role="alert">
            {state.error}
          </div>
        ) : (
          <>
            {note?.text ? (
              <section className="map-side-text markdown-body" aria-label="Note text">
                <ReactMarkdown>{note.text}</ReactMarkdown>
              </section>
            ) : (
              <section className="map-side-text" aria-label="Note text">
                No text available
              </section>
            )}

            <section className="map-side-neighbors" aria-label="Connected notes">
              <div className="map-side-section-head">
                <span>Connected notes</span>
                <span className="map-count-pill">{note?.neighbors.length ?? 0}</span>
              </div>
              {!note || note.neighbors.length === 0 ? (
                <div className="map-side-state">No connected notes</div>
              ) : (
                <ol className="map-neighbor-list">
                  {note.neighbors.map((neighbor) => (
                    <li key={neighbor.note_id}>
                      <button
                        className="map-neighbor"
                        type="button"
                        style={
                          { '--map-source-color': sourceColor(neighbor.source) } as CSSProperties
                        }
                        onClick={() => onSelectNeighbor(neighbor.note_id)}
                      >
                        <span className="map-neighbor-dot" aria-hidden="true" />
                        <span className="map-neighbor-main">
                          <strong>{neighbor.title}</strong>
                          {neighbor.fact ? <span>{neighbor.fact}</span> : null}
                        </span>
                        <span className="map-neighbor-weight">{neighbor.weight.toFixed(2)}</span>
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </>
        )}
      </div>
    </aside>
  )
}

function readableNoteError(message: string): string {
  if (message.includes('404')) {
    return 'Note was not found'
  }
  return message
}
