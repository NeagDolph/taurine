import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { fetchNotePreview, type NotePreview } from '../lib/sidecar'
import { sourceColor, sourceLabel } from '../lib/graph'

interface PreviewViewProps {
  noteId: string
}

type PreviewState =
  | { status: 'loading'; note: null; error: '' }
  | { status: 'ready'; note: NotePreview; error: '' }
  | { status: 'error'; note: null; error: string }

export function PreviewView({ noteId }: PreviewViewProps): React.JSX.Element {
  const [state, setState] = useState<PreviewState>({ status: 'loading', note: null, error: '' })

  useEffect(() => {
    document.body.classList.add('preview-body')
    return () => document.body.classList.remove('preview-body')
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setState({ status: 'loading', note: null, error: '' })

    fetchNotePreview(noteId, { signal: controller.signal })
      .then((note) => setState({ status: 'ready', note, error: '' }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return
        }
        setState({
          status: 'error',
          note: null,
          error: error instanceof Error ? readablePreviewError(error.message) : 'Preview failed'
        })
      })

    return () => controller.abort()
  }, [noteId])

  if (state.status === 'loading') {
    return (
      <main className="preview-view" aria-busy="true">
        <div className="preview-state">Loading preview</div>
      </main>
    )
  }

  if (state.status === 'error') {
    return (
      <main className="preview-view">
        <div className="preview-state preview-state-error" role="alert">
          {state.error}
        </div>
      </main>
    )
  }

  return <PreviewContent note={state.note} />
}

export function previewNoteIdFromHash(hash: string): string | null {
  const prefix = '#/preview/'
  if (!hash.startsWith(prefix)) {
    return null
  }

  const encoded = hash.slice(prefix.length)
  if (!encoded) {
    return null
  }

  try {
    return decodeURIComponent(encoded)
  } catch {
    return encoded
  }
}

function PreviewContent({ note }: { note: NotePreview }): React.JSX.Element {
  const sourceStyle = useMemo(
    () => ({ '--source-color': sourceColor(note.source) }) as CSSProperties,
    [note.source]
  )
  const maxWeight = Math.max(1, ...note.neighbors.map((neighbor) => neighbor.weight))

  return (
    <main className="preview-view">
      <header className="preview-header">
        <span className="preview-source-badge" style={sourceStyle}>
          <span aria-hidden="true" />
          {sourceLabel(note.source)}
        </span>
        <h1>{note.title}</h1>
        {note.gist ? <p>{note.gist}</p> : null}
      </header>

      <section className="preview-text" aria-label="Note text">
        {note.text || 'No text available'}
      </section>

      <section className="preview-neighbors" aria-label="Connected notes">
        <div className="preview-neighbors-header">
          <span className="section-label">Connected notes</span>
          <span className="count-pill">{note.neighbors.length}</span>
        </div>
        {note.neighbors.length === 0 ? (
          <div className="empty-surface">No connected notes</div>
        ) : (
          <ol className="preview-neighbor-list">
            {note.neighbors.map((neighbor) => {
              const fill = Math.round(Math.max(0, Math.min(1, neighbor.weight / maxWeight)) * 100)
              const style = {
                '--score-fill': `${fill}%`,
                '--source-color': sourceColor(neighbor.source)
              } as CSSProperties

              return (
                <li key={neighbor.note_id}>
                  <button
                    className="preview-neighbor"
                    type="button"
                    style={style}
                    onClick={() => openPreviewNeighbor(neighbor.note_id)}
                  >
                    <span className="preview-neighbor-source" aria-hidden="true" />
                    <span className="preview-neighbor-main">
                      <strong>{neighbor.title}</strong>
                      <span>{sourceLabel(neighbor.source)}</span>
                      {neighbor.fact ? (
                        <span className="preview-neighbor-fact">{neighbor.fact}</span>
                      ) : null}
                    </span>
                    <span className="preview-neighbor-weight">{formatWeight(neighbor.weight)}</span>
                    <span className="score-track" aria-hidden="true">
                      <span className="score-fill" />
                    </span>
                  </button>
                </li>
              )
            })}
          </ol>
        )}
      </section>
    </main>
  )
}

function openPreviewNeighbor(noteId: string): void {
  window.location.hash = `/preview/${encodeURIComponent(noteId)}`
}

function readablePreviewError(message: string): string {
  if (message.includes('HTTP 404')) {
    return 'Note preview was not found'
  }
  return message
}

function formatWeight(weight: number): string {
  if (weight >= 10) {
    return weight.toFixed(0)
  }

  return weight.toFixed(2)
}
