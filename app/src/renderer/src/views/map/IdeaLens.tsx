import { useState } from 'react'

export type IdeaLensStatus = 'idle' | 'pending' | 'ready' | 'error'

interface IdeaLensProps {
  status: IdeaLensStatus
  /** the idea being traced / currently applied (pending | ready | error) */
  activeIdea: string
  /** human-readable failure message when status === 'error' */
  error: string
  onSubmit: (idea: string) => void
  onCancel: () => void
  onClear: () => void
}

/**
 * Compact toolbar affordance for the idea lens: an agentic concept search
 * over the map, distinct from the literal note search next to it.
 */
export function IdeaLens({
  status,
  activeIdea,
  error,
  onSubmit,
  onCancel,
  onClear
}: IdeaLensProps): React.JSX.Element {
  const [draft, setDraft] = useState('')

  if (status === 'pending') {
    return (
      <div className="map-idea" aria-label="Idea lens">
        <div className="map-idea-progress" role="status">
          <span className="map-idea-pulse" aria-hidden="true" />
          <span className="map-idea-progress-text">tracing idea… (~10s)</span>
          <button
            className="map-icon-btn"
            type="button"
            aria-label="Cancel idea trace"
            title="Cancel"
            onClick={onCancel}
          >
            ×
          </button>
        </div>
      </div>
    )
  }

  if (status === 'ready') {
    return (
      <div className="map-idea" aria-label="Idea lens">
        <button
          className="map-idea-active"
          type="button"
          title={`Idea lens active: ${activeIdea} — click to clear`}
          onClick={onClear}
        >
          <span className="map-idea-glyph" aria-hidden="true">
            ✦
          </span>
          <span className="map-idea-active-text">{activeIdea}</span>
          <span className="map-idea-active-x" aria-hidden="true">
            ×
          </span>
        </button>
      </div>
    )
  }

  return (
    <div className="map-idea" aria-label="Idea lens">
      <label className="map-idea-field" title="Idea lens: trace a concept across your notes">
        <span className="map-idea-glyph" aria-hidden="true">
          ✦
        </span>
        <span className="map-sr-only">Trace an idea across your notes</span>
        <input
          type="text"
          value={draft}
          placeholder="Trace an idea across your notes…"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              const idea = draft.trim()
              if (idea.length > 0) {
                event.preventDefault()
                onSubmit(idea)
              }
            } else if (event.key === 'Escape' && status === 'error') {
              onClear()
            }
          }}
        />
      </label>
      {status === 'error' && error ? (
        <div className="map-idea-error" role="alert">
          <span>{error}</span>
          <button
            className="map-icon-btn"
            type="button"
            aria-label="Dismiss error"
            onClick={onClear}
          >
            ×
          </button>
        </div>
      ) : null}
    </div>
  )
}
