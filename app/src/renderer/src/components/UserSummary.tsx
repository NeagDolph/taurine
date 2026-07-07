import { useCallback, useEffect, useState } from 'react'
import { fetchProfile, saveProfile } from '../lib/sidecar'

interface UserSummaryProps {
  visible: boolean
}

type SummaryState = 'loading' | 'ready' | 'missing' | 'error'

export function UserSummary({ visible }: UserSummaryProps): React.JSX.Element {
  const [collapsed, setCollapsed] = useState(false)
  const [state, setState] = useState<SummaryState>('loading')
  const [card, setCard] = useState('')
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [loadedOnce, setLoadedOnce] = useState(false)

  const load = useCallback(async () => {
    setState((current) => (current === 'ready' ? current : 'loading'))
    try {
      const value = await fetchProfile()
      if (value && value.trim()) {
        setCard(value)
        setState('ready')
      } else {
        setState('missing')
      }
    } catch {
      setState('error')
    }
  }, [])

  useEffect(() => {
    if (visible && !loadedOnce) {
      setLoadedOnce(true)
      void load()
    }
  }, [visible, loadedOnce, load])

  const beginEdit = useCallback(() => {
    setDraft(card)
    setError('')
    setEditing(true)
    setCollapsed(false)
  }, [card])

  const cancelEdit = useCallback(() => {
    setEditing(false)
    setError('')
  }, [])

  const save = useCallback(async () => {
    if (!draft.trim() || saving) {
      return
    }
    setSaving(true)
    setError('')
    try {
      await saveProfile(draft.trim())
      setCard(draft.trim())
      setState('ready')
      setEditing(false)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }, [draft, saving])

  return (
    <section className="panel-card user-summary" aria-label="User summary">
      <header className="panel-card-header">
        <button
          type="button"
          className="collapse-toggle"
          aria-expanded={!collapsed}
          onClick={() => setCollapsed((current) => !current)}
        >
          <span className={`chevron ${collapsed ? '' : 'open'}`} aria-hidden="true" />
          <span className="section-label">User summary</span>
        </button>
        <div className="panel-card-actions">
          {state === 'ready' && !editing ? (
            <button type="button" className="secondary-button compact" onClick={beginEdit}>
              Edit
            </button>
          ) : null}
          {editing ? (
            <>
              <button
                type="button"
                className="secondary-button compact"
                onClick={cancelEdit}
                disabled={saving}
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary-button compact"
                onClick={save}
                disabled={saving || !draft.trim()}
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
            </>
          ) : null}
        </div>
      </header>

      {collapsed ? null : (
        <div className="user-summary-body">
          {editing ? (
            <>
              <textarea
                className="user-summary-editor"
                value={draft}
                rows={14}
                spellCheck={false}
                onChange={(event) => setDraft(event.target.value)}
              />
              <div className="user-summary-meta">
                <span>{draft.length.toLocaleString()} chars</span>
                {error ? <span className="inline-error">{error}</span> : null}
              </div>
            </>
          ) : state === 'loading' ? (
            <p className="muted-note">Loading…</p>
          ) : state === 'error' ? (
            <p className="inline-error">
              Could not load the user summary.{' '}
              <button type="button" className="link-button" onClick={load}>
                Retry
              </button>
            </p>
          ) : state === 'missing' ? (
            <p className="muted-note">
              No user summary yet — run a re-enrich from the Connectors tab to generate one from
              your notes, or{' '}
              <button type="button" className="link-button" onClick={beginEdit}>
                write it by hand
              </button>
              .
            </p>
          ) : (
            <pre className="user-summary-text">{card}</pre>
          )}
        </div>
      )}
    </section>
  )
}
