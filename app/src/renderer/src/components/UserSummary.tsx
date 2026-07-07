import { useCallback, useEffect, useState } from 'react'
import { saveProfile } from '../lib/sidecar'

interface UserSummaryProps {
  // undefined = updates channel hasn't delivered a snapshot yet;
  // null / '' = no card generated; string = the card
  card: string | null | undefined
}

export function UserSummary({ card }: UserSummaryProps): React.JSX.Element {
  const [collapsed, setCollapsed] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  // bridges the gap between a successful PUT and the pushed profile update
  const [savedOverride, setSavedOverride] = useState<string | null>(null)

  useEffect(() => {
    // once the updates channel reflects any card, the pushed value wins
    if (typeof card === 'string' && card.trim()) {
      setSavedOverride(null)
    }
  }, [card])

  const shownCard = savedOverride ?? (typeof card === 'string' ? card : '')
  const state: 'loading' | 'missing' | 'ready' =
    card === undefined && savedOverride === null
      ? 'loading'
      : shownCard.trim()
        ? 'ready'
        : 'missing'

  const beginEdit = useCallback(() => {
    setDraft(shownCard)
    setError('')
    setEditing(true)
    setCollapsed(false)
  }, [shownCard])

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
      setSavedOverride(draft.trim())
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
            <pre className="user-summary-text">{shownCard}</pre>
          )}
        </div>
      )}
    </section>
  )
}
