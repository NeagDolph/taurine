import { useEffect, useMemo } from 'react'
import type { TriggerAction, TriggerEvent, TriggerMatch } from '../lib/sidecar'

interface TriggerInspectorProps {
  trigger: TriggerEvent | null
  onClose: () => void
  onOpenNote: (noteId: string) => void
}

const actionLabels: Record<TriggerAction, string> = {
  grounded: 'Grounded',
  none: 'No action'
}

const exactActionContext: Record<TriggerAction, string> = {
  grounded: 'entered the grounding set',
  none: 'no action'
}

export function TriggerInspector({
  trigger,
  onClose,
  onOpenNote
}: TriggerInspectorProps): React.JSX.Element | null {
  useEffect(() => {
    if (!trigger) {
      return undefined
    }

    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose, trigger])

  const matches = useMemo(() => (trigger ? orderedMatches(trigger) : []), [trigger])
  const titleByNoteId = useMemo(() => {
    const titles = new Map<string, string>()
    trigger?.matches.forEach((match) => titles.set(match.note_id, match.title))
    return titles
  }, [trigger])

  if (!trigger) {
    return null
  }

  const isSearch = trigger.kind === 'search'
  const columnCount = isSearch ? 4 : 3

  return (
    <aside className="trigger-inspector" role="dialog" aria-label="Trigger inspector">
      <div className="trigger-inspector-header">
        <div>
          <span className="section-label">Trigger</span>
          <h2>{trigger.kind === 'exact' ? 'Exact match' : 'Search window'}</h2>
        </div>
        <button
          className="icon-button"
          type="button"
          aria-label="Close trigger inspector"
          onClick={onClose}
        >
          &times;
        </button>
      </div>

      <div className="trigger-inspector-badges">
        <span className={`trigger-kind-badge trigger-kind-${trigger.kind}`}>{trigger.kind}</span>
        <span className={`trigger-action-badge trigger-action-${trigger.action}`}>
          {actionLabels[trigger.action]}
        </span>
      </div>

      <div className="trigger-heard">
        <span>heard</span>
        <p>{trigger.span}</p>
      </div>

      <div className="trigger-match-table-wrap">
        <table className="trigger-match-table">
          <thead>
            <tr>
              <th scope="col">Title</th>
              {isSearch ? <th scope="col">Rank</th> : null}
              <th scope="col">Score</th>
              <th scope="col">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {matches.length === 0 ? (
              <tr>
                <td className="trigger-empty-cell" colSpan={columnCount}>
                  No matches
                </td>
              </tr>
            ) : (
              matches.map((match) => {
                const viaTitle = match.via_note_id
                  ? (titleByNoteId.get(match.via_note_id) ?? match.via_note_id)
                  : ''

                return (
                  <tr
                    className={
                      isExpandedSearchMatch(trigger, match) ? 'trigger-match-expanded' : ''
                    }
                    key={matchKey(match)}
                  >
                    <td>
                      <div className="trigger-match-title">{match.title}</div>
                      {match.via_note_id ? (
                        <div className="trigger-match-via">
                          via {viaTitle} (hop {match.hop ?? 1})
                        </div>
                      ) : null}
                      {trigger.kind === 'exact' ? (
                        <div className="trigger-match-context">
                          {exactActionContext[trigger.action]}
                        </div>
                      ) : null}
                    </td>
                    {isSearch ? (
                      <td>{typeof match.rank === 'number' ? `#${match.rank}` : 'linked'}</td>
                    ) : null}
                    <td>{formatScore(match.score)}</td>
                    <td>
                      <button
                        className="secondary-button compact"
                        type="button"
                        onClick={() => onOpenNote(match.note_id)}
                      >
                        Open in Notes
                      </button>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      <footer className="trigger-inspector-footer">
        <code>{trigger.trigger_id}</code>
        <code>seqs {trigger.seqs.join(', ')}</code>
      </footer>
    </aside>
  )
}

function orderedMatches(trigger: TriggerEvent): TriggerMatch[] {
  if (trigger.kind !== 'search') {
    return trigger.matches
  }

  return [...trigger.matches].sort((left, right) => {
    const leftHasRank = typeof left.rank === 'number'
    const rightHasRank = typeof right.rank === 'number'

    if (leftHasRank && rightHasRank) {
      return (left.rank ?? 0) - (right.rank ?? 0)
    }

    if (leftHasRank !== rightHasRank) {
      return leftHasRank ? -1 : 1
    }

    return right.score - left.score || left.title.localeCompare(right.title)
  })
}

function isExpandedSearchMatch(trigger: TriggerEvent, match: TriggerMatch): boolean {
  return trigger.kind === 'search' && typeof match.rank !== 'number'
}

function matchKey(match: TriggerMatch): string {
  return [
    match.note_id,
    match.rank ?? 'linked',
    match.via_note_id ?? 'direct',
    match.hop ?? 0
  ].join('-')
}

function formatScore(score: number): string {
  if (score >= 10) {
    return score.toFixed(0)
  }

  return score.toFixed(2)
}
