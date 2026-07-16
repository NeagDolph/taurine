import { SummaryPanel } from '../../components/SummaryPanel'
import type { ListeningState } from '../../components/SessionControls'
import type { Actionable, FinalSummaryStatusEvent, SummaryEvent } from '../../lib/sidecar'

interface SummaryCardProps {
  summary: SummaryEvent | null
  actionables: Actionable[]
  finalSummaryStatus: FinalSummaryStatusEvent | null
  listeningState: ListeningState
  onActionableStatus(id: string, status: 'confirmed' | 'dismissed'): void
  onAddToNotes(summaryMd: string): void
  onOpenNote(noteId: string): void
}

/**
 * Slim rolling-summary card shown while a meeting is live. The final summary
 * never renders here — it arrives in the notepad as a pending suggestion.
 */
export function SummaryCard({
  summary,
  actionables,
  finalSummaryStatus,
  listeningState,
  onActionableStatus,
  onAddToNotes,
  onOpenNote
}: SummaryCardProps): React.JSX.Element | null {
  const live = listeningState === 'listening' || listeningState === 'starting'
  const finalRunning = finalSummaryStatus?.state === 'running'
  const finalError =
    finalSummaryStatus?.state === 'error'
      ? finalSummaryStatus.detail || 'Final summary could not be generated.'
      : ''

  if (finalRunning) {
    return (
      <div
        className="mb-4 flex items-center gap-2.5 rounded-xl border border-white/[0.08] bg-white/[0.03] px-3.5 py-2.5"
        aria-label="Summarizing meeting"
      >
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
        <span className="text-xs text-fg/60">Writing your meeting summary…</span>
      </div>
    )
  }
  if (finalError) {
    return (
      <div className="mb-4 rounded-xl border border-bad/25 bg-bad/[0.06] px-3.5 py-2.5 text-xs text-bad">
        {finalError}
      </div>
    )
  }
  if (!live || !summary?.summary_md) return null

  const visibleActionables = actionables.filter((item) => item.status !== 'dismissed')

  return (
    <section className="mb-4 max-h-[32vh] overflow-y-auto">
      <div className="relative">
        <SummaryPanel summary={summary} onOpenNote={onOpenNote} onShowInMap={onOpenNote} />
        <button
          className="secondary-button compact absolute right-3 top-2.5"
          type="button"
          onClick={() => onAddToNotes(summary.summary_md)}
        >
          Add to notes
        </button>
      </div>
      {visibleActionables.length > 0 ? (
        <div className="mt-2.5 flex flex-wrap gap-1.5 border-t border-white/[0.07] pt-2.5">
          {visibleActionables.map((item) => (
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] ${
                item.status === 'confirmed'
                  ? 'border-ok/40 bg-ok/[0.08] text-fg/85'
                  : 'border-white/[0.1] bg-white/[0.03] text-fg/70'
              }`}
              key={item.id}
              title={item.quote}
            >
              {item.title}
              {item.status !== 'confirmed' ? (
                <button
                  className="text-ok/80 hover:text-ok"
                  type="button"
                  aria-label={`Confirm ${item.title}`}
                  onClick={() => onActionableStatus(item.id, 'confirmed')}
                >
                  ✓
                </button>
              ) : null}
              <button
                className="text-fg/35 hover:text-fg/70"
                type="button"
                aria-label={`Dismiss ${item.title}`}
                onClick={() => onActionableStatus(item.id, 'dismissed')}
              >
                ✕
              </button>
            </span>
          ))}
        </div>
      ) : null}
    </section>
  )
}
