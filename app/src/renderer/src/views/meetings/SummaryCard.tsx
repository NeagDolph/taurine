import ReactMarkdown from 'react-markdown'
import { ActionableRail } from '../../components/ActionableRail'
import type { ListeningState } from '../../components/SessionControls'
import { SummaryPanel } from '../../components/SummaryPanel'
import type { Actionable, SummaryEvent } from '../../lib/sidecar'

interface SummaryCardProps {
  summary: SummaryEvent | null
  actionables: Actionable[]
  finalSummaryMd: string
  listeningState: ListeningState
  onActionableStatus(id: string, status: 'confirmed' | 'dismissed'): void
  onAddToNotes(summaryMd: string): void
}

export function SummaryCard({
  summary,
  actionables,
  finalSummaryMd,
  listeningState,
  onActionableStatus,
  onAddToNotes
}: SummaryCardProps): React.JSX.Element | null {
  const markdown = summary?.summary_md || finalSummaryMd
  const visibleActionables = actionables.filter((item) => item.status !== 'dismissed')
  const active = listeningState !== 'idle' && listeningState !== 'error'
  const noteChangesDisabled = listeningState === 'stopping'
  if (!active && !markdown && visibleActionables.length === 0) return null

  if (summary) {
    return (
      <section className="grid max-h-[34vh] flex-none grid-cols-[minmax(0,3fr)_minmax(240px,2fr)] gap-3 overflow-y-auto border-b border-white/[0.08] bg-surface-2/30 p-3">
        <div className="min-w-0">
          <div className="mb-2 flex justify-end">
            <button
              className="secondary-button compact"
              type="button"
              disabled={noteChangesDisabled}
              onClick={() => onAddToNotes(markdown)}
            >
              Add to notes
            </button>
          </div>
          <SummaryPanel summary={summary} onOpenNote={ignoreNote} />
        </div>
        <ActionableRail
          actionables={actionables}
          evidence={summary.evidence ?? null}
          onStatus={onActionableStatus}
          onOpenNote={ignoreNote}
        />
      </section>
    )
  }

  return (
    <section className="grid max-h-[34vh] flex-none gap-3 overflow-y-auto border-b border-white/[0.08] bg-surface-2/50 px-5 py-3">
      <div className="flex items-center justify-between gap-3">
        <span className="section-label">Meeting summary</span>
        {markdown ? (
          <button
            className="secondary-button compact"
            type="button"
            disabled={noteChangesDisabled}
            onClick={() => onAddToNotes(markdown)}
          >
            Add to notes
          </button>
        ) : null}
      </div>
      {markdown ? (
        <div className="markdown-body text-[13px] leading-normal text-fg">
          <ReactMarkdown>{markdown}</ReactMarkdown>
        </div>
      ) : (
        <p className="m-0 text-xs text-fg/45">
          A live summary will appear as the meeting develops.
        </p>
      )}
      {visibleActionables.length > 0 ? (
        <div className="flex flex-wrap gap-2 border-t border-white/[0.08] pt-2.5">
          {visibleActionables.map((item) => (
            <article
              className="flex min-w-[220px] flex-1 items-start justify-between gap-2 rounded-lg border border-white/[0.08] bg-white/[0.03] p-2.5"
              key={item.id}
            >
              <div className="min-w-0">
                <h3 className="text-xs font-semibold text-fg">{item.title}</h3>
                <p className="mt-1 line-clamp-2 text-[11px] text-fg/55">{item.quote}</p>
              </div>
              <div className="flex flex-none gap-1">
                {item.status !== 'confirmed' ? (
                  <button
                    className="secondary-button compact"
                    type="button"
                    onClick={() => onActionableStatus(item.id, 'confirmed')}
                  >
                    Confirm
                  </button>
                ) : null}
                <button
                  className="icon-button"
                  type="button"
                  aria-label={`Dismiss ${item.title}`}
                  onClick={() => onActionableStatus(item.id, 'dismissed')}
                >
                  &times;
                </button>
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}

function ignoreNote(): void {
  // Meeting-local citation navigation can be added without expanding the coordinator contract.
}
