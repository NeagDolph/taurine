import type { MeetingListRow } from '../../lib/meetings'

interface MeetingSidebarProps {
  meetings: MeetingListRow[]
  selectedMeetingId: string | null
  liveMeetingId: string | null
  onSelect(id: string): void
  onCreate(): void
  onDelete(id: string): void
}

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  month: 'short',
  day: 'numeric',
  year: 'numeric'
})

export function MeetingSidebar({
  meetings,
  selectedMeetingId,
  liveMeetingId,
  onSelect,
  onCreate,
  onDelete
}: MeetingSidebarProps): React.JSX.Element {
  return (
    <aside className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] border-r border-white/[0.08] bg-surface-1/70">
      <div className="border-b border-white/[0.08] p-3">
        <button className="primary-button w-full" type="button" onClick={onCreate}>
          + New note
        </button>
      </div>

      {meetings.length === 0 ? (
        <div className="m-3 self-start rounded-lg border border-dashed border-white/10 p-3 text-center text-xs text-fg/45">
          Meeting notes will appear here.
        </div>
      ) : (
        <ol className="m-0 min-h-0 list-none overflow-y-auto p-2">
          {meetings.map((meeting) => {
            const selected = meeting.meetingId === selectedMeetingId
            const live = meeting.meetingId === liveMeetingId || meeting.status === 'live'
            return (
              <li className="group relative mb-1" key={meeting.meetingId}>
                <button
                  className={`grid w-full grid-cols-[auto_minmax(0,1fr)] items-start gap-2 rounded-lg border py-2 pl-2.5 pr-9 text-left transition-colors ${
                    selected
                      ? 'border-accent/35 bg-accent/10'
                      : 'border-transparent bg-transparent hover:bg-white/[0.05]'
                  }`}
                  type="button"
                  onClick={() => onSelect(meeting.meetingId)}
                >
                  <span
                    className={`mt-1.5 h-2 w-2 rounded-full ${statusDotClass(meeting.status, live)}`}
                    title={live ? 'live' : meeting.status}
                  />
                  <span className="min-w-0">
                    <span className="block truncate text-[12.5px] font-semibold text-fg">
                      {meeting.title.trim() || 'Untitled meeting'}
                    </span>
                    <span className="mt-0.5 block text-[10.5px] text-fg/40">
                      {DATE_FORMAT.format(new Date(meeting.createdAt * 1000))}
                    </span>
                  </span>
                </button>
                <button
                  className="icon-button absolute right-1.5 top-2 opacity-0 group-focus-within:opacity-100 group-hover:opacity-100"
                  type="button"
                  aria-label={`Delete ${meeting.title.trim() || 'untitled meeting'}`}
                  onClick={() => onDelete(meeting.meetingId)}
                >
                  &times;
                </button>
              </li>
            )
          })}
        </ol>
      )}
    </aside>
  )
}

function statusDotClass(status: MeetingListRow['status'], live: boolean): string {
  if (live) return 'bg-ok shadow-[0_0_0_3px_rgba(87,217,138,0.12)]'
  if (status === 'summarizing') return 'bg-accent animate-pulse'
  if (status === 'done') return 'bg-fg/35'
  return 'border border-fg/25 bg-transparent'
}
