import type { IndexStatus, AsrState, SidecarConnectionState } from '../lib/sidecar'

interface StatusBarProps {
  asr: AsrState
  connection: SidecarConnectionState
  detail: string
  index: IndexStatus
  isSyncing: boolean
  syncError: string
  onSync: () => void
}

export function StatusBar({
  asr,
  connection,
  detail,
  index,
  isSyncing,
  syncError,
  onSync
}: StatusBarProps): React.JSX.Element {
  const failure = index.sync?.error ? `Last sync failed: ${index.sync.error}` : ''
  const stale = !index.sync?.running && index.sync?.stale ? 'Graph out of date — press Sync' : ''
  const detailText = syncError || failure || syncDetail(index) || stale || detail
  const controlsDisabled = isSyncing

  return (
    <footer className="flex min-h-9 flex-none items-center justify-between gap-3 rounded-[10px] border border-white/[0.08] bg-surface-1/70 px-3 py-1.5 backdrop-blur-md">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <StatusPill
          label="Sidecar"
          state={connection}
          tone={connection === 'connected' ? 'good' : 'warn'}
        />
        <StatusPill
          label="ASR"
          state={asr}
          tone={asr === 'connected' ? 'good' : asr === 'error' ? 'bad' : 'warn'}
        />
        <span className={METRIC_CLASSES}>{index.notes} notes</span>
        <span className={METRIC_CLASSES}>Last sync {formatLastSync(index.last_sync)}</span>
      </div>
      <div className="flex flex-none flex-wrap items-center justify-end gap-2">
        {detailText ? (
          <span className="max-w-[340px] truncate text-[11px] text-fg/45">{detailText}</span>
        ) : null}
        <button
          className="secondary-button compact"
          type="button"
          onClick={onSync}
          disabled={controlsDisabled}
        >
          {isSyncing ? 'Syncing' : 'Sync now'}
        </button>
      </div>
    </footer>
  )
}

const METRIC_CLASSES =
  'inline-flex min-h-[22px] items-center gap-1.5 whitespace-nowrap rounded-full border border-white/[0.08] bg-white/[0.03] px-2.5 text-[11px] text-fg/65'

const TONE_BORDER: Record<'good' | 'warn' | 'bad', string> = {
  good: 'border-ok/40',
  warn: 'border-accent/50',
  bad: 'border-bad/40'
}

const TONE_TEXT: Record<'good' | 'warn' | 'bad', string> = {
  good: 'text-ok',
  warn: 'text-accent-hi',
  bad: 'text-bad'
}

function StatusPill({
  label,
  state,
  tone
}: {
  label: string
  state: string
  tone: 'good' | 'warn' | 'bad'
}): React.JSX.Element {
  return (
    <span
      className={`inline-flex min-h-[22px] items-center gap-1.5 whitespace-nowrap rounded-full border bg-white/[0.03] px-2.5 text-[11px] text-fg/65 ${TONE_BORDER[tone]}`}
    >
      <span>{label}</span>
      <strong className={`text-[11px] font-bold ${TONE_TEXT[tone]}`}>{state}</strong>
    </span>
  )
}

function formatLastSync(lastSync: number | null | undefined): string {
  if (!lastSync) {
    return 'never'
  }

  const millis = lastSync > 1_000_000_000_000 ? lastSync : lastSync * 1000
  return new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  }).format(new Date(millis))
}

function syncDetail(index: IndexStatus): string {
  const status = index.sync
  if (!status?.running) {
    return ''
  }

  const phase = status.phase === 'indexing' ? 'Indexing' : 'Syncing sources'
  return status.detail ? `${phase} — ${status.detail}` : phase
}
