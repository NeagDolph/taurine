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
  const detailText = syncError || detail

  return (
    <footer className="status-bar">
      <div className="status-group">
        <StatusPill label="Sidecar" state={connection} tone={connection === 'connected' ? 'good' : 'warn'} />
        <StatusPill label="ASR" state={asr} tone={asr === 'connected' ? 'good' : asr === 'error' ? 'bad' : 'warn'} />
        <span className="status-metric">{index.notes} notes</span>
        <span className="status-metric">Last sync {formatLastSync(index.last_sync)}</span>
      </div>
      <div className="status-group status-group-right">
        {detailText ? <span className="status-detail">{detailText}</span> : null}
        <button className="secondary-button compact" type="button" onClick={onSync} disabled={isSyncing}>
          {isSyncing ? 'Syncing' : 'Sync now'}
        </button>
      </div>
    </footer>
  )
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
    <span className={`status-pill status-${tone}`}>
      <span>{label}</span>
      <strong>{state}</strong>
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
