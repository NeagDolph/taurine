import { ConnectionsPanel } from '../components/ConnectionsPanel'
import type { ConnectorInfo, SyncStatus } from '../lib/sidecar'

interface ConnectorsViewProps {
  visible: boolean
  connectors: ConnectorInfo[] | null
  syncing: boolean
  syncStatus: SyncStatus | null
  syncError: string
  onSync: () => void
  onCancelSync: () => void
}

export function ConnectorsView({
  visible,
  connectors,
  syncing,
  syncStatus,
  syncError,
  onSync,
  onCancelSync
}: ConnectorsViewProps): React.JSX.Element {
  return (
    <div
      className="flex h-full min-h-0 flex-col gap-2.5 overflow-y-auto px-px pb-4 pt-px"
      aria-label="Connectors"
      aria-hidden={!visible}
    >
      <header>
        <h2 className="text-base font-bold tracking-tight text-fg">Connectors</h2>
        <p className="mt-1 max-w-[640px] text-xs leading-normal text-fg/65">
          Local sources for the index. Enabling a source only marks it for import — nothing is read
          or indexed until you press Sync, which imports every enabled source and rebuilds the
          knowledge index in one pass.
        </p>
      </header>
      <ConnectionsPanel
        connectors={connectors}
        syncing={syncing}
        syncStatus={syncStatus}
        syncError={syncError}
        onSync={onSync}
        onCancelSync={onCancelSync}
      />
    </div>
  )
}
