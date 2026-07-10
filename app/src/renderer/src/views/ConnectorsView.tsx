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
    <div className="page connectors-page" aria-label="Connectors" aria-hidden={!visible}>
      <header className="page-intro">
        <h2>Connectors</h2>
        <p>
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
