import { ConnectionsPanel } from '../components/ConnectionsPanel'
import type { ConnectorInfo } from '../lib/sidecar'

interface ConnectorsViewProps {
  visible: boolean
  connectors: ConnectorInfo[] | null
  reenriching: boolean
  reenrichError: string
  reenrichDisabled: boolean
  reenrichTarget: string | null
  onReenrich: (source?: string) => void
}

export function ConnectorsView({
  visible,
  connectors,
  reenriching,
  reenrichError,
  reenrichDisabled,
  reenrichTarget,
  onReenrich
}: ConnectorsViewProps): React.JSX.Element {
  return (
    <div className="page connectors-page" aria-label="Connectors">
      <header className="page-intro">
        <h2>Connectors</h2>
        <p>
          Local sources feeding the index. Enable a connector and sync it to pull its documents
          into the graph — everything downstream (enrichment, retrieval, the map) treats every
          source the same. Disabling keeps the data stored but drops it from the graph and
          retrieval until re-enabled.
        </p>
      </header>
      <ConnectionsPanel
        connectors={connectors}
        onReenrich={onReenrich}
        reenrichDisabled={reenrichDisabled}
        reenrichError={reenrichError}
        reenriching={reenriching}
        reenrichTarget={reenrichTarget}
        visible={visible}
      />
    </div>
  )
}
