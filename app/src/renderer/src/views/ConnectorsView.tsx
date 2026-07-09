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
    <div className="page connectors-page" aria-label="Connectors" aria-hidden={!visible}>
      <header className="page-intro">
        <h2>Connectors</h2>
        <p>
          Local sources feeding the index. Every enabled source flows through the same pipeline —
          enrichment, retrieval, and the map treat all documents alike. Disabling keeps the data
          stored but drops it from the graph until re-enabled.
        </p>
      </header>
      <ConnectionsPanel
        connectors={connectors}
        onReenrich={onReenrich}
        reenrichDisabled={reenrichDisabled}
        reenrichError={reenrichError}
        reenriching={reenriching}
        reenrichTarget={reenrichTarget}
      />
    </div>
  )
}
