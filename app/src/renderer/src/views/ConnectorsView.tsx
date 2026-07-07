import { ConnectionsPanel } from '../components/ConnectionsPanel'

interface ConnectorsViewProps {
  visible: boolean
  reenriching: boolean
  reenrichError: string
  reenrichDisabled: boolean
  onReenrich: () => void
  onDataChanged: () => void
}

export function ConnectorsView({
  visible,
  reenriching,
  reenrichError,
  reenrichDisabled,
  onReenrich,
  onDataChanged
}: ConnectorsViewProps): React.JSX.Element {
  return (
    <div className="page connectors-page" aria-label="Connectors">
      <header className="page-intro">
        <h2>Connectors</h2>
        <p>
          Local sources feeding the index. Enable a connector and sync it to pull its documents
          into the graph — everything downstream (enrichment, retrieval, the map) treats every
          source the same.
        </p>
      </header>
      <ConnectionsPanel
        onDataChanged={onDataChanged}
        onReenrich={onReenrich}
        reenrichDisabled={reenrichDisabled}
        reenrichError={reenrichError}
        reenriching={reenriching}
        visible={visible}
      />
    </div>
  )
}
