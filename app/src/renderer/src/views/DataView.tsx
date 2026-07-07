import { useCallback, useState } from 'react'
import { ConnectionsPanel } from '../components/ConnectionsPanel'
import type { RetrievalEvent } from '../lib/sidecar'
import { MapView } from './MapView'

interface DataViewProps {
  visible: boolean
  lastSync?: number | null
  latestFinalText: string
  retrieval: RetrievalEvent | null
  pinnedNoteIds: string[]
  reenriching: boolean
  reenrichError: string
  reenrichDisabled: boolean
  onReenrich: () => void
}

export function DataView({
  visible,
  lastSync,
  latestFinalText,
  retrieval,
  pinnedNoteIds,
  reenriching,
  reenrichError,
  reenrichDisabled,
  onReenrich
}: DataViewProps): React.JSX.Element {
  const [refreshKey, setRefreshKey] = useState(0)
  const refreshMap = useCallback(() => setRefreshKey((current) => current + 1), [])

  return (
    <section className="data-view" aria-label="Data">
      <MapView
        latestFinalText={latestFinalText}
        lastSync={lastSync}
        pinnedNoteIds={pinnedNoteIds}
        refreshKey={refreshKey}
        retrieval={retrieval}
        visible={visible}
      />
      <ConnectionsPanel
        onDataChanged={refreshMap}
        onReenrich={onReenrich}
        reenrichDisabled={reenrichDisabled}
        reenrichError={reenrichError}
        reenriching={reenriching}
        visible={visible}
      />
    </section>
  )
}
