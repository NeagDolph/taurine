import { useCallback, useEffect, useRef, useState } from 'react'
import { UserSummary } from '../components/UserSummary'
import type { RetrievalEvent } from '../lib/sidecar'
import { MapView } from './MapView'

interface DataViewProps {
  visible: boolean
  lastSync?: number | null
  latestFinalText: string
  retrieval: RetrievalEvent | null
  pinnedNoteIds: string[]
  refreshKey: number
}

const MAP_HEIGHT_KEY = 'taurine.mapHeight'
const MIN_MAP_HEIGHT = 320
const MAX_MAP_HEIGHT = 1400

function initialMapHeight(): number {
  const stored = Number(window.localStorage.getItem(MAP_HEIGHT_KEY))
  if (Number.isFinite(stored) && stored >= MIN_MAP_HEIGHT && stored <= MAX_MAP_HEIGHT) {
    return stored
  }
  return Math.max(MIN_MAP_HEIGHT, Math.round(window.innerHeight * 0.62))
}

export function DataView({
  visible,
  lastSync,
  latestFinalText,
  retrieval,
  pinnedNoteIds,
  refreshKey
}: DataViewProps): React.JSX.Element {
  const [mapHeight, setMapHeight] = useState(initialMapHeight)
  const dragRef = useRef<{ startY: number; startHeight: number } | null>(null)

  const onDragStart = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault()
      dragRef.current = { startY: event.clientY, startHeight: mapHeight }
      event.currentTarget.setPointerCapture(event.pointerId)
    },
    [mapHeight]
  )

  const onDragMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag) {
      return
    }
    const next = Math.min(
      MAX_MAP_HEIGHT,
      Math.max(MIN_MAP_HEIGHT, drag.startHeight + (event.clientY - drag.startY))
    )
    setMapHeight(next)
  }, [])

  const onDragEnd = useCallback(() => {
    dragRef.current = null
  }, [])

  useEffect(() => {
    window.localStorage.setItem(MAP_HEIGHT_KEY, String(mapHeight))
  }, [mapHeight])

  return (
    <div className="page data-page" aria-label="Data">
      <section className="panel-card map-panel" style={{ height: mapHeight }}>
        <MapView
          latestFinalText={latestFinalText}
          lastSync={lastSync}
          pinnedNoteIds={pinnedNoteIds}
          refreshKey={refreshKey}
          retrieval={retrieval}
          visible={visible}
        />
      </section>

      <div
        className="resize-handle"
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize map"
        title="Drag to resize the map"
        onPointerDown={onDragStart}
        onPointerMove={onDragMove}
        onPointerUp={onDragEnd}
        onPointerCancel={onDragEnd}
      >
        <span className="resize-grip" aria-hidden="true" />
      </div>

      <UserSummary visible={visible} />
    </div>
  )
}
