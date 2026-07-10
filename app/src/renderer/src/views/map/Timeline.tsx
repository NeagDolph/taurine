import { useCallback, useRef, type CSSProperties } from 'react'
import { formatTimestamp } from './render'

export type TimeRange = [number, number]

interface TimelineProps {
  /** full [min, max] extent of note timestamps (unix seconds), min < max */
  domain: TimeRange
  /** note counts per bucket across the domain (sparkline histogram) */
  histogram: number[]
  range: TimeRange | null
  onRangeChange: (range: TimeRange | null) => void
}

type DragKind = 'create' | 'left' | 'right' | 'move'

interface DragState {
  kind: DragKind
  anchor: number
  startRange: TimeRange | null
}

const DAY_S = 24 * 60 * 60

export function MapTimeline({
  domain,
  histogram,
  range,
  onRangeChange
}: TimelineProps): React.JSX.Element {
  const trackRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<DragState | null>(null)

  const [minTs, maxTs] = domain
  const span = Math.max(1, maxTs - minTs)
  const maxCount = Math.max(1, ...histogram)

  const timeAt = useCallback(
    (clientX: number): number => {
      const track = trackRef.current
      if (!track) {
        return minTs
      }
      const rect = track.getBoundingClientRect()
      const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)))
      return minTs + fraction * span
    },
    [minTs, span]
  )

  const beginDrag = useCallback(
    (event: React.PointerEvent<HTMLElement>, kind: DragKind) => {
      event.preventDefault()
      event.stopPropagation()
      dragRef.current = { kind, anchor: timeAt(event.clientX), startRange: range }
      trackRef.current?.setPointerCapture(event.pointerId)
    },
    [range, timeAt]
  )

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current
      if (!drag) {
        return
      }

      const t = timeAt(event.clientX)
      if (drag.kind === 'create') {
        const lo = Math.min(drag.anchor, t)
        const hi = Math.max(drag.anchor, t)
        if (hi - lo > span * 0.004) {
          onRangeChange([lo, hi])
        }
      } else if (drag.kind === 'left' && drag.startRange) {
        onRangeChange([Math.min(t, drag.startRange[1] - span * 0.004), drag.startRange[1]])
      } else if (drag.kind === 'right' && drag.startRange) {
        onRangeChange([drag.startRange[0], Math.max(t, drag.startRange[0] + span * 0.004)])
      } else if (drag.kind === 'move' && drag.startRange) {
        const width = drag.startRange[1] - drag.startRange[0]
        let lo = drag.startRange[0] + (t - drag.anchor)
        lo = Math.max(minTs, Math.min(maxTs - width, lo))
        onRangeChange([lo, lo + width])
      }
    },
    [maxTs, minTs, onRangeChange, span, timeAt]
  )

  const onPointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current
      dragRef.current = null
      if (!drag) {
        return
      }

      // a click (no meaningful create-drag) clears the brush back to All
      if (drag.kind === 'create') {
        const t = timeAt(event.clientX)
        if (Math.abs(t - drag.anchor) <= span * 0.004) {
          onRangeChange(null)
        }
      }
    },
    [onRangeChange, span, timeAt]
  )

  const presetRange = useCallback(
    (preset: 'today' | 'week' | 'month'): TimeRange => {
      const now = Date.now() / 1000
      const start =
        preset === 'today'
          ? new Date(new Date().setHours(0, 0, 0, 0)).getTime() / 1000
          : preset === 'week'
            ? now - 7 * DAY_S
            : now - 30 * DAY_S
      // hi extends past the domain so notes modified "now" always pass
      return [start, Math.max(maxTs, now)]
    },
    [maxTs]
  )

  const isPresetActive = (preset: 'today' | 'week' | 'month'): boolean => {
    if (!range) {
      return false
    }
    const [lo, hi] = presetRange(preset)
    const epsilon = span * 0.002 + 60
    return Math.abs(range[0] - lo) < epsilon && Math.abs(range[1] - hi) < epsilon
  }

  const leftPct = range ? Math.max(0, Math.min(100, ((range[0] - minTs) / span) * 100)) : 0
  const rightPct = range ? Math.max(0, Math.min(100, ((range[1] - minTs) / span) * 100)) : 100

  return (
    <div className="map-timeline" aria-label="Timeline filter">
      <div className="map-timeline-presets" role="group" aria-label="Time presets">
        <button
          className={`map-chip${range === null ? ' active' : ''}`}
          type="button"
          onClick={() => onRangeChange(null)}
        >
          All
        </button>
        {(['today', 'week', 'month'] as const).map((preset) => (
          <button
            key={preset}
            className={`map-chip${isPresetActive(preset) ? ' active' : ''}`}
            type="button"
            onClick={() => onRangeChange(presetRange(preset))}
          >
            {preset === 'today' ? 'Today' : preset === 'week' ? 'Week' : 'Month'}
          </button>
        ))}
        <span className="map-timeline-label">
          {range
            ? `${formatTimestamp(range[0])} – ${formatTimestamp(range[1])}`
            : `${formatTimestamp(minTs)} – ${formatTimestamp(maxTs)}`}
        </span>
      </div>

      <div
        className="map-timeline-track"
        ref={trackRef}
        onPointerDown={(event) => beginDrag(event, 'create')}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          dragRef.current = null
        }}
      >
        <div className="map-timeline-bars" aria-hidden="true">
          {histogram.map((count, index) => (
            <span
              key={index}
              style={{ '--map-bar-height': `${(count / maxCount) * 100}%` } as CSSProperties}
            />
          ))}
        </div>

        {range ? (
          <div
            className="map-timeline-window"
            style={{ left: `${leftPct}%`, width: `${Math.max(0.5, rightPct - leftPct)}%` }}
            onPointerDown={(event) => beginDrag(event, 'move')}
          >
            <span
              className="map-timeline-handle left"
              onPointerDown={(event) => beginDrag(event, 'left')}
            />
            <span
              className="map-timeline-handle right"
              onPointerDown={(event) => beginDrag(event, 'right')}
            />
          </div>
        ) : null}
      </div>
    </div>
  )
}
