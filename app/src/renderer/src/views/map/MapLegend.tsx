import type { CSSProperties } from 'react'
import { RECENCY_LEGEND_GRADIENT, communityColor, sourceColor, sourceLabel } from '../../lib/graph'
import { formatTimestamp } from './render'
import type { ColorMode } from './storage'

interface MapLegendProps {
  colorMode: ColorMode
  sourceIds: string[]
  /** [communityId, noteCount] sorted by count desc */
  communities: Array<[number, number]>
  timeDomain: [number, number] | null
}

const MAX_LEGEND_COMMUNITIES = 6

export function MapLegend({
  colorMode,
  sourceIds,
  communities,
  timeDomain
}: MapLegendProps): React.JSX.Element {
  return (
    <div className="map-legend" aria-label="Map legend">
      <span className="map-legend-title">Legend</span>
      <span className="map-legend-row">
        <span className="map-legend-gradient" aria-hidden="true" />
        rank hot → cold
      </span>
      <span className="map-legend-row">
        <span className="map-legend-glow" aria-hidden="true" />
        graph traversal
      </span>
      <span className="map-legend-row">
        <span className="map-legend-ring" aria-hidden="true" />
        grounding note
      </span>
      <span className="map-legend-row">
        <span className="map-legend-heat" aria-hidden="true" />
        session trail
      </span>

      {colorMode === 'source'
        ? sourceIds.map((source) => (
            <span className="map-legend-row" key={`legend-${source}`}>
              <span
                className="map-legend-swatch"
                style={{ '--map-source-color': sourceColor(source) } as CSSProperties}
                aria-hidden="true"
              />
              {sourceLabel(source)}
            </span>
          ))
        : null}

      {colorMode === 'community' ? (
        <>
          {communities.slice(0, MAX_LEGEND_COMMUNITIES).map(([community, count]) => (
            <span className="map-legend-row" key={`legend-community-${community}`}>
              <span
                className="map-legend-swatch"
                style={{ '--map-source-color': communityColor(community) } as CSSProperties}
                aria-hidden="true"
              />
              {community < 0 ? 'unclustered' : `cluster ${community}`} · {count}
            </span>
          ))}
          {communities.length > MAX_LEGEND_COMMUNITIES ? (
            <span className="map-legend-row map-legend-more">
              +{communities.length - MAX_LEGEND_COMMUNITIES} more clusters
            </span>
          ) : null}
        </>
      ) : null}

      {colorMode === 'recency' ? (
        <span className="map-legend-row">
          <span
            className="map-legend-gradient"
            style={
              {
                background: `linear-gradient(90deg, ${RECENCY_LEGEND_GRADIENT})`
              } as CSSProperties
            }
            aria-hidden="true"
          />
          {timeDomain
            ? `${formatTimestamp(timeDomain[0])} → ${formatTimestamp(timeDomain[1])}`
            : 'old → recent'}
        </span>
      ) : null}
    </div>
  )
}
