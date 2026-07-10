import type { CSSProperties } from 'react'
import {
  RECENCY_LEGEND_GRADIENT,
  communityColor,
  sourceColor,
  sourceLabel,
  type GraphCluster
} from '../../lib/graph'
import { formatTimestamp } from './render'
import type { ColorMode } from './storage'

interface MapLegendProps {
  colorMode: ColorMode
  sourceIds: string[]
  /** [communityId, noteCount] sorted by count desc */
  communities: Array<[number, number]>
  /** cluster id → LLM-generated name/description/size */
  clusterById: Map<number, GraphCluster>
  /** cluster id currently highlighted via the legend, if any */
  highlightedCluster: number | null
  onToggleCluster: (community: number) => void
  timeDomain: [number, number] | null
}

const MAX_LEGEND_COMMUNITIES = 12

export function MapLegend({
  colorMode,
  sourceIds,
  communities,
  clusterById,
  highlightedCluster,
  onToggleCluster,
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
          {communities.slice(0, MAX_LEGEND_COMMUNITIES).map(([community, count]) => {
            const cluster = community >= 0 ? clusterById.get(community) : undefined
            const name = community < 0 ? 'Unclustered' : cluster?.name || `Cluster ${community + 1}`
            const tooltip = cluster?.description
              ? `${cluster.description} · ${cluster.size || count} notes`
              : `${count} note${count === 1 ? '' : 's'}`
            const active = highlightedCluster === community

            return (
              <button
                className={`map-legend-cluster${active ? ' active' : ''}`}
                key={`legend-community-${community}`}
                type="button"
                aria-pressed={active}
                title={tooltip}
                onClick={() => onToggleCluster(community)}
              >
                <span
                  className="map-legend-swatch"
                  style={{ '--map-source-color': communityColor(community) } as CSSProperties}
                  aria-hidden="true"
                />
                <span className="map-legend-cluster-name">{name}</span>
                <span className="map-legend-cluster-count">{count}</span>
              </button>
            )
          })}
          {communities.length > MAX_LEGEND_COMMUNITIES ? (
            <span className="map-legend-row map-legend-more">
              …{communities.length - MAX_LEGEND_COMMUNITIES} more
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
