import type { CSSProperties } from 'react'
import type { LinkObject, NodeObject } from 'react-force-graph-2d'
import { NOW_NODE_ID, sourceColor, sourceLabel, type MapLink, type MapNode } from '../../lib/graph'
import { endpointNode } from './render'

type GraphLinkObject = LinkObject<MapNode, MapLink>
type GraphNodeObject = NodeObject<MapNode>

export function MapHoverPanel({
  hoverNode,
  hoverLink,
  nowLabel
}: {
  hoverNode: GraphNodeObject | null
  hoverLink: GraphLinkObject | null
  nowLabel: string
}): React.JSX.Element | null {
  if (hoverNode) {
    const isNow = String(hoverNode.id) === NOW_NODE_ID
    if (isNow) {
      return (
        <aside className="map-hover-panel" aria-label="Hovered element">
          <div className="map-hover-kicker">
            <span
              className="map-hover-dot"
              style={{ '--map-dot-color': '#f5c518' } as CSSProperties}
              aria-hidden="true"
            />
            Live transcript
          </div>
          <h3 className="map-hover-title">Now</h3>
          <p className="map-hover-gist">&ldquo;&hellip;{nowLabel}&rdquo;</p>
        </aside>
      )
    }

    return (
      <aside className="map-hover-panel" aria-label="Hovered note">
        <div className="map-hover-kicker">
          <span
            className="map-hover-dot"
            style={{ '--map-dot-color': sourceColor(hoverNode.source) } as CSSProperties}
            aria-hidden="true"
          />
          {sourceLabel(hoverNode.source)}
        </div>
        <h3 className="map-hover-title">{hoverNode.title}</h3>
        <p className={`map-hover-gist${hoverNode.gist ? '' : ' muted'}`}>
          {hoverNode.gist || 'No summary yet'}
        </p>
        <div className="map-hover-meta">
          <span>
            {hoverNode.degree} connection{hoverNode.degree === 1 ? '' : 's'}
          </span>
          <span>click to select</span>
        </div>
      </aside>
    )
  }

  if (hoverLink) {
    const source = endpointNode(hoverLink.source)
    const target = endpointNode(hoverLink.target)

    if (hoverLink.kind === 'rank') {
      return (
        <aside className="map-hover-panel" aria-label="Hovered retrieval edge">
          <div className="map-hover-kicker">
            <span className="map-hover-edge-swatch rank" aria-hidden="true" />
            Live retrieval · rank {hoverLink.rank}
          </div>
          {target ? <HoverEndpoint node={target} /> : null}
        </aside>
      )
    }

    if (hoverLink.kind !== 'static') {
      return null
    }

    return (
      <aside className="map-hover-panel" aria-label="Hovered connection">
        <div className="map-hover-kicker">
          <span className="map-hover-edge-swatch" aria-hidden="true" />
          Connection · {hoverLink.edgeType} · w {hoverLink.weight.toFixed(2)}
        </div>
        {hoverLink.fact ? (
          <p className="map-hover-fact">{hoverLink.fact}</p>
        ) : hoverLink.shared.length > 0 ? (
          <p className="map-hover-fact muted">Shared: {hoverLink.shared.join(', ')}</p>
        ) : null}
        <div className="map-hover-endpoints">
          {source ? <HoverEndpoint node={source} /> : null}
          {target ? <HoverEndpoint node={target} /> : null}
        </div>
        <div className="map-hover-meta">
          <span>right-click for supporting facts</span>
        </div>
      </aside>
    )
  }

  return null
}

function HoverEndpoint({ node }: { node: MapNode }): React.JSX.Element {
  return (
    <div className="map-hover-endpoint">
      <span
        className="map-hover-dot"
        style={{ '--map-dot-color': sourceColor(node.source) } as CSSProperties}
        aria-hidden="true"
      />
      <div className="map-hover-endpoint-body">
        <strong>{node.title}</strong>
        {node.gist ? <p>{node.gist}</p> : null}
      </div>
    </div>
  )
}
