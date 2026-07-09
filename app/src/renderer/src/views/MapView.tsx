import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import ForceGraph2D, {
  type ForceGraphMethods,
  type LinkObject,
  type NodeObject
} from 'react-force-graph-2d'
import {
  LIVE_EDGE_FADE_MS,
  NOW_NODE_ID,
  TOUCH_FLASH_MS,
  fadingLiveLinks,
  fetchGraph,
  graphResponseToMapData,
  liveGraphFromRetrieval,
  liveLinkOpacity,
  mergeGraphPositions,
  nodeRadius,
  nowLabelFromText,
  sourceColor,
  sourceLabel,
  sourcesFromMapData,
  staticLinkOpacity,
  type LiveGraphLink,
  type MapGraphData,
  type MapLink,
  type MapNode
} from '../lib/graph'
import type { RetrievalEvent } from '../lib/sidecar'

interface MapViewProps {
  visible: boolean
  refreshKey?: number
  latestFinalText: string
  retrieval: RetrievalEvent | null
  groundedNoteIds: string[]
}

type GraphLinkObject = LinkObject<MapNode, MapLink>
type GraphNodeObject = NodeObject<MapNode>

interface GlowEntry {
  until: number
  intensity: number
}

const EMPTY_GRAPH: MapGraphData = graphResponseToMapData({ nodes: [], edges: [] }, 'Now')
const GRAPH_REFETCH_DEBOUNCE_MS = 800
// zoom (globalScale) windows over which node labels fade in: hubs surface
// first so a zoomed-out map reads as landmarks instead of a text mass
const HUB_LABEL_FADE: readonly [number, number] = [0.8, 1.4]
const NODE_LABEL_FADE: readonly [number, number] = [1.6, 2.7]
// ~760 nodes need a few hundred ticks to untangle from d3's initial spiral;
// the old 80 froze half-laid-out graphs whenever a refetch reset positions
const LAYOUT_COOLDOWN_TICKS = 300

export function MapView({
  visible,
  refreshKey = 0,
  latestFinalText,
  retrieval,
  groundedNoteIds
}: MapViewProps): React.JSX.Element {
  const graphRef = useRef<ForceGraphMethods<MapNode, MapLink> | undefined>(undefined)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const isInitialGraphLoadRef = useRef(true)
  const needsReheatRef = useRef(false)
  const processedRetrievalRef = useRef<RetrievalEvent | null>(null)
  const pruneTimerRef = useRef<number | null>(null)
  const flashTimerRef = useRef<number | null>(null)
  const glowTimerRef = useRef<number | null>(null)

  const [baseGraph, setBaseGraph] = useState<MapGraphData>(EMPTY_GRAPH)
  const [liveLinks, setLiveLinks] = useState<LiveGraphLink[]>([])
  const [flashUntilByNode, setFlashUntilByNode] = useState<Map<string, number>>(new Map())
  const [glowByNode, setGlowByNode] = useState<Map<string, GlowEntry>>(new Map())
  const [hoverNode, setHoverNode] = useState<GraphNodeObject | null>(null)
  const [hoverLink, setHoverLink] = useState<GraphLinkObject | null>(null)
  const [size, setSize] = useState({ width: 720, height: 480 })
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [retryNonce, setRetryNonce] = useState(0)
  const [query, setQuery] = useState('')
  const [selectedSources, setSelectedSources] = useState<Set<string>>(() => new Set())

  const nowLabel = useMemo(() => nowLabelFromText(latestFinalText), [latestFinalText])
  const pinnedSet = useMemo(() => new Set(groundedNoteIds), [groundedNoteIds])
  const sourceIds = useMemo(() => sourcesFromMapData(baseGraph), [baseGraph])
  const hubDegree = useMemo(() => degreePercentile(baseGraph.nodes, 0.75), [baseGraph.nodes])
  const hoverNodeId = hoverNode ? String(hoverNode.id) : null
  const hoverLinkId = hoverLink ? hoverLink.id : null

  const schedulePrune = useCallback(() => {
    if (pruneTimerRef.current !== null) {
      window.clearTimeout(pruneTimerRef.current)
    }
    pruneTimerRef.current = window.setTimeout(() => {
      pruneTimerRef.current = null
      setLiveLinks((current) =>
        current.filter((link) => link.fadeStartedAt === undefined || liveLinkOpacity(link) > 0)
      )
    }, LIVE_EDGE_FADE_MS + 120)
  }, [])

  const scheduleFlashPrune = useCallback(() => {
    if (flashTimerRef.current !== null) {
      window.clearTimeout(flashTimerRef.current)
    }
    flashTimerRef.current = window.setTimeout(() => {
      flashTimerRef.current = null
      const now = Date.now()
      setFlashUntilByNode((current) => new Map([...current].filter(([, until]) => until > now)))
    }, TOUCH_FLASH_MS + 80)
  }, [])

  const scheduleGlowPrune = useCallback(() => {
    if (glowTimerRef.current !== null) {
      window.clearTimeout(glowTimerRef.current)
    }
    glowTimerRef.current = window.setTimeout(() => {
      glowTimerRef.current = null
      const now = Date.now()
      setGlowByNode((current) => new Map([...current].filter(([, entry]) => entry.until > now)))
    }, LIVE_EDGE_FADE_MS + 120)
  }, [])

  const graphData = useMemo(() => {
    const nodeIds = new Set(baseGraph.nodes.map((node) => node.id))

    return {
      nodes: baseGraph.nodes,
      links: [
        ...baseGraph.links.map((link) =>
          link.kind === 'static' ? { ...link, shared: [...link.shared] } : { ...link }
        ),
        ...liveLinks
          .filter((link) => nodeIds.has(link.source) && nodeIds.has(link.target))
          .map((link) => ({ ...link }))
      ]
    }
  }, [baseGraph, liveLinks])

  const normalizedQuery = query.trim().toLocaleLowerCase()
  const hasActiveMapFilter = normalizedQuery.length > 0 || selectedSources.size > 0
  const matchedNodeIds = useMemo(() => {
    const sourceFilterActive = selectedSources.size > 0
    const matches = new Set<string>()

    for (const node of baseGraph.nodes) {
      if (node.kind !== 'note') {
        continue
      }

      const matchesSource = !sourceFilterActive || selectedSources.has(node.source)
      const matchesTitle =
        normalizedQuery.length === 0 || node.title.toLocaleLowerCase().includes(normalizedQuery)

      if (matchesSource && matchesTitle) {
        matches.add(node.id)
      }
    }

    return matches
  }, [baseGraph.nodes, normalizedQuery, selectedSources])
  const matchedNodeKey = useMemo(() => [...matchedNodeIds].sort().join('\u0000'), [matchedNodeIds])

  const toggleSource = useCallback((source: string) => {
    setSelectedSources((current) => {
      if (current.size === 0) {
        return new Set([source])
      }

      const next = new Set(current)
      if (next.has(source)) {
        next.delete(source)
      } else {
        next.add(source)
      }
      return next
    })
  }, [])

  const clearFilters = useCallback(() => {
    setQuery('')
    setSelectedSources(new Set())
  }, [])

  useEffect(() => {
    const element = containerRef.current
    if (!element) {
      return
    }

    const observer = new ResizeObserver(([entry]) => {
      const width = Math.max(320, Math.floor(entry.contentRect.width))
      const height = Math.max(300, Math.floor(entry.contentRect.height))
      setSize((current) =>
        current.width === width && current.height === height ? current : { width, height }
      )
    })

    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const validSources = new Set(sourceIds)
    setSelectedSources((current) => {
      const next = new Set([...current].filter((source) => validSources.has(source)))
      return next.size === current.size ? current : next
    })
  }, [sourceIds])

  useEffect(() => {
    let cancelled = false
    let loadingTimer: number | null = null

    // a multi-connector sync bumps the graph version once per connector;
    // debounce the refetches so the layout is only disturbed once at the end
    const fetchTimer = window.setTimeout(
      () => {
        loadingTimer = window.setTimeout(() => {
          if (!cancelled) {
            setIsLoading(true)
            setError('')
          }
        }, 120)

        fetchGraph()
          .then((response) => {
            if (!cancelled) {
              setError('')
              setBaseGraph((current) =>
                mergeGraphPositions(current, graphResponseToMapData(response, 'Now'))
              )
              needsReheatRef.current = true
            }
          })
          .catch((fetchError: unknown) => {
            if (!cancelled) {
              setError(fetchError instanceof Error ? fetchError.message : 'Graph fetch failed')
            }
          })
          .finally(() => {
            if (loadingTimer !== null) {
              window.clearTimeout(loadingTimer)
            }
            if (!cancelled) {
              setIsLoading(false)
            }
          })
      },
      isInitialGraphLoadRef.current ? 0 : GRAPH_REFETCH_DEBOUNCE_MS
    )
    isInitialGraphLoadRef.current = false

    return () => {
      cancelled = true
      window.clearTimeout(fetchTimer)
      if (loadingTimer !== null) {
        window.clearTimeout(loadingTimer)
      }
    }
  }, [refreshKey, retryNonce])

  // if the data changed while this tab was hidden (or rAF was throttled),
  // the engine can hit force-graph's wall-clock cooldownTime with almost no
  // ticks executed and freeze in the initial layout — reheat on reveal
  useEffect(() => {
    if (!visible || !needsReheatRef.current) {
      return
    }
    needsReheatRef.current = false
    graphRef.current?.d3ReheatSimulation()
  }, [visible, baseGraph])

  useEffect(() => {
    if (!error) {
      return
    }
    const timeout = window.setTimeout(() => setRetryNonce((nonce) => nonce + 1), 4000)
    return () => window.clearTimeout(timeout)
  }, [error, retryNonce])

  useEffect(() => {
    if (!retrieval || processedRetrievalRef.current === retrieval) {
      return
    }

    processedRetrievalRef.current = retrieval
    const now = Date.now()
    const update = liveGraphFromRetrieval(retrieval, now)

    setLiveLinks((current) => [
      ...fadingLiveLinks(
        current.filter((link) => link.fadeStartedAt === undefined),
        now
      ),
      ...current.filter(
        (link) => link.fadeStartedAt !== undefined && liveLinkOpacity(link, now) > 0
      ),
      ...update.links
    ])

    setFlashUntilByNode((current) => {
      const next = new Map([...current].filter(([, until]) => until > now))
      for (const nodeId of update.touchedNodeIds) {
        next.set(nodeId, now + TOUCH_FLASH_MS)
      }
      return next
    })

    setGlowByNode((current) => {
      const next = new Map([...current].filter(([, entry]) => entry.until > now))
      for (const [nodeId, intensity] of update.traversalGlow) {
        const existing = next.get(nodeId)
        next.set(nodeId, {
          until: now + LIVE_EDGE_FADE_MS,
          intensity: Math.max(intensity, existing?.intensity ?? 0)
        })
      }
      return next
    })

    schedulePrune()
    scheduleFlashPrune()
    scheduleGlowPrune()

    return undefined
  }, [retrieval, scheduleFlashPrune, scheduleGlowPrune, schedulePrune])

  useEffect(() => {
    if (!visible || hasActiveMapFilter || baseGraph.nodes.length <= 1) {
      return
    }

    const timeout = window.setTimeout(() => {
      graphRef.current?.zoomToFit(420, 56)
    }, 120)

    return () => window.clearTimeout(timeout)
  }, [baseGraph.nodes.length, hasActiveMapFilter, visible])

  useEffect(() => {
    if (!visible || !hasActiveMapFilter || matchedNodeIds.size === 0) {
      return
    }

    const timeout = window.setTimeout(() => {
      graphRef.current?.zoomToFit(
        360,
        64,
        (node) => node.kind === 'note' && matchedNodeIds.has(String(node.id))
      )
    }, 100)

    return () => window.clearTimeout(timeout)
  }, [hasActiveMapFilter, matchedNodeIds, matchedNodeKey, visible])

  useEffect(() => {
    return () => {
      if (pruneTimerRef.current !== null) {
        window.clearTimeout(pruneTimerRef.current)
      }
      if (flashTimerRef.current !== null) {
        window.clearTimeout(flashTimerRef.current)
      }
      if (glowTimerRef.current !== null) {
        window.clearTimeout(glowTimerRef.current)
      }
    }
  }, [])

  const drawNode = useCallback(
    (node: GraphNodeObject, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const id = String(node.id)
      const isNow = id === NOW_NODE_ID
      const isHovered = id === hoverNodeId
      const radius = isNow ? 14 : nodeRadius(node.degree)
      const flashOpacity = nodeFlashOpacity(id, flashUntilByNode)
      const glow = nodeGlow(id, glowByNode)
      const isPinned = !isNow && pinnedSet.has(id)
      const opacity = nodeOpacity(node, hasActiveMapFilter, matchedNodeIds)
      const x = node.x ?? 0
      const y = node.y ?? 0

      ctx.save()
      ctx.globalAlpha = opacity

      if (glow > 0) {
        // PPR traversal arrival: soft amber halo, mass-weighted
        const glowRadius = radius + 6 + glow * 9
        const gradient = ctx.createRadialGradient(x, y, radius * 0.6, x, y, glowRadius)
        gradient.addColorStop(0, rgba('#f5c518', glow * 0.5))
        gradient.addColorStop(1, rgba('#f5c518', 0))
        ctx.beginPath()
        ctx.arc(x, y, glowRadius, 0, Math.PI * 2)
        ctx.fillStyle = gradient
        ctx.fill()
      }

      if (flashOpacity > 0) {
        ctx.beginPath()
        ctx.arc(x, y, radius + 7, 0, Math.PI * 2)
        ctx.strokeStyle = rgba('#ffe08a', flashOpacity * 0.78)
        ctx.lineWidth = Math.max(1.4 / globalScale, 1)
        ctx.stroke()
      }

      if (isPinned) {
        // yellow ring for grounding notes
        ctx.beginPath()
        ctx.arc(x, y, radius + 5, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(245, 197, 24, 0.95)'
        ctx.lineWidth = Math.max(2 / globalScale, 1.2)
        ctx.stroke()
      }

      if (isHovered && !isNow) {
        ctx.beginPath()
        ctx.arc(x, y, radius + 3, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(236, 235, 230, 0.92)'
        ctx.lineWidth = Math.max(1.6 / globalScale, 1.1)
        ctx.stroke()
      }

      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fillStyle = isNow ? '#f5c518' : sourceColor(node.source)
      ctx.fill()

      ctx.beginPath()
      ctx.arc(x, y, Math.max(2.4, radius * 0.32), 0, Math.PI * 2)
      ctx.fillStyle = isNow ? '#231b06' : '#0b0b0c'
      ctx.fill()

      const matched = hasActiveMapFilter && node.kind === 'note' && matchedNodeIds.has(id)
      const alpha =
        isNow || isHovered || matched ? 1 : labelAlpha(globalScale, node.degree >= hubDegree)
      if (alpha > 0.02) {
        const label = isNow ? nowLabel : node.title
        drawNodeLabel(ctx, label, x, y + radius + 4, isNow, globalScale, alpha)
      }
      ctx.restore()
    },
    [
      flashUntilByNode,
      glowByNode,
      hasActiveMapFilter,
      hoverNodeId,
      hubDegree,
      matchedNodeIds,
      nowLabel,
      pinnedSet
    ]
  )

  const drawLink = useCallback(
    (link: GraphLinkObject, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const source = endpoint(link.source)
      const target = endpoint(link.target)
      if (!source || !target) {
        return
      }

      const isStatic = link.kind === 'static'
      const isHovered = link.id === hoverLinkId
      const baseOpacity = isStatic ? staticLinkOpacity(link.weight) : liveLinkOpacity(link)
      const opacity = isHovered
        ? Math.max(baseOpacity, 0.85)
        : linkOpacity(link, baseOpacity, hasActiveMapFilter, matchedNodeIds)
      if (opacity <= 0) {
        return
      }

      ctx.save()
      ctx.beginPath()
      ctx.moveTo(source.x, source.y)
      ctx.lineTo(target.x, target.y)
      ctx.strokeStyle = isStatic
        ? isHovered
          ? `rgba(236, 235, 230, ${opacity})`
          : `rgba(108, 116, 128, ${opacity})`
        : rgba(link.color, opacity)
      ctx.lineWidth = isStatic
        ? Math.max((isHovered ? 1.8 : 0.75) / globalScale, isHovered ? 0.9 : 0.35)
        : Math.max((isHovered ? 3.4 : 2.6) / globalScale, 0.8)
      ctx.stroke()
      ctx.restore()
    },
    [hasActiveMapFilter, hoverLinkId, matchedNodeIds]
  )

  const noteCount = Math.max(0, baseGraph.nodes.length - 1)
  const matchCount = hasActiveMapFilter ? matchedNodeIds.size : noteCount
  const overlayText = error
    ? error
    : isLoading
      ? 'Loading map…'
      : noteCount === 0
        ? 'No notes indexed'
        : hasActiveMapFilter
          ? `${matchCount}/${noteCount} matches · ${baseGraph.links.length} links`
          : `${noteCount} notes · ${baseGraph.links.length} links`

  return (
    <section aria-hidden={!visible} aria-label="Map view" className="map-view">
      <div className="map-controls">
        <label className="map-search-field">
          <span className="sr-only">Search map titles</span>
          <input
            type="search"
            value={query}
            placeholder="Search titles"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <div className="map-source-filters" aria-label="Source filters">
          {sourceIds.map((source) => {
            const active = selectedSources.size === 0 || selectedSources.has(source)
            const style = { '--source-color': sourceColor(source) } as CSSProperties

            return (
              <button
                className={`map-source-chip${active ? ' active' : ''}`}
                key={source}
                type="button"
                aria-pressed={active}
                style={style}
                onClick={() => toggleSource(source)}
              >
                <span aria-hidden="true" />
                {sourceLabel(source)}
              </button>
            )
          })}
        </div>
        {hasActiveMapFilter ? (
          <button className="secondary-button compact" type="button" onClick={clearFilters}>
            Clear
          </button>
        ) : null}
      </div>

      <div className="map-stage" ref={containerRef}>
        <ForceGraph2D<MapNode, MapLink>
          ref={graphRef}
          autoPauseRedraw={!visible}
          backgroundColor="#0b0b0c"
          cooldownTicks={LAYOUT_COOLDOWN_TICKS}
          d3AlphaDecay={0.025}
          d3VelocityDecay={0.34}
          enableNodeDrag={false}
          graphData={graphData}
          height={size.height}
          linkCanvasObject={drawLink}
          linkCanvasObjectMode={() => 'replace'}
          linkHoverPrecision={6}
          linkLabel={() => ''}
          linkSource="source"
          linkTarget="target"
          nodeCanvasObject={drawNode}
          nodeCanvasObjectMode={() => 'replace'}
          nodeId="id"
          nodeLabel={() => ''}
          nodeRelSize={1}
          nodeVal={(node) => node.val}
          showPointerCursor={(item) => isNoteNodeObject(item)}
          onLinkHover={(link) => {
            setHoverLink(link ?? null)
            if (link) {
              setHoverNode(null)
            }
          }}
          onNodeClick={(node) => openPreview(node)}
          onNodeHover={(node) => {
            setHoverNode(node ?? null)
            if (node) {
              setHoverLink(null)
            }
          }}
          width={size.width}
        />

        <MapHoverPanel hoverLink={hoverLink} hoverNode={hoverNode} nowLabel={nowLabel} />

        <div
          aria-live="polite"
          role={error ? 'alert' : 'status'}
          className={`map-overlay-chip${error ? ' map-overlay-error' : ''}`}
        >
          {overlayText}
        </div>

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
          {sourceIds.map((source) => (
            <span className="map-legend-row" key={`legend-${source}`}>
              <span
                className="map-legend-swatch"
                style={{ '--source-color': sourceColor(source) } as CSSProperties}
                aria-hidden="true"
              />
              {sourceLabel(source)}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}

function openPreview(node: GraphNodeObject): void {
  if (node.kind !== 'note' || !node.note_id) {
    return
  }

  void window.api.openPreview(node.note_id).catch((error: unknown) => {
    console.error('Failed to open preview window', error)
  })
}

function isNoteNodeObject(value: unknown): value is GraphNodeObject {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  return (value as Partial<MapNode>).kind === 'note'
}

function nodeOpacity(
  node: GraphNodeObject,
  hasActiveMapFilter: boolean,
  matchedNodeIds: Set<string>
): number {
  if (!hasActiveMapFilter) {
    return 1
  }

  return node.kind === 'note' && matchedNodeIds.has(String(node.id)) ? 1 : 0.06
}

function linkOpacity(
  link: GraphLinkObject,
  baseOpacity: number,
  hasActiveMapFilter: boolean,
  matchedNodeIds: Set<string>
): number {
  if (!hasActiveMapFilter) {
    return baseOpacity
  }

  if (link.kind === 'static') {
    const sourceId = endpointId(link.source)
    const targetId = endpointId(link.target)
    return sourceId && targetId && matchedNodeIds.has(sourceId) && matchedNodeIds.has(targetId)
      ? baseOpacity
      : 0.06
  }

  const sourceId = endpointId(link.source)
  const targetId = endpointId(link.target)
  const noteIds = [sourceId, targetId].filter((id): id is string =>
    Boolean(id && id !== NOW_NODE_ID)
  )
  return noteIds.some((id) => matchedNodeIds.has(id)) ? baseOpacity : 0.06
}

function nodeFlashOpacity(nodeId: string, flashUntilByNode: Map<string, number>): number {
  const until = flashUntilByNode.get(nodeId)
  if (!until) {
    return 0
  }

  return Math.max(0, Math.min(1, (until - Date.now()) / TOUCH_FLASH_MS))
}

function endpoint(value: unknown): { x: number; y: number } | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const maybeNode = value as GraphNodeObject
  return typeof maybeNode.x === 'number' && typeof maybeNode.y === 'number'
    ? { x: maybeNode.x, y: maybeNode.y }
    : null
}

function endpointId(value: unknown): string | null {
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value)
  }

  if (typeof value !== 'object' || value === null) {
    return null
  }

  const maybeNode = value as { id?: unknown }
  if (maybeNode.id === undefined) {
    return null
  }

  return String(maybeNode.id)
}

function drawNodeLabel(
  ctx: CanvasRenderingContext2D,
  label: string,
  x: number,
  y: number,
  isNow: boolean,
  globalScale: number,
  alpha: number
): void {
  const text = truncateLabel(label, isNow ? 38 : 28)
  const fontSize = (isNow ? 12 : 9.5) / globalScale
  ctx.font = `${isNow ? 700 : 650} ${fontSize}px Inter, system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'

  const width = ctx.measureText(text).width
  const paddingX = 4 / globalScale
  const paddingY = 2 / globalScale
  const height = fontSize + paddingY * 2

  ctx.fillStyle = isNow ? `rgba(35, 27, 6, ${0.86 * alpha})` : `rgba(11, 11, 12, ${0.72 * alpha})`
  roundedRect(ctx, x - width / 2 - paddingX, y, width + paddingX * 2, height, 4 / globalScale)
  ctx.fill()

  ctx.fillStyle = isNow ? rgba('#ffe9a6', alpha) : `rgba(226, 224, 218, ${0.82 * alpha})`
  ctx.fillText(text, x, y + paddingY)
}

/** Zoom-based label level-of-detail: 0 hidden → 1 fully visible. */
function labelAlpha(globalScale: number, isHub: boolean): number {
  const [start, end] = isHub ? HUB_LABEL_FADE : NODE_LABEL_FADE
  return Math.max(0, Math.min(1, (globalScale - start) / (end - start)))
}

function degreePercentile(nodes: MapNode[], percentile: number): number {
  const degrees = nodes
    .filter((node) => node.kind === 'note')
    .map((node) => node.degree)
    .sort((left, right) => left - right)
  if (degrees.length === 0) {
    return Number.POSITIVE_INFINITY
  }

  const index = Math.min(degrees.length - 1, Math.floor(degrees.length * percentile))
  return Math.max(1, degrees[index])
}

function nodeGlow(nodeId: string, glowByNode: Map<string, GlowEntry>): number {
  const entry = glowByNode.get(nodeId)
  if (!entry) {
    return 0
  }

  const remaining = (entry.until - Date.now()) / LIVE_EDGE_FADE_MS
  return remaining <= 0 ? 0 : Math.min(1, remaining) * entry.intensity
}

function endpointNode(value: unknown): MapNode | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const maybeNode = value as Partial<MapNode>
  return typeof maybeNode.id === 'string' && typeof maybeNode.title === 'string'
    ? (maybeNode as MapNode)
    : null
}

function MapHoverPanel({
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
              style={{ '--dot-color': '#f5c518' } as CSSProperties}
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
            style={{ '--dot-color': sourceColor(hoverNode.source) } as CSSProperties}
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
          <span>click to open</span>
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
        style={{ '--dot-color': sourceColor(node.source) } as CSSProperties}
        aria-hidden="true"
      />
      <div className="map-hover-endpoint-body">
        <strong>{node.title}</strong>
        {node.gist ? <p>{node.gist}</p> : null}
      </div>
    </div>
  )
}

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  const r = Math.min(radius, width / 2, height / 2)
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + width - r, y)
  ctx.quadraticCurveTo(x + width, y, x + width, y + r)
  ctx.lineTo(x + width, y + height - r)
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height)
  ctx.lineTo(x + r, y + height)
  ctx.quadraticCurveTo(x, y + height, x, y + height - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}

function truncateLabel(label: string, maxLength: number): string {
  return label.length <= maxLength ? label : `${label.slice(0, maxLength - 3)}...`
}

function rgba(hex: string, alpha: number): string {
  const value = hex.replace('#', '')
  const r = Number.parseInt(value.slice(0, 2), 16)
  const g = Number.parseInt(value.slice(2, 4), 16)
  const b = Number.parseInt(value.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`
}
