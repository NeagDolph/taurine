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
  communityColor,
  fadingLiveLinks,
  fetchGraph,
  fetchLocalGraph,
  graphResponseToMapData,
  liveGraphFromRetrieval,
  liveLinkOpacity,
  localSubgraphFromMapData,
  mergeGraphPositions,
  nodeRadius,
  nowLabelFromText,
  recencyColor,
  sourceColor,
  sourceLabel,
  sourcesFromMapData,
  staticLinkOpacity,
  type LiveGraphLink,
  type MapGraphData,
  type MapLink,
  type MapNode,
  type StaticGraphLink
} from '../../lib/graph'
import type { RetrievalEvent } from '../../lib/sidecar'
import { MapContextMenu, type MenuTarget } from './ContextMenu'
import { MapHoverPanel } from './HoverPanel'
import { MapLegend } from './MapLegend'
import { MapSearch, type SearchResult } from './MapSearch'
import { SidePanel } from './SidePanel'
import { MapTimeline, type TimeRange } from './Timeline'
import { degreePercentile, drawNodeLabel, endpoint, endpointId, labelAlpha, rgba } from './render'
import {
  applyPinnedPositions,
  loadColorMode,
  loadMinDegree,
  loadPinnedPositions,
  saveColorMode,
  saveMinDegree,
  savePinnedPositions,
  type ColorMode,
  type PinnedPositions
} from './storage'
import './map.css'

export interface MapFocusRequest {
  noteId: string
  nonce: number
}

interface MapViewProps {
  visible: boolean
  refreshKey?: number
  latestFinalText: string
  retrieval: RetrievalEvent | null
  groundedNoteIds: string[]
  focusRequest: MapFocusRequest | null
  onOpenNote: (noteId: string) => void
  onAskAbout: (noteId: string, title: string) => void
}

type GraphLinkObject = LinkObject<MapNode, MapLink>
type GraphNodeObject = NodeObject<MapNode>

interface GlowEntry {
  until: number
  intensity: number
}

interface EgoNav {
  center: string
  depth: number
  /** previously visited centers, oldest first (breadcrumb trail) */
  history: string[]
}

const EMPTY_GRAPH: MapGraphData = graphResponseToMapData({ nodes: [], edges: [] }, 'Now')
const GRAPH_REFETCH_DEBOUNCE_MS = 800
// ~760 nodes need a few hundred ticks to untangle from d3's initial spiral;
// the old 80 froze half-laid-out graphs whenever a refetch reset positions
const LAYOUT_COOLDOWN_TICKS = 300
const DIM_OPACITY = 0.06
// session heat trail: touched notes keep a faint warm tint that decays over
// ~5 minutes so a glance shows where the conversation has walked
const HEAT_TRAIL_MS = 5 * 60 * 1000
const SEARCH_HALO_COLOR = '#9edcff'
const HEAT_TRAIL_COLOR = '#ff8a3d'
const MAX_SEARCH_RESULTS = 8
// zoomToFit on one or two nodes otherwise zooms in until a node fills the
// screen — clamp the post-fit scale to keep the neighborhood readable
const MAX_FIT_ZOOM = 3

export function MapView({
  visible,
  refreshKey = 0,
  latestFinalText,
  retrieval,
  groundedNoteIds,
  focusRequest,
  onOpenNote,
  onAskAbout
}: MapViewProps): React.JSX.Element {
  const graphRef = useRef<ForceGraphMethods<MapNode, MapLink> | undefined>(undefined)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const isInitialGraphLoadRef = useRef(true)
  const needsReheatRef = useRef(false)
  const processedRetrievalRef = useRef<RetrievalEvent | null>(null)
  const processedFocusNonceRef = useRef(0)
  const pruneTimerRef = useRef<number | null>(null)
  const flashTimerRef = useRef<number | null>(null)
  const glowTimerRef = useRef<number | null>(null)
  const heatByNodeRef = useRef<Map<string, number>>(new Map())
  const egoFitPendingRef = useRef(false)

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

  const [egoNav, setEgoNav] = useState<EgoNav | null>(null)
  const [egoGraph, setEgoGraph] = useState<MapGraphData | null>(null)
  const [egoLoading, setEgoLoading] = useState(false)
  const [egoError, setEgoError] = useState('')
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [menuTarget, setMenuTarget] = useState<MenuTarget | null>(null)
  const [hiddenNodeIds, setHiddenNodeIds] = useState<Set<string>>(() => new Set())
  const [pins, setPins] = useState<PinnedPositions>(() => loadPinnedPositions())
  const [colorMode, setColorMode] = useState<ColorMode>(() => loadColorMode())
  const [minDegree, setMinDegree] = useState<number>(() => loadMinDegree())
  const [timeRange, setTimeRange] = useState<TimeRange | null>(null)

  // mirrors for async callbacks (ego fetch fallback, pin rehydration)
  const baseGraphRef = useRef(baseGraph)
  const pinsRef = useRef(pins)
  useEffect(() => {
    baseGraphRef.current = baseGraph
  }, [baseGraph])
  useEffect(() => {
    pinsRef.current = pins
  }, [pins])

  const nowLabel = useMemo(() => nowLabelFromText(latestFinalText), [latestFinalText])
  const groundedSet = useMemo(() => new Set(groundedNoteIds), [groundedNoteIds])
  const isEgoMode = egoNav !== null
  const displayedGraph = isEgoMode && egoGraph ? egoGraph : baseGraph
  const sourceIds = useMemo(() => sourcesFromMapData(displayedGraph), [displayedGraph])
  const hubDegree = useMemo(
    () => degreePercentile(displayedGraph.nodes, 0.75),
    [displayedGraph.nodes]
  )
  const hoverNodeId = hoverNode ? String(hoverNode.id) : null
  const hoverLinkId = hoverLink ? hoverLink.id : null

  const titleById = useMemo(() => {
    const titles = new Map<string, string>()
    for (const node of baseGraph.nodes) {
      titles.set(node.id, node.title)
    }
    if (egoGraph) {
      for (const node of egoGraph.nodes) {
        titles.set(node.id, node.title)
      }
    }
    return titles
  }, [baseGraph, egoGraph])

  // ── filtering ──────────────────────────────────────────────────────────

  const degreeFilterActive = !isEgoMode && minDegree > 0
  const visibleNodes = useMemo(() => {
    return displayedGraph.nodes.filter((node) => {
      if (node.kind !== 'note') {
        return true
      }
      if (hiddenNodeIds.has(node.id)) {
        return false
      }
      if (degreeFilterActive && node.degree < minDegree && node.id !== selectedNodeId) {
        return false
      }
      return true
    })
  }, [degreeFilterActive, displayedGraph.nodes, hiddenNodeIds, minDegree, selectedNodeId])

  const graphData = useMemo(() => {
    const nodeIds = new Set(visibleNodes.map((node) => node.id))

    return {
      nodes: visibleNodes,
      links: [
        ...displayedGraph.links
          .filter((link) => {
            if (link.kind !== 'static') {
              return false
            }
            const src = endpointId(link.source)
            const dst = endpointId(link.target)
            return Boolean(src && dst && nodeIds.has(src) && nodeIds.has(dst))
          })
          .map((link) =>
            link.kind === 'static' ? { ...link, shared: [...link.shared] } : { ...link }
          ),
        ...liveLinks
          .filter((link) => nodeIds.has(link.source) && nodeIds.has(link.target))
          .map((link) => ({ ...link }))
      ]
    }
  }, [displayedGraph.links, liveLinks, visibleNodes])

  const normalizedQuery = query.trim().toLocaleLowerCase()
  const queryActive = normalizedQuery.length > 0
  // selections from sources no longer present in the graph are ignored (not
  // pruned from state) so a transient refetch cannot wipe the user's filter
  const activeSources = useMemo(() => {
    const valid = new Set(sourceIds)
    return new Set([...selectedSources].filter((source) => valid.has(source)))
  }, [selectedSources, sourceIds])
  const hasActiveMapFilter = queryActive || activeSources.size > 0

  const queryMatchedIds = useMemo(() => {
    const matches = new Set<string>()
    if (!queryActive) {
      return matches
    }
    for (const node of visibleNodes) {
      if (node.kind === 'note' && queryMatchKind(node, normalizedQuery)) {
        matches.add(node.id)
      }
    }
    return matches
  }, [normalizedQuery, queryActive, visibleNodes])

  const matchedNodeIds = useMemo(() => {
    const sourceFilterActive = activeSources.size > 0
    const matches = new Set<string>()

    for (const node of visibleNodes) {
      if (node.kind !== 'note') {
        continue
      }

      const matchesSource = !sourceFilterActive || activeSources.has(node.source)
      const matchesQuery = !queryActive || queryMatchedIds.has(node.id)

      if (matchesSource && matchesQuery) {
        matches.add(node.id)
      }
    }

    return matches
  }, [activeSources, queryActive, queryMatchedIds, visibleNodes])
  const matchedNodeKey = useMemo(() => [...matchedNodeIds].sort().join('\u0000'), [matchedNodeIds])

  const searchResults: SearchResult[] = useMemo(() => {
    if (!queryActive) {
      return []
    }

    const rows: Array<{ result: SearchResult; score: number }> = []
    for (const node of visibleNodes) {
      if (node.kind !== 'note') {
        continue
      }
      const via = queryMatchKind(node, normalizedQuery)
      if (!via) {
        continue
      }
      const score = via === 'title' ? 3 : via === 'entity' ? 2 : 1
      rows.push({ result: { node, via }, score })
    }

    rows.sort((left, right) =>
      right.score !== left.score
        ? right.score - left.score
        : right.result.node.degree - left.result.node.degree
    )
    return rows.slice(0, MAX_SEARCH_RESULTS).map((row) => row.result)
  }, [normalizedQuery, queryActive, visibleNodes])

  // ── temporal lens ──────────────────────────────────────────────────────

  const timeDomain = useMemo<TimeRange | null>(() => {
    let min = Number.POSITIVE_INFINITY
    let max = Number.NEGATIVE_INFINITY
    for (const node of displayedGraph.nodes) {
      const time = nodeTime(node)
      if (time > 0) {
        min = Math.min(min, time)
        max = Math.max(max, time)
      }
    }
    return min < max ? [min, max] : null
  }, [displayedGraph.nodes])

  const histogram = useMemo(() => {
    if (!timeDomain) {
      return []
    }
    const bucketCount = 72
    const buckets = new Array<number>(bucketCount).fill(0)
    const span = Math.max(1, timeDomain[1] - timeDomain[0])
    for (const node of displayedGraph.nodes) {
      const time = nodeTime(node)
      if (time > 0) {
        const index = Math.min(
          bucketCount - 1,
          Math.floor(((time - timeDomain[0]) / span) * bucketCount)
        )
        buckets[index] += 1
      }
    }
    return buckets
  }, [displayedGraph.nodes, timeDomain])

  // derived: a brushed range only applies while a time domain exists
  const activeTimeRange = timeDomain ? timeRange : null

  // ── hover neighborhood ─────────────────────────────────────────────────

  const adjacency = useMemo(() => {
    const map = new Map<string, Set<string>>()
    const connect = (a: string | null, b: string | null): void => {
      if (!a || !b) {
        return
      }
      if (!map.has(a)) map.set(a, new Set())
      if (!map.has(b)) map.set(b, new Set())
      map.get(a)?.add(b)
      map.get(b)?.add(a)
    }
    for (const link of displayedGraph.links) {
      connect(endpointId(link.source), endpointId(link.target))
    }
    for (const link of liveLinks) {
      connect(link.source, link.target)
    }
    return map
  }, [displayedGraph.links, liveLinks])

  const hoverNeighborhood = useMemo(() => {
    if (!hoverNodeId) {
      return null
    }
    const set = new Set<string>([hoverNodeId])
    for (const neighbor of adjacency.get(hoverNodeId) ?? []) {
      set.add(neighbor)
    }
    return set
  }, [adjacency, hoverNodeId])

  // ── ego navigation ─────────────────────────────────────────────────────

  const enterEgo = useCallback((noteId: string, depth?: number) => {
    setEgoNav((current) => {
      if (!current) {
        return { center: noteId, depth: depth ?? 2, history: [] }
      }
      if (current.center === noteId) {
        const nextDepth = depth ?? current.depth
        return nextDepth === current.depth ? current : { ...current, depth: nextDepth }
      }
      return {
        center: noteId,
        depth: depth ?? current.depth,
        history: [...current.history, current.center]
      }
    })
  }, [])

  const exitEgo = useCallback(() => {
    setEgoNav(null)
    setEgoGraph(null)
    setEgoError('')
    needsReheatRef.current = true
  }, [])

  const egoBack = useCallback(() => {
    setEgoNav((current) => {
      if (!current || current.history.length === 0) {
        return current
      }
      return {
        center: current.history[current.history.length - 1],
        depth: current.depth,
        history: current.history.slice(0, -1)
      }
    })
  }, [])

  const egoJumpTo = useCallback((index: number) => {
    setEgoNav((current) => {
      if (!current || index < 0 || index >= current.history.length) {
        return current
      }
      return {
        center: current.history[index],
        depth: current.depth,
        history: current.history.slice(0, index)
      }
    })
  }, [])

  const setEgoDepth = useCallback((depth: number) => {
    setEgoNav((current) => (current ? { ...current, depth } : current))
  }, [])

  const adoptEgoGraph = useCallback((next: MapGraphData) => {
    setEgoGraph((previous) => {
      seedPositions(next, previous, baseGraphRef.current)
      applyPinnedPositions(next, pinsRef.current)
      return previous ? mergeGraphPositions(previous, next) : next
    })
    needsReheatRef.current = true
    egoFitPendingRef.current = true
  }, [])

  useEffect(() => {
    if (!egoNav) {
      return undefined
    }

    let cancelled = false
    const timer = window.setTimeout(() => {
      setEgoLoading(true)
      setEgoError('')

      fetchLocalGraph(egoNav.center, egoNav.depth)
        .then((response) => {
          if (!cancelled) {
            adoptEgoGraph(graphResponseToMapData(response, '', { includeNow: false }))
          }
        })
        .catch(() => {
          if (cancelled) {
            return
          }
          // older backend without /graph/local: BFS the already-loaded global map
          const fallback = localSubgraphFromMapData(
            baseGraphRef.current,
            egoNav.center,
            egoNav.depth
          )
          if (fallback.nodes.length > 0) {
            adoptEgoGraph(fallback)
          } else {
            setEgoError('Neighborhood unavailable')
          }
        })
        .finally(() => {
          if (!cancelled) {
            setEgoLoading(false)
          }
        })
    }, 0)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [adoptEgoGraph, egoNav, refreshKey])

  useEffect(() => {
    if (!visible || !egoFitPendingRef.current || !egoGraph) {
      return undefined
    }
    egoFitPendingRef.current = false
    const timeout = window.setTimeout(() => {
      graphRef.current?.zoomToFit(420, 64)
    }, 420)
    const clampTimeout = window.setTimeout(() => clampZoom(graphRef.current), 900)
    return () => {
      window.clearTimeout(timeout)
      window.clearTimeout(clampTimeout)
    }
  }, [egoGraph, visible])

  // ── selection / focus ──────────────────────────────────────────────────

  const findDisplayedNode = useCallback(
    (noteId: string): MapNode | null =>
      displayedGraph.nodes.find((node) => node.id === noteId) ?? null,
    [displayedGraph.nodes]
  )

  const centerOnNode = useCallback(
    (noteId: string) => {
      const node = findDisplayedNode(noteId)
      if (!node || typeof node.x !== 'number' || typeof node.y !== 'number') {
        return
      }
      graphRef.current?.centerAt(node.x, node.y, 500)
      const zoom = graphRef.current?.zoom()
      if (zoom !== undefined && zoom < 1.8) {
        graphRef.current?.zoom(1.8, 500)
      }
    },
    [findDisplayedNode]
  )

  const pulseNode = useCallback((noteId: string) => {
    setFlashUntilByNode((current) => {
      const next = new Map(current)
      next.set(noteId, Date.now() + TOUCH_FLASH_MS + 800)
      return next
    })
  }, [])

  const selectNode = useCallback(
    (noteId: string, options: { center?: boolean } = {}) => {
      setSelectedNodeId(noteId)
      if (egoNav) {
        enterEgo(noteId)
      } else if (options.center) {
        centerOnNode(noteId)
      }
    },
    [centerOnNode, egoNav, enterEgo]
  )

  const focusNode = useCallback(
    (noteId: string, depth?: number) => {
      setSelectedNodeId(noteId)
      enterEgo(noteId, depth)
    },
    [enterEgo]
  )

  // ── pins / hide ────────────────────────────────────────────────────────

  useEffect(() => {
    savePinnedPositions(pins)
  }, [pins])

  useEffect(() => {
    saveColorMode(colorMode)
  }, [colorMode])

  useEffect(() => {
    saveMinDegree(minDegree)
  }, [minDegree])

  const togglePin = useCallback(
    (noteId: string) => {
      const node = findDisplayedNode(noteId)
      setPins((current) => {
        const next = { ...current }
        if (next[noteId]) {
          delete next[noteId]
          if (node) {
            node.fx = undefined
            node.fy = undefined
          }
          needsReheatRef.current = true
        } else if (node && typeof node.x === 'number' && typeof node.y === 'number') {
          node.fx = node.x
          node.fy = node.y
          next[noteId] = { x: node.x, y: node.y }
        }
        return next
      })
    },
    [findDisplayedNode]
  )

  const hideNode = useCallback((noteId: string) => {
    setHiddenNodeIds((current) => new Set(current).add(noteId))
    setSelectedNodeId((current) => (current === noteId ? null : current))
  }, [])

  const unhideAll = useCallback(() => {
    setHiddenNodeIds(new Set())
  }, [])

  // ── live schedule/prune plumbing ───────────────────────────────────────

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
    }, TOUCH_FLASH_MS + 900)
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

  // citations / chat pills: land in a depth-1 neighborhood and pulse the note
  useEffect(() => {
    if (!focusRequest || focusRequest.nonce === processedFocusNonceRef.current) {
      return
    }
    processedFocusNonceRef.current = focusRequest.nonce
    // deferred so the tab switch commits before the map re-navigates; the
    // nonce guard above makes the scheduled work run exactly once
    window.setTimeout(() => {
      setSelectedNodeId(focusRequest.noteId)
      setEgoNav({ center: focusRequest.noteId, depth: 1, history: [] })
      pulseNode(focusRequest.noteId)
      scheduleFlashPrune()
    }, 0)
  }, [focusRequest, pulseNode, scheduleFlashPrune])

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
    setTimeRange(null)
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
                applyPinnedPositions(
                  mergeGraphPositions(current, graphResponseToMapData(response, 'Now')),
                  pinsRef.current
                )
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
  }, [visible, baseGraph, egoGraph])

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

    // session heat trail residue (drawn directly from the ref each frame)
    const heat = heatByNodeRef.current
    for (const nodeId of update.touchedNodeIds) {
      heat.set(nodeId, now)
    }
    for (const [nodeId, until] of heat) {
      if (now - until > HEAT_TRAIL_MS) {
        heat.delete(nodeId)
      }
    }

    schedulePrune()
    scheduleFlashPrune()
    scheduleGlowPrune()

    return undefined
  }, [retrieval, scheduleFlashPrune, scheduleGlowPrune, schedulePrune])

  useEffect(() => {
    if (!visible || isEgoMode || hasActiveMapFilter || baseGraph.nodes.length <= 1) {
      return
    }

    const timeout = window.setTimeout(() => {
      graphRef.current?.zoomToFit(420, 56)
    }, 120)

    return () => window.clearTimeout(timeout)
  }, [baseGraph.nodes.length, hasActiveMapFilter, isEgoMode, visible])

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
    const clampTimeout = window.setTimeout(() => clampZoom(graphRef.current), 540)

    return () => {
      window.clearTimeout(timeout)
      window.clearTimeout(clampTimeout)
    }
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

  // ── interactions ───────────────────────────────────────────────────────

  const openMenu = useCallback(
    (
      event: MouseEvent,
      spec: { kind: 'node'; node: MapNode } | { kind: 'edge'; link: StaticGraphLink }
    ) => {
      const rect = containerRef.current?.getBoundingClientRect()
      const x = Math.min(event.clientX - (rect?.left ?? 0), Math.max(0, size.width - 250))
      const y = Math.min(event.clientY - (rect?.top ?? 0), Math.max(0, size.height - 200))
      setMenuTarget(spec.kind === 'node' ? { ...spec, x, y } : { ...spec, x, y })
    },
    [size.height, size.width]
  )

  const handleNodeClick = useCallback(
    (node: GraphNodeObject) => {
      if (node.kind !== 'note') {
        return
      }
      selectNode(String(node.id))
    },
    [selectNode]
  )

  const handleNodeDragEnd = useCallback((node: GraphNodeObject) => {
    if (typeof node.x !== 'number' || typeof node.y !== 'number') {
      return
    }
    node.fx = node.x
    node.fy = node.y
    if (node.kind === 'note') {
      const { x, y } = node
      setPins((current) => ({ ...current, [String(node.id)]: { x, y } }))
    }
  }, [])

  const inTimeRange = useCallback(
    (node: MapNode): boolean => {
      if (!activeTimeRange) {
        return true
      }
      const time = nodeTime(node)
      return time > 0 && time >= activeTimeRange[0] && time <= activeTimeRange[1]
    },
    [activeTimeRange]
  )

  const nodeFillColor = useCallback(
    (node: MapNode): string => {
      if (colorMode === 'community') {
        return communityColor(node.community)
      }
      if (colorMode === 'recency') {
        return recencyColor(nodeTime(node), timeDomain?.[0] ?? 0, timeDomain?.[1] ?? 0)
      }
      return sourceColor(node.source)
    },
    [colorMode, timeDomain]
  )

  // ── canvas painters ────────────────────────────────────────────────────

  const drawNode = useCallback(
    (node: GraphNodeObject, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const id = String(node.id)
      const isNow = id === NOW_NODE_ID
      const isHovered = id === hoverNodeId
      const isSelected = id === selectedNodeId
      const radius = isNow ? 14 : nodeRadius(node.degree)
      const flashOpacity = nodeFlashOpacity(id, flashUntilByNode)
      const glow = nodeGlow(id, glowByNode)
      const isGrounded = !isNow && groundedSet.has(id)
      const isLayoutPinned = !isNow && pins[id] !== undefined
      const opacity = nodeOpacityFor(
        node,
        id,
        hoverNeighborhood,
        hasActiveMapFilter,
        matchedNodeIds,
        activeTimeRange !== null,
        inTimeRange
      )
      const x = node.x ?? 0
      const y = node.y ?? 0

      ctx.save()
      ctx.globalAlpha = opacity

      const heat = heatIntensity(id, heatByNodeRef.current)
      if (heat > 0 && glow <= 0) {
        // session trail residue: faint warm disc, decays over ~5 minutes
        ctx.beginPath()
        ctx.arc(x, y, radius + 4 + heat * 3, 0, Math.PI * 2)
        ctx.fillStyle = rgba(HEAT_TRAIL_COLOR, 0.05 + heat * 0.13)
        ctx.fill()
      }

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

      const isQueryMatch = queryActive && node.kind === 'note' && queryMatchedIds.has(id)
      if (isQueryMatch) {
        // search hit: cool halo, distinct from the amber live glows
        const haloRadius = radius + 7
        const gradient = ctx.createRadialGradient(x, y, radius * 0.7, x, y, haloRadius + 4)
        gradient.addColorStop(0, rgba(SEARCH_HALO_COLOR, 0.4))
        gradient.addColorStop(1, rgba(SEARCH_HALO_COLOR, 0))
        ctx.beginPath()
        ctx.arc(x, y, haloRadius + 4, 0, Math.PI * 2)
        ctx.fillStyle = gradient
        ctx.fill()
        ctx.beginPath()
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2)
        ctx.strokeStyle = rgba(SEARCH_HALO_COLOR, 0.75)
        ctx.lineWidth = Math.max(1.2 / globalScale, 0.9)
        ctx.stroke()
      }

      if (flashOpacity > 0) {
        ctx.beginPath()
        ctx.arc(x, y, radius + 7, 0, Math.PI * 2)
        ctx.strokeStyle = rgba('#ffe08a', flashOpacity * 0.78)
        ctx.lineWidth = Math.max(1.4 / globalScale, 1)
        ctx.stroke()
      }

      if (isGrounded) {
        // yellow ring for grounding notes
        ctx.beginPath()
        ctx.arc(x, y, radius + 5, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(245, 197, 24, 0.95)'
        ctx.lineWidth = Math.max(2 / globalScale, 1.2)
        ctx.stroke()
      }

      if (isSelected && !isNow) {
        ctx.beginPath()
        ctx.arc(x, y, radius + 3.5, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)'
        ctx.lineWidth = Math.max(2.2 / globalScale, 1.4)
        ctx.stroke()
      } else if (isHovered && !isNow) {
        ctx.beginPath()
        ctx.arc(x, y, radius + 3, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(236, 235, 230, 0.92)'
        ctx.lineWidth = Math.max(1.6 / globalScale, 1.1)
        ctx.stroke()
      }

      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fillStyle = isNow ? '#f5c518' : nodeFillColor(node)
      ctx.fill()

      ctx.beginPath()
      ctx.arc(x, y, Math.max(2.4, radius * 0.32), 0, Math.PI * 2)
      ctx.fillStyle = isNow ? '#231b06' : '#0b0b0c'
      ctx.fill()

      if (isLayoutPinned) {
        // pin marker: small accent stud on the node's shoulder
        const px = x + radius * 0.78
        const py = y - radius * 0.78
        ctx.beginPath()
        ctx.arc(px, py, Math.max(2, radius * 0.3), 0, Math.PI * 2)
        ctx.fillStyle = '#f5c518'
        ctx.fill()
        ctx.beginPath()
        ctx.arc(px, py, Math.max(2, radius * 0.3), 0, Math.PI * 2)
        ctx.strokeStyle = '#0b0b0c'
        ctx.lineWidth = Math.max(1 / globalScale, 0.7)
        ctx.stroke()
      }

      const alpha =
        isNow || isHovered || isSelected || isQueryMatch
          ? 1
          : labelAlpha(globalScale, node.degree >= hubDegree)
      if (alpha > 0.02) {
        const label = isNow ? nowLabel : node.title
        drawNodeLabel(ctx, label, x, y + radius + 4, isNow, globalScale, alpha)
      }
      ctx.restore()
    },
    [
      flashUntilByNode,
      glowByNode,
      groundedSet,
      hasActiveMapFilter,
      hoverNeighborhood,
      hoverNodeId,
      hubDegree,
      inTimeRange,
      matchedNodeIds,
      nodeFillColor,
      nowLabel,
      pins,
      queryActive,
      queryMatchedIds,
      selectedNodeId,
      activeTimeRange
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
      let opacity: number
      if (isHovered) {
        opacity = Math.max(baseOpacity, 0.85)
      } else if (hoverNeighborhood && hoverNodeId) {
        const srcId = endpointId(link.source)
        const dstId = endpointId(link.target)
        const incident = srcId === hoverNodeId || dstId === hoverNodeId
        opacity = incident ? Math.max(baseOpacity, 0.5) : DIM_OPACITY * baseOpacity * 6
      } else {
        opacity = linkOpacityFor(
          link,
          baseOpacity,
          hasActiveMapFilter,
          matchedNodeIds,
          activeTimeRange !== null,
          inTimeRange
        )
      }
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
    [
      hasActiveMapFilter,
      hoverLinkId,
      hoverNeighborhood,
      hoverNodeId,
      inTimeRange,
      matchedNodeIds,
      activeTimeRange
    ]
  )

  // ── overlay text ───────────────────────────────────────────────────────

  const noteCount = visibleNodes.filter((node) => node.kind === 'note').length
  const matchCount = hasActiveMapFilter ? matchedNodeIds.size : noteCount
  const centerTitle = egoNav ? (titleById.get(egoNav.center) ?? egoNav.center) : ''
  const overlayText = error
    ? error
    : isLoading
      ? 'Loading map…'
      : egoNav
        ? egoLoading
          ? 'Loading neighborhood…'
          : egoError
            ? egoError
            : `${truncateTitle(centerTitle)} · depth ${egoNav.depth} · ${noteCount} notes`
        : noteCount === 0
          ? 'No notes indexed'
          : hasActiveMapFilter
            ? `${matchCount}/${noteCount} matches · ${graphData.links.length} links`
            : `${noteCount} notes · ${graphData.links.length} links`

  const communities = useMemo(() => {
    const counts = new Map<number, number>()
    for (const node of visibleNodes) {
      if (node.kind === 'note') {
        counts.set(node.community, (counts.get(node.community) ?? 0) + 1)
      }
    }
    return [...counts.entries()].sort((left, right) => right[1] - left[1])
  }, [visibleNodes])

  const selectedNode = selectedNodeId ? findDisplayedNode(selectedNodeId) : null

  return (
    <section aria-hidden={!visible} aria-label="Map view" className="map-view">
      <div className="map-controls">
        <MapSearch
          query={query}
          results={searchResults}
          onQueryChange={setQuery}
          onPick={(noteId) => selectNode(noteId, { center: true })}
        />

        <div className="map-source-filters" aria-label="Source filters">
          {sourceIds.map((source) => {
            const active = activeSources.size === 0 || activeSources.has(source)
            const style = { '--map-source-color': sourceColor(source) } as CSSProperties

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

        <div className="map-color-modes" role="group" aria-label="Color mode">
          {(['source', 'community', 'recency'] as const).map((mode) => (
            <button
              key={mode}
              className={`map-seg${colorMode === mode ? ' active' : ''}`}
              type="button"
              aria-pressed={colorMode === mode}
              onClick={() => setColorMode(mode)}
            >
              {mode === 'source' ? 'Source' : mode === 'community' ? 'Clusters' : 'Recency'}
            </button>
          ))}
        </div>

        {!isEgoMode ? (
          <label className="map-slider-control" title="Hide notes with fewer connections">
            <span>Min links</span>
            <input
              type="range"
              min={0}
              max={6}
              step={1}
              value={minDegree}
              onChange={(event) => setMinDegree(Number(event.target.value))}
            />
            <span className="map-slider-value">{minDegree === 0 ? 'all' : `${minDegree}+`}</span>
          </label>
        ) : null}

        {hiddenNodeIds.size > 0 ? (
          <button className="map-chip" type="button" onClick={unhideAll}>
            Unhide {hiddenNodeIds.size}
          </button>
        ) : null}

        {hasActiveMapFilter || activeTimeRange !== null ? (
          <button className="map-btn" type="button" onClick={clearFilters}>
            Clear
          </button>
        ) : null}
      </div>

      {egoNav ? (
        <div className="map-ego-bar" aria-label="Neighborhood navigation">
          <button className="map-btn" type="button" onClick={exitEgo}>
            Global
          </button>
          <button
            className="map-btn"
            type="button"
            disabled={egoNav.history.length === 0}
            onClick={egoBack}
          >
            ← Back
          </button>
          <nav className="map-breadcrumbs" aria-label="Visited notes">
            {egoNav.history.map((noteId, index) => (
              <span className="map-crumb" key={`${noteId}-${index}`}>
                <button type="button" onClick={() => egoJumpTo(index)}>
                  {truncateTitle(titleById.get(noteId) ?? noteId)}
                </button>
                <span aria-hidden="true">›</span>
              </span>
            ))}
            <span className="map-crumb current">{truncateTitle(centerTitle)}</span>
          </nav>
          <label className="map-slider-control" title="Neighborhood depth">
            <span>Depth</span>
            <input
              type="range"
              min={1}
              max={3}
              step={1}
              value={egoNav.depth}
              onChange={(event) => setEgoDepth(Number(event.target.value))}
            />
            <span className="map-slider-value">{egoNav.depth}</span>
          </label>
        </div>
      ) : null}

      <div className={`map-stage${selectedNodeId ? ' has-panel' : ''}`} ref={containerRef}>
        <ForceGraph2D<MapNode, MapLink>
          ref={graphRef}
          autoPauseRedraw={!visible}
          backgroundColor="#0b0b0c"
          cooldownTicks={LAYOUT_COOLDOWN_TICKS}
          d3AlphaDecay={0.025}
          d3VelocityDecay={0.34}
          enableNodeDrag={true}
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
          onBackgroundClick={() => {
            setMenuTarget(null)
            setSelectedNodeId(null)
          }}
          onBackgroundRightClick={() => setMenuTarget(null)}
          onLinkHover={(link) => {
            setHoverLink(link ?? null)
            if (link) {
              setHoverNode(null)
            }
          }}
          onLinkRightClick={(link, event) => {
            if (link.kind === 'static') {
              openMenu(event, { kind: 'edge', link: link as StaticGraphLink })
            }
          }}
          onNodeClick={handleNodeClick}
          onNodeDragEnd={handleNodeDragEnd}
          onNodeHover={(node) => {
            setHoverNode(node ?? null)
            if (node) {
              setHoverLink(null)
            }
          }}
          onNodeRightClick={(node, event) => {
            if (node.kind === 'note') {
              openMenu(event, { kind: 'node', node: node as MapNode })
            }
          }}
          width={size.width}
        />

        {!selectedNodeId ? (
          <MapHoverPanel hoverLink={hoverLink} hoverNode={hoverNode} nowLabel={nowLabel} />
        ) : null}

        <div
          aria-live="polite"
          role={error ? 'alert' : 'status'}
          className={`map-overlay-chip${error ? ' map-overlay-error' : ''}`}
        >
          {overlayText}
        </div>

        <MapLegend
          colorMode={colorMode}
          communities={communities}
          sourceIds={sourceIds}
          timeDomain={timeDomain}
        />

        {menuTarget ? (
          <MapContextMenu
            target={menuTarget}
            isPinned={menuTarget.kind === 'node' && pins[menuTarget.node.id] !== undefined}
            onClose={() => setMenuTarget(null)}
            onOpenNote={onOpenNote}
            onFocus={focusNode}
            onAskAbout={onAskAbout}
            onTogglePin={togglePin}
            onHideNode={hideNode}
          />
        ) : null}

        {selectedNodeId ? (
          <SidePanel
            key={selectedNodeId}
            noteId={selectedNodeId}
            node={selectedNode}
            onClose={() => setSelectedNodeId(null)}
            onFocus={focusNode}
            onAskAbout={onAskAbout}
            onOpenNote={onOpenNote}
            onSelectNeighbor={(noteId) => selectNode(noteId, { center: true })}
          />
        ) : null}
      </div>

      {timeDomain ? (
        <MapTimeline
          domain={timeDomain}
          histogram={histogram}
          range={activeTimeRange}
          onRangeChange={setTimeRange}
        />
      ) : null}
    </section>
  )
}

// ── module helpers ───────────────────────────────────────────────────────

function isNoteNodeObject(value: unknown): value is GraphNodeObject {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  return (value as Partial<MapNode>).kind === 'note'
}

function nodeTime(node: MapNode): number {
  return node.modified > 0 ? node.modified : node.created
}

function queryMatchKind(
  node: MapNode,
  normalizedQuery: string
): 'title' | 'entity' | 'gist' | null {
  if (node.title.toLocaleLowerCase().includes(normalizedQuery)) {
    return 'title'
  }
  if (node.entities.some((entity) => entity.toLocaleLowerCase().includes(normalizedQuery))) {
    return 'entity'
  }
  if (node.gist.toLocaleLowerCase().includes(normalizedQuery)) {
    return 'gist'
  }
  return null
}

function nodeOpacityFor(
  node: GraphNodeObject,
  id: string,
  hoverNeighborhood: Set<string> | null,
  hasActiveMapFilter: boolean,
  matchedNodeIds: Set<string>,
  timeFilterActive: boolean,
  inTimeRange: (node: MapNode) => boolean
): number {
  if (hoverNeighborhood) {
    return hoverNeighborhood.has(id) ? 1 : DIM_OPACITY
  }

  const dimByFilter = hasActiveMapFilter && !(node.kind === 'note' && matchedNodeIds.has(id))
  const dimByTime = timeFilterActive && node.kind === 'note' && !inTimeRange(node)
  return dimByFilter || dimByTime ? DIM_OPACITY : 1
}

function linkOpacityFor(
  link: GraphLinkObject,
  baseOpacity: number,
  hasActiveMapFilter: boolean,
  matchedNodeIds: Set<string>,
  timeFilterActive: boolean,
  inTimeRange: (node: MapNode) => boolean
): number {
  if (!hasActiveMapFilter && !timeFilterActive) {
    return baseOpacity
  }

  const sourceId = endpointId(link.source)
  const targetId = endpointId(link.target)

  if (timeFilterActive) {
    const sourceNode = asMapNode(link.source)
    const targetNode = asMapNode(link.target)
    const sourceOut = sourceNode?.kind === 'note' && !inTimeRange(sourceNode)
    const targetOut = targetNode?.kind === 'note' && !inTimeRange(targetNode)
    if (sourceOut || targetOut) {
      return DIM_OPACITY
    }
  }

  if (!hasActiveMapFilter) {
    return baseOpacity
  }

  if (link.kind === 'static') {
    return sourceId && targetId && matchedNodeIds.has(sourceId) && matchedNodeIds.has(targetId)
      ? baseOpacity
      : DIM_OPACITY
  }

  const noteIds = [sourceId, targetId].filter((id): id is string =>
    Boolean(id && id !== NOW_NODE_ID)
  )
  return noteIds.some((id) => matchedNodeIds.has(id)) ? baseOpacity : DIM_OPACITY
}

function asMapNode(value: unknown): MapNode | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }
  const maybe = value as Partial<MapNode>
  return typeof maybe.id === 'string' && typeof maybe.kind === 'string' ? (maybe as MapNode) : null
}

function nodeFlashOpacity(nodeId: string, flashUntilByNode: Map<string, number>): number {
  const until = flashUntilByNode.get(nodeId)
  if (!until) {
    return 0
  }

  return Math.max(0, Math.min(1, (until - Date.now()) / TOUCH_FLASH_MS))
}

function nodeGlow(nodeId: string, glowByNode: Map<string, GlowEntry>): number {
  const entry = glowByNode.get(nodeId)
  if (!entry) {
    return 0
  }

  const remaining = (entry.until - Date.now()) / LIVE_EDGE_FADE_MS
  return remaining <= 0 ? 0 : Math.min(1, remaining) * entry.intensity
}

function heatIntensity(nodeId: string, heatByNode: Map<string, number>): number {
  const touchedAt = heatByNode.get(nodeId)
  if (touchedAt === undefined) {
    return 0
  }

  const age = Date.now() - touchedAt
  return age >= HEAT_TRAIL_MS ? 0 : 1 - age / HEAT_TRAIL_MS
}

/** Copy simulated coordinates from prior graphs so new data starts in place. */
function seedPositions(data: MapGraphData, ...sources: Array<MapGraphData | null>): MapGraphData {
  const positions = new Map<string, { x: number; y: number }>()
  for (const source of sources) {
    if (!source) {
      continue
    }
    for (const node of source.nodes) {
      if (typeof node.x === 'number' && typeof node.y === 'number' && !positions.has(node.id)) {
        positions.set(node.id, { x: node.x, y: node.y })
      }
    }
  }

  for (const node of data.nodes) {
    const position = positions.get(node.id)
    if (position && node.x === undefined) {
      node.x = position.x
      node.y = position.y
    }
  }
  return data
}

function truncateTitle(title: string, maxLength = 34): string {
  return title.length <= maxLength ? title : `${title.slice(0, maxLength - 1)}…`
}

function clampZoom(graph: ForceGraphMethods<MapNode, MapLink> | undefined): void {
  if (!graph) {
    return
  }
  const zoom = graph.zoom()
  if (typeof zoom === 'number' && zoom > MAX_FIT_ZOOM) {
    graph.zoom(MAX_FIT_ZOOM, 220)
  }
}
