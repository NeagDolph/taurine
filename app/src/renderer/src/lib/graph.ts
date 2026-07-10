import { SIDECAR_PORT, type RetrievalEvent } from './sidecar'

export const NOW_NODE_ID = '__taurine_now__'
export const SIDECAR_GRAPH_URL = `http://127.0.0.1:${SIDECAR_PORT}/graph`
export const SIDECAR_GRAPH_LOCAL_URL = `http://127.0.0.1:${SIDECAR_PORT}/graph/local`
export const SIDECAR_FACTS_URL = `http://127.0.0.1:${SIDECAR_PORT}/facts`
export const LIVE_EDGE_FADE_MS = 3000
export const TOUCH_FLASH_MS = 1800
export const FACTS_FETCH_CAP = 50

export type GraphEdgeType = 'semantic' | 'topic' | 'entity' | 'mixed'

export interface GraphNode {
  note_id: string
  title: string
  degree: number
  source: string
  gist: string
  /** unix seconds, 0 when unknown */
  created: number
  /** unix seconds, 0 when unknown */
  modified: number
  folder: string
  /** stable label-propagation community id, -1 = none */
  community: number
  value_score: number
  /** top traversal-eligible canonical entity labels */
  entities: string[]
}

export interface GraphEdge {
  src: string
  dst: string
  type: GraphEdgeType
  weight: number
  shared: string[]
  fact: string
  fact_provenance: string
  fact_ids: string[]
  /** {"semantic": w, "entity": w, "fact": w} */
  mechanisms: Record<string, number>
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface LocalGraphResponse extends GraphResponse {
  center: string
  depth: number
}

export interface FactRecord {
  fact_id: string
  subject: string
  predicate: string
  object: string
  source_note_id: string
  confidence: number
  observed_at: number
}

export type MapNodeKind = 'note' | 'now'

export interface MapNode {
  id: string
  note_id?: string
  title: string
  label: string
  degree: number
  source: string
  gist: string
  created: number
  modified: number
  folder: string
  community: number
  value_score: number
  entities: string[]
  kind: MapNodeKind
  val: number
  fx?: number
  fy?: number
  x?: number
  y?: number
  vx?: number
  vy?: number
}

export interface StaticGraphLink {
  id: string
  kind: 'static'
  source: string
  target: string
  edgeType: GraphEdgeType
  weight: number
  shared: string[]
  fact: string
  factProvenance: string
  factIds: string[]
  mechanisms: Record<string, number>
}

export interface LiveRankLink {
  id: string
  kind: 'rank'
  source: string
  target: string
  color: string
  rank: number
  score: number
  createdAt: number
  fadeStartedAt?: number
}

export type LiveGraphLink = LiveRankLink
export type MapLink = StaticGraphLink | LiveGraphLink

export interface MapGraphData {
  nodes: MapNode[]
  links: MapLink[]
}

export interface LiveGraphUpdate {
  links: LiveGraphLink[]
  touchedNodeIds: Set<string>
  /** PPR traversal arrivals: note_id → normalized mass (0..1] within this event */
  traversalGlow: Map<string, number>
}

export async function fetchGraph(
  url: string = SIDECAR_GRAPH_URL,
  fetcher: typeof fetch = globalThis.fetch
): Promise<GraphResponse> {
  const response = await fetcher(url)
  if (!response.ok) {
    throw new Error(`Graph fetch failed with HTTP ${response.status}`)
  }

  return parseGraphResponse(await response.json())
}

export function parseGraphResponse(value: unknown): GraphResponse {
  const record = asRecord(value)
  if (!record || !Array.isArray(record.nodes) || !Array.isArray(record.edges)) {
    throw new Error('Invalid graph response')
  }

  const nodes = record.nodes.map(parseGraphNode)
  const edges = record.edges.map(parseGraphEdge)

  return { nodes, edges }
}

export async function fetchLocalGraph(
  center: string,
  depth: number,
  url: string = SIDECAR_GRAPH_LOCAL_URL,
  fetcher: typeof fetch = globalThis.fetch
): Promise<LocalGraphResponse> {
  const boundedDepth = Math.max(1, Math.min(3, Math.round(depth)))
  const response = await fetcher(
    `${url}?center=${encodeURIComponent(center)}&depth=${boundedDepth}`
  )
  if (!response.ok) {
    throw new Error(`Local graph fetch failed with HTTP ${response.status}`)
  }

  return parseLocalGraphResponse(await response.json(), center, boundedDepth)
}

export function parseLocalGraphResponse(
  value: unknown,
  fallbackCenter: string,
  fallbackDepth: number
): LocalGraphResponse {
  const record = asRecord(value)
  const graph = parseGraphResponse(value)
  return {
    ...graph,
    center: typeof record?.center === 'string' ? record.center : fallbackCenter,
    depth: typeof record?.depth === 'number' ? record.depth : fallbackDepth
  }
}

export async function fetchFacts(
  ids: string[],
  url: string = SIDECAR_FACTS_URL,
  fetcher: typeof fetch = globalThis.fetch
): Promise<FactRecord[]> {
  const unique = [...new Set(ids.filter((id) => id.length > 0))].slice(0, FACTS_FETCH_CAP)
  if (unique.length === 0) {
    return []
  }

  const response = await fetcher(`${url}?ids=${encodeURIComponent(unique.join(','))}`)
  if (!response.ok) {
    throw new Error(`Facts fetch failed with HTTP ${response.status}`)
  }

  return parseFactsResponse(await response.json())
}

export function parseFactsResponse(value: unknown): FactRecord[] {
  const record = asRecord(value)
  if (!record || !Array.isArray(record.facts)) {
    return []
  }

  const facts: FactRecord[] = []
  for (const row of record.facts) {
    const fact = parseFactRecord(row)
    if (fact) {
      facts.push(fact)
    }
  }
  return facts
}

export interface GraphToMapOptions {
  /** include the pinned live "Now" node (global map); ego graphs omit it */
  includeNow?: boolean
}

export function graphResponseToMapData(
  response: GraphResponse,
  nowLabel: string,
  options: GraphToMapOptions = {}
): MapGraphData {
  const includeNow = options.includeNow !== false
  const nodeIds = new Set(response.nodes.map((node) => node.note_id))
  const nowNodes: MapNode[] = includeNow
    ? [
        {
          id: NOW_NODE_ID,
          title: 'Now',
          label: nowLabel || 'Now',
          degree: 0,
          source: 'live',
          gist: '',
          created: 0,
          modified: 0,
          folder: '',
          community: -1,
          value_score: 0,
          entities: [],
          kind: 'now',
          val: nodeRadius(9),
          fx: 0,
          fy: -90
        }
      ]
    : []
  const nodes: MapNode[] = [
    ...nowNodes,
    ...response.nodes.map((node) => ({
      id: node.note_id,
      note_id: node.note_id,
      title: node.title,
      label: node.title,
      degree: node.degree,
      source: node.source,
      gist: node.gist,
      created: node.created,
      modified: node.modified,
      folder: node.folder,
      community: node.community,
      value_score: node.value_score,
      entities: [...node.entities],
      kind: 'note' as const,
      val: nodeRadius(node.degree)
    }))
  ]

  const links: StaticGraphLink[] = response.edges
    .filter((edge) => nodeIds.has(edge.src) && nodeIds.has(edge.dst))
    .map((edge) => ({
      id: `static:${edge.src}:${edge.dst}:${edge.type}`,
      kind: 'static',
      source: edge.src,
      target: edge.dst,
      edgeType: edge.type,
      weight: edge.weight,
      shared: [...edge.shared],
      fact: edge.fact,
      factProvenance: edge.fact_provenance,
      factIds: [...edge.fact_ids],
      mechanisms: { ...edge.mechanisms }
    }))

  return { nodes, links }
}

/**
 * Client-side BFS fallback for ego mode: builds a local subgraph from the
 * already-loaded global map when GET /graph/local is unavailable (older
 * backend). Node objects are cloned so ego-layout simulation does not disturb
 * the global layout; positions are copied as starting seeds.
 */
export function localSubgraphFromMapData(
  data: MapGraphData,
  center: string,
  depth: number
): MapGraphData {
  const adjacency = new Map<string, Set<string>>()
  const staticLinks = data.links.filter((link): link is StaticGraphLink => link.kind === 'static')
  for (const link of staticLinks) {
    const src = linkEndpointId(link.source)
    const dst = linkEndpointId(link.target)
    if (!src || !dst) continue
    if (!adjacency.has(src)) adjacency.set(src, new Set())
    if (!adjacency.has(dst)) adjacency.set(dst, new Set())
    adjacency.get(src)?.add(dst)
    adjacency.get(dst)?.add(src)
  }

  const included = new Set<string>([center])
  let frontier = [center]
  for (let hop = 0; hop < depth; hop += 1) {
    const next: string[] = []
    for (const nodeId of frontier) {
      for (const neighbor of adjacency.get(nodeId) ?? []) {
        if (!included.has(neighbor)) {
          included.add(neighbor)
          next.push(neighbor)
        }
      }
    }
    frontier = next
  }

  const nodes = data.nodes
    .filter((node) => node.kind === 'note' && included.has(node.id))
    .map((node) => ({ ...node }))
  const links: StaticGraphLink[] = []
  for (const link of staticLinks) {
    const src = linkEndpointId(link.source)
    const dst = linkEndpointId(link.target)
    if (!src || !dst || !included.has(src) || !included.has(dst)) {
      continue
    }
    links.push({
      ...link,
      source: src,
      target: dst,
      shared: [...link.shared],
      factIds: [...link.factIds],
      mechanisms: { ...link.mechanisms }
    })
  }

  return { nodes, links }
}

/** Endpoint id from a link source/target that force-graph may have objectified. */
export function linkEndpointId(value: unknown): string | null {
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value)
  }
  const record = asRecord(value)
  if (!record || record.id === undefined) {
    return null
  }
  return String(record.id as string | number)
}

/**
 * Merge a freshly fetched graph into the previous one, reusing existing node
 * objects so their simulated x/y/vx/vy survive. Without this, every re-sync
 * refetch replaced all node objects and d3 re-seeded the whole corpus into
 * its phyllotaxis spiral — which reads as concentric rings per connector,
 * because /graph returns notes ordered by namespaced note_id.
 */
export function mergeGraphPositions(previous: MapGraphData, next: MapGraphData): MapGraphData {
  const prevById = new Map(previous.nodes.map((node) => [node.id, node]))

  return {
    nodes: next.nodes.map((node) => {
      const existing = prevById.get(node.id)
      if (!existing) {
        return node
      }
      // fresh nodes carry no x/y keys, so assign refreshes titles/degree/val
      // while the simulation coordinates on the reused object stay put
      return Object.assign(existing, node)
    }),
    links: next.links
  }
}

export function nowLabelFromText(text: string, wordCount = 6): string {
  const words = text.trim().split(/\s+/).filter(Boolean)
  return words.length === 0 ? 'Now' : words.slice(-wordCount).join(' ')
}

export function liveGraphFromRetrieval(
  event: RetrievalEvent,
  createdAt = Date.now()
): LiveGraphUpdate {
  const touchedNodeIds = new Set<string>()
  const totalRanks = Math.max(1, event.hits.length)
  const links: LiveGraphLink[] = []

  for (const hit of event.hits) {
    touchedNodeIds.add(hit.note_id)
    links.push({
      id: `rank:${createdAt}:${hit.rank}:${hit.note_id}`,
      kind: 'rank',
      source: NOW_NODE_ID,
      target: hit.note_id,
      color: rankColor(hit.rank, totalRanks),
      rank: hit.rank,
      score: hit.score,
      createdAt
    })
  }

  // PPR traversal arrivals (spec 2026-07-08) carry no hop chain — render them
  // as a mass-weighted glow on the arrived-at node instead of edges
  const traversalGlow = new Map<string, number>()
  const maxMass = event.expansions.reduce((max, expansion) => Math.max(max, expansion.score), 0)
  for (const expansion of event.expansions) {
    if (!expansion.note_id) {
      continue
    }
    touchedNodeIds.add(expansion.note_id)
    const intensity = maxMass > 0 ? clamp(expansion.score / maxMass, 0.25, 1) : 0.25
    traversalGlow.set(
      expansion.note_id,
      Math.max(intensity, traversalGlow.get(expansion.note_id) ?? 0)
    )
  }

  return { links, touchedNodeIds, traversalGlow }
}

export function fadingLiveLinks(
  links: LiveGraphLink[],
  fadeStartedAt = Date.now()
): LiveGraphLink[] {
  return links.map((link) => ({
    ...link,
    fadeStartedAt: link.fadeStartedAt ?? fadeStartedAt
  }))
}

export function liveLinkOpacity(link: LiveGraphLink, now = Date.now()): number {
  const baseOpacity = 0.82
  if (link.fadeStartedAt === undefined) {
    return baseOpacity
  }

  const progress = clamp((now - link.fadeStartedAt) / LIVE_EDGE_FADE_MS, 0, 1)
  return round2(baseOpacity * (1 - progress))
}

export function staticLinkOpacity(weight: number): number {
  return round2(clamp(0.06 + weight * 0.24, 0.06, 0.32))
}

export function nodeRadius(degree: number): number {
  return round2(clamp(5 + Math.sqrt(Math.max(0, degree)) * 2.1, 5, 16))
}

export function rankColor(rank: number, totalRanks: number): string {
  // rank-hot = yellow accent, rank-cold = cool gray-blue
  const t = totalRanks <= 1 ? 0 : clamp((rank - 1) / (totalRanks - 1), 0, 1)
  return mixHex('#ffd75e', '#5f7690', t)
}

export function communityColor(community: number): string {
  if (community < 0) {
    return COMMUNITY_NONE_COLOR
  }

  return COMMUNITY_PALETTE[community % COMMUNITY_PALETTE.length]
}

/**
 * Recency lens with an exponential window: recency differences among the most
 * recent notes dominate the palette (gray → dark yellow → bright warm accent)
 * while everything older than roughly half the time range compresses into
 * near-identical muted grays. Unknown timestamps are neutral gray.
 */
export function recencyColor(timestamp: number, minTs: number, maxTs: number): string {
  if (timestamp <= 0 || maxTs <= minTs) {
    return RECENCY_UNKNOWN_COLOR
  }

  const age = clamp((maxTs - timestamp) / (maxTs - minTs), 0, 1)
  return mixStops(RECENCY_STOPS, recencyWeight(age))
}

export const RECENCY_OLD_COLOR = '#46484d'
export const RECENCY_MID_COLOR = '#8a7a3a'
export const RECENCY_NEW_COLOR = '#ffb454'
export const RECENCY_UNKNOWN_COLOR = '#55585f'

// tau = span / 6.5 — the newest ~20% of the time range covers ~75% of the
// color distance; ages beyond ~half the span land within a few percent of 0.
const RECENCY_TAU = 1 / 6.5
const RECENCY_WEIGHT_FLOOR = Math.exp(-1 / RECENCY_TAU)

/** Maps normalized age (0 = newest, 1 = oldest) to a 0..1 palette weight. */
function recencyWeight(age: number): number {
  const w = (Math.exp(-age / RECENCY_TAU) - RECENCY_WEIGHT_FLOOR) / (1 - RECENCY_WEIGHT_FLOOR)
  return clamp(w, 0, 1)
}

const RECENCY_STOPS: Array<[number, string]> = [
  [0, RECENCY_OLD_COLOR],
  [0.55, RECENCY_MID_COLOR],
  [1, RECENCY_NEW_COLOR]
]

/**
 * CSS gradient stop list for the recency legend, sampled along a *linear*
 * time axis so the gradient itself shows the exponential window: mostly
 * gray, with the bright ramp compressed against the right edge.
 */
export const RECENCY_LEGEND_GRADIENT = [0, 0.45, 0.65, 0.8, 0.88, 0.94, 1]
  .map((t) => `${mixStops(RECENCY_STOPS, recencyWeight(1 - t))} ${Math.round(t * 100)}%`)
  .join(', ')

function mixStops(stops: Array<[number, string]>, t: number): string {
  for (let i = 1; i < stops.length; i++) {
    const [pos, color] = stops[i]
    if (t <= pos) {
      const [prevPos, prevColor] = stops[i - 1]
      const local = pos <= prevPos ? 1 : (t - prevPos) / (pos - prevPos)
      return mixHex(prevColor, color, local)
    }
  }

  return stops[stops.length - 1][1]
}

const COMMUNITY_NONE_COLOR = '#5a5e66'
const COMMUNITY_PALETTE = [
  '#7aa7ff',
  '#5fd18d',
  '#f0a35e',
  '#c99cff',
  '#6dd6d6',
  '#ff8da1',
  '#e4d96f',
  '#8edfd2',
  '#ff9d6b',
  '#9cc7d8',
  '#b7e07a',
  '#d98cd6',
  '#7fc4ff',
  '#f2b8c6',
  '#a3a8f0',
  '#8fd6a8'
]

export function sourceColor(source: string): string {
  const knownSourceColor = SOURCE_COLORS[source]
  if (knownSourceColor) {
    return knownSourceColor
  }

  return SOURCE_PALETTE[hashString(source) % SOURCE_PALETTE.length]
}

export function sourceLabel(source: string): string {
  const knownSourceLabel = SOURCE_LABELS[source]
  if (knownSourceLabel) {
    return knownSourceLabel
  }

  return source
    .split(/[_:-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toLocaleUpperCase() + part.slice(1))
    .join(' ')
}

export function sourcesFromMapData(data: MapGraphData): string[] {
  const sources = new Set(
    data.nodes.filter((node) => node.kind === 'note').map((node) => node.source || 'apple_notes')
  )

  return [...sources].sort((left, right) => {
    if (left === 'apple_notes') {
      return -1
    }
    if (right === 'apple_notes') {
      return 1
    }
    return sourceLabel(left).localeCompare(sourceLabel(right))
  })
}

function parseGraphNode(value: unknown): GraphNode {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.note_id !== 'string' ||
    typeof record.title !== 'string' ||
    typeof record.degree !== 'number'
  ) {
    throw new Error('Invalid graph node')
  }

  return {
    note_id: record.note_id,
    title: record.title,
    degree: record.degree,
    source: typeof record.source === 'string' ? record.source : 'apple_notes',
    gist: typeof record.gist === 'string' ? record.gist : '',
    created: typeof record.created === 'number' ? record.created : 0,
    modified: typeof record.modified === 'number' ? record.modified : 0,
    folder: typeof record.folder === 'string' ? record.folder : '',
    community: typeof record.community === 'number' ? Math.trunc(record.community) : -1,
    value_score: typeof record.value_score === 'number' ? record.value_score : 0,
    entities: stringArray(record.entities)
  }
}

function parseGraphEdge(value: unknown): GraphEdge {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.src !== 'string' ||
    typeof record.dst !== 'string' ||
    !isGraphEdgeType(record.type) ||
    typeof record.weight !== 'number' ||
    !Array.isArray(record.shared) ||
    !record.shared.every((term) => typeof term === 'string')
  ) {
    throw new Error('Invalid graph edge')
  }

  return {
    src: record.src,
    dst: record.dst,
    type: record.type,
    weight: record.weight,
    shared: record.shared,
    fact: typeof record.fact === 'string' ? record.fact : '',
    fact_provenance: typeof record.fact_provenance === 'string' ? record.fact_provenance : '',
    fact_ids: stringArray(record.fact_ids),
    mechanisms: numberRecord(record.mechanisms)
  }
}

function parseFactRecord(value: unknown): FactRecord | null {
  const record = asRecord(value)
  if (!record || typeof record.fact_id !== 'string') {
    return null
  }

  return {
    fact_id: record.fact_id,
    subject: typeof record.subject === 'string' ? record.subject : '',
    predicate: typeof record.predicate === 'string' ? record.predicate : '',
    object: typeof record.object === 'string' ? record.object : '',
    source_note_id: typeof record.source_note_id === 'string' ? record.source_note_id : '',
    confidence: typeof record.confidence === 'number' ? record.confidence : 0,
    observed_at: typeof record.observed_at === 'number' ? record.observed_at : 0
  }
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

function numberRecord(value: unknown): Record<string, number> {
  const record = asRecord(value)
  if (!record) {
    return {}
  }

  const result: Record<string, number> = {}
  for (const [key, entry] of Object.entries(record)) {
    if (typeof entry === 'number') {
      result[key] = entry
    }
  }
  return result
}

function isGraphEdgeType(value: unknown): value is GraphEdgeType {
  return value === 'semantic' || value === 'topic' || value === 'entity' || value === 'mixed'
}

function mixHex(start: string, end: string, t: number): string {
  const a = hexToRgb(start)
  const b = hexToRgb(end)
  return rgbToHex({
    r: Math.round(a.r + (b.r - a.r) * t),
    g: Math.round(a.g + (b.g - a.g) * t),
    b: Math.round(a.b + (b.b - a.b) * t)
  })
}

const SOURCE_COLORS: Record<string, string> = {
  apple_notes: '#7aa7ff',
  claude_mem: '#ff8da1',
  codex_mem: '#6dd6d6',
  gmail: '#ffffff',
  imessage: '#5fd18d',
  local_folders: '#f0a35e',
  obsidian: '#b98cff'
}

const SOURCE_LABELS: Record<string, string> = {
  apple_notes: 'Apple Notes',
  claude_mem: 'Claude',
  codex_mem: 'Codex',
  imessage: 'iMessage',
  local_folders: 'Folders',
  obsidian: 'Obsidian'
}

const SOURCE_PALETTE = ['#8ab4ff', '#8edfd2', '#c99cff', '#ff99ad', '#79d7a2', '#9cc7d8']

function hashString(value: string): number {
  let hash = 0
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0
  }
  return hash
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const value = hex.replace('#', '')
  return {
    r: Number.parseInt(value.slice(0, 2), 16),
    g: Number.parseInt(value.slice(2, 4), 16),
    b: Number.parseInt(value.slice(4, 6), 16)
  }
}

function rgbToHex(rgb: { r: number; g: number; b: number }): string {
  return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`
}

function toHex(value: number): string {
  return clamp(Math.round(value), 0, 255).toString(16).padStart(2, '0')
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null
  }
  return value as Record<string, unknown>
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function round2(value: number): number {
  return Math.round(value * 100) / 100
}
