import type { RetrievalEvent } from './sidecar'

export const NOW_NODE_ID = '__taurine_now__'
export const SIDECAR_GRAPH_URL = 'http://127.0.0.1:8930/graph'
export const LIVE_EDGE_FADE_MS = 3000
export const TOUCH_FLASH_MS = 1800

export type GraphEdgeType = 'semantic' | 'topic' | 'entity' | 'mixed'

export interface GraphNode {
  note_id: string
  title: string
  degree: number
  source: string
}

export interface GraphEdge {
  src: string
  dst: string
  type: GraphEdgeType
  weight: number
  shared: string[]
  fact: string
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export type MapNodeKind = 'note' | 'now'

export interface MapNode {
  id: string
  note_id?: string
  title: string
  label: string
  degree: number
  source: string
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

export interface LiveHopLink {
  id: string
  kind: 'hop'
  source: string
  target: string
  color: string
  fromNoteId: string
  hop: number
  score: number
  createdAt: number
  fadeStartedAt?: number
}

export type LiveGraphLink = LiveRankLink | LiveHopLink
export type MapLink = StaticGraphLink | LiveGraphLink

export interface MapGraphData {
  nodes: MapNode[]
  links: MapLink[]
}

export interface LiveGraphUpdate {
  links: LiveGraphLink[]
  touchedNodeIds: Set<string>
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

export function graphResponseToMapData(response: GraphResponse, nowLabel: string): MapGraphData {
  const nodeIds = new Set(response.nodes.map((node) => node.note_id))
  const nodes: MapNode[] = [
    {
      id: NOW_NODE_ID,
      title: 'Now',
      label: nowLabel || 'Now',
      degree: 0,
      source: 'live',
      kind: 'now',
      val: nodeRadius(9),
      fx: 0,
      fy: -90
    },
    ...response.nodes.map((node) => ({
      id: node.note_id,
      note_id: node.note_id,
      title: node.title,
      label: node.title,
      degree: node.degree,
      source: node.source,
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
      fact: edge.fact
    }))

  return { nodes, links }
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

  for (const expansion of event.expansions) {
    touchedNodeIds.add(expansion.from_note_id)
    touchedNodeIds.add(expansion.note_id)
    links.push({
      id: `hop:${createdAt}:${expansion.from_note_id}:${expansion.note_id}:${expansion.hop}`,
      kind: 'hop',
      source: expansion.from_note_id,
      target: expansion.note_id,
      color: hopColor(expansion.hop),
      fromNoteId: expansion.from_note_id,
      hop: expansion.hop,
      score: expansion.score,
      createdAt
    })
  }

  return { links, touchedNodeIds }
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
  const baseOpacity = link.kind === 'rank' ? 0.82 : 0.52
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

export function hopColor(hop: number): string {
  // dashed expansion edges: warm amber fading to muted slate as hops grow
  return mixHex('#e5b23c', '#5b6577', clamp((hop - 1) / 3, 0, 1))
}

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

export function linkTooltip(link: MapLink): string {
  if (link.kind === 'static') {
    const terms = link.shared.length > 0 ? `: ${link.shared.join(', ')}` : ''
    const base = `${link.edgeType}${terms}`
    return link.fact ? `${link.fact}\n(${base})` : base
  }

  if (link.kind === 'rank') {
    return `retrieval rank ${link.rank}`
  }

  return `graph expansion hop ${link.hop}`
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
    source: typeof record.source === 'string' ? record.source : 'apple_notes'
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
    fact: typeof record.fact === 'string' ? record.fact : ''
  }
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
  imessage: '#5fd18d',
  obsidian: '#b98cff'
}

const SOURCE_LABELS: Record<string, string> = {
  apple_notes: 'Apple Notes',
  claude_mem: 'Claude',
  codex_mem: 'Codex',
  imessage: 'iMessage',
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
