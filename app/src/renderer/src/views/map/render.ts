import type { MapNode } from '../../lib/graph'

// zoom (globalScale) windows over which node labels fade in: hubs surface
// first so a zoomed-out map reads as landmarks instead of a text mass
export const HUB_LABEL_FADE: readonly [number, number] = [0.8, 1.4]
export const NODE_LABEL_FADE: readonly [number, number] = [1.6, 2.7]

export function rgba(hex: string, alpha: number): string {
  const value = hex.replace('#', '')
  const r = Number.parseInt(value.slice(0, 2), 16)
  const g = Number.parseInt(value.slice(2, 4), 16)
  const b = Number.parseInt(value.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`
}

/**
 * Darkened stroke color derived from a node fill — used for the subtle
 * outline ring on the hovered node's neighbors. Falls back to translucent
 * black when the input is not a 6-digit hex color.
 */
export function darkenColor(color: string, factor: number): string {
  const match = /^#([0-9a-f]{6})$/i.exec(color)
  if (!match) {
    return 'rgba(0, 0, 0, 0.55)'
  }
  const value = match[1]
  const r = Math.round(Number.parseInt(value.slice(0, 2), 16) * factor)
  const g = Math.round(Number.parseInt(value.slice(2, 4), 16) * factor)
  const b = Math.round(Number.parseInt(value.slice(4, 6), 16) * factor)
  return `rgba(${r}, ${g}, ${b}, 0.9)`
}

export function roundedRect(
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

export function truncateLabel(label: string, maxLength: number): string {
  return label.length <= maxLength ? label : `${label.slice(0, maxLength - 3)}...`
}

export function drawNodeLabel(
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
export function labelAlpha(globalScale: number, isHub: boolean): number {
  const [start, end] = isHub ? HUB_LABEL_FADE : NODE_LABEL_FADE
  return Math.max(0, Math.min(1, (globalScale - start) / (end - start)))
}

export function degreePercentile(nodes: MapNode[], percentile: number): number {
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

export function endpoint(value: unknown): { x: number; y: number } | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const maybeNode = value as { x?: unknown; y?: unknown }
  return typeof maybeNode.x === 'number' && typeof maybeNode.y === 'number'
    ? { x: maybeNode.x, y: maybeNode.y }
    : null
}

export function endpointId(value: unknown): string | null {
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

  return String(maybeNode.id as string | number)
}

export function endpointNode(value: unknown): MapNode | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const maybeNode = value as Partial<MapNode>
  return typeof maybeNode.id === 'string' && typeof maybeNode.title === 'string'
    ? (maybeNode as MapNode)
    : null
}

const DAY_MS = 24 * 60 * 60 * 1000

export function formatTimestamp(unixSeconds: number): string {
  if (unixSeconds <= 0) {
    return ''
  }

  const date = new Date(unixSeconds * 1000)
  const withYear = Date.now() - date.getTime() > 300 * DAY_MS
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    ...(withYear ? { year: 'numeric' } : {})
  })
}
