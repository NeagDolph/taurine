import type { MapGraphData } from '../../lib/graph'

export const PINNED_POSITIONS_KEY = 'taurine.map.pins'
export const COLOR_MODE_KEY = 'taurine.map.colorMode'
export const MIN_DEGREE_KEY = 'taurine.map.minDegree'
export const PANEL_WIDTH_KEY = 'taurine.map.panelWidth'

export const PANEL_MIN_WIDTH = 280
export const PANEL_MAX_WIDTH = 640
export const PANEL_DEFAULT_WIDTH = 332

export type ColorMode = 'source' | 'community' | 'recency'

export interface PinnedPosition {
  x: number
  y: number
}

export type PinnedPositions = Record<string, PinnedPosition>

export function loadPinnedPositions(storage: Storage = window.localStorage): PinnedPositions {
  try {
    const raw = storage.getItem(PINNED_POSITIONS_KEY)
    if (!raw) {
      return {}
    }

    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return {}
    }

    const pins: PinnedPositions = {}
    for (const [noteId, value] of Object.entries(parsed as Record<string, unknown>)) {
      const record = value as { x?: unknown; y?: unknown } | null
      if (
        record &&
        typeof record === 'object' &&
        typeof record.x === 'number' &&
        Number.isFinite(record.x) &&
        typeof record.y === 'number' &&
        Number.isFinite(record.y)
      ) {
        pins[noteId] = { x: record.x, y: record.y }
      }
    }
    return pins
  } catch {
    return {}
  }
}

export function savePinnedPositions(
  pins: PinnedPositions,
  storage: Storage = window.localStorage
): void {
  try {
    if (Object.keys(pins).length === 0) {
      storage.removeItem(PINNED_POSITIONS_KEY)
    } else {
      storage.setItem(PINNED_POSITIONS_KEY, JSON.stringify(pins))
    }
  } catch {
    // storage full / unavailable: pins simply do not persist
  }
}

/** Rehydrate saved layout pins onto freshly fetched node objects. */
export function applyPinnedPositions(data: MapGraphData, pins: PinnedPositions): MapGraphData {
  for (const node of data.nodes) {
    if (node.kind !== 'note') {
      continue
    }

    const pin = pins[node.id]
    if (pin) {
      node.fx = pin.x
      node.fy = pin.y
      if (node.x === undefined) {
        node.x = pin.x
        node.y = pin.y
      }
    }
  }
  return data
}

export function loadColorMode(storage: Storage = window.localStorage): ColorMode {
  try {
    const raw = storage.getItem(COLOR_MODE_KEY)
    return raw === 'community' || raw === 'recency' ? raw : 'source'
  } catch {
    return 'source'
  }
}

export function saveColorMode(mode: ColorMode, storage: Storage = window.localStorage): void {
  try {
    storage.setItem(COLOR_MODE_KEY, mode)
  } catch {
    // non-fatal
  }
}

export const MIN_DEGREE_DEFAULT = 2

export function loadMinDegree(storage: Storage = window.localStorage): number {
  try {
    const raw = storage.getItem(MIN_DEGREE_KEY)
    if (raw === null) {
      return MIN_DEGREE_DEFAULT
    }
    const value = Number(raw)
    return Number.isFinite(value) && value >= 0 && value <= 8
      ? Math.round(value)
      : MIN_DEGREE_DEFAULT
  } catch {
    return MIN_DEGREE_DEFAULT
  }
}

export function saveMinDegree(minDegree: number, storage: Storage = window.localStorage): void {
  try {
    // always store explicitly — with a non-zero default, "all" (0) must persist
    storage.setItem(MIN_DEGREE_KEY, String(Math.round(minDegree)))
  } catch {
    // non-fatal
  }
}

/** Clamp a side-panel width to [min, max]; `max` can shrink with the stage. */
export function clampPanelWidth(width: number, max: number = PANEL_MAX_WIDTH): number {
  const upper = Math.max(PANEL_MIN_WIDTH, Math.min(PANEL_MAX_WIDTH, max))
  return Math.round(Math.min(Math.max(width, PANEL_MIN_WIDTH), upper))
}

export function loadPanelWidth(storage: Storage = window.localStorage): number {
  try {
    const value = Number(storage.getItem(PANEL_WIDTH_KEY))
    return Number.isFinite(value) && value > 0 ? clampPanelWidth(value) : PANEL_DEFAULT_WIDTH
  } catch {
    return PANEL_DEFAULT_WIDTH
  }
}

export function savePanelWidth(width: number, storage: Storage = window.localStorage): void {
  try {
    const rounded = Math.round(width)
    if (rounded === PANEL_DEFAULT_WIDTH) {
      storage.removeItem(PANEL_WIDTH_KEY)
    } else {
      storage.setItem(PANEL_WIDTH_KEY, String(rounded))
    }
  } catch {
    // non-fatal
  }
}
