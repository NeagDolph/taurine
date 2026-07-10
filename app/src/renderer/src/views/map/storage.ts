import type { MapGraphData } from '../../lib/graph'

export const PINNED_POSITIONS_KEY = 'taurine.map.pins'
export const COLOR_MODE_KEY = 'taurine.map.colorMode'
export const MIN_DEGREE_KEY = 'taurine.map.minDegree'

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

export function loadMinDegree(storage: Storage = window.localStorage): number {
  try {
    const value = Number(storage.getItem(MIN_DEGREE_KEY))
    return Number.isFinite(value) && value >= 0 && value <= 8 ? Math.round(value) : 0
  } catch {
    return 0
  }
}

export function saveMinDegree(minDegree: number, storage: Storage = window.localStorage): void {
  try {
    if (minDegree <= 0) {
      storage.removeItem(MIN_DEGREE_KEY)
    } else {
      storage.setItem(MIN_DEGREE_KEY, String(minDegree))
    }
  } catch {
    // non-fatal
  }
}
