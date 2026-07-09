import {
  SIDECAR_WS_URL,
  parseConnectorInfo,
  parseProfileState,
  parseReenrichStatus
} from './sidecar'
import type { ConnectorInfo, ProfileState, ReenrichStatus } from './sidecar'

// Long-lived state channel: the sidecar pushes connector/sync/re-enrich/graph/
// profile changes here so the renderer never polls REST endpoints. REST is
// only used for actions and for fetching the (large) graph payload when the
// pushed graph version changes.

export const UPDATES_WS_URL = SIDECAR_WS_URL.replace(/\/session$/, '/updates')

export interface ConnectorsUpdate {
  type: 'connectors'
  connectors: ConnectorInfo[]
}

export interface ReenrichUpdate {
  type: 'reenrich'
  status: ReenrichStatus
}

export interface GraphUpdate {
  type: 'graph'
  version: number
}

export interface ProfileUpdate extends ProfileState {
  type: 'profile'
}

export type UpdateEvent = ConnectorsUpdate | ReenrichUpdate | GraphUpdate | ProfileUpdate

type Callback = (event: UpdateEvent) => void
type Unsubscribe = () => void

const INITIAL_RECONNECT_MS = 500
const MAX_RECONNECT_MS = 8000

export class UpdatesClient {
  private socket: WebSocket | null = null
  private reconnectTimer: number | null = null
  private reconnectAttempt = 0
  private closedByClient = false
  private readonly callbacks = new Set<Callback>()

  constructor(private readonly url: string = UPDATES_WS_URL) {}

  connect(): void {
    if (
      this.socket?.readyState === WebSocket.OPEN ||
      this.socket?.readyState === WebSocket.CONNECTING
    ) {
      return
    }

    this.closedByClient = false
    this.clearReconnectTimer()
    this.openSocket()
  }

  disconnect(): void {
    this.closedByClient = true
    this.clearReconnectTimer()
    this.socket?.close()
    this.socket = null
  }

  onUpdate(callback: Callback): Unsubscribe {
    this.callbacks.add(callback)
    return () => this.callbacks.delete(callback)
  }

  private openSocket(): void {
    const socket = new WebSocket(this.url)
    this.socket = socket

    socket.onopen = (): void => {
      this.reconnectAttempt = 0
    }

    socket.onmessage = (message): void => {
      if (typeof message.data !== 'string') {
        return
      }

      const event = parseUpdateEvent(message.data)
      if (event) {
        this.callbacks.forEach((callback) => callback(event))
      }
    }

    socket.onclose = (): void => {
      if (this.socket === socket) {
        this.socket = null
      }

      if (!this.closedByClient) {
        this.scheduleReconnect()
      }
    }

    socket.onerror = (): void => {
      socket.close()
    }
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer()
    this.reconnectAttempt += 1

    const delay = Math.min(
      MAX_RECONNECT_MS,
      INITIAL_RECONNECT_MS * 2 ** (this.reconnectAttempt - 1)
    )
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null
      this.openSocket()
    }, delay)
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }
}

export function parseUpdateEvent(raw: string): UpdateEvent | null {
  let value: unknown

  try {
    value = JSON.parse(raw)
  } catch {
    return null
  }

  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null
  }

  const record = value as Record<string, unknown>
  switch (record.type) {
    case 'connectors': {
      if (!Array.isArray(record.connectors)) {
        return null
      }
      try {
        return {
          type: 'connectors',
          connectors: record.connectors.map(parseConnectorInfo)
        }
      } catch {
        return null
      }
    }
    case 'reenrich': {
      const status = parseReenrichStatus(record.status)
      return status ? { type: 'reenrich', status } : null
    }
    case 'graph': {
      return typeof record.version === 'number' ? { type: 'graph', version: record.version } : null
    }
    case 'profile': {
      const profile = parseProfileState(record)
      return profile ? { type: 'profile', ...profile } : null
    }
    default:
      return null
  }
}
