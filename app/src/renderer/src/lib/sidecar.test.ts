import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SidecarClient } from './sidecar'

class FakeWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  static instances: FakeWebSocket[] = []

  binaryType = ''
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((message: { data: unknown }) => void) | null = null
  onopen: (() => void) | null = null
  readyState = FakeWebSocket.CONNECTING
  readonly sent: unknown[] = []

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this)
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.()
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.()
  }

  send(value: unknown): void {
    this.sent.push(value)
  }
}

describe('SidecarClient reconnect', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeWebSocket.instances = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal('window', globalThis)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('replays the complete last start message after reconnecting', () => {
    const client = new SidecarClient('ws://127.0.0.1:8930/session')
    client.connect()
    const first = FakeWebSocket.instances[0]
    first.open()

    client.start(
      [
        { id: 0, label: 'mic' },
        { id: 1, label: 'system' }
      ],
      'meeting-42'
    )
    first.close()
    vi.advanceTimersByTime(500)

    const second = FakeWebSocket.instances[1]
    second.open()

    expect(second.sent).toEqual([
      JSON.stringify({
        type: 'start',
        channels: [
          { id: 0, label: 'mic' },
          { id: 1, label: 'system' }
        ],
        meeting_id: 'meeting-42'
      })
    ])
  })
})
