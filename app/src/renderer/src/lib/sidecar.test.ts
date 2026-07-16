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

  it('sends the current notepad markdown with its meeting id', () => {
    const client = new SidecarClient('ws://127.0.0.1:8930/session')
    client.connect()
    const socket = FakeWebSocket.instances[0]
    socket.open()

    client.sendNotepad('meeting-42', '# Decisions\n\nShip it.')

    expect(socket.sent).toEqual([
      JSON.stringify({
        type: 'notepad',
        meeting_id: 'meeting-42',
        text_md: '# Decisions\n\nShip it.'
      })
    ])
  })

  it('drops notepad updates while the socket is down', () => {
    const client = new SidecarClient('ws://127.0.0.1:8930/session')
    client.connect()
    const socket = FakeWebSocket.instances[0]

    client.sendNotepad('meeting-42', 'offline edit')

    expect(socket.sent).toEqual([])
  })

  it('parses tolerant final-summary status events', () => {
    const client = new SidecarClient('ws://127.0.0.1:8930/session')
    const statuses: unknown[] = []
    client.onFinalSummaryStatus((event) => statuses.push(event))
    client.connect()
    const socket = FakeWebSocket.instances[0]
    socket.open()

    socket.onmessage?.({
      data: JSON.stringify({
        type: 'final_summary_status',
        meeting_id: 'meeting-42',
        state: 'running'
      })
    })

    expect(statuses).toEqual([
      {
        type: 'final_summary_status',
        meetingId: 'meeting-42',
        state: 'running',
        detail: ''
      }
    ])
  })

  it('maps final summaries and drops malformed suggested edits', () => {
    const client = new SidecarClient('ws://127.0.0.1:8930/session')
    const summaries: unknown[] = []
    client.onFinalSummary((event) => summaries.push(event))
    client.connect()
    const socket = FakeWebSocket.instances[0]
    socket.open()

    socket.onmessage?.({
      data: JSON.stringify({
        type: 'final_summary',
        meeting_id: 'meeting-42',
        title: 'Planning',
        final_summary_md: '## Summary',
        base_rev: 7,
        suggested_edits: [
          {
            id: 'edit-1',
            anchor: 'Decision',
            occurrence: 1,
            insert_md: '\n\nMore detail.',
            rationale: 'Adds context',
            status: 'pending'
          },
          { id: 'bad-edit' }
        ],
        ignored_future_field: true
      })
    })

    expect(summaries).toEqual([
      {
        type: 'final_summary',
        meetingId: 'meeting-42',
        title: 'Planning',
        finalSummaryMd: '## Summary',
        baseRev: 7,
        suggestedEdits: [
          {
            id: 'edit-1',
            anchor: 'Decision',
            occurrence: 1,
            insertMd: '\n\nMore detail.',
            rationale: 'Adds context',
            status: 'pending'
          }
        ]
      }
    ])
  })
})
