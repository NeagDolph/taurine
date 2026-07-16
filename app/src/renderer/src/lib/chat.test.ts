import { afterEach, describe, expect, it, vi } from 'vitest'
import { ChatClient } from './chat'

class FakeSocket {
  static readonly OPEN = 1
  readyState = FakeSocket.OPEN
  sent: string[] = []

  send(payload: string): void {
    this.sent.push(payload)
  }
}

describe('ChatClient meeting scope', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('adds meeting_id only for a meeting-scoped send', () => {
    vi.stubGlobal('WebSocket', FakeSocket)
    const socket = new FakeSocket()
    const client = new ChatClient()
    ;(client as unknown as { socket: FakeSocket }).socket = socket

    expect(client.send('chat-1', 'plain')).toBe(true)
    expect(client.send('chat-2', 'meeting', [], 'meeting-1')).toBe(true)

    expect(socket.sent.map((payload) => JSON.parse(payload) as unknown)).toEqual([
      { type: 'chat_send', chat_id: 'chat-1', text: 'plain' },
      {
        type: 'chat_send',
        chat_id: 'chat-2',
        text: 'meeting',
        meeting_id: 'meeting-1'
      }
    ])
  })
})
