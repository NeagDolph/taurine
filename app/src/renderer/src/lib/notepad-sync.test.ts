import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DebouncedNotepadPush, appendSummaryToNotes, flushCurrentNotepad } from './notepad-sync'

describe('DebouncedNotepadPush', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('window', globalThis)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('sends only the latest live edit after two idle seconds', () => {
    const send = vi.fn()
    const push = new DebouncedNotepadPush(send)

    push.schedule('meeting-42', 'first')
    vi.advanceTimersByTime(1_500)
    push.schedule('meeting-42', 'second')
    vi.advanceTimersByTime(1_999)
    expect(send).not.toHaveBeenCalled()

    vi.advanceTimersByTime(1)
    expect(send).toHaveBeenCalledOnce()
    expect(send).toHaveBeenCalledWith('meeting-42', 'second')
  })

  it('flushes the current editor value immediately and cancels the debounce', () => {
    const send = vi.fn()
    const push = new DebouncedNotepadPush(send)

    push.schedule('meeting-42', 'last debounced value')
    push.flush('meeting-42', 'current editor value')
    vi.advanceTimersByTime(2_000)

    expect(send).toHaveBeenCalledOnce()
    expect(send).toHaveBeenCalledWith('meeting-42', 'current editor value')
  })

  it('drains a pending edit before stop when no editor flush is registered', () => {
    const send = vi.fn()
    const push = new DebouncedNotepadPush(send)

    push.schedule('meeting-42', 'pending value')
    push.flushPending()
    vi.advanceTimersByTime(2_000)

    expect(send).toHaveBeenCalledOnce()
    expect(send).toHaveBeenCalledWith('meeting-42', 'pending value')
  })

  it('cancels pending work during StrictMode cleanup', () => {
    const send = vi.fn()
    const push = new DebouncedNotepadPush(send)

    push.schedule('meeting-42', 'discard me')
    push.cancel()
    vi.advanceTimersByTime(2_000)

    expect(send).not.toHaveBeenCalled()
  })
})

describe('notepad editor helpers', () => {
  it('awaits the notes PATCH before the final websocket push', async () => {
    const events: string[] = []

    const saved = await flushCurrentNotepad({
      meetingId: 'meeting-42',
      currentTextMd: 'current editor value',
      savedTextMd: 'last autosaved value',
      save: async () => {
        events.push('patch')
      },
      push: () => events.push('notepad')
    })

    expect(events).toEqual(['patch', 'notepad'])
    expect(saved).toBe('current editor value')
  })

  it('still sends the final current value when REST is already current', async () => {
    const save = vi.fn()
    const push = vi.fn()

    await flushCurrentNotepad({
      meetingId: 'meeting-42',
      currentTextMd: 'already saved',
      savedTextMd: 'already saved',
      save,
      push
    })

    expect(save).not.toHaveBeenCalled()
    expect(push).toHaveBeenCalledWith('meeting-42', 'already saved')
  })

  it('does not flush an editor before its meeting has loaded', async () => {
    const save = vi.fn()
    const push = vi.fn()

    const saved = await flushCurrentNotepad({
      meetingId: null,
      currentTextMd: '',
      savedTextMd: 'last known value',
      save,
      push
    })

    expect(save).not.toHaveBeenCalled()
    expect(push).not.toHaveBeenCalled()
    expect(saved).toBe('last known value')
  })

  it('appends a summary using the normal markdown change value', () => {
    expect(appendSummaryToNotes('# Notes', '## Summary\nA decision.')).toBe(
      '# Notes\n\n## Summary\nA decision.\n'
    )
  })
})
