import { afterEach, describe, expect, test, vi } from 'vitest'

vi.mock('electron', () => ({
  app: {
    isPackaged: false,
    getAppPath: () => '/tmp/taurine'
  }
}))

import { MicActivityDebouncer, parseMicWatchLine } from './micwatch'

afterEach(() => {
  vi.useRealTimers()
})

describe('parseMicWatchLine', () => {
  test('accepts only JSON objects with a boolean active field', () => {
    expect(parseMicWatchLine('{"active":true}')).toBe(true)
    expect(parseMicWatchLine('  {"active":false,"extra":1}  ')).toBe(false)
    expect(parseMicWatchLine('{"active":"true"}')).toBeNull()
    expect(parseMicWatchLine('not json')).toBeNull()
    expect(parseMicWatchLine('[]')).toBeNull()
  })
})

describe('MicActivityDebouncer', () => {
  test('drops short activations and forwards stable edges', () => {
    vi.useFakeTimers()
    const edges: boolean[] = []
    const debouncer = new MicActivityDebouncer((active) => edges.push(active), 1500)

    debouncer.accept(true)
    vi.advanceTimersByTime(1499)
    debouncer.accept(false)
    expect(edges).toEqual([false])

    debouncer.accept(true)
    vi.advanceTimersByTime(1500)
    expect(edges).toEqual([false, true])

    debouncer.accept(true)
    debouncer.accept(false)
    expect(edges).toEqual([false, true, false])
  })

  test('dispose cancels a pending activation', () => {
    vi.useFakeTimers()
    const callback = vi.fn()
    const debouncer = new MicActivityDebouncer(callback, 1500)

    debouncer.accept(true)
    debouncer.dispose()
    vi.advanceTimersByTime(1500)

    expect(callback).not.toHaveBeenCalled()
  })
})
