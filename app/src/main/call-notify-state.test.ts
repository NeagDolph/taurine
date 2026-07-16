import { describe, expect, it } from 'vitest'
import {
  callNotifyReducer,
  INITIAL_CALL_NOTIFY_STATE,
  type CallNotifyEvent,
  type CallNotifyState
} from './call-notify-state'

function run(events: CallNotifyEvent[]): { state: CallNotifyState; effects: string[] } {
  let state = INITIAL_CALL_NOTIFY_STATE
  const effects: string[] = []
  for (const event of events) {
    const [next, effect] = callNotifyReducer(state, event)
    state = next
    if (effect) effects.push(effect)
  }
  return { state, effects }
}

describe('callNotifyReducer', () => {
  it('shows the panel on a mic edge while idle', () => {
    const { state, effects } = run([{ type: 'mic', active: true }])
    expect(effects).toEqual(['show'])
    expect(state.panelVisible).toBe(true)
  })

  it('never shows while Taurine itself is listening', () => {
    const { effects } = run([
      { type: 'listening', listening: true },
      { type: 'mic', active: true }
    ])
    expect(effects).toEqual([])
  })

  it('one prompt per call: resolution suppresses until re-armed', () => {
    const { effects } = run([
      { type: 'mic', active: true },
      { type: 'panel-resolved' },
      { type: 'mic', active: false },
      { type: 'mic', active: true } // same-ish call flapping: still suppressed
    ])
    expect(effects).toEqual(['show', 'hide'])
  })

  it('re-arms only after the mic has been idle', () => {
    const { effects } = run([
      { type: 'mic', active: true },
      { type: 'panel-resolved' },
      { type: 'mic', active: false },
      { type: 'rearm' },
      { type: 'mic', active: true }
    ])
    expect(effects).toEqual(['show', 'hide', 'show'])
  })

  it('rearm is ignored while the mic is still active', () => {
    const { state } = run([
      { type: 'mic', active: true },
      { type: 'panel-resolved' },
      { type: 'rearm' }
    ])
    expect(state.suppressed).toBe(true)
  })

  it('hides when the call ends before a response', () => {
    const { state, effects } = run([
      { type: 'mic', active: true },
      { type: 'mic', active: false }
    ])
    expect(effects).toEqual(['show', 'hide'])
    expect(state.suppressed).toBe(false) // unanswered ≠ dismissed
  })

  it('hides and suppresses when listening starts from the panel', () => {
    const { effects, state } = run([
      { type: 'mic', active: true },
      { type: 'listening', listening: true }
    ])
    expect(effects).toEqual(['show', 'hide'])
    expect(state.suppressed).toBe(true)
  })
})
