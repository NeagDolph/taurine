import { describe, expect, test } from 'vitest'
import {
  callDetectionReducer,
  INITIAL_CALL_DETECTION_STATE,
  shouldShowCallNotification
} from './call-detection'

describe('call detection notification state', () => {
  test('shows for an active mic only while Taurine is idle', () => {
    const active = callDetectionReducer(INITIAL_CALL_DETECTION_STATE, {
      type: 'mic',
      active: true,
      listeningState: 'idle'
    })

    expect(shouldShowCallNotification(active, 'idle')).toBe(true)
    expect(shouldShowCallNotification(active, 'listening')).toBe(false)
  })

  test('ignores micwatch events while Taurine is listening', () => {
    const state = callDetectionReducer(INITIAL_CALL_DETECTION_STATE, {
      type: 'mic',
      active: true,
      listeningState: 'listening'
    })

    expect(state).toBe(INITIAL_CALL_DETECTION_STATE)
  })

  test('stays suppressed until an inactive mic is explicitly re-armed', () => {
    const active = callDetectionReducer(INITIAL_CALL_DETECTION_STATE, {
      type: 'mic',
      active: true,
      listeningState: 'idle'
    })
    const suppressed = callDetectionReducer(active, { type: 'suppress' })
    const stillActive = callDetectionReducer(suppressed, { type: 'rearm' })
    expect(shouldShowCallNotification(stillActive, 'idle')).toBe(false)

    const inactive = callDetectionReducer(suppressed, {
      type: 'mic',
      active: false,
      listeningState: 'idle'
    })
    const rearmed = callDetectionReducer(inactive, { type: 'rearm' })
    const nextCall = callDetectionReducer(rearmed, {
      type: 'mic',
      active: true,
      listeningState: 'idle'
    })

    expect(shouldShowCallNotification(nextCall, 'idle')).toBe(true)
  })

  test('starting a meeting suppresses a prompt that was visible', () => {
    const active = callDetectionReducer(INITIAL_CALL_DETECTION_STATE, {
      type: 'mic',
      active: true,
      listeningState: 'idle'
    })
    const starting = callDetectionReducer(active, {
      type: 'listening',
      listeningState: 'starting'
    })

    expect(starting.suppressed).toBe(true)
    expect(shouldShowCallNotification(starting, 'idle')).toBe(false)
  })
})
