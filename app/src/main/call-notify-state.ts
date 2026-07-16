/**
 * Pure decision core for the OS-level call notification. The notifier owns
 * timers and windows; this owns the rules:
 * - show the panel on a mic-activity edge only when Taurine is idle and the
 *   current call has not been dismissed/answered;
 * - one prompt per call: any panel resolution suppresses until the mic has
 *   been idle long enough to re-arm (the notifier fires `rearm` after 5 s);
 * - the panel hides itself if the call ends or Taurine starts listening.
 */
export interface CallNotifyState {
  micActive: boolean
  listening: boolean
  suppressed: boolean
  panelVisible: boolean
}

export const INITIAL_CALL_NOTIFY_STATE: CallNotifyState = {
  micActive: false,
  listening: false,
  suppressed: false,
  panelVisible: false
}

export type CallNotifyEvent =
  | { type: 'mic'; active: boolean }
  | { type: 'listening'; listening: boolean }
  | { type: 'panel-resolved' } // start, dismiss, or countdown expiry
  | { type: 'rearm' }

export type CallNotifyEffect = 'show' | 'hide' | null

export function callNotifyReducer(
  state: CallNotifyState,
  event: CallNotifyEvent
): [CallNotifyState, CallNotifyEffect] {
  switch (event.type) {
    case 'mic': {
      const next = { ...state, micActive: event.active }
      if (event.active) {
        if (!state.listening && !state.suppressed && !state.panelVisible) {
          return [{ ...next, panelVisible: true }, 'show']
        }
        return [next, null]
      }
      if (state.panelVisible) return [{ ...next, panelVisible: false }, 'hide']
      return [next, null]
    }
    case 'listening': {
      const next = { ...state, listening: event.listening }
      if (event.listening && state.panelVisible) {
        return [{ ...next, panelVisible: false, suppressed: true }, 'hide']
      }
      return [next, null]
    }
    case 'panel-resolved': {
      if (!state.panelVisible) return [state, null]
      return [{ ...state, panelVisible: false, suppressed: true }, 'hide']
    }
    case 'rearm': {
      if (state.micActive || state.panelVisible) return [state, null]
      return [{ ...state, suppressed: false }, null]
    }
  }
}
