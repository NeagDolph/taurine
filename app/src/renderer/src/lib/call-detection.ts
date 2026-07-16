import type { ListeningState } from '../components/SessionControls'

export interface CallDetectionState {
  active: boolean
  suppressed: boolean
}

export type CallDetectionEvent =
  | { type: 'mic'; active: boolean; listeningState: ListeningState }
  | { type: 'listening'; listeningState: ListeningState }
  | { type: 'suppress' }
  | { type: 'rearm' }

export const INITIAL_CALL_DETECTION_STATE: CallDetectionState = {
  active: false,
  suppressed: false
}

export function callDetectionReducer(
  state: CallDetectionState,
  event: CallDetectionEvent
): CallDetectionState {
  switch (event.type) {
    case 'mic':
      if (event.listeningState !== 'idle' || event.active === state.active) return state
      return { ...state, active: event.active }
    case 'listening':
      if (event.listeningState === 'idle' || !state.active || state.suppressed) return state
      return { ...state, suppressed: true }
    case 'suppress':
      return state.suppressed ? state : { ...state, suppressed: true }
    case 'rearm':
      return state.active || !state.suppressed ? state : { ...state, suppressed: false }
  }
}

export function shouldShowCallNotification(
  state: CallDetectionState,
  listeningState: ListeningState
): boolean {
  return state.active && listeningState === 'idle' && !state.suppressed
}
