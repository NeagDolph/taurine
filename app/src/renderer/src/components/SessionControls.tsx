export type ListeningState = 'idle' | 'starting' | 'listening' | 'stopping' | 'error'

interface SessionControlsProps {
  state: ListeningState
  error: string
  onStart: () => void
  onStop: () => void
}

export function SessionControls({ state, error, onStart, onStop }: SessionControlsProps): React.JSX.Element {
  const isStarting = state === 'starting'
  const isStopping = state === 'stopping'
  const isListening = state === 'listening'
  const isBusy = isStarting || isStopping

  return (
    <div className="session-controls">
      <div className={`mic-state mic-${state}`}>
        <span className="mic-dot" />
        <span>{labelForState(state)}</span>
      </div>
      <button
        className={`primary-button ${isListening ? 'stop-button' : ''}`}
        type="button"
        onClick={isListening || isStarting ? onStop : onStart}
        disabled={isBusy}
      >
        {isListening || isStarting ? 'Stop' : 'Start'}
      </button>
      {error ? <span className="control-error">{error}</span> : null}
    </div>
  )
}

function labelForState(state: ListeningState): string {
  switch (state) {
    case 'starting':
      return 'Starting'
    case 'listening':
      return 'Listening'
    case 'stopping':
      return 'Stopping'
    case 'error':
      return 'Mic error'
    case 'idle':
      return 'Idle'
  }
}
