export type ListeningState = 'idle' | 'starting' | 'listening' | 'stopping' | 'error'

interface SessionControlsProps {
  state: ListeningState
  error: string
  onStart: () => void
  onStop: () => void
}

const MIC_STATE_CLASSES: Record<ListeningState, string> = {
  idle: 'border-white/[0.08] text-fg/60',
  starting: 'border-accent/50 text-accent-hi',
  stopping: 'border-accent/50 text-accent-hi',
  listening: 'border-ok/40 text-ok',
  error: 'border-bad/40 text-bad'
}

const MIC_DOT_CLASSES: Record<ListeningState, string> = {
  idle: 'bg-fg/40',
  starting: 'bg-accent',
  stopping: 'bg-accent',
  listening: 'bg-ok animate-[ring-pulse_1.6s_ease-out_infinite] [--ring:rgba(87,217,138,0.5)]',
  error: 'bg-bad'
}

export function SessionControls({
  state,
  error,
  onStart,
  onStop
}: SessionControlsProps): React.JSX.Element {
  const isStarting = state === 'starting'
  const isStopping = state === 'stopping'
  const isListening = state === 'listening'
  const isBusy = isStarting || isStopping

  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <div
        className={`inline-flex min-w-[96px] items-center gap-2 rounded-full border bg-white/[0.03] px-2.5 py-1 text-[11px] font-semibold ${MIC_STATE_CLASSES[state]}`}
      >
        <span className={`h-2 w-2 flex-none rounded-full ${MIC_DOT_CLASSES[state]}`} />
        <span>{labelForState(state)}</span>
      </div>
      <button
        className={isListening || isStarting ? 'secondary-button danger' : 'primary-button'}
        type="button"
        onClick={isListening || isStarting ? onStop : onStart}
        disabled={isBusy}
      >
        {isListening || isStarting ? 'Stop listening' : 'Start listening'}
      </button>
      {error ? (
        <span className="max-w-[220px] truncate text-xs text-bad" role="alert">
          {error}
        </span>
      ) : null}
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
