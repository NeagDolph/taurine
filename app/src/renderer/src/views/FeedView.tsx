import type { TranscriptEvent } from '../lib/sidecar'

export interface FeedTranscript extends TranscriptEvent {
  id: string
}

interface FeedViewProps {
  chunks: FeedTranscript[]
  interims: TranscriptEvent[]
}

export function FeedView({ chunks, interims }: FeedViewProps): React.JSX.Element {
  return (
    <section className="feed-view" aria-label="Transcript feed">
      {chunks.length === 0 && interims.length === 0 ? (
        <div className="empty-surface feed-empty">No transcript yet</div>
      ) : (
        <div className="transcript-list" aria-live="polite">
          {chunks.map((chunk) => (
            <TranscriptRow chunk={chunk} key={chunk.id} />
          ))}
          {interims.map((chunk) => (
            <TranscriptRow chunk={chunk} key={`interim-${chunk.channel}`} interim />
          ))}
        </div>
      )}
    </section>
  )
}

function TranscriptRow({
  chunk,
  interim = false
}: {
  chunk: TranscriptEvent
  interim?: boolean
}): React.JSX.Element {
  return (
    <article className={`transcript-row ${interim ? 'transcript-interim' : ''}`}>
      <div className="transcript-meta">
        <span>{channelLabel(chunk.channel)}</span>
        <time>{formatTime(chunk.ts)}</time>
      </div>
      <p>{chunk.text}</p>
    </article>
  )
}

function channelLabel(channel: number): string {
  return channel === 0 ? 'Mic' : `Ch ${channel}`
}

function formatTime(ts: number): string {
  const millis = ts > 1_000_000_000_000 ? ts : ts * 1000
  return new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit'
  }).format(new Date(millis))
}
