import { useMemo, useState } from 'react'
import { AmbientList } from '../../components/AmbientList'
import { TriggerInspector } from '../../components/TriggerInspector'
import type { Meeting, TranscriptChunk } from '../../lib/meetings'
import type { LiveGroundingEvent, TranscriptEvent } from '../../lib/sidecar'
import { FeedView, type FeedTranscript } from '../FeedView'

interface TranscriptPanelProps {
  meeting: Meeting
  liveMeetingId: string | null
  live: {
    chunks: TranscriptEvent[]
    interims: Record<number, TranscriptEvent>
    liveGrounding: LiveGroundingEvent | null
  }
  onClose(): void
}

export function TranscriptPanel({
  meeting,
  liveMeetingId,
  live,
  onClose
}: TranscriptPanelProps): React.JSX.Element {
  const [inspect, setInspect] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const isLive = meeting.meetingId === liveMeetingId
  const chunks = useMemo(
    () =>
      isLive ? live.chunks.map(toFeedTranscript) : meeting.transcript.map(storedToFeedTranscript),
    [isLive, live.chunks, meeting.transcript]
  )
  const interims = isLive ? Object.values(live.interims) : []
  const grounding = isLive ? live.liveGrounding : null

  return (
    <aside className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] border-l border-white/[0.08] bg-surface-1/75">
      <div className="flex h-11 items-center justify-between gap-2 border-b border-white/[0.08] px-3">
        <span className="section-label">Transcript</span>
        <div className="flex items-center gap-1.5">
          <button
            className={`turn-action ${inspect ? 'text-accent-hi' : ''}`}
            type="button"
            aria-pressed={inspect}
            onClick={() => setInspect((current) => !current)}
          >
            Inspect
          </button>
          <button
            className="icon-button"
            type="button"
            aria-label="Hide transcript"
            onClick={onClose}
          >
            &times;
          </button>
        </div>
      </div>

      {inspect ? (
        <div className="grid min-h-0 grid-rows-[minmax(0,1fr)] p-2.5">
          <AmbientList
            notes={grounding?.display_notes ?? []}
            passages={grounding?.display_passages ?? []}
            noNewEvidence={grounding?.no_new_evidence}
            onInspect={grounding ? () => setInspectorOpen(true) : undefined}
            onOpenNote={() => undefined}
          />
        </div>
      ) : chunks.length === 0 && interims.length === 0 && !isLive ? (
        <div className="flex items-center justify-center p-5 text-center text-xs text-fg/45">
          No transcript was saved for this meeting.
        </div>
      ) : (
        <FeedView
          chunks={chunks}
          interims={interims}
          triggersBySeq={EMPTY_TRIGGERS}
          onInspectTrigger={() => undefined}
        />
      )}

      <TriggerInspector
        grounding={inspectorOpen ? grounding : null}
        onClose={() => setInspectorOpen(false)}
        onOpenNote={() => undefined}
      />
    </aside>
  )
}

const EMPTY_TRIGGERS = new Map()

function toFeedTranscript(event: TranscriptEvent): FeedTranscript {
  return { ...event, id: event.seq > 0 ? `seq-${event.seq}` : `${event.channel}-${event.ts}` }
}

function storedToFeedTranscript(chunk: TranscriptChunk): FeedTranscript {
  return {
    type: 'transcript',
    ...chunk,
    final: true,
    id: chunk.seq > 0 ? `seq-${chunk.seq}` : `${chunk.channel}-${chunk.ts}`
  }
}
