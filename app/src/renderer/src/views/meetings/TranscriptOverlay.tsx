import { useEffect, useMemo, useRef, useState } from 'react'
import { AmbientList } from '../../components/AmbientList'
import { TriggerInspector } from '../../components/TriggerInspector'
import type { Meeting, TranscriptChunk } from '../../lib/meetings'
import { mergeBubbles, type Bubble } from '../../lib/transcript-bubbles'
import type { LiveGroundingEvent, TranscriptEvent } from '../../lib/sidecar'

interface TranscriptOverlayProps {
  open: boolean
  meeting: Meeting
  isLive: boolean
  live: {
    chunks: TranscriptEvent[]
    interims: Record<number, TranscriptEvent>
    liveGrounding: LiveGroundingEvent | null
  }
  onClose(): void
  onOpenNote(noteId: string): void
}

/**
 * Granola-style floating transcript sheet: compact chat bubbles — your mic on
 * the right, system audio on the left. Stays mounted while closed so scroll
 * position survives open/close and inspect toggles.
 */
export function TranscriptOverlay({
  open,
  meeting,
  isLive,
  live,
  onClose,
  onOpenNote
}: TranscriptOverlayProps): React.JSX.Element {
  const [query, setQuery] = useState('')
  const [inspect, setInspect] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const pinnedRef = useRef(true)

  const chunks: TranscriptChunk[] = useMemo(
    () =>
      isLive
        ? live.chunks.map((event) => ({
            channel: event.channel,
            text: event.text,
            ts: event.ts,
            seq: event.seq
          }))
        : meeting.transcript,
    [isLive, live.chunks, meeting.transcript]
  )

  const bubbles = useMemo(() => {
    const merged = mergeBubbles(chunks)
    if (!query.trim()) return merged
    const needle = query.trim().toLowerCase()
    return merged.filter((bubble) => bubble.text.toLowerCase().includes(needle))
  }, [chunks, query])

  const interimBubbles: Bubble[] = isLive
    ? Object.values(live.interims)
        .filter((event) => event.text.trim())
        .map((event) => ({
          key: `interim-${event.channel}`,
          channel: event.channel,
          text: event.text,
          interim: true
        }))
    : []

  // Autoscroll only while live AND the user is already pinned to the bottom.
  // Opening the sheet or toggling inspect never moves the scroll position.
  useEffect(() => {
    const node = scrollRef.current
    if (!node || !isLive || !pinnedRef.current) return
    node.scrollTop = node.scrollHeight
  }, [bubbles, interimBubbles.length, isLive])

  const elapsed = useElapsed(meeting, isLive)

  const copyTranscript = (): void => {
    const text = chunks.map((chunk) => `${speakerLabel(chunk.channel)}: ${chunk.text}`).join('\n')
    void navigator.clipboard?.writeText(text)
  }

  return (
    <section
      className={`absolute bottom-[84px] left-1/2 z-30 flex h-[min(56vh,540px)] w-[min(660px,94%)] -translate-x-1/2 flex-col overflow-hidden rounded-2xl border border-white/[0.1] bg-surface-2/95 shadow-2xl backdrop-blur-xl transition-all duration-150 ${
        open ? 'visible translate-y-0 opacity-100' : 'invisible translate-y-2 opacity-0'
      }`}
      aria-hidden={!open}
      aria-label="Meeting transcript"
    >
      <header className="flex h-11 flex-none items-center gap-2 border-b border-white/[0.07] px-3">
        <span className="flex items-center gap-1.5 text-xs tabular-nums text-fg/60">
          {isLive ? <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ok" /> : null}
          {elapsed}
        </span>
        <input
          className="min-w-0 flex-1 rounded-md border-0 bg-white/[0.04] px-2.5 py-1 text-xs text-fg outline-none placeholder:text-fg/30 focus:bg-white/[0.07]"
          value={query}
          placeholder="Search transcript"
          aria-label="Search transcript"
          onChange={(event) => setQuery(event.target.value)}
        />
        <button
          className={`turn-action ${inspect ? 'text-accent-hi' : ''}`}
          type="button"
          aria-pressed={inspect}
          title="Retrieved context for this conversation"
          onClick={() => setInspect((current) => !current)}
        >
          Insights
        </button>
        <button
          className="icon-button"
          type="button"
          title="Copy transcript"
          aria-label="Copy transcript"
          onClick={copyTranscript}
        >
          ⧉
        </button>
        <button
          className="icon-button"
          type="button"
          aria-label="Minimize transcript"
          onClick={onClose}
        >
          −
        </button>
      </header>

      <div className={`min-h-0 flex-1 ${inspect ? '' : 'hidden'} overflow-y-auto p-2.5`}>
        <AmbientList
          notes={isLive ? (live.liveGrounding?.display_notes ?? []) : []}
          passages={isLive ? (live.liveGrounding?.display_passages ?? []) : []}
          noNewEvidence={isLive ? live.liveGrounding?.no_new_evidence : undefined}
          onInspect={isLive && live.liveGrounding ? () => setInspectorOpen(true) : undefined}
          onOpenNote={onOpenNote}
        />
      </div>

      <div
        ref={scrollRef}
        className={`min-h-0 flex-1 ${inspect ? 'hidden' : ''} overflow-y-auto px-4 py-3`}
        onScroll={(event) => {
          const node = event.currentTarget
          pinnedRef.current = node.scrollTop + node.clientHeight >= node.scrollHeight - 48
        }}
      >
        <p className="mb-3 text-center text-[11px] text-fg/35">
          Always get consent when transcribing others.
        </p>
        {bubbles.length === 0 && interimBubbles.length === 0 ? (
          <p className="mt-8 text-center text-xs text-fg/35">
            {isLive ? 'Listening…' : 'No transcript was saved for this meeting.'}
          </p>
        ) : (
          <div className="flex flex-col gap-1.5">
            {bubbles.map((bubble) => (
              <TranscriptBubble key={bubble.key} bubble={bubble} />
            ))}
            {interimBubbles.map((bubble) => (
              <TranscriptBubble key={bubble.key} bubble={bubble} />
            ))}
          </div>
        )}
      </div>

      <TriggerInspector
        grounding={inspectorOpen && isLive ? live.liveGrounding : null}
        onClose={() => setInspectorOpen(false)}
        onOpenNote={onOpenNote}
      />
    </section>
  )
}

function TranscriptBubble({ bubble }: { bubble: Bubble }): React.JSX.Element {
  const mine = bubble.channel === 0
  return (
    <div className={`flex ${mine ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[78%] rounded-[12px] px-2.5 py-1.5 text-[12.5px] leading-5 ${
          mine ? 'bg-accent/[0.13] text-fg' : 'bg-white/[0.06] text-fg/90'
        } ${bubble.interim ? 'italic opacity-60' : ''}`}
      >
        {bubble.text}
      </div>
    </div>
  )
}

function speakerLabel(channel: number): string {
  if (channel === 0) return 'You'
  if (channel === 1) return 'Them'
  return `Ch ${channel}`
}

function useElapsed(meeting: Meeting, isLive: boolean): string {
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!isLive) return undefined
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => window.clearInterval(timer)
  }, [isLive])

  const start = meeting.startedAt
  if (!start) return '00:00'
  const end = isLive ? now : (meeting.endedAt ?? start)
  const total = Math.max(0, Math.floor(end - start))
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  const hours = Math.floor(minutes / 60)
  const pad = (value: number): string => String(value).padStart(2, '0')
  return hours > 0
    ? `${hours}:${pad(minutes % 60)}:${pad(seconds)}`
    : `${pad(minutes)}:${pad(seconds)}`
}
