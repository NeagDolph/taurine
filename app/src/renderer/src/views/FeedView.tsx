import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import type { TranscriptEvent, TriggerEvent } from '../lib/sidecar'

export interface FeedTranscript extends TranscriptEvent {
  id: string
}

interface FeedViewProps {
  chunks: FeedTranscript[]
  interims: TranscriptEvent[]
  triggersBySeq: Map<number, TriggerEvent[]>
  onInspectTrigger: (trigger: TriggerEvent) => void
}

interface SpanRange {
  start: number
  end: number
  trigger: TriggerEvent
}

interface TriggerAnnotations {
  chips: TriggerEvent[]
  hasSearchTrigger: boolean
  ranges: SpanRange[]
}

const AUTOSCROLL_THRESHOLD_PX = 48

export function FeedView({
  chunks,
  interims,
  triggersBySeq,
  onInspectTrigger
}: FeedViewProps): React.JSX.Element {
  const listRef = useRef<HTMLDivElement | null>(null)
  const [atBottom, setAtBottom] = useState(true)
  const isEmpty = chunks.length === 0 && interims.length === 0

  const computeAtBottom = useCallback((): boolean => {
    const el = listRef.current
    if (!el) {
      return true
    }
    return el.scrollHeight - el.scrollTop - el.clientHeight <= AUTOSCROLL_THRESHOLD_PX
  }, [])

  const scrollToLatest = useCallback((behavior: ScrollBehavior = 'smooth') => {
    const el = listRef.current
    if (!el) {
      return
    }
    el.scrollTo({ top: el.scrollHeight, behavior })
    setAtBottom(true)
  }, [])

  const handleScroll = useCallback(() => {
    setAtBottom(computeAtBottom())
  }, [computeAtBottom])

  // Auto-follow newest content only when the user is already pinned to the bottom.
  useLayoutEffect(() => {
    if (atBottom) {
      const el = listRef.current
      if (el) {
        el.scrollTop = el.scrollHeight
      }
    }
  }, [chunks, interims, atBottom])

  useEffect(() => {
    setAtBottom(computeAtBottom())
  }, [computeAtBottom, isEmpty])

  return (
    <section className="feed-view" aria-label="Transcript feed">
      {isEmpty ? (
        <div className="feed-empty">
          <span className="empty-glyph" aria-hidden="true">◎</span>
          <p className="empty-title">Listening…</p>
          <p className="empty-hint">Transcript and matched notes appear here as you talk.</p>
        </div>
      ) : (
        <div className="transcript-scroll">
          <div className="transcript-list" aria-live="polite" ref={listRef} onScroll={handleScroll}>
            {chunks.map((chunk) => (
              <TranscriptRow
                chunk={chunk}
                key={chunk.id}
                onInspectTrigger={onInspectTrigger}
                triggers={triggersBySeq.get(chunk.seq) ?? []}
              />
            ))}
            {interims.map((chunk) => (
              <TranscriptRow chunk={chunk} key={`interim-${chunk.channel}`} interim />
            ))}
          </div>
          {atBottom ? null : (
            <button
              className="jump-latest"
              type="button"
              onClick={() => scrollToLatest()}
            >
              Jump to latest ↓
            </button>
          )}
        </div>
      )}
    </section>
  )
}

function TranscriptRow({
  chunk,
  interim = false,
  onInspectTrigger,
  triggers = []
}: {
  chunk: TranscriptEvent
  interim?: boolean
  onInspectTrigger?: (trigger: TriggerEvent) => void
  triggers?: TriggerEvent[]
}): React.JSX.Element {
  const annotations = interim ? emptyAnnotations() : buildTriggerAnnotations(chunk, triggers)
  const rowClasses = [
    'transcript-row',
    interim ? 'transcript-interim' : '',
    annotations.hasSearchTrigger ? 'transcript-row-search-accent' : ''
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <article className={rowClasses}>
      <div className="transcript-meta">
        <span>{channelLabel(chunk.channel)}</span>
        <time>{formatTime(chunk.ts)}</time>
      </div>
      <p className="transcript-text">{renderTranscriptText(chunk.text, annotations.ranges, onInspectTrigger)}</p>
      {annotations.chips.length > 0 ? (
        <div className="trigger-chip-row" aria-label="Trigger annotations">
          {annotations.chips.map((trigger) => (
            <button
              className={`trigger-chip trigger-chip-${trigger.kind} trigger-action-${trigger.action}`}
              key={`${chunk.seq}-${trigger.trigger_id}`}
              type="button"
              title={chipLabel(trigger)}
              aria-label={`Inspect ${chipLabel(trigger)}`}
              onClick={() => onInspectTrigger?.(trigger)}
            >
              {chipLabel(trigger)}
            </button>
          ))}
        </div>
      ) : null}
    </article>
  )
}

function buildTriggerAnnotations(chunk: TranscriptEvent, triggers: TriggerEvent[]): TriggerAnnotations {
  const ranges: SpanRange[] = []
  const chips: TriggerEvent[] = []
  const hasSearchTrigger = triggers.some((trigger) => trigger.kind === 'search')

  for (const trigger of triggers) {
    if (trigger.kind !== 'exact') {
      continue
    }

    const spanRange = findSpanRange(chunk.text, trigger.span)
    if (!spanRange || ranges.some((range) => rangesOverlap(range, spanRange))) {
      chips.push(trigger)
      continue
    }

    ranges.push({ ...spanRange, trigger })
  }

  for (const trigger of triggers) {
    if (trigger.kind === 'search' && isLastSeqInTrigger(chunk.seq, trigger)) {
      chips.push(trigger)
    }
  }

  return {
    chips,
    hasSearchTrigger,
    ranges: ranges.sort((left, right) => left.start - right.start)
  }
}

function emptyAnnotations(): TriggerAnnotations {
  return { chips: [], hasSearchTrigger: false, ranges: [] }
}

function renderTranscriptText(
  text: string,
  ranges: SpanRange[],
  onInspectTrigger?: (trigger: TriggerEvent) => void
): ReactNode[] {
  if (ranges.length === 0) {
    return [text]
  }

  const nodes: ReactNode[] = []
  let cursor = 0

  ranges.forEach((range) => {
    if (range.start > cursor) {
      nodes.push(text.slice(cursor, range.start))
    }

    nodes.push(
      <button
        className={`trigger-span trigger-action-${range.trigger.action}`}
        key={`${range.trigger.trigger_id}-${range.start}-${range.end}`}
        type="button"
        onClick={() => onInspectTrigger?.(range.trigger)}
      >
        {text.slice(range.start, range.end)}
      </button>
    )
    cursor = range.end
  })

  if (cursor < text.length) {
    nodes.push(text.slice(cursor))
  }

  return nodes
}

function findSpanRange(text: string, span: string): { start: number; end: number } | null {
  const needle = span.trim()
  if (!needle) {
    return null
  }

  const exactStart = text.toLocaleLowerCase().indexOf(needle.toLocaleLowerCase())
  if (exactStart >= 0) {
    return { start: exactStart, end: exactStart + needle.length }
  }

  const normalizedText = normalizeForSearch(text)
  const normalizedNeedle = normalizeForSearch(needle)
  if (!normalizedNeedle.value) {
    return null
  }

  const normalizedStart = normalizedText.value.indexOf(normalizedNeedle.value)
  if (normalizedStart < 0) {
    return null
  }

  const normalizedEnd = normalizedStart + normalizedNeedle.value.length - 1
  const start = normalizedText.indexMap[normalizedStart]
  const end = normalizedText.indexMap[normalizedEnd] + 1

  if (start === undefined || end === undefined || end <= start) {
    return null
  }

  return { start, end }
}

function normalizeForSearch(value: string): { value: string; indexMap: number[] } {
  let normalized = ''
  const indexMap: number[] = []
  let lastWasSpace = true

  for (let index = 0; index < value.length; index += 1) {
    const char = value[index].toLocaleLowerCase()

    if (/[a-z0-9]/i.test(char)) {
      normalized += char
      indexMap.push(index)
      lastWasSpace = false
      continue
    }

    if (!lastWasSpace && normalized.length > 0) {
      normalized += ' '
      indexMap.push(index)
      lastWasSpace = true
    }
  }

  if (normalized.endsWith(' ')) {
    normalized = normalized.slice(0, -1)
    indexMap.pop()
  }

  return { value: normalized, indexMap }
}

function rangesOverlap(left: { start: number; end: number }, right: { start: number; end: number }): boolean {
  return left.start < right.end && right.start < left.end
}

function isLastSeqInTrigger(seq: number, trigger: TriggerEvent): boolean {
  return seq === Math.max(...trigger.seqs)
}

function chipLabel(trigger: TriggerEvent): string {
  if (trigger.kind === 'search') {
    return `Search ${trigger.matches.length} ${trigger.matches.length === 1 ? 'note' : 'notes'}`
  }

  return `Exact: ${trigger.span}`
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
