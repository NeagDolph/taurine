import { useMemo, useState } from 'react'
import type { ChatActivityStep, ChatMessage } from '../../lib/chat'
import type { EvidencePack } from '../../lib/sidecar'
import { sourceLabel } from '../../lib/labels'
import { MessageMarkdown, type CitationTarget } from './MessageMarkdown'

export type TurnPhase = 'retrieving' | 'thinking' | 'answering' | 'done' | 'error'

export interface StreamingTurn {
  messageId: number
  phase: TurnPhase
  thinking: string
  thinkingStartedAt: number | null
  thinkingEndedAt: number | null
  activity: ChatActivityStep[]
  content: string
  evidence: EvidencePack | null
  retrievedIds: string[]
  consultedIds: string[]
  error: string
}

interface AssistantTurnProps {
  message?: ChatMessage // persisted form (done)
  streaming?: StreamingTurn // in-flight form
  isLast: boolean
  onOpenNote: (noteId: string) => void
  onShowInMap?: (noteId: string) => void
  onRegenerate: () => void
  onRetry?: () => void
}

export function AssistantTurn({
  message,
  streaming,
  isLast,
  onOpenNote,
  onShowInMap,
  onRegenerate,
  onRetry
}: AssistantTurnProps): React.JSX.Element {
  const thinking = streaming?.thinking ?? message?.thinking ?? ''
  const content = streaming?.content ?? message?.content ?? ''
  const activity = streaming?.activity ?? message?.activity ?? []
  const evidence = streaming?.evidence ?? message?.evidence ?? null
  const retrievedIds = streaming?.retrievedIds ?? message?.retrieved_ids ?? []
  const consultedIds = streaming?.consultedIds ?? message?.consulted_ids ?? []
  const phase: TurnPhase = streaming?.phase ?? 'done'
  const isStreaming = streaming !== undefined && phase !== 'done' && phase !== 'error'

  const citations = useMemo(() => {
    const map = new Map<string, CitationTarget>()
    for (const item of evidence?.notes ?? []) {
      map.set(item.note_ref, { title: item.title, noteId: item.note_id })
    }
    for (const passage of evidence?.passages ?? []) {
      map.set(passage.evidence_id, { title: passage.text, noteId: passage.note_id })
    }
    for (const fact of evidence?.facts ?? []) {
      const source = evidence?.passages.find((row) =>
        fact.source_evidence_ids.includes(row.evidence_id)
      )
      map.set(fact.fact_ref, {
        title: `${fact.subject} ${fact.predicate} ${fact.object}`,
        noteId: source?.note_id ?? ''
      })
    }
    for (const path of evidence?.paths ?? []) {
      map.set(path.path_id, { title: path.rendered, noteId: '' })
    }
    return map
  }, [evidence])

  const thinkingSeconds = useMemo(() => {
    if (streaming?.thinkingStartedAt && streaming.thinkingEndedAt) {
      return Math.max(1, Math.round(streaming.thinkingEndedAt - streaming.thinkingStartedAt))
    }
    if (message && message.usage.total_ms > 0 && message.thinking) {
      return Math.max(1, Math.round(message.usage.total_ms / 1000))
    }
    return null
  }, [streaming, message])

  return (
    <article className="assistant-turn">
      {phase === 'retrieving' ? (
        <div className="turn-status shimmer">Searching your notes…</div>
      ) : null}

      {thinking ? (
        <ThinkingAccordion text={thinking} live={phase === 'thinking'} seconds={thinkingSeconds} />
      ) : null}

      {activity.length > 0 ? (
        <ol className="activity-steps">
          {activity.map((step, index) => (
            <li key={index} className={`activity-step ${step.status}`}>
              <span className="activity-icon" aria-hidden="true">
                {step.kind === 'search' ? '🔍' : step.kind === 'read' ? '📄' : '🕸'}
              </span>
              <button
                className="activity-label"
                type="button"
                disabled={step.note_ids.length === 0}
                onClick={() => step.note_ids[0] && onOpenNote(step.note_ids[0])}
              >
                {step.label}
              </button>
              {step.status === 'running' ? <span className="activity-spinner" /> : null}
            </li>
          ))}
        </ol>
      ) : null}

      {content ? (
        <MessageMarkdown
          markdown={content}
          citations={citations}
          onOpenNote={onOpenNote}
          streaming={phase === 'answering'}
        />
      ) : null}

      {phase === 'error' ? (
        <div className="turn-error">
          <span>{streaming?.error || 'Something went wrong'}</span>
          {onRetry ? (
            <button className="secondary-button" type="button" onClick={onRetry}>
              Retry
            </button>
          ) : null}
        </div>
      ) : null}

      {message?.cancelled ? <div className="turn-stopped">stopped</div> : null}

      {!isStreaming && message ? (
        <TurnFooter
          message={message}
          evidence={evidence}
          citedIds={message.cited_ids}
          consultedIds={consultedIds.length > 0 ? consultedIds : retrievedIds}
          isLast={isLast}
          onOpenNote={onOpenNote}
          onShowInMap={onShowInMap}
          onRegenerate={onRegenerate}
        />
      ) : null}
    </article>
  )
}

function ThinkingAccordion({
  text,
  live,
  seconds
}: {
  text: string
  live: boolean
  seconds: number | null
}): React.JSX.Element {
  const [expanded, setExpanded] = useState(false)
  const open = live || expanded

  return (
    <div className={`thinking-accordion${live ? ' live' : ''}`}>
      <button
        className="thinking-header"
        type="button"
        aria-expanded={open}
        onClick={() => !live && setExpanded((current) => !current)}
      >
        {live ? (
          <span className="shimmer">Thinking…</span>
        ) : (
          <span>Thought{seconds ? ` for ${seconds}s` : ''}</span>
        )}
        {!live ? <span className="thinking-caret">{expanded ? '▾' : '▸'}</span> : null}
      </button>
      {open ? (
        <div className="thinking-body" ref={live ? autoScrollRef : undefined}>
          {text}
        </div>
      ) : null}
    </div>
  )
}

function autoScrollRef(node: HTMLDivElement | null): void {
  if (node) {
    node.scrollTop = node.scrollHeight
  }
}

function TurnFooter({
  message,
  evidence,
  citedIds,
  consultedIds,
  isLast,
  onOpenNote,
  onShowInMap,
  onRegenerate
}: {
  message: ChatMessage
  evidence: EvidencePack | null
  citedIds: string[]
  consultedIds: string[]
  isLast: boolean
  onOpenNote: (noteId: string) => void
  onShowInMap?: (noteId: string) => void
  onRegenerate: () => void
}): React.JSX.Element {
  const [copied, setCopied] = useState(false)
  const [sourcesExpanded, setSourcesExpanded] = useState(false)
  const cited = new Set(citedIds)
  const consulted = new Set(consultedIds)
  const citedPassages = new Set(
    (evidence?.facts ?? [])
      .filter((fact) => cited.has(fact.fact_ref))
      .flatMap((fact) => fact.source_evidence_ids)
  )
  const sourceNotes = (evidence?.notes ?? []).filter(
    (note) =>
      cited.has(note.note_ref) ||
      note.evidence_ids.some((id) => cited.has(id) || citedPassages.has(id))
  )
  const consultedNotes = (evidence?.notes ?? []).filter(
    (note) =>
      !sourceNotes.includes(note) &&
      (consulted.has(note.note_ref) || note.evidence_ids.some((id) => consulted.has(id)))
  )
  const allVisible = [...sourceNotes, ...consultedNotes]
  const visibleSources = sourcesExpanded ? allVisible : allVisible.slice(0, 4)

  const copy = (): void => {
    void navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    })
  }

  const totalTokens = message.usage.completion + message.usage.reasoning
  const seconds = message.usage.total_ms / 1000

  return (
    <div className="turn-footer">
      <div className="turn-actions">
        <button className="turn-action" type="button" onClick={copy}>
          {copied ? 'Copied' : 'Copy'}
        </button>
        {isLast ? (
          <button className="turn-action" type="button" onClick={onRegenerate}>
            Regenerate
          </button>
        ) : null}
        {totalTokens > 0 ? (
          <span className="turn-meta">
            {formatTokens(totalTokens)} tok · {seconds >= 1 ? `${seconds.toFixed(0)}s` : '<1s'}
          </span>
        ) : null}
      </div>
      {allVisible.length > 0 ? (
        <div className="turn-sources">
          {visibleSources.map((item) => (
            <span className="map-pill-wrap" key={item.note_id}>
              <button
                className="source-pill"
                type="button"
                title={item.title}
                onClick={() => onOpenNote(item.note_id)}
              >
                <span className="citation-chip citation-note">{item.note_ref}</span>
                <span className="source-pill-title">{item.title}</span>
                <span className={`source-chip source-${item.source}`}>
                  {sourceLabel(item.source)}
                </span>
                <span className="source-state">
                  {sourceNotes.includes(item) ? 'source' : 'consulted'}
                </span>
              </button>
              {onShowInMap ? (
                <button
                  className="map-cite-btn"
                  type="button"
                  title="Show in map"
                  aria-label={`Show ${item.title} in map`}
                  onClick={() => onShowInMap(item.note_id)}
                >
                  ⌖
                </button>
              ) : null}
            </span>
          ))}
          {allVisible.length > 4 && !sourcesExpanded ? (
            <button className="turn-action" type="button" onClick={() => setSourcesExpanded(true)}>
              +{allVisible.length - 4} more
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function formatTokens(count: number): string {
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : `${count}`
}
