import { useMemo, useState } from 'react'
import type { ChatActivityStep, ChatMessage } from '../../lib/chat'
import type { GroundedFact, GroundingItem } from '../../lib/sidecar'
import { ageLabel, sourceLabel } from '../../lib/labels'
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
  sources: GroundingItem[]
  facts: GroundedFact[]
  error: string
}

interface AssistantTurnProps {
  message?: ChatMessage // persisted form (done)
  streaming?: StreamingTurn // in-flight form
  isLast: boolean
  onOpenNote: (noteId: string) => void
  onRegenerate: () => void
  onRetry?: () => void
}

export function AssistantTurn({
  message,
  streaming,
  isLast,
  onOpenNote,
  onRegenerate,
  onRetry
}: AssistantTurnProps): React.JSX.Element {
  const thinking = streaming?.thinking ?? message?.thinking ?? ''
  const content = streaming?.content ?? message?.content ?? ''
  const activity = streaming?.activity ?? message?.activity ?? []
  const sources = streaming?.sources ?? message?.sources ?? []
  const facts = streaming?.facts ?? message?.facts ?? []
  const phase: TurnPhase = streaming?.phase ?? 'done'
  const isStreaming = streaming !== undefined && phase !== 'done' && phase !== 'error'

  const citations = useMemo(() => {
    const map = new Map<string, CitationTarget>()
    for (const item of sources) {
      if (item.short_id) {
        map.set(item.short_id, { title: item.title, noteId: item.note_id })
      }
    }
    for (const fact of facts) {
      map.set(fact.fact_id, { title: fact.text, noteId: fact.src })
    }
    return map
  }, [sources, facts])

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
          sources={sources}
          isLast={isLast}
          onOpenNote={onOpenNote}
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
  sources,
  isLast,
  onOpenNote,
  onRegenerate
}: {
  message: ChatMessage
  sources: GroundingItem[]
  isLast: boolean
  onOpenNote: (noteId: string) => void
  onRegenerate: () => void
}): React.JSX.Element {
  const [copied, setCopied] = useState(false)
  const [sourcesExpanded, setSourcesExpanded] = useState(false)
  const visibleSources = sourcesExpanded ? sources : sources.slice(0, 4)

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
      {sources.length > 0 ? (
        <div className="turn-sources">
          {visibleSources.map((item) => (
            <button
              key={item.note_id}
              className="source-pill"
              type="button"
              title={item.gist}
              onClick={() => onOpenNote(item.note_id)}
            >
              <span className="citation-chip citation-note">{item.short_id || '·'}</span>
              <span className="source-pill-title">{item.title}</span>
              <span className={`source-chip source-${item.source}`}>
                {sourceLabel(item.source)}
              </span>
              <span className="age-chip">{ageLabel(item.age_days)}</span>
            </button>
          ))}
          {sources.length > 4 && !sourcesExpanded ? (
            <button className="turn-action" type="button" onClick={() => setSourcesExpanded(true)}>
              +{sources.length - 4} more
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
