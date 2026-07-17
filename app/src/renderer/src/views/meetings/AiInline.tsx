import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatClient, type ChatEvent } from '../../lib/chat'
import type { ChatNotesEdit } from './MeetingChat'

export interface AiAnchor {
  top: number
  left: number
  width: number
}

interface AiInlineProps {
  meetingId: string
  chatId: string | null
  open: boolean
  anchor: AiAnchor | null
  selection: string
  onClose(): void
  onNotesEdit(edit: ChatNotesEdit): void
  /** Streamed answer markdown — written directly into the note as gray lines. */
  onAnswer(md: string, done: boolean): void
  /** A new ask is starting — the editor resets its stream anchor. */
  onAskStarted(): void
}

type Phase = 'idle' | 'running' | 'done' | 'error'

/**
 * In-editor "Ask AI" (slash-menu): a slim prompt anchored at the cursor that
 * runs a turn of the meeting chat agent. The answer never renders here — it
 * streams straight into the note as AI-authored (gray) content, Granola
 * style. Shares the meeting's chat thread, so asks appear in the drawer too.
 */
export function AiInline({
  meetingId,
  chatId,
  open,
  anchor,
  selection,
  onClose,
  onNotesEdit,
  onAnswer,
  onAskStarted
}: AiInlineProps): React.JSX.Element | null {
  const [prompt, setPrompt] = useState('')
  const [phase, setPhase] = useState<Phase>('idle')
  const [activity, setActivity] = useState('')
  const [error, setError] = useState('')
  const [connected, setConnected] = useState(false)
  const clientRef = useRef<ChatClient | null>(null)
  const chatIdRef = useRef<string | null>(chatId)
  const messageIdRef = useRef<number | null>(null)
  const answerRef = useRef('')
  const inputRef = useRef<HTMLInputElement | null>(null)
  const onAnswerRef = useRef(onAnswer)
  const onNotesEditRef = useRef(onNotesEdit)
  useEffect(() => {
    onAnswerRef.current = onAnswer
    onNotesEditRef.current = onNotesEdit
  }, [onAnswer, onNotesEdit])

  const handleEvent = useCallback(
    (event: ChatEvent): void => {
      switch (event.type) {
        case 'chat_ack':
          chatIdRef.current = event.chat_id
          messageIdRef.current = event.message_id
          return
        case 'chat_delta':
          if (event.message_id === messageIdRef.current) {
            answerRef.current += event.delta
            onAnswerRef.current(answerRef.current, false)
          }
          return
        case 'chat_activity':
          if (event.message_id === messageIdRef.current && event.step.status === 'running') {
            setActivity(event.step.label)
          }
          return
        case 'chat_notes_edit':
          if (event.meetingId === meetingId) {
            onNotesEditRef.current({
              editId: event.editId,
              anchor: event.anchor,
              occurrence: event.occurrence,
              insertMd: event.insertMd
            })
            setActivity('Edited the meeting notes')
          }
          return
        case 'chat_done':
          if (event.message_id === messageIdRef.current) {
            // The validated final content is authoritative (repair/stripping
            // may have changed it relative to the streamed deltas).
            answerRef.current = event.message.content
            onAnswerRef.current(answerRef.current, true)
            setPhase('done')
            setActivity('')
          }
          return
        case 'chat_error':
          setPhase('error')
          setError(event.detail || event.code)
          return
        default:
          return
      }
    },
    [meetingId]
  )
  const handleEventRef = useRef(handleEvent)
  useEffect(() => {
    handleEventRef.current = handleEvent
  }, [handleEvent])

  // One socket per notepad, connected on first open.
  const [everOpened, setEverOpened] = useState(false)
  if (open && !everOpened) setEverOpened(true)
  useEffect(() => {
    if (!everOpened) return undefined
    const client = new ChatClient()
    clientRef.current = client
    const unsubscribeEvent = client.onEvent((event) => handleEventRef.current(event))
    const unsubscribeConnection = client.onConnection((state) =>
      setConnected(state === 'connected')
    )
    client.connect()
    return () => {
      unsubscribeEvent()
      unsubscribeConnection()
      client.disconnect()
      clientRef.current = null
    }
  }, [everOpened])

  // Reset per-open state during render (React-sanctioned adjust pattern).
  const [openedFor, setOpenedFor] = useState(false)
  if (open !== openedFor) {
    setOpenedFor(open)
    if (open) {
      setPrompt('')
      setPhase('idle')
      setError('')
      setActivity('')
    }
  }
  useEffect(() => {
    if (open) window.setTimeout(() => inputRef.current?.focus(), 30)
  }, [open])

  useEffect(() => {
    if (!open) return undefined
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, open])

  const ask = (): void => {
    const question = prompt.trim()
    const client = clientRef.current
    if (!question || !client || phase === 'running') return
    const framing =
      '[Inline ask — your reply is written directly into the meeting notes at the cursor. ' +
      'Write note-ready markdown: no preamble, no "here is", just the content.]'
    const wrapped = selection
      ? `${framing}\nRegarding this part of my notes:\n> ${selection.replaceAll('\n', '\n> ')}\n\n${question}`
      : `${framing}\n${question}`
    onAskStarted()
    if (!client.send(chatIdRef.current ?? '', wrapped, [], meetingId)) {
      setError('Not connected')
      setPhase('error')
      return
    }
    answerRef.current = ''
    setPhase('running')
    setError('')
    setActivity('Thinking…')
  }

  if (!open || !anchor) return null

  return (
    <div
      className="absolute z-40 w-[min(540px,94%)] rounded-xl border border-white/[0.12] bg-surface-2/95 shadow-2xl backdrop-blur-xl"
      style={{ top: anchor.top, left: Math.max(0, Math.min(anchor.left, anchor.width - 540)) }}
      role="dialog"
      aria-label="Ask AI"
    >
      <div className="flex items-center gap-2 px-3 py-1.5">
        <span className="text-[13px]">✨</span>
        <input
          ref={inputRef}
          className="h-8 min-w-0 flex-1 border-0 bg-transparent text-[13px] text-fg outline-none placeholder:text-fg/35"
          value={prompt}
          placeholder={
            selection
              ? 'Ask about the selected text…'
              : 'Ask — the answer is written into your notes…'
          }
          aria-label="Ask AI"
          disabled={phase === 'running'}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') ask()
          }}
        />
        <button className="icon-button" type="button" aria-label="Close Ask AI" onClick={onClose}>
          ✕
        </button>
      </div>

      {phase === 'running' ? (
        <div className="flex items-center gap-2 border-t border-white/[0.07] px-3.5 py-2">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
          <span className="text-xs text-fg/55">{activity || 'Writing into your notes…'}</span>
        </div>
      ) : null}

      {phase === 'done' ? (
        <div className="flex items-center gap-2 border-t border-white/[0.07] px-3.5 py-2">
          <span className="text-xs text-ok">✓ Added to your notes</span>
          <button
            className="turn-action"
            type="button"
            disabled={!connected}
            onClick={() => {
              setPrompt('')
              setPhase('idle')
              inputRef.current?.focus()
            }}
          >
            Ask again
          </button>
        </div>
      ) : null}

      {phase === 'error' ? (
        <p className="m-0 border-t border-white/[0.07] px-3.5 py-2 text-xs text-bad">{error}</p>
      ) : null}
    </div>
  )
}
