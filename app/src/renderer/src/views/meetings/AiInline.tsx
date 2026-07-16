import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MessageMarkdown, type CitationTarget } from '../../components/chat/MessageMarkdown'
import { ChatClient, type ChatEvent, type ChatMessage } from '../../lib/chat'
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
  onOpenNote(noteId: string): void
  onInsert(md: string): void
}

type Phase = 'idle' | 'running' | 'done' | 'error'

/**
 * In-editor "Ask AI" (slash-menu) — a floating prompt anchored at the cursor
 * that runs a turn of the meeting chat agent (KB search, [M#] meeting
 * context, edit_notes). Shares the meeting's chat thread, so inline asks
 * appear in the bottom drawer history too.
 */
export function AiInline({
  meetingId,
  chatId,
  open,
  anchor,
  selection,
  onClose,
  onNotesEdit,
  onOpenNote,
  onInsert
}: AiInlineProps): React.JSX.Element | null {
  const [prompt, setPrompt] = useState('')
  const [phase, setPhase] = useState<Phase>('idle')
  const [answer, setAnswer] = useState('')
  const [activity, setActivity] = useState('')
  const [message, setMessage] = useState<ChatMessage | null>(null)
  const [error, setError] = useState('')
  const [connected, setConnected] = useState(false)
  const clientRef = useRef<ChatClient | null>(null)
  const chatIdRef = useRef<string | null>(chatId)
  const messageIdRef = useRef<number | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const handleEvent = useCallback(
    (event: ChatEvent): void => {
      switch (event.type) {
        case 'chat_ack':
          chatIdRef.current = event.chat_id
          messageIdRef.current = event.message_id
          return
        case 'chat_delta':
          if (event.message_id === messageIdRef.current) {
            setAnswer((current) => current + event.delta)
          }
          return
        case 'chat_activity':
          if (event.message_id === messageIdRef.current && event.step.status === 'running') {
            setActivity(event.step.label)
          }
          return
        case 'chat_notes_edit':
          if (event.meetingId === meetingId) {
            onNotesEdit({
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
            setMessage(event.message)
            setAnswer(event.message.content)
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
    [meetingId, onNotesEdit]
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
      setAnswer('')
      setMessage(null)
      setError('')
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
    const wrapped = selection
      ? `Regarding this part of my notes:\n> ${selection.replaceAll('\n', '\n> ')}\n\n${question}`
      : question
    if (!client.send(chatIdRef.current ?? '', wrapped, [], meetingId)) {
      setError('Not connected')
      setPhase('error')
      return
    }
    setPhase('running')
    setAnswer('')
    setMessage(null)
    setError('')
    setActivity('Thinking…')
  }

  const citations = useMemo(() => {
    const map = new Map<string, CitationTarget>()
    const evidence = message?.evidence
    for (const item of evidence?.notes ?? []) {
      map.set(item.note_ref, { title: item.title, noteId: item.note_id })
    }
    for (const passage of evidence?.passages ?? []) {
      map.set(passage.evidence_id, { title: passage.text, noteId: passage.note_id })
    }
    return map
  }, [message])

  if (!open || !anchor) return null

  return (
    <div
      className="absolute z-40 w-[min(540px,94%)] rounded-xl border border-white/[0.12] bg-surface-2/95 shadow-2xl backdrop-blur-xl"
      style={{ top: anchor.top, left: Math.max(0, Math.min(anchor.left, anchor.width - 540)) }}
      role="dialog"
      aria-label="Ask AI"
    >
      <div className="flex items-center gap-2 px-3 pt-2.5">
        <span className="text-[13px]">✨</span>
        <input
          ref={inputRef}
          className="h-8 min-w-0 flex-1 border-0 bg-transparent text-[13px] text-fg outline-none placeholder:text-fg/35"
          value={prompt}
          placeholder={
            selection ? 'Ask about the selected text…' : 'Ask your knowledge base anything…'
          }
          aria-label="Ask AI"
          disabled={phase === 'running'}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') ask()
          }}
        />
        <button
          className="icon-button"
          type="button"
          aria-label="Close Ask AI"
          onClick={onClose}
        >
          ✕
        </button>
      </div>

      {phase === 'running' ? (
        <div className="flex items-center gap-2 px-3.5 pb-3 pt-1.5">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
          <span className="text-xs text-fg/55">{activity || 'Working…'}</span>
          {answer ? null : <span className="sr-only">Waiting for the first tokens</span>}
        </div>
      ) : null}

      {answer ? (
        <div className="max-h-[38vh] overflow-y-auto border-t border-white/[0.07] px-3.5 py-2.5 text-[13px] leading-relaxed">
          <MessageMarkdown
            markdown={answer}
            citations={citations}
            onOpenNote={onOpenNote}
            streaming={phase === 'running'}
          />
        </div>
      ) : null}

      {phase === 'error' ? (
        <p className="m-0 border-t border-white/[0.07] px-3.5 py-2.5 text-xs text-bad">{error}</p>
      ) : null}

      {phase === 'done' && answer ? (
        <div className="flex items-center gap-2 border-t border-white/[0.07] px-3 py-2">
          <button
            className="primary-button compact"
            type="button"
            onClick={() => {
              onInsert(answer)
              onClose()
            }}
          >
            Insert into notes
          </button>
          <button
            className="secondary-button compact"
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
    </div>
  )
}
