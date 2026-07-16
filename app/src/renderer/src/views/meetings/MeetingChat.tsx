import { useCallback, useEffect, useRef, useState } from 'react'
import { AssistantTurn, type StreamingTurn } from '../../components/chat/AssistantTurn'
import { Composer } from '../../components/chat/Composer'
import {
  ChatClient,
  fetchChat,
  type ChatConnectionState,
  type ChatEvent,
  type ChatMessage
} from '../../lib/chat'
import type { Meeting } from '../../lib/meetings'

const EMPTY_STREAM: Omit<StreamingTurn, 'messageId'> = {
  phase: 'retrieving',
  thinking: '',
  thinkingStartedAt: null,
  thinkingEndedAt: null,
  activity: [],
  content: '',
  evidence: null,
  retrievedIds: [],
  consultedIds: [],
  error: ''
}

interface MeetingChatProps {
  meeting: Meeting
  onOpenNote: (noteId: string) => void
}

export function MeetingChat({ meeting, onOpenNote }: MeetingChatProps): React.JSX.Element {
  const [expanded, setExpanded] = useState(false)
  const [everExpanded, setEverExpanded] = useState(false)
  const [chatId, setChatId] = useState<string | null>(meeting.chatId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState<StreamingTurn | null>(null)
  const [connection, setConnection] = useState<ChatConnectionState>('disconnected')
  const [error, setError] = useState('')
  const clientRef = useRef<ChatClient | null>(null)
  const pendingTextRef = useRef<string | null>(null)
  const loadedChatRef = useRef<string | null>(null)
  const threadRef = useRef<HTMLDivElement | null>(null)

  if (expanded && !everExpanded) setEverExpanded(true)

  useEffect(() => {
    if (!everExpanded) return undefined
    const client = new ChatClient()
    clientRef.current = client
    const unsubscribeEvent = client.onEvent((event) => {
      applyEvent(event, {
        pendingTextRef,
        loadedChatRef,
        setChatId,
        setMessages,
        setStreaming,
        setError
      })
    })
    const unsubscribeConnection = client.onConnection(setConnection)
    client.connect()
    return () => {
      unsubscribeEvent()
      unsubscribeConnection()
      client.disconnect()
      clientRef.current = null
    }
  }, [everExpanded])

  useEffect(() => {
    if (!everExpanded || !chatId || loadedChatRef.current === chatId) return
    let cancelled = false
    void fetchChat(chatId)
      .then(({ messages: loaded }) => {
        if (cancelled) return
        loadedChatRef.current = chatId
        setMessages(loaded.filter((message) => message.content || message.role === 'user'))
      })
      .catch(() => {
        if (!cancelled) setError('Could not load meeting chat')
      })
    return () => {
      cancelled = true
    }
  }, [chatId, everExpanded])

  useEffect(() => {
    if (!expanded) return undefined
    const collapse = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') setExpanded(false)
    }
    window.addEventListener('keydown', collapse)
    return () => window.removeEventListener('keydown', collapse)
  }, [expanded])

  useEffect(() => {
    const node = threadRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [messages, streaming])

  const send = useCallback(
    (text: string): void => {
      const client = clientRef.current
      const targetChatId = chatId ?? ''
      if (!client?.send(targetChatId, text, [], meeting.meetingId)) {
        setError('Not connected')
        return
      }
      setError('')
      if (chatId) {
        setMessages((current) => [...current, localUserMessage(chatId, text)])
      } else {
        pendingTextRef.current = text
      }
    },
    [chatId, meeting.meetingId]
  )

  const stop = useCallback((): void => {
    if (chatId) clientRef.current?.cancel(chatId)
  }, [chatId])

  const regenerate = useCallback((): void => {
    if (chatId && !streaming) clientRef.current?.regenerate(chatId)
  }, [chatId, streaming])

  if (!expanded) {
    return (
      <div className="border-t border-white/[0.08] p-3">
        <button
          className="flex h-9 w-full items-center rounded-lg border border-white/[0.08] bg-white/[0.025] px-3 text-left text-xs text-fg/45 transition-colors hover:border-white/[0.14] hover:text-fg/70"
          type="button"
          aria-expanded="false"
          onClick={() => setExpanded(true)}
        >
          Ask about this meeting…
        </button>
      </div>
    )
  }

  const lastAssistant = [...messages].reverse().find((message) => message.role === 'assistant')

  return (
    <section className="grid h-[min(42vh,380px)] min-h-[220px] grid-rows-[auto_minmax(0,1fr)_auto] border-t border-white/[0.08] bg-surface-1/95">
      <header className="flex h-9 items-center justify-between border-b border-white/[0.07] px-4">
        <div className="flex items-center gap-2 text-xs font-medium text-fg/70">
          <span>Ask about this meeting</span>
          <span className="text-[10px] font-normal text-fg/35">{connectionLabel(connection)}</span>
        </div>
        <button
          className="border-0 bg-transparent text-sm text-fg/45 hover:text-fg"
          type="button"
          aria-label="Collapse meeting chat"
          onClick={() => setExpanded(false)}
        >
          ⌄
        </button>
      </header>

      <div ref={threadRef} className="min-h-0 overflow-y-auto px-5 py-4">
        {messages.length === 0 && !streaming ? (
          <p className="m-0 text-xs text-fg/35">
            Ask about the notes, transcript, or summary. Answers can cite meeting evidence as [M#].
          </p>
        ) : (
          <div className="flex flex-col gap-4">
            {messages.map((message) =>
              message.role === 'user' ? (
                <div
                  key={message.id}
                  className="ml-auto max-w-[82%] rounded-xl bg-white/[0.07] px-3 py-2 text-[13px] leading-relaxed text-fg/85"
                >
                  {message.content}
                </div>
              ) : (
                <AssistantTurn
                  key={message.id}
                  message={message}
                  isLast={message.id === lastAssistant?.id}
                  onOpenNote={onOpenNote}
                  onRegenerate={regenerate}
                />
              )
            )}
            {streaming ? (
              <AssistantTurn
                streaming={streaming}
                isLast
                onOpenNote={onOpenNote}
                onRegenerate={regenerate}
              />
            ) : null}
          </div>
        )}
        {error ? <div className="mt-3 text-xs text-bad">{error}</div> : null}
      </div>

      <Composer
        disabled={connection !== 'connected'}
        streaming={streaming !== null && streaming.phase !== 'error'}
        onSend={send}
        onStop={stop}
        focusKey={`${meeting.meetingId}:${expanded ? 'open' : 'closed'}`}
      />
    </section>
  )
}

interface EventState {
  pendingTextRef: React.MutableRefObject<string | null>
  loadedChatRef: React.MutableRefObject<string | null>
  setChatId: React.Dispatch<React.SetStateAction<string | null>>
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>
  setStreaming: React.Dispatch<React.SetStateAction<StreamingTurn | null>>
  setError: React.Dispatch<React.SetStateAction<string>>
}

function applyEvent(event: ChatEvent, state: EventState): void {
  switch (event.type) {
    case 'chat_ack': {
      const pending = state.pendingTextRef.current
      if (pending !== null) {
        state.pendingTextRef.current = null
        state.loadedChatRef.current = event.chat_id
        state.setChatId(event.chat_id)
        state.setMessages([localUserMessage(event.chat_id, pending, event.user_message_id)])
      }
      state.setStreaming({ ...EMPTY_STREAM, messageId: event.message_id })
      return
    }
    case 'chat_retrieval':
      state.setStreaming((current) =>
        current?.messageId === event.message_id
          ? {
              ...current,
              evidence: event.evidence,
              retrievedIds: event.retrieved_ids,
              consultedIds: event.consulted_ids
            }
          : current
      )
      return
    case 'chat_thinking':
      state.setStreaming((current) =>
        current?.messageId === event.message_id
          ? {
              ...current,
              phase: 'thinking',
              thinking: current.thinking + event.delta,
              thinkingStartedAt: current.thinkingStartedAt ?? Date.now() / 1000,
              thinkingEndedAt: Date.now() / 1000
            }
          : current
      )
      return
    case 'chat_activity':
      state.setStreaming((current) => {
        if (!current || current.messageId !== event.message_id) return current
        const activity = [...current.activity]
        activity[event.step_index] = event.step
        return { ...current, activity }
      })
      return
    case 'chat_delta':
      state.setStreaming((current) =>
        current?.messageId === event.message_id
          ? { ...current, phase: 'answering', content: current.content + event.delta }
          : current
      )
      return
    case 'chat_done':
      state.setMessages((current) => [
        ...current.filter((message) => message.id !== event.message.id),
        event.message
      ])
      state.setStreaming(null)
      return
    case 'chat_error':
      if (event.code === 'busy') {
        state.setError('This meeting chat is already answering')
        return
      }
      state.setStreaming((current) =>
        current ? { ...current, phase: 'error', error: event.detail || event.code } : current
      )
      return
    default:
      return
  }
}

function localUserMessage(chatId: string, text: string, id = -Date.now()): ChatMessage {
  return {
    id,
    chat_id: chatId,
    role: 'user',
    content: text,
    thinking: '',
    evidence: null,
    retrieved_ids: [],
    consulted_ids: [],
    cited_ids: [],
    activity: [],
    usage: { prompt: 0, completion: 0, reasoning: 0, ttft_ms: 0, total_ms: 0 },
    cancelled: false,
    created: Date.now() / 1000
  }
}

function connectionLabel(connection: ChatConnectionState): string {
  if (connection === 'connected') return 'Connected'
  if (connection === 'reconnecting') return 'Reconnecting…'
  if (connection === 'connecting') return 'Connecting…'
  return 'Offline'
}
