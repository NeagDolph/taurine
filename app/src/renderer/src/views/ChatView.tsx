import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { AssistantTurn, type StreamingTurn } from '../components/chat/AssistantTurn'
import { ChatSidebar } from '../components/chat/ChatSidebar'
import { Composer } from '../components/chat/Composer'
import {
  ChatClient,
  deleteChat as deleteChatRequest,
  fetchChat,
  fetchChats,
  renameChat as renameChatRequest,
  type ChatConnectionState,
  type ChatEvent,
  type ChatMessage,
  type ChatMeta
} from '../lib/chat'

interface ChatState {
  chats: ChatMeta[]
  activeChatId: string | null
  messagesByChat: Record<string, ChatMessage[]>
  loadedChats: Set<string>
  streamingByChat: Record<string, StreamingTurn>
  pendingSend: string | null // text sent with chat_id="" awaiting ack
}

type ChatAction =
  | { kind: 'chats_loaded'; chats: ChatMeta[] }
  | { kind: 'chat_loaded'; chatId: string; messages: ChatMessage[] }
  | { kind: 'select'; chatId: string | null }
  | { kind: 'local_user_message'; chatId: string; text: string }
  | { kind: 'pending_send'; text: string }
  | { kind: 'meta'; chat: ChatMeta }
  | { kind: 'deleted'; chatId: string }
  | { kind: 'event'; event: ChatEvent; now: number }

const EMPTY_STREAM: Omit<StreamingTurn, 'messageId'> = {
  phase: 'retrieving',
  thinking: '',
  thinkingStartedAt: null,
  thinkingEndedAt: null,
  activity: [],
  content: '',
  sources: [],
  facts: [],
  error: ''
}

function reducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.kind) {
    case 'chats_loaded':
      return { ...state, chats: action.chats }
    case 'chat_loaded':
      return {
        ...state,
        messagesByChat: { ...state.messagesByChat, [action.chatId]: action.messages },
        loadedChats: new Set(state.loadedChats).add(action.chatId)
      }
    case 'select':
      return { ...state, activeChatId: action.chatId }
    case 'local_user_message': {
      const local: ChatMessage = {
        id: -Date.now(),
        chat_id: action.chatId,
        role: 'user',
        content: action.text,
        thinking: '',
        sources: [],
        facts: [],
        activity: [],
        usage: { prompt: 0, completion: 0, reasoning: 0, ttft_ms: 0, total_ms: 0 },
        cancelled: false,
        created: Date.now() / 1000
      }
      return appendMessage(state, action.chatId, local)
    }
    case 'pending_send':
      return { ...state, pendingSend: action.text }
    case 'meta': {
      const rest = state.chats.filter((chat) => chat.chat_id !== action.chat.chat_id)
      return { ...state, chats: sortChats([action.chat, ...rest]) }
    }
    case 'deleted': {
      const messagesByChat = { ...state.messagesByChat }
      delete messagesByChat[action.chatId]
      const streamingByChat = { ...state.streamingByChat }
      delete streamingByChat[action.chatId]
      const chats = state.chats.filter((chat) => chat.chat_id !== action.chatId)
      return {
        ...state,
        chats,
        messagesByChat,
        streamingByChat,
        activeChatId:
          state.activeChatId === action.chatId ? (chats[0]?.chat_id ?? null) : state.activeChatId
      }
    }
    case 'event':
      return applyEvent(state, action.event, action.now)
    default:
      return state
  }
}

function applyEvent(state: ChatState, event: ChatEvent, now: number): ChatState {
  switch (event.type) {
    case 'chat_ack': {
      let next = state
      // a send with chat_id="" just created this chat: adopt it
      if (state.pendingSend !== null) {
        next = {
          ...appendMessage(state, event.chat_id, {
            id: event.user_message_id,
            chat_id: event.chat_id,
            role: 'user',
            content: state.pendingSend,
            thinking: '',
            sources: [],
            facts: [],
            activity: [],
            usage: { prompt: 0, completion: 0, reasoning: 0, ttft_ms: 0, total_ms: 0 },
            cancelled: false,
            created: now
          }),
          pendingSend: null,
          activeChatId: event.chat_id
        }
        next.loadedChats = new Set(next.loadedChats).add(event.chat_id)
      }
      return {
        ...next,
        streamingByChat: {
          ...next.streamingByChat,
          [event.chat_id]: { ...EMPTY_STREAM, messageId: event.message_id }
        }
      }
    }
    case 'chat_retrieval':
      return patchStream(state, event.chat_id, event.message_id, (turn) => ({
        ...turn,
        sources: event.sources,
        facts: event.facts
      }))
    case 'chat_thinking':
      return patchStream(state, event.chat_id, event.message_id, (turn) => ({
        ...turn,
        phase: 'thinking',
        thinking: turn.thinking + event.delta,
        thinkingStartedAt: turn.thinkingStartedAt ?? now,
        thinkingEndedAt: now
      }))
    case 'chat_activity':
      return patchStream(state, event.chat_id, event.message_id, (turn) => {
        const activity = [...turn.activity]
        activity[event.step_index] = event.step
        return { ...turn, activity }
      })
    case 'chat_delta':
      return patchStream(state, event.chat_id, event.message_id, (turn) => ({
        ...turn,
        phase: 'answering',
        content: turn.content + event.delta
      }))
    case 'chat_done': {
      const streamingByChat = { ...state.streamingByChat }
      delete streamingByChat[event.chat_id]
      // regenerate replaces the previous assistant row with the same id
      const withoutOld = removeMessage(state, event.chat_id, event.message.id)
      return {
        ...appendMessage({ ...withoutOld, streamingByChat }, event.chat_id, event.message)
      }
    }
    case 'chat_error': {
      if (event.code === 'busy') {
        return state // toast handled by the view; stream state untouched
      }
      return patchStream(state, event.chat_id, event.message_id ?? -1, (turn) => ({
        ...turn,
        phase: 'error',
        error: event.detail || event.code
      }))
    }
    case 'chat_meta': {
      const rest = state.chats.filter((chat) => chat.chat_id !== event.chat.chat_id)
      return { ...state, chats: sortChats([event.chat, ...rest]) }
    }
    default:
      return state
  }
}

function patchStream(
  state: ChatState,
  chatId: string,
  messageId: number,
  patch: (turn: StreamingTurn) => StreamingTurn
): ChatState {
  const turn = state.streamingByChat[chatId]
  if (!turn || (messageId >= 0 && turn.messageId !== messageId)) {
    return state
  }
  return {
    ...state,
    streamingByChat: { ...state.streamingByChat, [chatId]: patch(turn) }
  }
}

function appendMessage(state: ChatState, chatId: string, message: ChatMessage): ChatState {
  const existing = state.messagesByChat[chatId] ?? []
  return {
    ...state,
    messagesByChat: { ...state.messagesByChat, [chatId]: [...existing, message] }
  }
}

function removeMessage(state: ChatState, chatId: string, messageId: number): ChatState {
  const existing = state.messagesByChat[chatId]
  if (!existing?.some((message) => message.id === messageId)) {
    return state
  }
  return {
    ...state,
    messagesByChat: {
      ...state.messagesByChat,
      [chatId]: existing.filter((message) => message.id !== messageId)
    }
  }
}

function sortChats(chats: ChatMeta[]): ChatMeta[] {
  return [...chats].sort((left, right) => right.updated - left.updated)
}

const EXAMPLE_PROMPTS = [
  'What did we decide about the Shopify rollout?',
  'Summarize everything I know about transcription capacity',
  'Who owns SMS compliance and what is still open?'
]

interface ChatViewProps {
  visible: boolean
  onOpenNote: (noteId: string) => void
}

export function ChatView({ visible, onOpenNote }: ChatViewProps): React.JSX.Element {
  const clientRef = useRef<ChatClient | null>(null)
  const [state, dispatch] = useReducer(reducer, {
    chats: [],
    activeChatId: null,
    messagesByChat: {},
    loadedChats: new Set<string>(),
    streamingByChat: {},
    pendingSend: null
  })
  const [connection, setConnection] = useState<ChatConnectionState>('disconnected')
  const [toast, setToast] = useState('')
  const [everVisible, setEverVisible] = useState(false)
  const threadRef = useRef<HTMLDivElement | null>(null)
  const pinnedToBottom = useRef(true)
  const [showJump, setShowJump] = useState(false)

  // one-way latch, adjusted during render (no effect needed)
  if (visible && !everVisible) {
    setEverVisible(true)
  }

  // lazy ws + chat list on first tab focus; connection persists afterwards
  useEffect(() => {
    if (!everVisible) {
      return undefined
    }
    const client = new ChatClient()
    clientRef.current = client
    const unsubEvent = client.onEvent((event) =>
      dispatch({ kind: 'event', event, now: Date.now() / 1000 })
    )
    const unsubConn = client.onConnection(setConnection)
    client.connect()
    void fetchChats()
      .then((chats) => dispatch({ kind: 'chats_loaded', chats }))
      .catch(() => setToast('Could not load chats'))
    return () => {
      unsubEvent()
      unsubConn()
      client.disconnect()
      clientRef.current = null
    }
  }, [everVisible])

  // lazy history load per chat
  const activeChatId = state.activeChatId
  useEffect(() => {
    if (!activeChatId || state.loadedChats.has(activeChatId)) {
      return
    }
    void fetchChat(activeChatId)
      .then(({ messages }) => dispatch({ kind: 'chat_loaded', chatId: activeChatId, messages }))
      .catch(() => setToast('Could not load chat history'))
  }, [activeChatId, state.loadedChats])

  // auto-follow scroll
  const messages = activeChatId ? (state.messagesByChat[activeChatId] ?? []) : []
  const streaming = activeChatId ? state.streamingByChat[activeChatId] : undefined
  useEffect(() => {
    const node = threadRef.current
    if (node && pinnedToBottom.current) {
      node.scrollTop = node.scrollHeight
    }
  }, [messages, streaming])

  const handleScroll = useCallback((): void => {
    const node = threadRef.current
    if (!node) {
      return
    }
    const nearBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 48
    pinnedToBottom.current = nearBottom
    setShowJump(!nearBottom)
  }, [])

  const jumpToLatest = useCallback((): void => {
    const node = threadRef.current
    if (node) {
      pinnedToBottom.current = true
      setShowJump(false)
      node.scrollTop = node.scrollHeight
    }
  }, [])

  const sendMessage = useCallback(
    (text: string): void => {
      const client = clientRef.current
      if (!client) {
        return
      }
      pinnedToBottom.current = true
      if (activeChatId) {
        if (!client.send(activeChatId, text)) {
          setToast('Not connected')
          return
        }
        dispatch({ kind: 'local_user_message', chatId: activeChatId, text })
      } else {
        if (!client.send('', text)) {
          setToast('Not connected')
          return
        }
        dispatch({ kind: 'pending_send', text })
      }
    },
    [activeChatId]
  )

  const stopStreaming = useCallback((): void => {
    if (activeChatId) {
      clientRef.current?.cancel(activeChatId)
    }
  }, [activeChatId])

  const regenerate = useCallback((): void => {
    if (activeChatId && !state.streamingByChat[activeChatId]) {
      clientRef.current?.regenerate(activeChatId)
    }
  }, [activeChatId, state.streamingByChat])

  const renameChat = useCallback((chatId: string, title: string): void => {
    void renameChatRequest(chatId, title)
      .then(() => fetchChats())
      .then((chats) => dispatch({ kind: 'chats_loaded', chats }))
      .catch(() => setToast('Rename failed'))
  }, [])

  const removeChat = useCallback((chatId: string): void => {
    void deleteChatRequest(chatId)
      .then(() => dispatch({ kind: 'deleted', chatId }))
      .catch(() => setToast('Delete failed'))
  }, [])

  useEffect(() => {
    if (!toast) {
      return undefined
    }
    const timer = window.setTimeout(() => setToast(''), 2500)
    return () => window.clearTimeout(timer)
  }, [toast])

  const streamingChatIds = useMemo(
    () => new Set(Object.keys(state.streamingByChat)),
    [state.streamingByChat]
  )

  const lastAssistantId = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant') {
        return messages[i].id
      }
    }
    return null
  }, [messages])

  const emptyThread = messages.length === 0 && !streaming && state.pendingSend === null

  return (
    <div className="page chat-page">
      <ChatSidebar
        chats={state.chats}
        activeChatId={activeChatId}
        streamingChatIds={streamingChatIds}
        onSelect={(chatId) => dispatch({ kind: 'select', chatId })}
        onNewChat={() => dispatch({ kind: 'select', chatId: null })}
        onRename={renameChat}
        onDelete={removeChat}
      />

      <div className="chat-main">
        {connection !== 'connected' && everVisible ? (
          <div className="disconnect-banner" role="status">
            Chat {connection}
          </div>
        ) : null}

        <div className="chat-thread" ref={threadRef} onScroll={handleScroll}>
          {emptyThread ? (
            <div className="chat-empty">
              <h2>Ask your notes anything</h2>
              <div className="chat-examples">
                {EXAMPLE_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    className="chat-example"
                    type="button"
                    onClick={() => sendMessage(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {state.pendingSend !== null && !activeChatId ? (
                <UserBubble text={state.pendingSend} />
              ) : null}
              {messages.map((message) =>
                message.role === 'user' ? (
                  <UserBubble key={message.id} text={message.content} />
                ) : (
                  <AssistantTurn
                    key={message.id}
                    message={message}
                    isLast={message.id === lastAssistantId}
                    onOpenNote={onOpenNote}
                    onRegenerate={regenerate}
                  />
                )
              )}
              {streaming ? (
                <AssistantTurn
                  streaming={streaming}
                  isLast={false}
                  onOpenNote={onOpenNote}
                  onRegenerate={regenerate}
                  onRetry={regenerate}
                />
              ) : null}
            </>
          )}
          {showJump ? (
            <button className="jump-to-latest" type="button" onClick={jumpToLatest}>
              ↓ Jump to latest
            </button>
          ) : null}
        </div>

        <Composer
          disabled={connection !== 'connected'}
          streaming={Boolean(activeChatId && state.streamingByChat[activeChatId])}
          onSend={sendMessage}
          onStop={stopStreaming}
          focusKey={activeChatId}
        />

        {toast ? (
          <div className="chat-toast" role="status">
            {toast}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function UserBubble({ text }: { text: string }): React.JSX.Element {
  const [copied, setCopied] = useState(false)
  const copy = (): void => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    })
  }
  return (
    <div className="user-bubble-row">
      <button className="turn-action bubble-copy" type="button" onClick={copy}>
        {copied ? 'Copied' : 'Copy'}
      </button>
      <div className="user-bubble">{text}</div>
    </div>
  )
}
