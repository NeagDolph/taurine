import type { EvidencePack } from './sidecar'
import { parseEvidencePack, SIDECAR_PORT, SIDECAR_REST_URL } from './sidecar'

export const CHAT_WS_URL = `ws://127.0.0.1:${SIDECAR_PORT}/chat`

export interface ChatMeta {
  chat_id: string
  title: string
  created: number
  updated: number
  message_count: number
}

export type ChatActivityKind = 'search' | 'read' | 'expand'

export interface ChatActivityStep {
  kind: ChatActivityKind
  label: string
  status: 'running' | 'done'
  note_ids: string[]
}

export interface ChatUsage {
  prompt: number
  completion: number
  reasoning: number
  ttft_ms: number
  total_ms: number
}

export interface ChatMessage {
  id: number
  chat_id: string
  role: 'user' | 'assistant'
  content: string
  thinking: string
  evidence: EvidencePack | null
  retrieved_ids: string[]
  consulted_ids: string[]
  cited_ids: string[]
  activity: ChatActivityStep[]
  usage: ChatUsage
  cancelled: boolean
  created: number
}

export interface ChatAckEvent {
  type: 'chat_ack'
  chat_id: string
  message_id: number
  user_message_id: number
  title: string
}

export interface ChatRetrievalEvent {
  type: 'chat_retrieval'
  chat_id: string
  message_id: number
  evidence: EvidencePack
  retrieved_ids: string[]
  consulted_ids: string[]
}

export interface ChatThinkingEvent {
  type: 'chat_thinking'
  chat_id: string
  message_id: number
  delta: string
}

export interface ChatActivityEvent {
  type: 'chat_activity'
  chat_id: string
  message_id: number
  step_index: number
  step: ChatActivityStep
}

export interface ChatDeltaEvent {
  type: 'chat_delta'
  chat_id: string
  message_id: number
  delta: string
}

export interface ChatDoneEvent {
  type: 'chat_done'
  chat_id: string
  message_id: number
  message: ChatMessage
}

export interface ChatErrorEvent {
  type: 'chat_error'
  chat_id: string
  message_id: number | null
  code: 'busy' | 'llm' | 'timeout'
  detail: string
}

export interface ChatMetaEvent {
  type: 'chat_meta'
  chat: ChatMeta
}

export type ChatEvent =
  | ChatAckEvent
  | ChatRetrievalEvent
  | ChatThinkingEvent
  | ChatActivityEvent
  | ChatDeltaEvent
  | ChatDoneEvent
  | ChatErrorEvent
  | ChatMetaEvent

export type ChatConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'

type Callback<T> = (event: T) => void
type Unsubscribe = () => void

const INITIAL_RECONNECT_MS = 500
const MAX_RECONNECT_MS = 8000

export class ChatClient {
  private socket: WebSocket | null = null
  private reconnectTimer: number | null = null
  private reconnectAttempt = 0
  private closedByClient = false

  private readonly eventCallbacks = new Set<Callback<ChatEvent>>()
  private readonly connectionCallbacks = new Set<Callback<ChatConnectionState>>()

  constructor(private readonly url: string = CHAT_WS_URL) {}

  connect(): void {
    if (
      this.socket?.readyState === WebSocket.OPEN ||
      this.socket?.readyState === WebSocket.CONNECTING
    ) {
      return
    }

    this.closedByClient = false
    this.clearReconnectTimer()
    this.openSocket()
  }

  disconnect(): void {
    this.closedByClient = true
    this.clearReconnectTimer()
    this.socket?.close()
    this.socket = null
    this.emitConnection('disconnected')
  }

  send(chatId: string, text: string): boolean {
    return this.sendJson({ type: 'chat_send', chat_id: chatId, text })
  }

  cancel(chatId: string): boolean {
    return this.sendJson({ type: 'chat_cancel', chat_id: chatId })
  }

  regenerate(chatId: string): boolean {
    return this.sendJson({ type: 'chat_regenerate', chat_id: chatId })
  }

  onEvent(callback: Callback<ChatEvent>): Unsubscribe {
    this.eventCallbacks.add(callback)
    return () => this.eventCallbacks.delete(callback)
  }

  onConnection(callback: Callback<ChatConnectionState>): Unsubscribe {
    this.connectionCallbacks.add(callback)
    return () => this.connectionCallbacks.delete(callback)
  }

  private openSocket(): void {
    this.emitConnection(this.reconnectAttempt > 0 ? 'reconnecting' : 'connecting')

    const socket = new WebSocket(this.url)
    this.socket = socket

    socket.onopen = (): void => {
      this.reconnectAttempt = 0
      this.emitConnection('connected')
    }

    socket.onmessage = (message): void => {
      if (typeof message.data !== 'string') {
        return
      }
      const event = parseChatEvent(message.data)
      if (event) {
        this.eventCallbacks.forEach((callback) => callback(event))
      }
    }

    socket.onclose = (): void => {
      if (this.socket === socket) {
        this.socket = null
      }
      if (this.closedByClient) {
        this.emitConnection('disconnected')
        return
      }
      this.scheduleReconnect()
    }

    socket.onerror = (): void => {
      socket.close()
    }
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer()
    this.reconnectAttempt += 1
    const delay = Math.min(
      MAX_RECONNECT_MS,
      INITIAL_RECONNECT_MS * 2 ** (this.reconnectAttempt - 1)
    )
    this.emitConnection('reconnecting')
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null
      this.openSocket()
    }, delay)
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  private sendJson(payload: Record<string, unknown>): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      return false
    }
    this.socket.send(JSON.stringify(payload))
    return true
  }

  private emitConnection(state: ChatConnectionState): void {
    this.connectionCallbacks.forEach((callback) => callback(state))
  }
}

// ── REST ────────────────────────────────────────────────────────────────
export async function fetchChats(): Promise<ChatMeta[]> {
  const payload = await requestJson('/chats', { method: 'GET' })
  if (!Array.isArray(payload)) {
    throw new Error('Invalid chats response')
  }
  return payload.map(parseChatMetaStrict)
}

export async function fetchChat(
  chatId: string
): Promise<{ chat: ChatMeta; messages: ChatMessage[] }> {
  const payload = await requestJson(`/chats/${encodeURIComponent(chatId)}`, { method: 'GET' })
  const record = asRecord(payload)
  const chat = record ? parseChatMeta(record.chat) : null
  const messages = record && Array.isArray(record.messages) ? record.messages : null
  if (!chat || !messages) {
    throw new Error('Invalid chat response')
  }
  return { chat, messages: messages.map(parseChatMessageStrict) }
}

export async function createChat(): Promise<ChatMeta> {
  return parseChatMetaStrict(await requestJson('/chats', { method: 'POST' }))
}

export async function renameChat(chatId: string, title: string): Promise<void> {
  await requestJson(`/chats/${encodeURIComponent(chatId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  })
}

export async function deleteChat(chatId: string): Promise<void> {
  await requestJson(`/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' })
}

async function requestJson(path: string, init: RequestInit): Promise<unknown> {
  const response = await fetch(`${SIDECAR_REST_URL}${path}`, init)
  if (!response.ok) {
    throw new Error(`Chat request ${path} failed with HTTP ${response.status}`)
  }
  if (response.status === 204) {
    return null
  }
  const text = await response.text()
  return text.trim() ? (JSON.parse(text) as unknown) : null
}

// ── parsers (house style: tolerant hand-rolled guards) ─────────────────
export function parseChatEvent(raw: string): ChatEvent | null {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return null
  }
  const record = asRecord(value)
  if (!record || typeof record.type !== 'string') {
    return null
  }

  switch (record.type) {
    case 'chat_ack':
      if (typeof record.chat_id !== 'string' || typeof record.message_id !== 'number') {
        return null
      }
      return {
        type: 'chat_ack',
        chat_id: record.chat_id,
        message_id: record.message_id,
        user_message_id: typeof record.user_message_id === 'number' ? record.user_message_id : 0,
        title: typeof record.title === 'string' ? record.title : ''
      }
    case 'chat_retrieval': {
      if (typeof record.chat_id !== 'string' || typeof record.message_id !== 'number') {
        return null
      }
      const evidence = parseEvidencePack(record.evidence)
      if (!evidence) {
        return null
      }
      return {
        type: 'chat_retrieval',
        chat_id: record.chat_id,
        message_id: record.message_id,
        evidence,
        retrieved_ids: stringArray(record.retrieved_ids),
        consulted_ids: stringArray(record.consulted_ids)
      }
    }
    case 'chat_thinking':
    case 'chat_delta': {
      if (
        typeof record.chat_id !== 'string' ||
        typeof record.message_id !== 'number' ||
        typeof record.delta !== 'string'
      ) {
        return null
      }
      return {
        type: record.type,
        chat_id: record.chat_id,
        message_id: record.message_id,
        delta: record.delta
      } as ChatThinkingEvent | ChatDeltaEvent
    }
    case 'chat_activity': {
      const step = parseActivityStep(record.step)
      if (
        typeof record.chat_id !== 'string' ||
        typeof record.message_id !== 'number' ||
        typeof record.step_index !== 'number' ||
        !step
      ) {
        return null
      }
      return {
        type: 'chat_activity',
        chat_id: record.chat_id,
        message_id: record.message_id,
        step_index: record.step_index,
        step
      }
    }
    case 'chat_done': {
      const message = parseChatMessage(record.message)
      if (typeof record.chat_id !== 'string' || typeof record.message_id !== 'number' || !message) {
        return null
      }
      return {
        type: 'chat_done',
        chat_id: record.chat_id,
        message_id: record.message_id,
        message
      }
    }
    case 'chat_error': {
      if (typeof record.chat_id !== 'string' || !isErrorCode(record.code)) {
        return null
      }
      return {
        type: 'chat_error',
        chat_id: record.chat_id,
        message_id: typeof record.message_id === 'number' ? record.message_id : null,
        code: record.code,
        detail: typeof record.detail === 'string' ? record.detail : ''
      }
    }
    case 'chat_meta': {
      const chat = parseChatMeta(record.chat)
      return chat ? { type: 'chat_meta', chat } : null
    }
    default:
      return null
  }
}

function parseChatMeta(value: unknown): ChatMeta | null {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.chat_id !== 'string' ||
    typeof record.created !== 'number' ||
    typeof record.updated !== 'number'
  ) {
    return null
  }
  return {
    chat_id: record.chat_id,
    title: typeof record.title === 'string' ? record.title : '',
    created: record.created,
    updated: record.updated,
    message_count: typeof record.message_count === 'number' ? record.message_count : 0
  }
}

function parseChatMetaStrict(value: unknown): ChatMeta {
  const meta = parseChatMeta(value)
  if (!meta) {
    throw new Error('Invalid chat meta')
  }
  return meta
}

function parseChatMessage(value: unknown): ChatMessage | null {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.chat_id !== 'string' ||
    (record.role !== 'user' && record.role !== 'assistant') ||
    typeof record.content !== 'string'
  ) {
    return null
  }
  const usage = asRecord(record.usage)
  const evidence =
    record.evidence === null || record.evidence === undefined
      ? null
      : parseEvidencePack(record.evidence)
  if (record.evidence !== null && record.evidence !== undefined && !evidence) {
    return null
  }
  return {
    id: typeof record.id === 'number' ? record.id : 0,
    chat_id: record.chat_id,
    role: record.role,
    content: record.content,
    thinking: typeof record.thinking === 'string' ? record.thinking : '',
    evidence,
    retrieved_ids: stringArray(record.retrieved_ids),
    consulted_ids: stringArray(record.consulted_ids),
    cited_ids: stringArray(record.cited_ids),
    activity: parseArray(record.activity, parseActivityStep),
    usage: {
      prompt: numberOr(usage?.prompt, 0),
      completion: numberOr(usage?.completion, 0),
      reasoning: numberOr(usage?.reasoning, 0),
      ttft_ms: numberOr(usage?.ttft_ms, 0),
      total_ms: numberOr(usage?.total_ms, 0)
    },
    cancelled: record.cancelled === true,
    created: numberOr(record.created, 0)
  }
}

function parseChatMessageStrict(value: unknown): ChatMessage {
  const message = parseChatMessage(value)
  if (!message) {
    throw new Error('Invalid chat message')
  }
  return message
}

function parseActivityStep(value: unknown): ChatActivityStep | null {
  const record = asRecord(value)
  if (
    !record ||
    (record.kind !== 'search' && record.kind !== 'read' && record.kind !== 'expand') ||
    typeof record.label !== 'string'
  ) {
    return null
  }
  return {
    kind: record.kind,
    label: record.label,
    status: record.status === 'running' ? 'running' : 'done',
    note_ids: Array.isArray(record.note_ids)
      ? record.note_ids.filter((id): id is string => typeof id === 'string')
      : []
  }
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((row): row is string => typeof row === 'string') : []
}

function parseArray<T>(value: unknown, parse: (item: unknown) => T | null): T[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.map(parse).filter((item): item is T => item !== null)
}

function isErrorCode(value: unknown): value is 'busy' | 'llm' | 'timeout' {
  return value === 'busy' || value === 'llm' || value === 'timeout'
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' ? value : fallback
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null
  }
  return value as Record<string, unknown>
}
