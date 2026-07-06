export const SIDECAR_WS_URL = 'ws://127.0.0.1:8930/session'

export type AsrState = 'disconnected' | 'connecting' | 'connected' | 'error'
export type CardProvenance = 'exact' | 'search' | 'linked'
export type SidecarConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'

export interface TranscriptEvent {
  type: 'transcript'
  channel: number
  text: string
  final: boolean
  ts: number
}

export interface AmbientItem {
  note_id: string
  title: string
  gist: string
  score: number
}

export interface AmbientEvent {
  type: 'ambient'
  items: AmbientItem[]
}

export interface Card {
  card_id: string
  note_id: string
  title: string
  summary: string
  provenance: CardProvenance
  heard: string
}

export interface PinEvent {
  type: 'pin'
  card: Card
}

export interface UnpinEvent {
  type: 'unpin'
  card_id: string
}

export interface Hit {
  note_id: string
  rank: number
  score: number
}

export interface Expansion {
  from_note_id: string
  note_id: string
  hop: number
  score: number
}

export interface RetrievalEvent {
  type: 'retrieval'
  heard: string
  hits: Hit[]
  expansions: Expansion[]
}

export interface IndexStatus {
  notes: number
  last_sync?: number | null
}

export interface StatusEvent {
  type: 'status'
  asr: AsrState
  index: IndexStatus
  detail: string
}

export type SidecarEvent =
  | TranscriptEvent
  | AmbientEvent
  | PinEvent
  | UnpinEvent
  | RetrievalEvent
  | StatusEvent

export interface SidecarConnectionEvent {
  state: SidecarConnectionState
  retryInMs?: number
}

type Callback<T> = (event: T) => void
type Unsubscribe = () => void
type JsonControlMessage =
  | { type: 'start'; channels: Array<{ id: number; label: string }> }
  | { type: 'stop' }
  | { type: 'dismiss'; card_id: string }
  | { type: 'open_note'; note_id: string }

const INITIAL_RECONNECT_MS = 500
const MAX_RECONNECT_MS = 8000

export class SidecarClient {
  private socket: WebSocket | null = null
  private reconnectTimer: number | null = null
  private reconnectAttempt = 0
  private closedByClient = false
  private sessionActive = false
  private readonly restBaseUrl: string

  private readonly transcriptCallbacks = new Set<Callback<TranscriptEvent>>()
  private readonly ambientCallbacks = new Set<Callback<AmbientEvent>>()
  private readonly pinCallbacks = new Set<Callback<PinEvent>>()
  private readonly unpinCallbacks = new Set<Callback<UnpinEvent>>()
  private readonly retrievalCallbacks = new Set<Callback<RetrievalEvent>>()
  private readonly statusCallbacks = new Set<Callback<StatusEvent>>()
  private readonly connectionCallbacks = new Set<Callback<SidecarConnectionEvent>>()

  constructor(private readonly url: string = SIDECAR_WS_URL) {
    this.restBaseUrl = restBaseFromWsUrl(url)
  }

  connect(): void {
    if (this.socket?.readyState === WebSocket.OPEN || this.socket?.readyState === WebSocket.CONNECTING) {
      return
    }

    this.closedByClient = false
    this.clearReconnectTimer()
    this.openSocket()
  }

  disconnect(): void {
    this.closedByClient = true
    this.sessionActive = false
    this.clearReconnectTimer()
    this.socket?.close()
    this.socket = null
    this.emitConnection({ state: 'disconnected' })
  }

  start(channels: Array<{ id: number; label: string }> = [{ id: 0, label: 'mic' }]): void {
    this.sessionActive = true
    this.sendJson({ type: 'start', channels })
  }

  stop(): void {
    this.sessionActive = false
    this.sendJson({ type: 'stop' })
  }

  dismiss(cardId: string): void {
    this.sendJson({ type: 'dismiss', card_id: cardId })
  }

  openNote(noteId: string): void {
    this.sendJson({ type: 'open_note', note_id: noteId })
  }

  sendAudioFrame(frame: ArrayBuffer): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      return false
    }

    this.socket.send(frame)
    return true
  }

  async syncIndex(): Promise<void> {
    const response = await fetch(`${this.restBaseUrl}/index/sync`, { method: 'POST' })
    if (!response.ok) {
      throw new Error(`Sync failed with HTTP ${response.status}`)
    }
  }

  onTranscript(callback: Callback<TranscriptEvent>): Unsubscribe {
    return this.addCallback(this.transcriptCallbacks, callback)
  }

  onAmbient(callback: Callback<AmbientEvent>): Unsubscribe {
    return this.addCallback(this.ambientCallbacks, callback)
  }

  onPin(callback: Callback<PinEvent>): Unsubscribe {
    return this.addCallback(this.pinCallbacks, callback)
  }

  onUnpin(callback: Callback<UnpinEvent>): Unsubscribe {
    return this.addCallback(this.unpinCallbacks, callback)
  }

  onRetrieval(callback: Callback<RetrievalEvent>): Unsubscribe {
    return this.addCallback(this.retrievalCallbacks, callback)
  }

  onStatus(callback: Callback<StatusEvent>): Unsubscribe {
    return this.addCallback(this.statusCallbacks, callback)
  }

  onConnection(callback: Callback<SidecarConnectionEvent>): Unsubscribe {
    return this.addCallback(this.connectionCallbacks, callback)
  }

  private openSocket(): void {
    this.emitConnection({ state: this.reconnectAttempt > 0 ? 'reconnecting' : 'connecting' })

    const socket = new WebSocket(this.url)
    socket.binaryType = 'arraybuffer'
    this.socket = socket

    socket.onopen = (): void => {
      this.reconnectAttempt = 0
      this.emitConnection({ state: 'connected' })

      if (this.sessionActive) {
        this.sendJson({ type: 'start', channels: [{ id: 0, label: 'mic' }] })
      }
    }

    socket.onmessage = (message): void => {
      if (typeof message.data !== 'string') {
        return
      }

      const event = parseSidecarEvent(message.data)
      if (!event) {
        return
      }

      this.emitEvent(event)
    }

    socket.onclose = (): void => {
      if (this.socket === socket) {
        this.socket = null
      }

      if (this.closedByClient) {
        this.emitConnection({ state: 'disconnected' })
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

    const delay = Math.min(MAX_RECONNECT_MS, INITIAL_RECONNECT_MS * 2 ** (this.reconnectAttempt - 1))
    this.emitConnection({ state: 'reconnecting', retryInMs: delay })
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

  private sendJson(message: JsonControlMessage): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      return false
    }

    this.socket.send(JSON.stringify(message))
    return true
  }

  private emitEvent(event: SidecarEvent): void {
    switch (event.type) {
      case 'transcript':
        this.emit(this.transcriptCallbacks, event)
        break
      case 'ambient':
        this.emit(this.ambientCallbacks, event)
        break
      case 'pin':
        this.emit(this.pinCallbacks, event)
        break
      case 'unpin':
        this.emit(this.unpinCallbacks, event)
        break
      case 'retrieval':
        this.emit(this.retrievalCallbacks, event)
        break
      case 'status':
        this.emit(this.statusCallbacks, event)
        break
    }
  }

  private emitConnection(event: SidecarConnectionEvent): void {
    this.emit(this.connectionCallbacks, event)
  }

  private emit<T>(callbacks: Set<Callback<T>>, event: T): void {
    callbacks.forEach((callback) => callback(event))
  }

  private addCallback<T>(callbacks: Set<Callback<T>>, callback: Callback<T>): Unsubscribe {
    callbacks.add(callback)
    return () => callbacks.delete(callback)
  }
}

function parseSidecarEvent(raw: string): SidecarEvent | null {
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
    case 'transcript':
      return parseTranscript(record)
    case 'ambient':
      return parseAmbient(record)
    case 'pin':
      return parsePin(record)
    case 'unpin':
      return parseUnpin(record)
    case 'retrieval':
      return parseRetrieval(record)
    case 'status':
      return parseStatus(record)
    default:
      return null
  }
}

function parseTranscript(record: Record<string, unknown>): TranscriptEvent | null {
  if (
    typeof record.channel !== 'number' ||
    typeof record.text !== 'string' ||
    typeof record.final !== 'boolean' ||
    typeof record.ts !== 'number'
  ) {
    return null
  }

  return {
    type: 'transcript',
    channel: record.channel,
    text: record.text,
    final: record.final,
    ts: record.ts
  }
}

function parseAmbient(record: Record<string, unknown>): AmbientEvent | null {
  if (!Array.isArray(record.items)) {
    return null
  }

  const items = record.items.map(parseAmbientItem)
  if (items.some((item) => item === null)) {
    return null
  }

  return { type: 'ambient', items: items as AmbientItem[] }
}

function parseAmbientItem(value: unknown): AmbientItem | null {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.note_id !== 'string' ||
    typeof record.title !== 'string' ||
    typeof record.gist !== 'string' ||
    typeof record.score !== 'number'
  ) {
    return null
  }

  return {
    note_id: record.note_id,
    title: record.title,
    gist: record.gist,
    score: record.score
  }
}

function parsePin(record: Record<string, unknown>): PinEvent | null {
  const card = parseCard(record.card)
  if (!card) {
    return null
  }

  return { type: 'pin', card }
}

function parseCard(value: unknown): Card | null {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.card_id !== 'string' ||
    typeof record.note_id !== 'string' ||
    typeof record.title !== 'string' ||
    typeof record.summary !== 'string' ||
    !isCardProvenance(record.provenance) ||
    typeof record.heard !== 'string'
  ) {
    return null
  }

  return {
    card_id: record.card_id,
    note_id: record.note_id,
    title: record.title,
    summary: record.summary,
    provenance: record.provenance,
    heard: record.heard
  }
}

function parseUnpin(record: Record<string, unknown>): UnpinEvent | null {
  if (typeof record.card_id !== 'string') {
    return null
  }

  return { type: 'unpin', card_id: record.card_id }
}

function parseRetrieval(record: Record<string, unknown>): RetrievalEvent | null {
  if (typeof record.heard !== 'string' || !Array.isArray(record.hits) || !Array.isArray(record.expansions)) {
    return null
  }

  const hits = record.hits.map(parseHit)
  const expansions = record.expansions.map(parseExpansion)
  if (hits.some((hit) => hit === null) || expansions.some((expansion) => expansion === null)) {
    return null
  }

  return {
    type: 'retrieval',
    heard: record.heard,
    hits: hits as Hit[],
    expansions: expansions as Expansion[]
  }
}

function parseHit(value: unknown): Hit | null {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.note_id !== 'string' ||
    typeof record.rank !== 'number' ||
    typeof record.score !== 'number'
  ) {
    return null
  }

  return {
    note_id: record.note_id,
    rank: record.rank,
    score: record.score
  }
}

function parseExpansion(value: unknown): Expansion | null {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.from_note_id !== 'string' ||
    typeof record.note_id !== 'string' ||
    typeof record.hop !== 'number' ||
    typeof record.score !== 'number'
  ) {
    return null
  }

  return {
    from_note_id: record.from_note_id,
    note_id: record.note_id,
    hop: record.hop,
    score: record.score
  }
}

function parseStatus(record: Record<string, unknown>): StatusEvent | null {
  const index = asRecord(record.index)
  if (!isAsrState(record.asr) || !index || typeof index.notes !== 'number') {
    return null
  }

  const lastSync = index.last_sync
  if (lastSync !== undefined && lastSync !== null && typeof lastSync !== 'number') {
    return null
  }

  return {
    type: 'status',
    asr: record.asr,
    index: {
      notes: index.notes,
      last_sync: lastSync
    },
    detail: typeof record.detail === 'string' ? record.detail : ''
  }
}

function isCardProvenance(value: unknown): value is CardProvenance {
  return value === 'exact' || value === 'search' || value === 'linked'
}

function isAsrState(value: unknown): value is AsrState {
  return value === 'disconnected' || value === 'connecting' || value === 'connected' || value === 'error'
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return null
  }

  return value as Record<string, unknown>
}

function restBaseFromWsUrl(url: string): string {
  const parsed = new URL(url)
  parsed.protocol = parsed.protocol === 'wss:' ? 'https:' : 'http:'
  parsed.pathname = ''
  parsed.search = ''
  parsed.hash = ''
  return parsed.toString().replace(/\/$/, '')
}
