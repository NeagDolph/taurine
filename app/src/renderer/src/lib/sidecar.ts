export const SIDECAR_WS_URL = 'ws://127.0.0.1:8930/session'
export const SIDECAR_REST_URL = restBaseFromWsUrl(SIDECAR_WS_URL)

export type AsrState = 'disconnected' | 'connecting' | 'connected' | 'error'
export type CardProvenance = 'exact' | 'search' | 'linked'
export type SidecarConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
export type TriggerAction = 'pinned' | 'slated' | 'none'
export type TriggerKind = 'exact' | 'search'
export type ReenrichPhase = 'idle' | 'profile' | 'notes' | 'finalizing'
export type ConnectorKind = 'local' | 'oauth'

export interface TranscriptEvent {
  type: 'transcript'
  channel: number
  text: string
  final: boolean
  ts: number
  seq: number
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
  retrieval_id: string
  seqs: number[]
}

export interface TriggerMatch {
  note_id: string
  title: string
  rank?: number | null
  score: number
  via_note_id?: string | null
  hop?: number | null
}

export interface TriggerEvent {
  type: 'trigger'
  trigger_id: string
  kind: TriggerKind
  seqs: number[]
  span: string
  matches: TriggerMatch[]
  action: TriggerAction
}

export interface ReenrichStatus {
  running: boolean
  phase: ReenrichPhase
  done: number
  total: number
}

export interface IndexStatus {
  notes: number
  last_sync?: number | null
  reenrich?: ReenrichStatus | null
}

export interface ConnectorInfo {
  connector_id: string
  label: string
  kind: ConnectorKind
  enabled: boolean
  available: boolean
  detail: string
  docs: number
  last_sync?: number | null
  syncing: boolean
  config: Record<string, unknown>
}

export interface ConnectorSyncReport {
  connector_id: string
  added: number
  changed: number
  removed: number
  enrich_failures: number
  error: string
}

export interface NeighborRef {
  note_id: string
  title: string
  source: string
  weight: number
}

export interface NotePreview {
  note_id: string
  title: string
  source: string
  gist: string
  text: string
  neighbors: NeighborRef[]
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
  | TriggerEvent
  | StatusEvent

export interface SidecarConnectionEvent {
  state: SidecarConnectionState
  retryInMs?: number
}

interface SidecarRequestOptions {
  baseUrl?: string
  fetcher?: typeof fetch
  signal?: AbortSignal
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
  private readonly triggerCallbacks = new Set<Callback<TriggerEvent>>()
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
    await requestSidecar('/index/sync', { method: 'POST' }, { baseUrl: this.restBaseUrl })
  }

  async reenrichIndex(): Promise<ReenrichStatus> {
    const payload = await requestSidecarJson('/index/reenrich', { method: 'POST' }, { baseUrl: this.restBaseUrl })
    return parseReenrichStatus(payload) ?? {
      running: true,
      phase: 'profile',
      done: 0,
      total: 0
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

  onTrigger(callback: Callback<TriggerEvent>): Unsubscribe {
    return this.addCallback(this.triggerCallbacks, callback)
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
      case 'trigger':
        this.emit(this.triggerCallbacks, event)
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

export async function fetchConnectors(
  options: SidecarRequestOptions = {}
): Promise<ConnectorInfo[]> {
  const payload = await requestSidecarJson('/connectors', { method: 'GET' }, options)
  if (!Array.isArray(payload)) {
    throw new Error('Invalid connectors response')
  }

  return payload.map(parseConnectorInfo)
}

export async function enableConnector(
  connectorId: string,
  options: SidecarRequestOptions = {}
): Promise<void> {
  await requestSidecar(`/connectors/${encodeURIComponent(connectorId)}/enable`, { method: 'POST' }, options)
}

export async function disableConnector(
  connectorId: string,
  purge: boolean,
  options: SidecarRequestOptions = {}
): Promise<void> {
  const suffix = purge ? '?purge=true' : ''
  await requestSidecar(
    `/connectors/${encodeURIComponent(connectorId)}/disable${suffix}`,
    { method: 'POST' },
    options
  )
}

export async function syncConnector(
  connectorId: string,
  options: SidecarRequestOptions = {}
): Promise<ConnectorSyncReport | null> {
  const response = await requestSidecar(
    `/connectors/${encodeURIComponent(connectorId)}/sync`,
    { method: 'POST' },
    options
  )
  const payload = await responseJsonOrNull(response)
  return payload === null ? null : parseConnectorSyncReport(payload)
}

export async function updateConnectorConfig(
  connectorId: string,
  config: Record<string, unknown>,
  options: SidecarRequestOptions = {}
): Promise<void> {
  await requestSidecar(
    `/connectors/${encodeURIComponent(connectorId)}/config`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    },
    options
  )
}

export async function fetchNotePreview(
  noteId: string,
  options: SidecarRequestOptions = {}
): Promise<NotePreview> {
  const payload = await requestSidecarJson(`/notes/${encodeURIComponent(noteId)}`, { method: 'GET' }, options)
  return parseNotePreview(payload)
}

async function requestSidecarJson(
  path: string,
  init: RequestInit,
  options: SidecarRequestOptions
): Promise<unknown> {
  const response = await requestSidecar(path, init, options)
  return responseJsonOrNull(response)
}

async function requestSidecar(
  path: string,
  init: RequestInit,
  options: SidecarRequestOptions
): Promise<Response> {
  const baseUrl = options.baseUrl ?? SIDECAR_REST_URL
  const fetcher = options.fetcher ?? globalThis.fetch
  const response = await fetcher(`${baseUrl}${path}`, { ...init, signal: options.signal })

  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, path))
  }

  return response
}

async function responseJsonOrNull(response: Response): Promise<unknown> {
  if (response.status === 204) {
    return null
  }

  const text = await response.text()
  if (!text.trim()) {
    return null
  }

  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new Error('Invalid sidecar JSON response')
  }
}

async function responseErrorMessage(response: Response, path: string): Promise<string> {
  const fallback = `Sidecar request ${path} failed with HTTP ${response.status}`
  const text = await response.text().catch(() => '')
  if (!text.trim()) {
    return response.status === 409 ? 'Connector is already syncing' : fallback
  }

  try {
    const payload = JSON.parse(text) as unknown
    const record = asRecord(payload)
    if (record && typeof record.detail === 'string' && record.detail.trim()) {
      return record.detail
    }
    if (record && typeof record.error === 'string' && record.error.trim()) {
      return record.error
    }
  } catch {
    // plain-text error body
  }

  return text.trim()
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
    case 'trigger':
      return parseTrigger(record)
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
    typeof record.ts !== 'number' ||
    typeof record.seq !== 'number'
  ) {
    return null
  }

  return {
    type: 'transcript',
    channel: record.channel,
    text: record.text,
    final: record.final,
    ts: record.ts,
    seq: record.seq
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
  const seqs = record.seqs === undefined ? [] : parseNumberArray(record.seqs)
  if (
    hits.some((hit) => hit === null) ||
    expansions.some((expansion) => expansion === null) ||
    seqs === null ||
    (record.retrieval_id !== undefined && typeof record.retrieval_id !== 'string')
  ) {
    return null
  }

  return {
    type: 'retrieval',
    heard: record.heard,
    hits: hits as Hit[],
    expansions: expansions as Expansion[],
    retrieval_id: typeof record.retrieval_id === 'string' ? record.retrieval_id : '',
    seqs
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

function parseTrigger(record: Record<string, unknown>): TriggerEvent | null {
  if (
    typeof record.trigger_id !== 'string' ||
    !isTriggerKind(record.kind) ||
    !Array.isArray(record.seqs) ||
    typeof record.span !== 'string' ||
    !Array.isArray(record.matches) ||
    !isTriggerAction(record.action)
  ) {
    return null
  }

  const seqs = parseNumberArray(record.seqs)
  const matches = record.matches.map(parseTriggerMatch)
  if (seqs === null || matches.some((match) => match === null)) {
    return null
  }

  return {
    type: 'trigger',
    trigger_id: record.trigger_id,
    kind: record.kind,
    seqs,
    span: record.span,
    matches: matches as TriggerMatch[],
    action: record.action
  }
}

function parseTriggerMatch(value: unknown): TriggerMatch | null {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.note_id !== 'string' ||
    typeof record.title !== 'string' ||
    typeof record.score !== 'number'
  ) {
    return null
  }

  if (record.rank !== undefined && record.rank !== null && typeof record.rank !== 'number') {
    return null
  }
  if (
    record.via_note_id !== undefined &&
    record.via_note_id !== null &&
    typeof record.via_note_id !== 'string'
  ) {
    return null
  }
  if (record.hop !== undefined && record.hop !== null && typeof record.hop !== 'number') {
    return null
  }

  return {
    note_id: record.note_id,
    title: record.title,
    rank: typeof record.rank === 'number' ? record.rank : null,
    score: record.score,
    via_note_id: typeof record.via_note_id === 'string' ? record.via_note_id : null,
    hop: typeof record.hop === 'number' ? record.hop : null
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

  const reenrich = parseReenrichStatus(index.reenrich)

  return {
    type: 'status',
    asr: record.asr,
    index: {
      notes: index.notes,
      last_sync: lastSync,
      reenrich
    },
    detail: typeof record.detail === 'string' ? record.detail : ''
  }
}

function parseConnectorInfo(value: unknown): ConnectorInfo {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.connector_id !== 'string' ||
    typeof record.label !== 'string' ||
    !isConnectorKind(record.kind) ||
    typeof record.enabled !== 'boolean' ||
    typeof record.available !== 'boolean'
  ) {
    throw new Error('Invalid connector info')
  }

  const lastSync = record.last_sync
  if (lastSync !== undefined && lastSync !== null && typeof lastSync !== 'number') {
    throw new Error('Invalid connector info')
  }

  return {
    connector_id: record.connector_id,
    label: record.label,
    kind: record.kind,
    enabled: record.enabled,
    available: record.available,
    detail: typeof record.detail === 'string' ? record.detail : '',
    docs: typeof record.docs === 'number' ? record.docs : 0,
    last_sync: typeof lastSync === 'number' ? lastSync : null,
    syncing: typeof record.syncing === 'boolean' ? record.syncing : false,
    config: asRecord(record.config) ?? {}
  }
}

function parseConnectorSyncReport(value: unknown): ConnectorSyncReport {
  const record = asRecord(value)
  if (!record || typeof record.connector_id !== 'string') {
    throw new Error('Invalid connector sync report')
  }

  return {
    connector_id: record.connector_id,
    added: typeof record.added === 'number' ? record.added : 0,
    changed: typeof record.changed === 'number' ? record.changed : 0,
    removed: typeof record.removed === 'number' ? record.removed : 0,
    enrich_failures: typeof record.enrich_failures === 'number' ? record.enrich_failures : 0,
    error: typeof record.error === 'string' ? record.error : ''
  }
}

function parseNotePreview(value: unknown): NotePreview {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.note_id !== 'string' ||
    typeof record.title !== 'string' ||
    typeof record.source !== 'string' ||
    typeof record.gist !== 'string' ||
    typeof record.text !== 'string' ||
    !Array.isArray(record.neighbors)
  ) {
    throw new Error('Invalid note preview')
  }

  return {
    note_id: record.note_id,
    title: record.title,
    source: record.source,
    gist: record.gist,
    text: record.text,
    neighbors: record.neighbors.map(parseNeighborRef)
  }
}

function parseNeighborRef(value: unknown): NeighborRef {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.note_id !== 'string' ||
    typeof record.title !== 'string' ||
    typeof record.source !== 'string' ||
    typeof record.weight !== 'number'
  ) {
    throw new Error('Invalid preview neighbor')
  }

  return {
    note_id: record.note_id,
    title: record.title,
    source: record.source,
    weight: record.weight
  }
}

function parseReenrichStatus(value: unknown): ReenrichStatus | null {
  if (value === undefined || value === null) {
    return null
  }

  const record = asRecord(value)
  if (
    !record ||
    typeof record.running !== 'boolean' ||
    !isReenrichPhase(record.phase) ||
    typeof record.done !== 'number' ||
    typeof record.total !== 'number'
  ) {
    return null
  }

  return {
    running: record.running,
    phase: record.phase,
    done: record.done,
    total: record.total
  }
}

function isCardProvenance(value: unknown): value is CardProvenance {
  return value === 'exact' || value === 'search' || value === 'linked'
}

function isTriggerKind(value: unknown): value is TriggerKind {
  return value === 'exact' || value === 'search'
}

function isTriggerAction(value: unknown): value is TriggerAction {
  return value === 'pinned' || value === 'slated' || value === 'none'
}

function isReenrichPhase(value: unknown): value is ReenrichPhase {
  return value === 'idle' || value === 'profile' || value === 'notes' || value === 'finalizing'
}

function isConnectorKind(value: unknown): value is ConnectorKind {
  return value === 'local' || value === 'oauth'
}

function isAsrState(value: unknown): value is AsrState {
  return value === 'disconnected' || value === 'connecting' || value === 'connected' || value === 'error'
}

function parseNumberArray(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'number')) {
    return null
  }

  return value
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
