import { SIDECAR_REST_URL } from './sidecar'

export type MeetingStatus = 'draft' | 'live' | 'summarizing' | 'done'
export type SuggestedEditStatus = 'pending' | 'accepted' | 'rejected'

export interface MeetingListRow {
  meetingId: string
  title: string
  createdAt: number
  updatedAt: number
  startedAt: number | null
  endedAt: number | null
  status: MeetingStatus
}

export interface SuggestedEdit {
  id: string
  anchor: string
  occurrence: number
  insertMd: string
  rationale: string
  status: SuggestedEditStatus
}

export interface TranscriptChunk {
  channel: number
  text: string
  ts: number
  seq: number
}

export interface Meeting extends MeetingListRow {
  notesMd: string
  notesRev: number
  finalSummaryMd: string
  suggestions: SuggestedEdit[]
  transcript: TranscriptChunk[]
  chatId: string | null
}

export interface MeetingPatch {
  title?: string
  notesMd?: string
  suggestions?: SuggestedEdit[]
}

interface MeetingsRequestOptions {
  baseUrl?: string
  fetcher?: typeof fetch
  signal?: AbortSignal
}

export async function listMeetings(
  options: MeetingsRequestOptions = {}
): Promise<MeetingListRow[]> {
  const payload = await requestMeetingsJson('/meetings', { method: 'GET' }, options)
  const record = asRecord(payload)
  if (!record || !Array.isArray(record.meetings)) {
    throw new Error('Invalid meetings response')
  }
  return record.meetings.map(parseMeetingListRow)
}

export async function createMeeting(
  title = '',
  options: MeetingsRequestOptions = {}
): Promise<Meeting> {
  const payload = await requestMeetingsJson(
    '/meetings',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(title ? { title } : {})
    },
    options
  )
  return parseMeeting(payload)
}

export async function getMeeting(
  meetingId: string,
  options: MeetingsRequestOptions = {}
): Promise<Meeting> {
  const payload = await requestMeetingsJson(
    `/meetings/${encodeURIComponent(meetingId)}`,
    { method: 'GET' },
    options
  )
  return parseMeeting(payload)
}

export async function patchMeeting(
  meetingId: string,
  patch: MeetingPatch,
  options: MeetingsRequestOptions = {}
): Promise<Meeting> {
  const body: Record<string, unknown> = {}
  if (patch.title !== undefined) body.title = patch.title
  if (patch.notesMd !== undefined) body.notes_md = patch.notesMd
  if (patch.suggestions !== undefined) {
    body.suggestions = patch.suggestions.map(suggestedEditToWire)
  }

  const payload = await requestMeetingsJson(
    `/meetings/${encodeURIComponent(meetingId)}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    },
    options
  )
  return parseMeeting(payload)
}

export async function deleteMeeting(
  meetingId: string,
  options: MeetingsRequestOptions = {}
): Promise<void> {
  await requestMeetings(`/meetings/${encodeURIComponent(meetingId)}`, { method: 'DELETE' }, options)
}

export function parseMeetingListRow(value: unknown): MeetingListRow {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.meeting_id !== 'string' ||
    typeof record.title !== 'string' ||
    typeof record.created_at !== 'number' ||
    typeof record.updated_at !== 'number' ||
    !isNullableNumber(record.started_at) ||
    !isNullableNumber(record.ended_at) ||
    !isMeetingStatus(record.status)
  ) {
    throw new Error('Invalid meeting list row')
  }

  return {
    meetingId: record.meeting_id,
    title: record.title,
    createdAt: record.created_at,
    updatedAt: record.updated_at,
    startedAt: record.started_at,
    endedAt: record.ended_at,
    status: record.status
  }
}

export function parseMeeting(value: unknown): Meeting {
  const record = asRecord(value)
  if (
    !record ||
    typeof record.notes_md !== 'string' ||
    typeof record.notes_rev !== 'number' ||
    typeof record.final_summary_md !== 'string' ||
    (record.chat_id !== null && typeof record.chat_id !== 'string')
  ) {
    throw new Error('Invalid meeting response')
  }

  const row = parseMeetingListRow(record)
  return {
    ...row,
    notesMd: record.notes_md,
    notesRev: record.notes_rev,
    finalSummaryMd: record.final_summary_md,
    suggestions: parseSuggestions(record.suggestions),
    transcript: parseTranscript(record.transcript),
    chatId: record.chat_id
  }
}

function parseSuggestions(value: unknown): SuggestedEdit[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    const record = asRecord(item)
    if (
      !record ||
      typeof record.id !== 'string' ||
      typeof record.anchor !== 'string' ||
      typeof record.occurrence !== 'number' ||
      typeof record.insert_md !== 'string' ||
      typeof record.rationale !== 'string' ||
      !isSuggestedEditStatus(record.status)
    ) {
      return []
    }
    return [
      {
        id: record.id,
        anchor: record.anchor,
        occurrence: record.occurrence,
        insertMd: record.insert_md,
        rationale: record.rationale,
        status: record.status
      }
    ]
  })
}

function parseTranscript(value: unknown): TranscriptChunk[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    const record = asRecord(item)
    if (
      !record ||
      typeof record.channel !== 'number' ||
      typeof record.text !== 'string' ||
      typeof record.ts !== 'number' ||
      typeof record.seq !== 'number'
    ) {
      return []
    }
    return [{ channel: record.channel, text: record.text, ts: record.ts, seq: record.seq }]
  })
}

function suggestedEditToWire(edit: SuggestedEdit): Record<string, unknown> {
  return {
    id: edit.id,
    anchor: edit.anchor,
    occurrence: edit.occurrence,
    insert_md: edit.insertMd,
    rationale: edit.rationale,
    status: edit.status
  }
}

async function requestMeetingsJson(
  path: string,
  init: RequestInit,
  options: MeetingsRequestOptions
): Promise<unknown> {
  const response = await requestMeetings(path, init, options)
  const text = await response.text()
  if (!text.trim()) return null
  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new Error('Invalid sidecar JSON response')
  }
}

async function requestMeetings(
  path: string,
  init: RequestInit,
  options: MeetingsRequestOptions
): Promise<Response> {
  const fetcher = options.fetcher ?? globalThis.fetch
  const response = await fetcher(`${options.baseUrl ?? SIDECAR_REST_URL}${path}`, {
    ...init,
    signal: options.signal
  })
  if (!response.ok) {
    const fallback = `Meeting request ${path} failed with HTTP ${response.status}`
    const text = await response.text().catch(() => '')
    if (!text.trim()) throw new Error(fallback)
    let payload: Record<string, unknown> | null = null
    try {
      payload = asRecord(JSON.parse(text) as unknown)
    } catch {
      // Preserve a plain-text error body below.
    }
    if (payload && typeof payload.detail === 'string' && payload.detail.trim()) {
      throw new Error(payload.detail)
    }
    throw new Error(text.trim() || fallback)
  }
  return response
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || typeof value === 'number'
}

function isMeetingStatus(value: unknown): value is MeetingStatus {
  return value === 'draft' || value === 'live' || value === 'summarizing' || value === 'done'
}

function isSuggestedEditStatus(value: unknown): value is SuggestedEditStatus {
  return value === 'pending' || value === 'accepted' || value === 'rejected'
}
