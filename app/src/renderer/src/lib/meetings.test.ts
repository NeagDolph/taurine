import { describe, expect, it } from 'vitest'
import { parseMeeting } from './meetings'
import { deserializeMarkdown, serializeMarkdown } from '../views/meetings/NotepadEditor'
import { parseUpdateEvent } from './updates'

describe('parseMeeting', () => {
  it('maps a complete meeting payload from snake case to camel case', () => {
    expect(
      parseMeeting({
        meeting_id: 'meeting-1',
        title: 'Weekly planning',
        created_at: 1,
        updated_at: 2,
        started_at: 3,
        ended_at: 4,
        status: 'done',
        notes_md: '# Notes',
        notes_rev: 7,
        final_summary_md: '## Summary',
        suggestions: [
          {
            id: 'suggestion-1',
            anchor: 'Notes',
            occurrence: 0,
            insert_md: '\nMore detail',
            rationale: 'Adds context',
            status: 'pending'
          }
        ],
        transcript: [{ channel: 1, text: 'Hello', ts: 5, seq: 6 }],
        chat_id: 'chat-1'
      })
    ).toEqual({
      meetingId: 'meeting-1',
      title: 'Weekly planning',
      createdAt: 1,
      updatedAt: 2,
      startedAt: 3,
      endedAt: 4,
      status: 'done',
      notesMd: '# Notes',
      notesRev: 7,
      finalSummaryMd: '## Summary',
      suggestions: [
        {
          id: 'suggestion-1',
          anchor: 'Notes',
          occurrence: 0,
          insertMd: '\nMore detail',
          rationale: 'Adds context',
          status: 'pending'
        }
      ],
      transcript: [{ channel: 1, text: 'Hello', ts: 5, seq: 6 }],
      chatId: 'chat-1'
    })
  })

  it('tolerates malformed suggestions by returning an empty list', () => {
    const meeting = parseMeeting({
      meeting_id: 'meeting-1',
      title: '',
      created_at: 1,
      updated_at: 1,
      started_at: null,
      ended_at: null,
      status: 'draft',
      notes_md: '',
      notes_rev: 0,
      final_summary_md: '',
      suggestions: { not: 'an array' },
      transcript: [],
      chat_id: null
    })

    expect(meeting.suggestions).toEqual([])
  })
})

describe('NotepadEditor markdown serialization', () => {
  it('round-trips headings, bold text, and lists', () => {
    const markdown = '# Planning\n\nA **bold** decision.\n\n- First\n- Second'
    const value = deserializeMarkdown(markdown)

    const serialized = serializeMarkdown(value)
    expect(serialized).toBe('# Planning\n\nA **bold** decision.\n\n* First\n* Second')
    expect(serializeMarkdown(deserializeMarkdown(serialized))).toBe(serialized)
  })

  it('does not leak suggestion-marked text into markdown', () => {
    const value = deserializeMarkdown('Keep this')
    const paragraph = value[0] as { children: Array<Record<string, unknown>> }
    paragraph.children.push({ text: ' pending insertion', suggestion: true })

    expect(serializeMarkdown(value)).toBe('Keep this')
  })
})

describe('meetings updates', () => {
  it('parses meeting list rows from the updates websocket', () => {
    expect(
      parseUpdateEvent(
        JSON.stringify({
          type: 'meetings',
          meetings: [
            {
              meeting_id: 'meeting-1',
              title: 'Planning',
              created_at: 1,
              updated_at: 2,
              started_at: null,
              ended_at: null,
              status: 'draft'
            }
          ]
        })
      )
    ).toEqual({
      type: 'meetings',
      meetings: [
        {
          meetingId: 'meeting-1',
          title: 'Planning',
          createdAt: 1,
          updatedAt: 2,
          startedAt: null,
          endedAt: null,
          status: 'draft'
        }
      ]
    })
  })
})
