import { describe, expect, it } from 'vitest'
import { parseMeeting } from './meetings'
import {
  anchorTail,
  blockPlainText,
  findAnchorBlockId,
  stripSuggestionBlocks,
  type BlockLike
} from './suggestion-blocks'
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
        suggestions_base_rev: 6,
        suggestions: [
          {
            id: 'suggestion-1',
            anchor: 'Notes',
            occurrence: 1,
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
      suggestionsBaseRev: 6,
      suggestions: [
        {
          id: 'suggestion-1',
          anchor: 'Notes',
          occurrence: 1,
          insertMd: '\nMore detail',
          rationale: 'Adds context',
          status: 'pending'
        }
      ],
      transcript: [{ channel: 1, text: 'Hello', ts: 5, seq: 6 }],
      aiLines: [],
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
      suggestions_base_rev: 0,
      suggestions: { not: 'an array' },
      transcript: [],
      chat_id: null
    })

    expect(meeting.suggestions).toEqual([])
  })
})

describe('suggestion block projection', () => {
  const block = (id: string, text: string, children: BlockLike[] = []): BlockLike => ({
    id,
    type: 'paragraph',
    content: [{ type: 'text', text }],
    children
  })

  it('extracts plain text from inline content', () => {
    expect(blockPlainText(block('a', 'Maya owns the rollout.'))).toBe('Maya owns the rollout.')
    expect(blockPlainText({ id: 'x', type: 'suggestion' })).toBe('')
  })

  it('strips list and heading markers from anchor tails', () => {
    expect(anchorTail('- Maya owns the rollout.')).toBe('Maya owns the rollout.')
    expect(anchorTail('## Agenda\n1. **Budget** review')).toBe('Budget review')
    expect(anchorTail('')).toBe('')
  })

  it('finds the anchored block by occurrence and skips suggestion blocks', () => {
    const blocks = [
      block('b1', 'Same line'),
      { id: 's1', type: 'suggestion' as const },
      block('b2', 'Same line'),
      block('b3', 'Other')
    ]
    const edit = (
      occurrence: number,
      anchor = '- Same line'
    ): Parameters<typeof findAnchorBlockId>[1] => ({
      id: 'e1',
      anchor,
      occurrence,
      insertMd: 'x',
      rationale: '',
      status: 'pending' as const
    })
    expect(findAnchorBlockId(blocks, edit(1))).toBe('b1')
    expect(findAnchorBlockId(blocks, edit(2))).toBe('b2')
    expect(findAnchorBlockId(blocks, edit(9))).toBe('b2')
    expect(findAnchorBlockId(blocks, edit(1, ''))).toBeNull()
    expect(findAnchorBlockId(blocks, edit(1, '- Missing'))).toBeNull()
  })

  it('finds anchors inside nested children', () => {
    const blocks = [block('top', 'Top', [block('child', 'Nested target')])]
    const edit = {
      id: 'e2',
      anchor: '- Nested target',
      occurrence: 1,
      insertMd: 'x',
      rationale: '',
      status: 'pending' as const
    }
    expect(findAnchorBlockId(blocks, edit)).toBe('child')
  })

  it('strips suggestion blocks recursively before serialization', () => {
    const blocks: BlockLike[] = [
      block('keep', 'Keep this', [{ id: 'nested-s', type: 'suggestion' }]),
      { id: 's', type: 'suggestion' }
    ]
    const stripped = stripSuggestionBlocks(blocks)
    expect(stripped).toHaveLength(1)
    expect(stripped[0].children).toHaveLength(0)
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
