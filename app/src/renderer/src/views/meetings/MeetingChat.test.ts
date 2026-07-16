import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Meeting } from '../../lib/meetings'
import { MeetingChat } from './MeetingChat'

describe('MeetingChat', () => {
  it('renders the collapsed meeting-scoped drawer for a persisted chat', () => {
    const meeting: Meeting = {
      meetingId: 'meeting-1',
      title: 'Launch review',
      createdAt: 1,
      updatedAt: 2,
      startedAt: null,
      endedAt: null,
      status: 'done',
      notesMd: '',
      notesRev: 0,
      finalSummaryMd: '',
      suggestionsBaseRev: 0,
      suggestions: [],
      transcript: [],
      chatId: 'chat-1'
    }

    const markup = renderToStaticMarkup(
      createElement(MeetingChat, {
        meeting,
        onOpenNote: () => undefined
      })
    )

    expect(markup).toContain('Ask about this meeting…')
    expect(markup).toContain('aria-expanded="false"')
  })
})
