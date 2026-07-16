import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Meeting } from '../../lib/meetings'
import { MeetingChat } from './MeetingChat'

const MEETING: Meeting = {
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

describe('MeetingChat', () => {
  it('renders nothing until the drawer has been opened once', () => {
    const markup = renderToStaticMarkup(
      createElement(MeetingChat, {
        meeting: MEETING,
        open: false,
        askRequest: null,
        onClose: () => undefined,
        onOpenNote: () => undefined
      })
    )
    expect(markup).toBe('')
  })

  it('renders the overlay panel when open', () => {
    const markup = renderToStaticMarkup(
      createElement(MeetingChat, {
        meeting: MEETING,
        open: true,
        askRequest: null,
        onClose: () => undefined,
        onOpenNote: () => undefined
      })
    )
    expect(markup).toContain('Ask about this meeting')
    expect(markup).toContain('aria-hidden="false"')
  })
})
