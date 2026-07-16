import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { SummaryCard } from './SummaryCard'

describe('SummaryCard', () => {
  it('offers to add the displayed summary to the meeting notes', () => {
    const markup = renderToStaticMarkup(
      createElement(SummaryCard, {
        summary: null,
        actionables: [],
        finalSummaryMd: '## Summary\nA decision.',
        finalSummaryStatus: null,
        listeningState: 'idle',
        onActionableStatus: vi.fn(),
        onAddToNotes: vi.fn()
      })
    )

    expect(markup).toContain('Add to notes')
  })

  it('does not accept note changes after stop has begun', () => {
    const markup = renderToStaticMarkup(
      createElement(SummaryCard, {
        summary: null,
        actionables: [],
        finalSummaryMd: '## Summary\nA decision.',
        finalSummaryStatus: null,
        listeningState: 'stopping',
        onActionableStatus: vi.fn(),
        onAddToNotes: vi.fn()
      })
    )

    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>Add to notes<\/button>/)
  })

  it('shows final-summary progress while the sidecar is working', () => {
    const markup = renderToStaticMarkup(
      createElement(SummaryCard, {
        summary: null,
        actionables: [],
        finalSummaryMd: '',
        finalSummaryStatus: {
          type: 'final_summary_status',
          meetingId: 'meeting-1',
          state: 'running',
          detail: ''
        },
        listeningState: 'idle',
        onActionableStatus: vi.fn(),
        onAddToNotes: vi.fn()
      })
    )

    expect(markup).toContain('Summarizing…')
    expect(markup).toContain('animate-pulse')
  })

  it('shows a final-summary error detail without hiding the meeting notes', () => {
    const markup = renderToStaticMarkup(
      createElement(SummaryCard, {
        summary: null,
        actionables: [],
        finalSummaryMd: '',
        finalSummaryStatus: {
          type: 'final_summary_status',
          meetingId: 'meeting-1',
          state: 'error',
          detail: 'Model unavailable'
        },
        listeningState: 'idle',
        onActionableStatus: vi.fn(),
        onAddToNotes: vi.fn()
      })
    )

    expect(markup).toContain('Model unavailable')
  })

  it('shows a useful fallback for an empty final-summary error detail', () => {
    const markup = renderToStaticMarkup(
      createElement(SummaryCard, {
        summary: null,
        actionables: [],
        finalSummaryMd: '',
        finalSummaryStatus: {
          type: 'final_summary_status',
          meetingId: 'meeting-1',
          state: 'error',
          detail: ''
        },
        listeningState: 'idle',
        onActionableStatus: vi.fn(),
        onAddToNotes: vi.fn()
      })
    )

    expect(markup).toContain('Final summary could not be generated.')
  })
})
