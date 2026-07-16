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
        listeningState: 'stopping',
        onActionableStatus: vi.fn(),
        onAddToNotes: vi.fn()
      })
    )

    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>Add to notes<\/button>/)
  })
})
