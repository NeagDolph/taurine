import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import type { SummaryEvent } from '../../lib/sidecar'
import { SummaryCard } from './SummaryCard'

const ROLLING: SummaryEvent = {
  type: 'summary',
  rev: 3,
  summary_md: 'The rollout is on track [T1].',
  highlights: [],
  actionables: [],
  facts: [],
  grounding: [],
  evidence: null,
  updated_at: 10
}

function render(overrides: Partial<Parameters<typeof SummaryCard>[0]>): string {
  return renderToStaticMarkup(
    createElement(SummaryCard, {
      summary: null,
      actionables: [],
      finalSummaryStatus: null,
      listeningState: 'idle',
      onActionableStatus: vi.fn(),
      onAddToNotes: vi.fn(),
      onOpenNote: vi.fn(),
      ...overrides
    })
  )
}

describe('SummaryCard', () => {
  it('renders nothing when idle with no rolling summary', () => {
    expect(render({})).toBe('')
  })

  it('offers to add the rolling summary to notes while live', () => {
    const markup = render({ summary: ROLLING, listeningState: 'listening' })
    expect(markup).toContain('Add to notes')
    expect(markup).toContain('on track')
  })

  it('never renders the final summary content (it lives in the notepad as a suggestion)', () => {
    const markup = render({ listeningState: 'idle' })
    expect(markup).not.toContain('Final summary')
  })

  it('shows final-summary progress while the sidecar is working', () => {
    const markup = render({
      finalSummaryStatus: {
        type: 'final_summary_status',
        meetingId: 'meeting-1',
        state: 'running',
        detail: ''
      }
    })
    expect(markup).toContain('Writing your meeting summary…')
  })

  it('shows a final-summary error detail with fallback copy', () => {
    const withDetail = render({
      finalSummaryStatus: {
        type: 'final_summary_status',
        meetingId: 'meeting-1',
        state: 'error',
        detail: 'Model unavailable'
      }
    })
    expect(withDetail).toContain('Model unavailable')

    const withoutDetail = render({
      finalSummaryStatus: {
        type: 'final_summary_status',
        meetingId: 'meeting-1',
        state: 'error',
        detail: ''
      }
    })
    expect(withoutDetail).toContain('Final summary could not be generated.')
  })
})
