import { describe, expect, it } from 'vitest'
import type { SuggestedEdit } from './meetings'
import {
  applySuggestion,
  locateAnchor,
  projectPendingSuggestions,
  revalidatePendingSuggestions
} from './suggestions'

function edit(overrides: Partial<SuggestedEdit> = {}): SuggestedEdit {
  return {
    id: 'suggestion-1',
    anchor: 'Decision',
    occurrence: 1,
    insertMd: '\n\nMore detail.',
    rationale: 'Adds context',
    status: 'pending',
    ...overrides
  }
}

describe('locateAnchor', () => {
  it('uses the 1-based occurrence to disambiguate repeated anchors', () => {
    const notes = 'one one one'

    expect(locateAnchor(notes, 'one', 1)).toBe(3)
    expect(locateAnchor(notes, 'one', 2)).toBe(7)
    expect(locateAnchor(notes, 'one', 3)).toBe(11)
  })

  it('locates an empty anchor at the end of the notes', () => {
    expect(locateAnchor('Existing notes', '', 1)).toBe('Existing notes'.length)
  })

  it('returns null when the requested occurrence is missing', () => {
    expect(locateAnchor('Only once', 'once', 2)).toBeNull()
    expect(locateAnchor('Only once', 'absent', 1)).toBeNull()
  })

  it('does not split unicode anchors', () => {
    const notes = '📌 café / 📌 café'
    const anchor = '📌 café'
    const secondStart = notes.indexOf(anchor, notes.indexOf(anchor) + anchor.length)

    expect(locateAnchor(notes, anchor, 2)).toBe(secondStart + anchor.length)
  })
})

describe('applySuggestion', () => {
  it('is idempotent when the same insertion is applied twice', () => {
    const suggestion = edit()
    const once = applySuggestion('Decision', suggestion)

    expect(once).toBe('Decision\n\nMore detail.')
    expect(applySuggestion(once, suggestion)).toBe(once)
  })

  it('degrades a missing anchor to an append with a blank line', () => {
    expect(
      applySuggestion('Existing notes', edit({ anchor: 'Missing', insertMd: 'Recovered detail.' }))
    ).toBe('Existing notes\n\nRecovered detail.')
  })

  it('appends empty-anchor suggestions without duplicating them', () => {
    const suggestion = edit({ anchor: '', insertMd: 'Closing note.' })
    const once = applySuggestion('Existing notes', suggestion)

    expect(once).toBe('Existing notes\n\nClosing note.')
    expect(applySuggestion(once, suggestion)).toBe(once)
  })

  it('inserts unicode content after a unicode anchor', () => {
    expect(
      applySuggestion('Discussed 📦 launch.', edit({ anchor: '📦', insertMd: '🚀', occurrence: 1 }))
    ).toBe('Discussed 📦🚀 launch.')
  })
})

describe('revalidatePendingSuggestions', () => {
  it('degrades stale missing anchors to append mode idempotently', () => {
    const suggestions = [
      edit({ id: 'found', anchor: 'Decision' }),
      edit({ id: 'missing', anchor: 'Old wording' }),
      edit({ id: 'resolved', anchor: 'Old wording', status: 'accepted' })
    ]

    const once = revalidatePendingSuggestions('Decision', suggestions)

    expect(once).toEqual([
      suggestions[0],
      { ...suggestions[1], anchor: '', occurrence: 1 },
      suggestions[2]
    ])
    expect(revalidatePendingSuggestions('Decision', once)).toEqual(once)
    expect(suggestions[1].anchor).toBe('Old wording')
  })
})

describe('projectPendingSuggestions', () => {
  it('projects decorations without mutating notes and is idempotent', () => {
    const notes = 'Decision'
    const suggestions = [edit(), edit({ id: 'resolved', status: 'rejected' })]

    const first = projectPendingSuggestions(notes, suggestions)
    const second = projectPendingSuggestions(notes, suggestions)

    expect(first).toEqual([
      {
        edit: suggestions[0],
        location: 'Decision'.length,
        appendMode: false
      }
    ])
    expect(second).toEqual(first)
    expect(notes).toBe('Decision')
  })
})
