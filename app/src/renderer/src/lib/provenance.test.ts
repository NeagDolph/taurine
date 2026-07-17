import { describe, expect, it } from 'vitest'
import { addAiLines, aiLinesFromMarkdown, normalizeLine, pruneAiLines } from './provenance'

describe('provenance', () => {
  it('normalizes markdown markers and whitespace', () => {
    expect(normalizeLine('- **Budget**: $48,000')).toBe('Budget: $48,000')
    expect(normalizeLine('## Summary')).toBe('Summary')
    expect(normalizeLine('  1. First   item ')).toBe('First item')
    expect(normalizeLine('> quoted `code`')).toBe('quoted code')
  })

  it('extracts non-empty normalized lines from an agent insertion', () => {
    expect(aiLinesFromMarkdown('\n## Summary\n\n- One\n- Two\n')).toEqual(['Summary', 'One', 'Two'])
  })

  it('accumulates without duplicates', () => {
    const lines = addAiLines(['One'], '- One\n- Two')
    expect(lines.sort()).toEqual(['One', 'Two'])
  })

  it('prunes lines the user edited or deleted', () => {
    const notes = '- One\n- Two edited by me'
    expect(pruneAiLines(['One', 'Two'], notes)).toEqual(['One'])
  })
})
