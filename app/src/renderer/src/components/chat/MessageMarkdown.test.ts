import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { MessageMarkdown } from './MessageMarkdown'

describe('MessageMarkdown meeting citations', () => {
  it('renders M references as meeting citation chips', () => {
    const markup = renderToStaticMarkup(
      createElement(MessageMarkdown, {
        markdown: 'The owner is Maya [M3].',
        citations: new Map(),
        onOpenNote: () => undefined
      })
    )

    expect(markup).toContain('citation-chip citation-meeting')
    expect(markup).toContain('>M3</button>')
  })
})
