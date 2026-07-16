import type { SuggestedEdit } from './meetings'

/**
 * Pure helpers for projecting pending suggestions into a BlockNote document.
 * Structurally typed so they stay unit-testable in a node environment.
 */
export interface BlockLike {
  id: string
  type: string
  content?: unknown
  children?: BlockLike[]
  props?: Record<string, unknown>
}

/** Concatenated plain text of a block's inline content (ignores children). */
export function blockPlainText(block: BlockLike): string {
  const content = block.content
  if (!Array.isArray(content)) return ''
  let text = ''
  for (const item of content) {
    if (item && typeof item === 'object') {
      const record = item as Record<string, unknown>
      if (typeof record.text === 'string') text += record.text
      else if (Array.isArray(record.content)) {
        text += (record.content as unknown[])
          .map((inner) =>
            inner &&
            typeof inner === 'object' &&
            typeof (inner as { text?: unknown }).text === 'string'
              ? ((inner as { text: string }).text ?? '')
              : ''
          )
          .join('')
      }
    }
  }
  return text
}

/**
 * Find the block a suggestion should render after. Anchors are exact
 * substrings of the markdown; the trailing anchor line, stripped of list and
 * heading markers, is matched against block text. Returns null for
 * append-mode (empty/unmatched anchors).
 */
export function findAnchorBlockId(blocks: BlockLike[], edit: SuggestedEdit): string | null {
  if (!edit.anchor) return null
  const tail = anchorTail(edit.anchor)
  if (!tail) return null

  const matches: string[] = []
  const walk = (nodes: BlockLike[]): void => {
    for (const node of nodes) {
      if (node.type !== 'suggestion' && blockPlainText(node).includes(tail)) {
        matches.push(node.id)
      }
      if (node.children?.length) walk(node.children)
    }
  }
  walk(blocks)
  if (matches.length === 0) return null
  const index = Math.min(Math.max(edit.occurrence, 1), matches.length) - 1
  return matches[index]
}

/** Last non-empty anchor line without markdown list/heading/quote markers. */
export function anchorTail(anchor: string): string {
  const lines = anchor.split('\n').filter((line) => line.trim().length > 0)
  const last = lines[lines.length - 1] ?? ''
  return last
    .replace(/^[\s>]*(?:[-*+]|\d+[.)]|#{1,6})?\s*/, '')
    .replace(/\*\*|__|`/g, '')
    .trim()
}

/** Remove all suggestion blocks (recursively) before serializing to markdown. */
export function stripSuggestionBlocks<T extends BlockLike>(blocks: T[]): T[] {
  return blocks
    .filter((block) => block.type !== 'suggestion')
    .map((block) =>
      block.children?.length ? { ...block, children: stripSuggestionBlocks(block.children) } : block
    )
}
