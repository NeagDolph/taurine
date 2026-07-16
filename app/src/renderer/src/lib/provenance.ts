/**
 * Per-line AI provenance for meeting notes. Lines authored by agents (chat
 * edits, accepted suggestions, added summaries) render gray until the user
 * edits them; anything the user types renders white. Provenance is tracked
 * as NORMALIZED line texts (markdown markers stripped) so it survives the
 * markdown ↔ editor round-trip; editing a line changes its normalized text,
 * which flips it back to user-authored.
 */

/** Strip list/heading/quote markers and inline emphasis; collapse whitespace. */
export function normalizeLine(line: string): string {
  return line
    .replace(/^[\s>]*(?:[-*+]|\d+[.)]|#{1,6})?\s*/, '')
    .replace(/\*\*|__|~~|`/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Normalized, non-empty lines of a markdown fragment an agent authored. */
export function aiLinesFromMarkdown(markdown: string): string[] {
  return markdown
    .split('\n')
    .map(normalizeLine)
    .filter((line) => line.length > 0)
}

/** Merge newly authored agent lines into the provenance set. */
export function addAiLines(current: string[], insertMd: string): string[] {
  const next = new Set(current)
  for (const line of aiLinesFromMarkdown(insertMd)) next.add(line)
  return [...next]
}

/** Drop provenance for lines no longer present in the note (edited/deleted). */
export function pruneAiLines(aiLines: string[], notesMd: string): string[] {
  if (aiLines.length === 0) return aiLines
  const present = new Set(aiLinesFromMarkdown(notesMd))
  return aiLines.filter((line) => present.has(line))
}
