import type { SuggestedEdit } from './meetings'

export interface SuggestionProjection {
  edit: SuggestedEdit
  location: number
  appendMode: boolean
}

/** Return the UTF-16 index immediately after a 1-based anchor occurrence. */
export function locateAnchor(notesMd: string, anchor: string, occurrence: number): number | null {
  if (anchor === '') return notesMd.length
  if (!Number.isInteger(occurrence) || occurrence < 1) return null

  let start = 0
  for (let current = 1; current <= occurrence; current += 1) {
    const position = notesMd.indexOf(anchor, start)
    if (position < 0) return null
    const end = position + anchor.length
    if (current === occurrence) return end
    start = end
  }

  return null
}

/** Apply one additive edit, degrading stale anchors to an end-of-note append. */
export function applySuggestion(notesMd: string, edit: SuggestedEdit): string {
  if (!edit.insertMd) return notesMd

  const location = locateAnchor(notesMd, edit.anchor, edit.occurrence)
  if (location !== null && edit.anchor !== '') {
    if (notesMd.startsWith(edit.insertMd, location)) return notesMd
    return `${notesMd.slice(0, location)}${edit.insertMd}${notesMd.slice(location)}`
  }

  return appendSuggestion(notesMd, edit.insertMd)
}

/** Re-anchor pending edits against newer notes; missing anchors become append suggestions. */
export function revalidatePendingSuggestions(
  notesMd: string,
  suggestions: SuggestedEdit[]
): SuggestedEdit[] {
  return suggestions.map((edit) => {
    if (
      edit.status !== 'pending' ||
      edit.anchor === '' ||
      locateAnchor(notesMd, edit.anchor, edit.occurrence) !== null
    ) {
      return edit
    }
    return { ...edit, anchor: '', occurrence: 1 }
  })
}

/** Build widget-only projections; pending text never enters the editor document tree. */
export function projectPendingSuggestions(
  notesMd: string,
  suggestions: SuggestedEdit[]
): SuggestionProjection[] {
  return suggestions.flatMap((edit) => {
    if (edit.status !== 'pending') return []
    const location = locateAnchor(notesMd, edit.anchor, edit.occurrence)
    return [
      {
        edit,
        location: location ?? notesMd.length,
        appendMode: edit.anchor === '' || location === null
      }
    ]
  })
}

function appendSuggestion(notesMd: string, insertMd: string): string {
  if (!notesMd) return insertMd.replace(/^\n+/, '')
  if (notesMd.endsWith(insertMd)) return notesMd

  const trailingNewlines = notesMd.match(/\n+$/)?.[0].length ?? 0
  const leadingNewlines = insertMd.match(/^\n+/)?.[0].length ?? 0
  const separator = '\n'.repeat(Math.max(0, 2 - trailingNewlines - leadingNewlines))
  return `${notesMd}${separator}${insertMd}`
}
