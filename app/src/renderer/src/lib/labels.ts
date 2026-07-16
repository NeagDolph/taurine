export function sourceLabel(source: string): string {
  switch (source) {
    case 'apple_notes':
      return 'Notes'
    case 'obsidian':
      return 'Obsidian'
    case 'imessage':
      return 'iMessage'
    case 'claude_mem':
      return 'Claude'
    case 'codex_mem':
      return 'Codex'
    case 'taurine':
      return 'Taurine'
    default:
      return source
  }
}

export function ageLabel(ageDays: number): string {
  if (ageDays < 1) {
    return 'today'
  }
  if (ageDays < 60) {
    return `${Math.round(ageDays)}d`
  }
  if (ageDays < 365) {
    return `${Math.round(ageDays / 30)}mo`
  }
  return `${(ageDays / 365).toFixed(1)}y`
}
