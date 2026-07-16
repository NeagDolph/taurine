const NOTEPAD_PUSH_DELAY_MS = 2_000

interface PendingNotepad {
  meetingId: string
  textMd: string
}

export class DebouncedNotepadPush {
  private timer: number | null = null
  private pending: PendingNotepad | null = null

  constructor(private readonly send: (meetingId: string, textMd: string) => void) {}

  schedule(meetingId: string, textMd: string): void {
    this.clearTimer()
    this.pending = { meetingId, textMd }
    this.timer = window.setTimeout(() => {
      this.timer = null
      const pending = this.pending
      this.pending = null
      if (pending) this.send(pending.meetingId, pending.textMd)
    }, NOTEPAD_PUSH_DELAY_MS)
  }

  flush(meetingId: string, textMd: string): void {
    this.cancel()
    this.send(meetingId, textMd)
  }

  flushPending(): void {
    const pending = this.pending
    this.cancel()
    if (pending) this.send(pending.meetingId, pending.textMd)
  }

  cancel(): void {
    this.clearTimer()
    this.pending = null
  }

  private clearTimer(): void {
    if (this.timer === null) return
    window.clearTimeout(this.timer)
    this.timer = null
  }
}

interface FlushCurrentNotepadOptions {
  meetingId: string | null
  currentTextMd: string
  savedTextMd: string
  save(meetingId: string, textMd: string): Promise<void>
  push(meetingId: string, textMd: string): void
}

export async function flushCurrentNotepad({
  meetingId,
  currentTextMd,
  savedTextMd,
  save,
  push
}: FlushCurrentNotepadOptions): Promise<string> {
  if (!meetingId) return savedTextMd
  if (currentTextMd !== savedTextMd) await save(meetingId, currentTextMd)
  push(meetingId, currentTextMd)
  return currentTextMd
}

export function appendSummaryToNotes(notesMd: string, summaryMd: string): string {
  return `${notesMd}\n\n${summaryMd}\n`
}
