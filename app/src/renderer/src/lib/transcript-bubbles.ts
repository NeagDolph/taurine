import type { TranscriptChunk } from './meetings'

export interface Bubble {
  key: string
  channel: number
  text: string
  interim?: boolean
}

/** Merge consecutive same-channel chunks into one bubble for compactness. */
export function mergeBubbles(chunks: TranscriptChunk[]): Bubble[] {
  const bubbles: Bubble[] = []
  for (const chunk of chunks) {
    const text = chunk.text.trim()
    if (!text) continue
    const last = bubbles[bubbles.length - 1]
    if (last && last.channel === chunk.channel && !last.interim) {
      last.text = `${last.text} ${text}`
    } else {
      bubbles.push({
        key: `${chunk.channel}-${chunk.seq}-${chunk.ts}`,
        channel: chunk.channel,
        text
      })
    }
  }
  return bubbles
}
