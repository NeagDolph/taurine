/**
 * Streams an agent's markdown answer directly into a BlockNote document as it
 * grows (Granola style): each throttled flush replaces the blocks written by
 * the previous flush, and every flush EXPLICITLY serializes the document and
 * reports it upward — persistence never depends on editor-internal change
 * events. Testable against a headless BlockNoteEditor.
 */

export interface AiStreamEditor {
  document: Array<{ id: string }>
  getBlock(id: string): unknown
  tryParseMarkdownToBlocks(md: string): unknown
  insertBlocks(
    blocks: unknown,
    reference: unknown,
    placement: 'before' | 'after'
  ): Array<{ id: string }>
  replaceBlocks(remove: string[], insert: unknown): { insertedBlocks: Array<{ id: string }> }
}

export interface AiStreamCallbacks {
  /** Serialize + propagate the document upward (autosave path). */
  emit(): void | Promise<void>
  /** Register AI-line provenance; `done` marks the final, validated write. */
  onProvenance(md: string, done: boolean): void
  /** Anchor block id to insert after (cursor block); null appends at end. */
  anchorBlockId(): string | null
  throttleMs?: number
  schedule?(callback: () => void, ms: number): number
  cancel?(handle: number): void
}

const CITATION_TOKENS = /\s*\[(?:N|E|F|P|T|M)\d+(?:,\s*(?:N|E|F|P|T|M)\d+)*\]/g

export class AiStreamWriter {
  private blockIds: string[] = []
  private latestMd = ''
  private writtenMd = ''
  private done = false
  private timer: number | null = null
  private writing = false

  constructor(
    private readonly editor: AiStreamEditor,
    private readonly callbacks: AiStreamCallbacks
  ) {}

  /** Start a fresh answer; follow-ups continue after the previous one. */
  reset(): string | null {
    if (this.timer !== null) this.cancelTimer()
    const lastStreamed = this.blockIds[this.blockIds.length - 1]
    const continueAfter = lastStreamed && this.editor.getBlock(lastStreamed) ? lastStreamed : null
    this.blockIds = []
    this.latestMd = ''
    this.writtenMd = ''
    this.done = false
    this.writing = false
    return continueAfter
  }

  handleAnswer(md: string, done: boolean): void {
    this.latestMd = md
    if (done) {
      this.done = true
      this.cancelTimer()
      void this.flush()
      return
    }
    if (this.timer !== null) return
    const schedule = this.callbacks.schedule ?? window.setTimeout.bind(window)
    this.timer = schedule(() => {
      this.timer = null
      void this.flush()
    }, this.callbacks.throttleMs ?? 300)
  }

  async flush(): Promise<void> {
    if (this.writing || this.latestMd === this.writtenMd) return
    this.writing = true
    const md = this.latestMd
    try {
      // A parse/insert failure abandons THIS snapshot (writtenMd advances) —
      // a later delta retries with fresh content; never loop on the same one.
      const cleaned = md.replace(CITATION_TOKENS, '').trim()
      if (!cleaned) {
        this.writtenMd = md
        return
      }
      const blocks = (await this.editor.tryParseMarkdownToBlocks(cleaned)) as Array<{
        id: string
      }>
      if (!Array.isArray(blocks) || blocks.length === 0) {
        this.writtenMd = md
        return
      }
      const existing = this.blockIds.filter((id) => this.editor.getBlock(id))
      if (existing.length > 0) {
        const replaced = this.editor.replaceBlocks(existing, blocks)
        this.blockIds = replaced.insertedBlocks.map((block) => block.id)
      } else {
        const anchorId = this.callbacks.anchorBlockId()
        const anchorBlock = anchorId ? this.editor.getBlock(anchorId) : undefined
        const reference = anchorBlock ?? this.editor.document[this.editor.document.length - 1]
        const inserted = reference
          ? this.editor.insertBlocks(blocks, reference, 'after')
          : this.editor.replaceBlocks(
              this.editor.document.map((block) => block.id),
              blocks
            ).insertedBlocks
        this.blockIds = inserted.map((block) => block.id)
      }
      this.writtenMd = md
      const isFinal = this.done && this.latestMd === md
      // Persist BEFORE registering provenance for the final write so the
      // save that provenance triggers sees the finished document.
      await this.callbacks.emit()
      this.callbacks.onProvenance(cleaned, isFinal)
    } catch {
      this.writtenMd = md
    } finally {
      this.writing = false
      if (this.latestMd !== this.writtenMd && this.timer === null) {
        void this.flush()
      }
    }
  }

  private cancelTimer(): void {
    if (this.timer === null) return
    const cancel = this.callbacks.cancel ?? window.clearTimeout.bind(window)
    cancel(this.timer)
    this.timer = null
  }
}
