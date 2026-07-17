// @vitest-environment jsdom
import { BlockNoteEditor } from '@blocknote/core'
import { describe, expect, it } from 'vitest'
import { AiStreamWriter, type AiStreamEditor } from './ai-stream'

interface Harness {
  editor: BlockNoteEditor
  writer: AiStreamWriter
  emits: number
  provenance: Array<{ md: string; done: boolean }>
  scheduled: Array<() => void>
  runScheduled(): void
  text(): string
}

function harness(initial = 'user line'): Harness {
  const editor = BlockNoteEditor.create({
    initialContent: [{ type: 'paragraph', content: initial }]
  })
  const state: Harness = {
    editor,
    emits: 0,
    provenance: [],
    scheduled: [],
    runScheduled() {
      const tasks = [...state.scheduled]
      state.scheduled = []
      tasks.forEach((task) => task())
    },
    text() {
      return editor.document
        .map((block) =>
          Array.isArray(block.content)
            ? block.content.map((item) => ('text' in item ? item.text : '')).join('')
            : ''
        )
        .join('\n')
    },
    writer: null as unknown as AiStreamWriter
  }
  state.writer = new AiStreamWriter(editor as unknown as AiStreamEditor, {
    emit: () => {
      state.emits += 1
    },
    onProvenance: (md, done) => state.provenance.push({ md, done }),
    anchorBlockId: () => editor.document[0]?.id ?? null,
    schedule: (callback) => {
      state.scheduled.push(callback)
      return state.scheduled.length
    },
    cancel: () => undefined
  })
  return state
}

const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0))

describe('AiStreamWriter', () => {
  it('writes the growing answer into the document and always emits for persistence', async () => {
    const h = harness()
    h.writer.handleAnswer('- first', false)
    h.runScheduled()
    await settle()
    expect(h.text()).toContain('first')
    expect(h.emits).toBe(1)

    h.writer.handleAnswer('- first\n- second', false)
    h.runScheduled()
    await settle()
    expect(h.text()).toContain('second')
    expect(h.editor.document).toHaveLength(3) // user line + 2 streamed bullets
    expect(h.emits).toBe(2)
  })

  it('marks only the final validated write as done and strips citation tokens', async () => {
    const h = harness()
    h.writer.handleAnswer('- budget is $48k [M3]', false)
    h.runScheduled()
    await settle()
    h.writer.handleAnswer('- budget is $48k, Priya backs up [M3, E2]', true)
    await settle()

    expect(h.text()).not.toContain('[M3')
    expect(h.provenance.at(-1)).toEqual({
      md: '- budget is $48k, Priya backs up',
      done: true
    })
    expect(h.provenance.filter((entry) => entry.done)).toHaveLength(1)
  })

  it('coalesces deltas that arrive during an in-flight write', async () => {
    const h = harness()
    h.writer.handleAnswer('- a', false)
    h.runScheduled()
    // arrives while the first flush is mid-air
    h.writer.handleAnswer('- a\n- b\n- c', true)
    await settle()
    await settle()
    expect(h.text()).toContain('c')
    expect(h.provenance.at(-1)?.done).toBe(true)
  })

  it('follow-up asks continue after the previous answer instead of overwriting it', async () => {
    const h = harness()
    h.writer.handleAnswer('- answer one', true)
    await settle()
    const continueAfter = h.writer.reset()
    expect(continueAfter).not.toBeNull()

    h.writer.handleAnswer('- answer two', true)
    await settle()
    expect(h.text()).toContain('answer one')
    expect(h.text()).toContain('answer two')
  })
})
