import '@blocknote/core/fonts/inter.css'
import '@blocknote/mantine/style.css'
import { BlockNoteSchema, defaultBlockSpecs, filterSuggestionItems, type Block } from '@blocknote/core'
import { BlockNoteView } from '@blocknote/mantine'
import {
  createReactBlockSpec,
  getDefaultReactSlashMenuItems,
  SuggestionMenuController,
  useCreateBlockNote
} from '@blocknote/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { SuggestedEdit } from '../../lib/meetings'
import { normalizeLine } from '../../lib/provenance'
import { AiInline, type AiAnchor } from './AiInline'
import type { ChatNotesEdit } from './MeetingChat'
import {
  blockPlainText,
  findAnchorBlockId,
  stripSuggestionBlocks,
  type BlockLike
} from '../../lib/suggestion-blocks'

export interface NotepadEditorProps {
  valueMd: string
  onChangeMd(md: string): void
  suggestions: SuggestedEdit[]
  onResolveSuggestion(id: string, accept: boolean): void
  onOpenNoteInMap?(noteId: string): void
  aiLines?: string[]
  readOnly?: boolean
  /** Meeting binding for the in-editor Ask AI (slash menu). */
  meetingId?: string
  chatId?: string | null
  onChatNotesEdit?(edit: ChatNotesEdit): void
  onAiInserted?(md: string): void
}

// Block specs are module-level and cannot close over component props, so
// accept/deny handlers reach block renders through the editor instance.
const suggestionResolvers = new WeakMap<object, (id: string, accept: boolean) => void>()

/**
 * Google-Docs-style inline suggestion block. Rendered inside the document at
 * its anchor position, but excluded from markdown serialization until
 * accepted — pending AI text can never leak into saved notes.
 */
const SuggestionBlock = createReactBlockSpec(
  {
    type: 'suggestion',
    propSchema: {
      suggestionId: { default: '' },
      md: { default: '' },
      rationale: { default: '' },
      big: { default: false }
    },
    content: 'none'
  },
  {
    render: (props) => {
      const { suggestionId, md, rationale, big } = props.block.props
      const resolve = (accept: boolean): void => {
        suggestionResolvers.get(props.editor as object)?.(suggestionId, accept)
      }
      return (
        <div
          className={`suggestion-block group relative my-1 rounded-md border-l-2 border-ok/60 bg-ok/[0.055] py-1.5 pl-3 pr-20 ${
            big ? 'mt-4' : ''
          }`}
          contentEditable={false}
          data-suggestion-id={suggestionId}
        >
          {big ? (
            <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-ok/80">
              Suggested summary
            </div>
          ) : null}
          <div className="markdown-body suggestion-md text-[13.5px] leading-6 text-fg/85">
            <ReactMarkdown>{md}</ReactMarkdown>
          </div>
          <div className="absolute right-1.5 top-1.5 flex items-center gap-0.5 opacity-0 transition-opacity duration-100 focus-within:opacity-100 group-hover:opacity-100">
            {rationale ? (
              <span
                className="flex h-6 w-6 cursor-help items-center justify-center rounded text-[11px] text-fg/40 hover:text-fg/70"
                title={rationale}
                aria-label={`Why: ${rationale}`}
              >
                ⓘ
              </span>
            ) : null}
            <button
              className="flex h-6 w-6 items-center justify-center rounded text-[13px] text-ok hover:bg-ok/15"
              type="button"
              title="Accept suggestion"
              aria-label="Accept suggestion"
              onClick={() => resolve(true)}
            >
              ✓
            </button>
            <button
              className="flex h-6 w-6 items-center justify-center rounded text-[13px] text-fg/45 hover:bg-white/10 hover:text-fg"
              type="button"
              title="Dismiss suggestion"
              aria-label="Dismiss suggestion"
              onClick={() => resolve(false)}
            >
              ✕
            </button>
          </div>
        </div>
      )
    }
  }
)

const schema = BlockNoteSchema.create({
  blockSpecs: {
    ...defaultBlockSpecs,
    suggestion: SuggestionBlock()
  }
})

type NotepadBlock = Block<typeof schema.blockSchema>

export function NotepadEditor({
  valueMd,
  onChangeMd,
  suggestions,
  onResolveSuggestion,
  onOpenNoteInMap,
  aiLines = [],
  readOnly = false,
  meetingId,
  chatId = null,
  onChatNotesEdit,
  onAiInserted
}: NotepadEditorProps): React.JSX.Element {
  const editor = useCreateBlockNote({ schema })
  const containerRef = useRef<HTMLDivElement | null>(null)
  const lastExternalMd = useRef<string | null>(null)
  const pending = useMemo(
    () => suggestions.filter((edit) => edit.status === 'pending'),
    [suggestions]
  )
  const pendingKey = useMemo(
    () => pending.map((edit) => `${edit.id}:${edit.anchor}:${edit.occurrence}`).join('|'),
    [pending]
  )
  const projectedKey = useRef<string | null>(null)

  useEffect(() => {
    suggestionResolvers.set(editor as object, onResolveSuggestion)
  }, [editor, onResolveSuggestion])

  // Per-line AI provenance: blocks whose normalized text matches an
  // agent-authored line render gray; the moment the user edits one, its
  // text diverges and it flips back to the default color.
  const paintProvenance = useCallback((): void => {
    const container = containerRef.current
    if (!container) return
    const aiSet = new Set(aiLines)
    const walk = (blocks: BlockLike[]): void => {
      for (const block of blocks) {
        const element = container.querySelector(`[data-id="${block.id}"]`)
        if (element && block.type !== 'suggestion') {
          const text = normalizeLine(blockPlainText(block))
          element.classList.toggle('ai-authored', text.length > 0 && aiSet.has(text))
        }
        if (block.children?.length) walk(block.children)
      }
    }
    walk(editor.document as unknown as BlockLike[])
  }, [aiLines, editor])

  useEffect(() => {
    const frame = requestAnimationFrame(paintProvenance)
    return () => cancelAnimationFrame(frame)
  }, [paintProvenance, valueMd, suggestions])

  // Rebuild the document whenever canonical markdown or the pending
  // suggestion set changes externally. User keystrokes round-trip through
  // onChange → parent → identical valueMd, so they never trigger a rebuild.
  useEffect(() => {
    if (valueMd === lastExternalMd.current && pendingKey === projectedKey.current) return
    lastExternalMd.current = valueMd
    projectedKey.current = pendingKey
    let cancelled = false
    void (async () => {
      const blocks = (await editor.tryParseMarkdownToBlocks(valueMd)) as NotepadBlock[]
      if (cancelled) return
      editor.replaceBlocks(editor.document, injectSuggestions(blocks, pending))
      requestAnimationFrame(paintProvenance)
    })()
    return () => {
      cancelled = true
    }
  }, [editor, paintProvenance, pending, pendingKey, valueMd])

  // In-editor Ask AI: opened from the slash menu, anchored under the block
  // that held the cursor; answers can be inserted there as AI-authored lines.
  const [aiOpen, setAiOpen] = useState(false)
  const [aiAnchor, setAiAnchor] = useState<AiAnchor | null>(null)
  const [aiSelection, setAiSelection] = useState('')
  const aiBlockIdRef = useRef<string | null>(null)


  // Streamed Ask-AI answers write straight into the note (Granola style):
  // the growing markdown replaces its own previously inserted blocks on a
  // throttle, and every flush registers provenance so the content renders
  // gray from the first paint.
  const aiStreamRef = useRef<{
    blockIds: string[]
    latestMd: string
    writtenMd: string
    timer: number | null
    writing: boolean
  }>({ blockIds: [], latestMd: '', writtenMd: '', timer: null, writing: false })

  const resetAiStream = useCallback((): void => {
    const stream = aiStreamRef.current
    if (stream.timer !== null) window.clearTimeout(stream.timer)
    // Follow-up asks continue after the previous answer, not over it.
    const lastStreamed = stream.blockIds[stream.blockIds.length - 1]
    if (lastStreamed && editor.getBlock(lastStreamed)) aiBlockIdRef.current = lastStreamed
    aiStreamRef.current = { blockIds: [], latestMd: '', writtenMd: '', timer: null, writing: false }
  }, [editor])

  const flushAiStream = useCallback((): void => {
    const stream = aiStreamRef.current
    if (stream.writing || stream.latestMd === stream.writtenMd) return
    stream.writing = true
    const md = stream.latestMd
    void (async () => {
      try {
        // Citation tokens are dead text outside the chat surface.
        const cleaned = md
          .replace(/\s*\[(?:N|E|F|P|T|M)\d+(?:,\s*(?:N|E|F|P|T|M)\d+)*\]/g, '')
          .trim()
        if (!cleaned) return
        onAiInserted?.(cleaned)
        const blocks = (await editor.tryParseMarkdownToBlocks(cleaned)) as NotepadBlock[]
        if (blocks.length === 0) return
        const existing = stream.blockIds.filter((id) => editor.getBlock(id))
        if (existing.length > 0) {
          const replaced = editor.replaceBlocks(existing, blocks)
          stream.blockIds = replaced.insertedBlocks.map((block) => block.id)
        } else {
          const anchorId = aiBlockIdRef.current
          const anchorBlock = anchorId ? editor.getBlock(anchorId) : undefined
          const reference = anchorBlock ?? editor.document[editor.document.length - 1]
          const inserted = reference
            ? editor.insertBlocks(blocks, reference, 'after')
            : editor.replaceBlocks(editor.document, blocks).insertedBlocks
          stream.blockIds = inserted.map((block) => block.id)
        }
        stream.writtenMd = md
      } finally {
        stream.writing = false
        // A newer delta (or the authoritative final content) arrived while
        // this write was in flight — flush again so nothing is dropped.
        if (stream.latestMd !== stream.writtenMd && stream.timer === null) {
          flushAiStream()
        }
      }
    })()
  }, [editor, onAiInserted])

  const handleAiAnswer = useCallback(
    (md: string, done: boolean): void => {
      const stream = aiStreamRef.current
      stream.latestMd = md
      if (done) {
        if (stream.timer !== null) {
          window.clearTimeout(stream.timer)
          stream.timer = null
        }
        flushAiStream()
        return
      }
      if (stream.timer !== null) return
      stream.timer = window.setTimeout(() => {
        stream.timer = null
        flushAiStream()
      }, 300)
    },
    [flushAiStream]
  )

  const openAskAi = useCallback((): void => {
    resetAiStream()
    const container = containerRef.current
    let anchor: AiAnchor = { top: 48, left: 0, width: container?.clientWidth ?? 540 }
    try {
      const block = editor.getTextCursorPosition().block
      aiBlockIdRef.current = block.id
      const element = container?.querySelector(`[data-id="${block.id}"]`)
      if (container && element) {
        const containerRect = container.getBoundingClientRect()
        const rect = element.getBoundingClientRect()
        anchor = {
          top: rect.bottom - containerRect.top + 6,
          left: rect.left - containerRect.left,
          width: container.clientWidth
        }
      }
    } catch {
      aiBlockIdRef.current = null
    }
    let selected = ''
    try {
      selected = editor.getSelectedText()
    } catch {
      selected = ''
    }
    setAiSelection(selected.trim())
    setAiAnchor(anchor)
    setAiOpen(true)
  }, [editor, resetAiStream])

  // One capture handler covers links in both note content and suggestion
  // blocks: note://<id> opens the note on the map, everything else goes to
  // the external browser (never in-app navigation).
  const handleLinkClicks = (event: React.MouseEvent): void => {
    const anchor = (event.target as HTMLElement).closest('a')
    if (!anchor) return
    event.preventDefault()
    event.stopPropagation()
    const href = anchor.getAttribute('href') ?? ''
    if (href.startsWith('note://')) {
      onOpenNoteInMap?.(decodeURIComponent(href.slice('note://'.length)))
    } else if (href && !href.startsWith('#')) {
      window.open(href)
    }
  }

  const askAiEnabled = Boolean(meetingId && onChatNotesEdit)

  return (
    <div ref={containerRef} className="relative" onClickCapture={handleLinkClicks}>
      <BlockNoteView
        editor={editor}
        editable={!readOnly}
        theme="dark"
        className="notepad-editor"
        data-color-scheme="dark"
        sideMenu={false}
        slashMenu={false}
        onChange={() => {
          requestAnimationFrame(paintProvenance)
          void (async () => {
            const canonical = stripSuggestionBlocks(
              editor.document as unknown as BlockLike[]
            ) as unknown as NotepadBlock[]
            const markdown = (await editor.blocksToMarkdownLossy(canonical)).trimEnd()
            if (markdown === lastExternalMd.current) return
            lastExternalMd.current = markdown
            onChangeMd(markdown)
          })()
        }}
      >
        <SuggestionMenuController
          triggerCharacter="/"
          getItems={async (query) =>
            filterSuggestionItems(
              [
                ...(askAiEnabled
                  ? [
                      {
                        title: 'Ask AI',
                        subtext: 'Answer from your knowledge base and this meeting',
                        aliases: ['ai', 'ask', 'agent'],
                        group: 'AI',
                        icon: <span>✨</span>,
                        onItemClick: () => openAskAi()
                      }
                    ]
                  : []),
                ...getDefaultReactSlashMenuItems(editor)
              ],
              query
            )
          }
        />
      </BlockNoteView>
      {askAiEnabled && meetingId ? (
        <AiInline
          meetingId={meetingId}
          chatId={chatId}
          open={aiOpen}
          anchor={aiAnchor}
          selection={aiSelection}
          onClose={() => setAiOpen(false)}
          onNotesEdit={onChatNotesEdit ?? (() => undefined)}
          onAnswer={handleAiAnswer}
          onAskStarted={resetAiStream}
        />
      ) : null}
    </div>
  )
}

function injectSuggestions(blocks: NotepadBlock[], pending: SuggestedEdit[]): NotepadBlock[] {
  if (pending.length === 0) return blocks
  const result = [...blocks]
  const appends: NotepadBlock[] = []

  for (const edit of pending) {
    const suggestionBlock = {
      type: 'suggestion',
      props: {
        suggestionId: edit.id,
        md: edit.insertMd.trim(),
        rationale: edit.rationale,
        big: edit.anchor === ''
      }
    } as unknown as NotepadBlock

    const anchorId = findAnchorBlockId(result as unknown as BlockLike[], edit)
    if (anchorId === null) {
      appends.push(suggestionBlock)
      continue
    }
    const index = result.findIndex((block) => block.id === anchorId)
    if (index >= 0) result.splice(index + 1, 0, suggestionBlock)
    else appends.push(suggestionBlock)
  }

  return [...result, ...appends]
}
