import '@blocknote/core/fonts/inter.css'
import '@blocknote/mantine/style.css'
import {
  BlockNoteSchema,
  defaultBlockSpecs,
  filterSuggestionItems,
  type Block
} from '@blocknote/core'
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
import { AiStreamWriter } from '../../lib/ai-stream'
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
  onAiInserted?(md: string, done: boolean): void
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
          className={`suggestion-block my-1.5 overflow-hidden rounded-md border border-ok/25 bg-ok/[0.05] ${
            big ? 'mt-4' : ''
          }`}
          contentEditable={false}
          data-suggestion-id={suggestionId}
        >
          <div className="flex min-h-7 items-center gap-2 border-b border-ok/[0.15] bg-ok/[0.06] py-1 pl-2.5 pr-1.5">
            <span className="text-[10px] font-bold uppercase tracking-wider text-ok/90">
              {big ? '+ Suggested summary' : '+ Suggested'}
            </span>
            {rationale ? (
              <span className="min-w-0 flex-1 truncate text-[11px] text-fg/45" title={rationale}>
                {rationale}
              </span>
            ) : (
              <span className="flex-1" />
            )}
            <button
              className="flex h-6 flex-none items-center gap-1 rounded px-2 text-[11px] font-semibold text-ok hover:bg-ok/15"
              type="button"
              aria-label="Accept suggestion"
              onClick={() => resolve(true)}
            >
              ✓ Accept
            </button>
            <button
              className="flex h-6 flex-none items-center gap-1 rounded px-2 text-[11px] text-fg/55 hover:bg-white/10 hover:text-fg"
              type="button"
              aria-label="Dismiss suggestion"
              onClick={() => resolve(false)}
            >
              ✕ Dismiss
            </button>
          </div>
          <div className="border-l-2 border-ok/60 py-1.5 pl-3 pr-3">
            <div className="markdown-body suggestion-md text-[13.5px] leading-6 text-fg/85">
              <ReactMarkdown>{md}</ReactMarkdown>
            </div>
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
  const onChangeMdRef = useRef(onChangeMd)
  useEffect(() => {
    onChangeMdRef.current = onChangeMd
  }, [onChangeMd])
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

  // Streamed Ask-AI answers write straight into the note (Granola style).
  // Every flush explicitly serializes and propagates the document upward —
  // persistence never depends on editor-internal change events.
  const emitEditorMarkdown = useCallback(async (): Promise<void> => {
    const canonical = stripSuggestionBlocks(
      editor.document as unknown as BlockLike[]
    ) as unknown as NotepadBlock[]
    const markdown = (await editor.blocksToMarkdownLossy(canonical)).trimEnd()
    if (markdown === lastExternalMd.current) return
    lastExternalMd.current = markdown
    onChangeMdRef.current(markdown)
  }, [editor])

  const onAiInsertedRef = useRef(onAiInserted)
  const emitEditorMarkdownRef = useRef(emitEditorMarkdown)
  useEffect(() => {
    onAiInsertedRef.current = onAiInserted
    emitEditorMarkdownRef.current = emitEditorMarkdown
  }, [emitEditorMarkdown, onAiInserted])

  const aiWriterRef = useRef<AiStreamWriter | null>(null)
  useEffect(() => {
    aiWriterRef.current = new AiStreamWriter(
      editor as unknown as ConstructorParameters<typeof AiStreamWriter>[0],
      {
        emit: () => emitEditorMarkdownRef.current(),
        onProvenance: (md, done) => onAiInsertedRef.current?.(md, done),
        anchorBlockId: () => aiBlockIdRef.current
      }
    )
    return () => {
      aiWriterRef.current = null
    }
  }, [editor])

  const resetAiStream = useCallback((): void => {
    const continueAfter = aiWriterRef.current?.reset() ?? null
    if (continueAfter) aiBlockIdRef.current = continueAfter
  }, [])

  const handleAiAnswer = useCallback((md: string, done: boolean): void => {
    aiWriterRef.current?.handleAnswer(md, done)
  }, [])

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
