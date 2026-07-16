import '@blocknote/core/fonts/inter.css'
import '@blocknote/mantine/style.css'
import { BlockNoteSchema, defaultBlockSpecs, type Block } from '@blocknote/core'
import { BlockNoteView } from '@blocknote/mantine'
import { createReactBlockSpec, useCreateBlockNote } from '@blocknote/react'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import type { SuggestedEdit } from '../../lib/meetings'
import { normalizeLine } from '../../lib/provenance'
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
  readOnly = false
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

  return (
    <div ref={containerRef} onClickCapture={handleLinkClicks}>
      <BlockNoteView
      editor={editor}
      editable={!readOnly}
      theme="dark"
      className="notepad-editor"
      data-color-scheme="dark"
      sideMenu={false}
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
      />
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
