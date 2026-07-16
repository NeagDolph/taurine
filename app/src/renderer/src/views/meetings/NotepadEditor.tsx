/* eslint-disable react-refresh/only-export-components -- Task 1.2 requires these helpers from this module. */
import {
  BasicBlocksPlugin,
  BasicMarksPlugin,
  BoldPlugin,
  H2Plugin,
  ItalicPlugin
} from '@platejs/basic-nodes/react'
import { ListStyleType, toggleList } from '@platejs/list'
import { ListPlugin } from '@platejs/list/react'
import { MarkdownPlugin, deserializeMd, serializeMd } from '@platejs/markdown'
import type { Descendant, RenderElementProps, RenderLeafProps, TElement, Value } from 'platejs'
import { Plate, PlateContent, createPlateEditor, useEditorRef, usePlateEditor } from 'platejs/react'
import { useEffect, useRef } from 'react'
import type { SuggestedEdit } from '../../lib/meetings'
import { projectPendingSuggestions, type SuggestionProjection } from '../../lib/suggestions'

export interface NotepadEditorProps {
  valueMd: string
  onChangeMd(md: string): void
  suggestions: SuggestedEdit[]
  onResolveSuggestion(id: string, accept: boolean): void
  readOnly?: boolean
}

const NOTEPAD_PLUGINS = [BasicBlocksPlugin, BasicMarksPlugin, ListPlugin, MarkdownPlugin]

export function deserializeMarkdown(markdown: string): Value {
  const editor = createPlateEditor({ plugins: NOTEPAD_PLUGINS })
  return deserializeMd(editor, markdown) as Value
}

export function serializeMarkdown(value: Value): string {
  const acceptedValue = filterPendingSuggestions(value)
  const editor = createPlateEditor({ plugins: NOTEPAD_PLUGINS, value: acceptedValue })
  return serializeMd(editor, { value: acceptedValue }).trimEnd()
}

export function NotepadEditor({
  valueMd,
  onChangeMd,
  suggestions,
  onResolveSuggestion,
  readOnly = false
}: NotepadEditorProps): React.JSX.Element {
  const editor = usePlateEditor({
    plugins: NOTEPAD_PLUGINS,
    value: deserializeMarkdown(valueMd)
  })
  const lastExternalValue = useRef(valueMd)

  useEffect(() => {
    if (valueMd === lastExternalValue.current) return
    lastExternalValue.current = valueMd
    editor.tf.setValue(deserializeMarkdown(valueMd))
  }, [editor, valueMd])

  return (
    <Plate
      editor={editor}
      readOnly={readOnly}
      onValueChange={({ value }) => {
        const markdown = serializeMarkdown(value)
        lastExternalValue.current = markdown
        onChangeMd(markdown)
      }}
    >
      <NotepadToolbar readOnly={readOnly} />
      <SuggestionDecorations
        projections={projectPendingSuggestions(valueMd, suggestions)}
        onResolveSuggestion={onResolveSuggestion}
        readOnly={readOnly}
      />
      <PlateContent
        className="min-h-full px-5 py-4 text-[14px] leading-7 text-fg outline-none"
        placeholder="Take notes…"
        renderElement={renderElement}
        renderLeaf={renderLeaf}
        spellCheck
      />
    </Plate>
  )
}

export function SuggestionDecorations({
  projections,
  onResolveSuggestion,
  readOnly
}: {
  projections: SuggestionProjection[]
  onResolveSuggestion(id: string, accept: boolean): void
  readOnly: boolean
}): React.JSX.Element | null {
  if (projections.length === 0) return null

  return (
    <aside
      className="grid gap-2 border-b border-accent/20 bg-accent/[0.035] px-4 py-3"
      aria-label="Suggested note edits"
    >
      {projections
        .slice()
        .sort((left, right) => left.location - right.location)
        .map(({ edit, appendMode }) => (
          <article
            className="rounded-lg border border-accent/25 bg-surface-2/90 px-3 py-2 shadow-sm"
            key={edit.id}
            title={edit.rationale || undefined}
          >
            <div className="flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="mb-1 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-accent-hi">
                  <span>
                    {appendMode ? 'Append to notes' : `After “${anchorLabel(edit.anchor)}”`}
                  </span>
                  {edit.rationale ? (
                    <span
                      className="cursor-help text-fg/35"
                      aria-label={`Rationale: ${edit.rationale}`}
                      title={edit.rationale}
                    >
                      ⓘ
                    </span>
                  ) : null}
                </div>
                <div className="whitespace-pre-wrap text-[12px] leading-5 text-fg/80">
                  {edit.insertMd.trim()}
                </div>
              </div>
              <div className="flex flex-none items-center gap-1">
                <button
                  className="primary-button compact"
                  type="button"
                  aria-label="Accept suggestion"
                  disabled={readOnly}
                  onClick={() => onResolveSuggestion(edit.id, true)}
                >
                  Accept
                </button>
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Dismiss suggestion"
                  disabled={readOnly}
                  onClick={() => onResolveSuggestion(edit.id, false)}
                >
                  &times;
                </button>
              </div>
            </div>
          </article>
        ))}
    </aside>
  )
}

function anchorLabel(anchor: string): string {
  const compact = anchor.replace(/\s+/g, ' ').trim()
  return compact.length > 48 ? `${compact.slice(0, 47)}…` : compact
}

function filterPendingSuggestions(value: Value): Value {
  return value
    .map(filterNode)
    .filter((node): node is TElement => node !== null && 'children' in node)
}

function filterNode(node: Descendant): Descendant | null {
  const record = node as Descendant & Record<string, unknown>
  if (isSuggestionNode(record)) return null
  if (!('children' in record) || !Array.isArray(record.children)) return node

  const children = (record.children as Descendant[])
    .map(filterNode)
    .filter((child): child is Descendant => child !== null)

  return {
    ...record,
    children: children.length > 0 ? children : [{ text: '' }]
  } as Descendant
}

function isSuggestionNode(node: Record<string, unknown>): boolean {
  return Boolean(node.suggestion) || Object.keys(node).some((key) => key.startsWith('suggestion_'))
}

function NotepadToolbar({ readOnly }: { readOnly: boolean }): React.JSX.Element {
  const editor = useEditorRef()
  const run =
    (action: () => void) =>
    (event: React.MouseEvent<HTMLButtonElement>): void => {
      event.preventDefault()
      action()
      editor.tf.focus()
    }

  return (
    <div className="flex h-9 items-center gap-1 border-b border-white/[0.08] px-3">
      <button
        className="icon-button font-bold"
        type="button"
        title="Bold"
        aria-label="Bold"
        disabled={readOnly}
        onMouseDown={run(() => editor.tf.toggleMark(BoldPlugin.key))}
      >
        B
      </button>
      <button
        className="icon-button italic"
        type="button"
        title="Italic"
        aria-label="Italic"
        disabled={readOnly}
        onMouseDown={run(() => editor.tf.toggleMark(ItalicPlugin.key))}
      >
        I
      </button>
      <button
        className="icon-button w-auto px-1.5 font-semibold"
        type="button"
        title="Heading"
        aria-label="Heading"
        disabled={readOnly}
        onMouseDown={run(() => editor.tf.toggleBlock(H2Plugin.key))}
      >
        H2
      </button>
      <button
        className="icon-button w-auto px-1.5"
        type="button"
        title="Bulleted list"
        aria-label="Bulleted list"
        disabled={readOnly}
        onMouseDown={run(() => toggleList(editor, { listStyleType: ListStyleType.Disc }))}
      >
        • List
      </button>
    </div>
  )
}

function renderElement({ attributes, children, element }: RenderElementProps): React.JSX.Element {
  const record = element as TElement & Record<string, unknown>
  switch (record.type) {
    case 'h1':
      return (
        <h1 className="mb-2 mt-4 text-2xl font-bold leading-tight" {...attributes}>
          {children}
        </h1>
      )
    case 'h2':
      return (
        <h2 className="mb-1.5 mt-4 text-xl font-bold leading-tight" {...attributes}>
          {children}
        </h2>
      )
    case 'h3':
      return (
        <h3 className="mb-1 mt-3 text-base font-semibold" {...attributes}>
          {children}
        </h3>
      )
    case 'blockquote':
      return (
        <blockquote className="my-2 border-l-2 border-accent/50 pl-3 text-fg/70" {...attributes}>
          {children}
        </blockquote>
      )
    default: {
      const listStyle = typeof record.listStyleType === 'string' ? record.listStyleType : null
      const indent = typeof record.indent === 'number' ? Math.max(1, record.indent) : 1
      if (listStyle) {
        const marker = listStyle === ListStyleType.Decimal ? `${record.listStart ?? 1}.` : '•'
        return (
          <div
            className="flex gap-2"
            style={{ paddingLeft: `${(indent - 1) * 20}px` }}
            {...attributes}
          >
            <span
              className="w-4 flex-none select-none text-right text-fg/50"
              contentEditable={false}
            >
              {marker}
            </span>
            <span className="min-w-0 flex-1">{children}</span>
          </div>
        )
      }
      return <p {...attributes}>{children}</p>
    }
  }
}

function renderLeaf({ attributes, children, leaf }: RenderLeafProps): React.JSX.Element {
  const record = leaf as Record<string, unknown>
  let content = children
  if (record.bold) content = <strong className="font-semibold">{content}</strong>
  if (record.italic) content = <em>{content}</em>
  if (record.code) {
    content = (
      <code className="rounded bg-black/30 px-1 py-0.5 font-mono text-[0.9em]">{content}</code>
    )
  }
  return <span {...attributes}>{content}</span>
}
