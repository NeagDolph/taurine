import { useEffect, useRef, useState } from 'react'

interface ComposerProps {
  disabled: boolean // no active chat target (never disabled for typing)
  streaming: boolean // in-flight turn: send morphs to stop
  onSend: (text: string) => void
  onStop: () => void
  focusKey: string | null // refocus when the active chat changes
  draft?: { text: string; nonce: number } | null // prefill (e.g. map "Ask about this")
}

export function Composer({
  disabled,
  streaming,
  onSend,
  onStop,
  focusKey,
  draft
}: ComposerProps): React.JSX.Element {
  const [text, setText] = useState('')
  const [appliedDraftNonce, setAppliedDraftNonce] = useState(0)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // draft prefill (e.g. map "Ask about this") — applied during render, like a
  // derived-state latch, so the textarea carries the text on this commit
  if (draft && draft.nonce !== appliedDraftNonce) {
    setAppliedDraftNonce(draft.nonce)
    setText(draft.text)
  }

  useEffect(() => {
    textareaRef.current?.focus()
  }, [focusKey])

  // resize/focus once the prefilled value has been committed to the DOM
  useEffect(() => {
    if (appliedDraftNonce === 0) {
      return
    }
    const node = textareaRef.current
    if (node) {
      node.style.height = 'auto'
      node.style.height = `${Math.min(node.scrollHeight, 8 * 24 + 16)}px`
      node.focus()
      node.setSelectionRange(node.value.length, node.value.length)
    }
  }, [appliedDraftNonce])

  const submit = (): void => {
    const trimmed = text.trim()
    if (!trimmed || streaming || disabled) {
      return
    }
    onSend(trimmed)
    setText('')
    resize()
  }

  const resize = (): void => {
    const node = textareaRef.current
    if (!node) {
      return
    }
    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, 8 * 24 + 16)}px`
  }

  return (
    <div className="flex items-end gap-2 px-4 pb-3.5 pt-2.5">
      <textarea
        ref={textareaRef}
        className="max-h-52 flex-1 resize-none rounded-xl border border-white/10 bg-surface-2/70 px-3.5 py-2.5 text-[13px] leading-normal text-fg outline-none transition-colors placeholder:text-fg/35 focus:border-accent/70"
        placeholder="Ask your notes anything…"
        rows={1}
        value={text}
        onChange={(event) => {
          setText(event.target.value)
          resize()
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            submit()
          }
          if (event.key === 'k' && (event.metaKey || event.ctrlKey)) {
            event.preventDefault()
            setText('')
            resize()
          }
        }}
      />
      {streaming ? (
        <button
          className="flex h-9 w-10 flex-none cursor-pointer items-center justify-center rounded-[10px] border-0 bg-bad/20 text-sm text-bad transition-colors hover:bg-bad/30"
          type="button"
          aria-label="Stop generating"
          onClick={onStop}
        >
          ⏹
        </button>
      ) : (
        <button
          className="flex h-9 w-10 flex-none cursor-pointer items-center justify-center rounded-[10px] border-0 bg-accent text-sm text-accent-ink transition-colors hover:bg-accent-hi disabled:cursor-default"
          type="button"
          aria-label="Send"
          disabled={disabled || !text.trim()}
          onClick={submit}
        >
          ⏎
        </button>
      )}
    </div>
  )
}
