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
    <div className="composer">
      <textarea
        ref={textareaRef}
        className="composer-input"
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
          className="composer-button stop"
          type="button"
          aria-label="Stop generating"
          onClick={onStop}
        >
          ⏹
        </button>
      ) : (
        <button
          className="composer-button send"
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
