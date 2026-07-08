import { useEffect, useRef, useState } from 'react'

interface ComposerProps {
  disabled: boolean // no active chat target (never disabled for typing)
  streaming: boolean // in-flight turn: send morphs to stop
  onSend: (text: string) => void
  onStop: () => void
  focusKey: string | null // refocus when the active chat changes
}

export function Composer({
  disabled,
  streaming,
  onSend,
  onStop,
  focusKey
}: ComposerProps): React.JSX.Element {
  const [text, setText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    textareaRef.current?.focus()
  }, [focusKey])

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
