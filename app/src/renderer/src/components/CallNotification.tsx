import { useCallback, useEffect, useRef, useState } from 'react'

interface CallNotificationProps {
  seconds: number
  onStart: () => void
  onDismiss: () => void
}

export function CallNotification({
  seconds,
  onStart,
  onDismiss
}: CallNotificationProps): React.JSX.Element {
  const [countingDown, setCountingDown] = useState(false)
  const handledRef = useRef(false)
  const durationMs = Math.max(0, seconds) * 1000

  const start = useCallback((): void => {
    if (handledRef.current) return
    handledRef.current = true
    onStart()
  }, [onStart])

  const dismiss = useCallback((): void => {
    if (handledRef.current) return
    handledRef.current = true
    onDismiss()
  }, [onDismiss])

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setCountingDown(true))
    const timer = window.setTimeout(start, durationMs)
    return () => {
      window.cancelAnimationFrame(frame)
      window.clearTimeout(timer)
    }
  }, [durationMs, start])

  return (
    <aside
      className="fixed bottom-5 right-5 z-[2147483647] w-[340px] overflow-hidden rounded-[10px] border border-white/[0.14] bg-surface-2 shadow-[0_20px_60px_rgba(0,0,0,0.58)]"
      role="alertdialog"
      aria-label="Call detected"
    >
      <div className="absolute inset-x-0 top-0 h-[3px] bg-white/[0.08]" aria-hidden="true">
        <div
          className="h-full w-full bg-accent"
          style={{
            transform: countingDown ? 'scaleX(0)' : 'scaleX(1)',
            transformOrigin: 'left',
            transitionDuration: `${durationMs}ms`,
            transitionProperty: 'transform',
            transitionTimingFunction: 'linear'
          }}
        />
      </div>
      <div className="px-4 pb-3.5 pt-4">
        <p className="text-[13px] font-semibold text-fg">Call detected — start meeting notes?</p>
        <div className="mt-3 flex justify-end gap-2">
          <button className="secondary-button" type="button" onClick={dismiss}>
            Dismiss
          </button>
          <button className="primary-button" type="button" onClick={start}>
            Start notes
          </button>
        </div>
      </div>
    </aside>
  )
}
