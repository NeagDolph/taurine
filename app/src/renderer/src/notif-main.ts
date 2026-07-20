/** Countdown panel logic for the OS-level call notification window. */

// No imports here — the empty export keeps this a module so its top-level
// names don't collide with the other panel script in the same TS program.
export {}

interface PanelApi {
  callNotifPanel: {
    action(action: 'start' | 'dismiss'): void
    onShow(callback: (seconds: number) => void): () => void
  }
}

const api = (window as unknown as { api: PanelApi }).api
const bar = document.getElementById('countdown') as HTMLDivElement
const startButton = document.getElementById('start') as HTMLButtonElement
const dismissButton = document.getElementById('dismiss') as HTMLButtonElement

let timer: ReturnType<typeof setTimeout> | null = null

function resolve(action: 'start' | 'dismiss'): void {
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  api.callNotifPanel.action(action)
}

startButton.addEventListener('click', () => resolve('start'))
dismissButton.addEventListener('click', () => resolve('dismiss'))

api.callNotifPanel.onShow((seconds) => {
  if (timer) clearTimeout(timer)
  // Restart the CSS countdown: snap to full width, reflow, then animate out.
  bar.classList.remove('running')
  bar.style.transitionDuration = '0s'
  void bar.offsetWidth
  bar.style.transitionDuration = `${seconds}s`
  bar.classList.add('running')
  timer = setTimeout(() => resolve('start'), seconds * 1000)
})
