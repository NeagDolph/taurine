/** Panel logic for the OS-level suggested-note popup. */

// No imports here — the empty export keeps this a module so its top-level
// names don't collide with the other panel script in the same TS program.
export {}

interface PanelApi {
  suggestNotifPanel: {
    action(action: 'open' | 'dismiss'): void
    onShow(
      callback: (payload: {
        title: string
        snippet: string
        source: string
        seconds: number
      }) => void
    ): () => void
  }
}

const api = (window as unknown as { api: PanelApi }).api
const bar = document.getElementById('countdown') as HTMLDivElement
const title = document.getElementById('title') as HTMLDivElement
const snippet = document.getElementById('snippet') as HTMLDivElement
const openButton = document.getElementById('open') as HTMLButtonElement
const dismissButton = document.getElementById('dismiss') as HTMLButtonElement

let timer: ReturnType<typeof setTimeout> | null = null

function resolve(action: 'open' | 'dismiss'): void {
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  api.suggestNotifPanel.action(action)
}

openButton.addEventListener('click', () => resolve('open'))
dismissButton.addEventListener('click', () => resolve('dismiss'))

api.suggestNotifPanel.onShow((payload) => {
  title.textContent = payload.title
  snippet.textContent = payload.snippet || 'Surfaced for this conversation'
  snippet.classList.toggle('empty', !payload.snippet)

  if (timer) clearTimeout(timer)
  // Restart the CSS countdown: snap to full width, reflow, then animate out.
  bar.classList.remove('running')
  bar.style.transitionDuration = '0s'
  void bar.offsetWidth
  bar.style.transitionDuration = `${payload.seconds}s`
  bar.classList.add('running')
  // Timeout auto-dismisses — never auto-opens the app.
  timer = setTimeout(() => resolve('dismiss'), payload.seconds * 1000)
})
