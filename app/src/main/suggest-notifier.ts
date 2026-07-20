import { BrowserWindow, ipcMain, screen } from 'electron'
import { is } from '@electron-toolkit/utils'
import { join } from 'path'

const PANEL_WIDTH = 344
const PANEL_HEIGHT = 168
const PANEL_MARGIN = 16
export const SUGGESTION_SECONDS = 12
const QUEUE_LIMIT = 3

export interface NoteSuggestion {
  noteId: string
  title: string
  snippet: string
  source: string
}

interface SuggestionHandles {
  /** "Open note" — surface Taurine and focus the note in the map. */
  openNote(noteId: string): void
}

/**
 * OS-level suggested-note popup: while a meeting is being transcribed, notes
 * surfaced by live retrieval appear bottom-right with their most important
 * snippet — independent of the main window, and (like every Taurine window)
 * invisible to screenshare via content protection.
 */
export class SuggestionNotifier {
  private panel: BrowserWindow | null = null
  private current: NoteSuggestion | null = null
  private queue: NoteSuggestion[] = []
  private disposed = false

  constructor(private readonly handles: SuggestionHandles) {
    ipcMain.on('suggestnotif:notify', (_event, payload: unknown) => {
      const suggestion = sanitize(payload)
      if (suggestion) this.notify(suggestion)
    })
    ipcMain.on('suggestnotif:panel-action', (_event, action: unknown) => {
      if (action === 'open' && this.current) {
        this.handles.openNote(this.current.noteId)
      }
      this.advance()
    })
  }

  notify(suggestion: NoteSuggestion): void {
    if (this.disposed) return
    if (this.current) {
      // A popup is on screen: queue, keeping only the newest few.
      this.queue.push(suggestion)
      if (this.queue.length > QUEUE_LIMIT) this.queue.shift()
      return
    }
    this.present(suggestion)
  }

  dispose(): void {
    this.disposed = true
    this.queue = []
    this.current = null
    this.panel?.destroy()
    this.panel = null
  }

  /** The visible suggestion resolved (open/dismiss/timeout): show the next. */
  private advance(): void {
    this.current = null
    const next = this.queue.shift()
    if (next) {
      this.present(next)
    } else {
      this.panel?.hide()
    }
  }

  private present(suggestion: NoteSuggestion): void {
    this.current = suggestion
    const panel = this.ensurePanel()
    const area = screen.getPrimaryDisplay().workArea
    panel.setPosition(
      area.x + area.width - PANEL_WIDTH - PANEL_MARGIN,
      area.y + area.height - PANEL_HEIGHT - PANEL_MARGIN
    )
    const payload = {
      title: suggestion.title,
      snippet: suggestion.snippet,
      source: suggestion.source,
      seconds: SUGGESTION_SECONDS
    }
    const contents = panel.webContents
    if (contents.isLoading()) {
      contents.once('did-finish-load', () => contents.send('suggestnotif:show', payload))
    } else {
      contents.send('suggestnotif:show', payload)
    }
    panel.showInactive()
  }

  private ensurePanel(): BrowserWindow {
    if (this.panel && !this.panel.isDestroyed()) return this.panel

    const panel = new BrowserWindow({
      width: PANEL_WIDTH,
      height: PANEL_HEIGHT,
      frame: false,
      transparent: true,
      resizable: false,
      movable: false,
      minimizable: false,
      maximizable: false,
      closable: false,
      // Non-activating, like the call panel: clicking the popup must never
      // activate the app while the user is mid-call in another window.
      focusable: false,
      show: false,
      skipTaskbar: true,
      hasShadow: false,
      webPreferences: {
        preload: join(__dirname, '../preload/index.js'),
        sandbox: false
      }
    })
    panel.setAlwaysOnTop(true, 'screen-saver')
    panel.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })

    if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
      void panel.loadURL(`${process.env['ELECTRON_RENDERER_URL']}/suggest.html`)
    } else {
      void panel.loadFile(join(__dirname, '../renderer/suggest.html'))
    }

    this.panel = panel
    return panel
  }
}

function sanitize(payload: unknown): NoteSuggestion | null {
  const record = payload as Record<string, unknown> | null
  if (!record || typeof record.noteId !== 'string' || !record.noteId) return null
  if (typeof record.title !== 'string' || !record.title.trim()) return null
  return {
    noteId: record.noteId,
    title: record.title,
    snippet: typeof record.snippet === 'string' ? record.snippet : '',
    source: typeof record.source === 'string' ? record.source : ''
  }
}
