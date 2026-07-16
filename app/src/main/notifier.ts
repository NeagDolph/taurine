import { BrowserWindow, ipcMain, screen } from 'electron'
import { is } from '@electron-toolkit/utils'
import { join } from 'path'
import {
  callNotifyReducer,
  INITIAL_CALL_NOTIFY_STATE,
  type CallNotifyEvent,
  type CallNotifyState
} from './call-notify-state'

const PANEL_WIDTH = 344
const PANEL_HEIGHT = 108
const PANEL_MARGIN = 16
const REARM_IDLE_MS = 5000
export const COUNTDOWN_SECONDS = 10

/**
 * OS-level "call detected" notification: a frameless always-on-top panel in
 * the bottom-right of the screen, fully independent of the main window —
 * visible even when Taurine is hidden, minimized, or on another Space.
 */
export class CallNotifier {
  private state: CallNotifyState = INITIAL_CALL_NOTIFY_STATE
  private panel: BrowserWindow | null = null
  private rearmTimer: ReturnType<typeof setTimeout> | null = null
  private disposed = false

  constructor(private readonly getMainWindow: () => BrowserWindow | null) {
    ipcMain.on('callnotif:panel-action', (_event, action: unknown) => {
      if (action === 'start') this.startMeeting()
      else this.resolvePanel()
    })
    ipcMain.on('session:listening', (_event, listening: unknown) => {
      this.dispatch({ type: 'listening', listening: Boolean(listening) })
    })
  }

  onMicActive(active: boolean): void {
    this.dispatch({ type: 'mic', active })
    if (!active) this.scheduleRearm()
  }

  dispose(): void {
    this.disposed = true
    if (this.rearmTimer) clearTimeout(this.rearmTimer)
    this.panel?.destroy()
    this.panel = null
  }

  private dispatch(event: CallNotifyEvent): void {
    if (this.disposed) return
    const [next, effect] = callNotifyReducer(this.state, event)
    this.state = next
    if (effect === 'show') this.showPanel()
    else if (effect === 'hide') this.hidePanel()
  }

  private startMeeting(): void {
    this.resolvePanel()
    const main = this.getMainWindow()
    if (!main) return
    if (main.isMinimized()) main.restore()
    main.show()
    main.focus()
    main.webContents.send('callnotif:start')
  }

  private resolvePanel(): void {
    this.dispatch({ type: 'panel-resolved' })
    this.scheduleRearm()
  }

  private scheduleRearm(): void {
    if (this.rearmTimer) clearTimeout(this.rearmTimer)
    this.rearmTimer = setTimeout(() => {
      this.rearmTimer = null
      this.dispatch({ type: 'rearm' })
    }, REARM_IDLE_MS)
  }

  private showPanel(): void {
    const panel = this.ensurePanel()
    const area = screen.getPrimaryDisplay().workArea
    panel.setPosition(
      area.x + area.width - PANEL_WIDTH - PANEL_MARGIN,
      area.y + area.height - PANEL_HEIGHT - PANEL_MARGIN
    )
    panel.webContents.send('callnotif:show', COUNTDOWN_SECONDS)
    panel.showInactive()
  }

  private hidePanel(): void {
    this.panel?.hide()
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
      focusable: true,
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
      void panel.loadURL(`${process.env['ELECTRON_RENDERER_URL']}/notif.html`)
    } else {
      void panel.loadFile(join(__dirname, '../renderer/notif.html'))
    }

    this.panel = panel
    return panel
  }
}
