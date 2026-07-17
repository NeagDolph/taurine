import { app, Menu, nativeImage, Tray } from 'electron'
import { join } from 'path'

interface TrayHandles {
  showWindow(): void
  startMeeting(): void
}

let tray: Tray | null = null // module-level: prevents GC from destroying the tray

/** Menu-bar presence: the app lives here while the window is hidden. */
export function createTray(handles: TrayHandles): void {
  const image = nativeImage.createFromPath(trayIconPath())
  // The "Template" filename suffix opts into macOS template rendering
  // (auto-recolored for light/dark menu bars); be explicit for safety.
  image.setTemplateImage(true)
  tray = new Tray(image)
  tray.setToolTip('Taurine')
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: 'Show Taurine', click: () => handles.showWindow() },
      { label: 'Start meeting', click: () => handles.startMeeting() },
      { type: 'separator' },
      { label: 'Quit Taurine', click: () => app.quit() }
    ])
  )
}

export function destroyTray(): void {
  tray?.destroy()
  tray = null
}

function trayIconPath(): string {
  return app.isPackaged
    ? join(process.resourcesPath, 'app.asar.unpacked', 'resources', 'trayTemplate.png')
    : join(app.getAppPath(), 'resources', 'trayTemplate.png')
}
