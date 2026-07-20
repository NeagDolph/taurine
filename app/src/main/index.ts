import { app, shell, BrowserWindow, dialog, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { startMicWatch } from './micwatch'
import { CallNotifier } from './notifier'
import { registerSystemAudioIpc } from './sysaudio'
import { createTray, destroyTray } from './tray'

// Menus, notifications, and app.getPath('userData') naming. The dev dock
// label itself comes from the Electron bundle's Info.plist, which
// scripts/rename-dev-electron.mjs rewrites to Taurine after install.
app.setName('Taurine')

let mainWindow: BrowserWindow | null = null
// Closing the window keeps the app alive in the tray/dock; only an explicit
// quit (Cmd+Q, tray Quit, updater) actually tears the window down.
let quitting = false

function showMainWindow(): void {
  if (mainWindow === null || mainWindow.isDestroyed()) {
    createWindow()
    return
  }
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.show()
  mainWindow.focus()
}

function startMeetingFromTray(): void {
  showMainWindow()
  const contents = mainWindow?.webContents
  if (!contents) return
  // Reuses the call-notification channel: the renderer starts a meeting and
  // opens the Meetings tab.
  if (contents.isLoading()) {
    contents.once('did-finish-load', () => contents.send('callnotif:start'))
  } else {
    contents.send('callnotif:start')
  }
}

if (is.dev) {
  // dev-only: lets tooling attach to the real renderer for debugging
  app.commandLine.appendSwitch('remote-debugging-port', '9223')
}

function createWindow(): void {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1180,
    minHeight: 760,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      // Transcription can start from the call notification while this window
      // is hidden — never throttle the renderer that pumps PCM frames.
      backgroundThrottling: false
    }
  })

  const window = mainWindow
  window.on('ready-to-show', () => {
    window.show()
  })
  window.on('close', (event) => {
    if (quitting) return
    event.preventDefault()
    window.hide()
  })
  window.on('closed', () => {
    if (mainWindow === window) mainWindow = null
  })

  window.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  loadRenderer(window)
}

function loadRenderer(window: BrowserWindow): void {
  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    window.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    window.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.taurine.app')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // IPC test
  ipcMain.on('ping', () => console.log('pong'))
  ipcMain.handle('dialog:choose-directory', async (event) => {
    const owner = BrowserWindow.fromWebContents(event.sender)
    const options = {
      properties: ['openDirectory', 'createDirectory'] as Array<
        'openDirectory' | 'createDirectory'
      >,
      message: 'Choose a folder to index'
    }
    const result = owner
      ? await dialog.showOpenDialog(owner, options)
      : await dialog.showOpenDialog(options)
    return result.canceled ? null : (result.filePaths[0] ?? null)
  })
  registerSystemAudioIpc()
  const notifier = new CallNotifier(() => mainWindow)
  const stopMicWatch = startMicWatch((active) => notifier.onMicActive(active))
  app.once('will-quit', () => {
    stopMicWatch()
    notifier.dispose()
    destroyTray()
  })
  if (process.platform === 'darwin' && is.dev) {
    // Packaged builds get the icns; dev runs show the stock Electron icon
    // unless the dock icon is set explicitly.
    app.dock?.setIcon(icon)
  }
  createTray({
    showWindow: showMainWindow,
    startMeeting: startMeetingFromTray
  })
  createWindow()

  app.on('activate', function () {
    // Dock click: surface the hidden window (or recreate it if gone).
    showMainWindow()
  })
})

app.on('before-quit', () => {
  quitting = true
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
