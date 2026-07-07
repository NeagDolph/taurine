import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

let previewWindow: BrowserWindow | null = null

if (is.dev) {
  // dev-only: lets tooling attach to the real renderer for debugging
  app.commandLine.appendSwitch('remote-debugging-port', '9223')
}

function createWindow(): void {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 980,
    minHeight: 640,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  loadRenderer(mainWindow)
}

function openPreviewWindow(noteId: string): void {
  const hash = `/preview/${encodeURIComponent(noteId)}`

  if (previewWindow && !previewWindow.isDestroyed()) {
    loadRenderer(previewWindow, hash)
    if (previewWindow.isMinimized()) {
      previewWindow.restore()
    }
    previewWindow.show()
    previewWindow.focus()
    return
  }

  previewWindow = new BrowserWindow({
    width: 520,
    height: 680,
    minWidth: 420,
    minHeight: 520,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#0b0b0c',
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  previewWindow.on('ready-to-show', () => {
    previewWindow?.show()
  })

  previewWindow.on('closed', () => {
    previewWindow = null
  })

  previewWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  loadRenderer(previewWindow, hash)
}

function loadRenderer(window: BrowserWindow, hash?: string): void {
  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    const rendererUrl = hash
      ? `${process.env['ELECTRON_RENDERER_URL']}#${hash}`
      : process.env['ELECTRON_RENDERER_URL']
    window.loadURL(rendererUrl)
  } else {
    window.loadFile(join(__dirname, '../renderer/index.html'), hash ? { hash } : undefined)
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // IPC test
  ipcMain.on('ping', () => console.log('pong'))
  ipcMain.handle('preview:open', (_event, noteId: unknown) => {
    if (typeof noteId !== 'string' || noteId.length === 0) {
      throw new Error('Invalid preview note id')
    }

    openPreviewWindow(noteId)
  })

  createWindow()

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
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
