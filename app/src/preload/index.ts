import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  chooseDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:choose-directory'),
  callNotif: {
    // Main window: the OS-level panel (or its countdown) chose "Start notes".
    onStart: (callback: () => void): (() => void) => {
      const listener = (): void => callback()
      ipcRenderer.on('callnotif:start', listener)
      return () => ipcRenderer.removeListener('callnotif:start', listener)
    }
  },
  callNotifPanel: {
    // Notification panel window only.
    action: (action: 'start' | 'dismiss'): void => {
      ipcRenderer.send('callnotif:panel-action', action)
    },
    onShow: (callback: (seconds: number) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, seconds: unknown): void => {
        callback(typeof seconds === 'number' && seconds > 0 ? seconds : 10)
      }
      ipcRenderer.on('callnotif:show', listener)
      return () => ipcRenderer.removeListener('callnotif:show', listener)
    }
  },
  suggestNotif: {
    // Main window: forward a live note suggestion to the OS-level popup.
    notify: (payload: {
      noteId: string
      title: string
      snippet: string
      source: string
    }): void => {
      ipcRenderer.send('suggestnotif:notify', payload)
    },
    // Main window: the popup chose "Open note".
    onOpen: (callback: (noteId: string) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, noteId: unknown): void => {
        if (typeof noteId === 'string' && noteId) callback(noteId)
      }
      ipcRenderer.on('suggestnotif:open', listener)
      return () => ipcRenderer.removeListener('suggestnotif:open', listener)
    }
  },
  suggestNotifPanel: {
    // Suggestion panel window only.
    action: (action: 'open' | 'dismiss'): void => {
      ipcRenderer.send('suggestnotif:panel-action', action)
    },
    onShow: (
      callback: (payload: { title: string; snippet: string; source: string; seconds: number }) => void
    ): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, payload: unknown): void => {
        const record = payload as Record<string, unknown> | null
        if (!record || typeof record.title !== 'string') return
        callback({
          title: record.title,
          snippet: typeof record.snippet === 'string' ? record.snippet : '',
          source: typeof record.source === 'string' ? record.source : '',
          seconds: typeof record.seconds === 'number' && record.seconds > 0 ? record.seconds : 12
        })
      }
      ipcRenderer.on('suggestnotif:show', listener)
      return () => ipcRenderer.removeListener('suggestnotif:show', listener)
    }
  },
  session: {
    reportListening: (listening: boolean): void => {
      ipcRenderer.send('session:listening', listening)
    }
  },
  sysAudio: {
    available: (): Promise<boolean> => ipcRenderer.invoke('sysaudio:available'),
    start: (): Promise<void> => ipcRenderer.invoke('sysaudio:start'),
    stop: (): Promise<void> => ipcRenderer.invoke('sysaudio:stop'),
    onChunk: (callback: (chunk: ArrayBuffer) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, payload: unknown): void => {
        const chunk = copyArrayBuffer(payload)
        if (chunk) callback(chunk)
      }
      ipcRenderer.on('sysaudio:chunk', listener)
      return () => ipcRenderer.removeListener('sysaudio:chunk', listener)
    },
    onError: (callback: (message: string) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, message: unknown): void => {
        callback(typeof message === 'string' ? message : String(message))
      }
      ipcRenderer.on('sysaudio:error', listener)
      return () => ipcRenderer.removeListener('sysaudio:error', listener)
    }
  }
}

function copyArrayBuffer(payload: unknown): ArrayBuffer | null {
  let source: Uint8Array
  if (payload instanceof ArrayBuffer) {
    source = new Uint8Array(payload)
  } else if (ArrayBuffer.isView(payload)) {
    source = new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength)
  } else {
    return null
  }

  const copy = new Uint8Array(source.byteLength)
  copy.set(source)
  return copy.buffer
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.api = api
}
