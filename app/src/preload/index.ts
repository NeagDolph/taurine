import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  chooseDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:choose-directory'),
  micWatch: {
    onActive: (callback: (active: boolean) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, active: unknown): void => {
        if (typeof active === 'boolean') callback(active)
      }
      ipcRenderer.on('micwatch:active', listener)
      return () => ipcRenderer.removeListener('micwatch:active', listener)
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
