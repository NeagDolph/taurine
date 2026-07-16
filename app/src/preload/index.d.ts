import { ElectronAPI } from '@electron-toolkit/preload'

export interface TaurineApi {
  chooseDirectory: () => Promise<string | null>
  sysAudio: {
    available: () => Promise<boolean>
    start: () => Promise<void>
    stop: () => Promise<void>
    onChunk: (callback: (chunk: ArrayBuffer) => void) => () => void
    onError: (callback: (message: string) => void) => () => void
  }
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: TaurineApi
  }
}
