import { ElectronAPI } from '@electron-toolkit/preload'

export interface TaurineApi {
  openPreview: (noteId: string) => Promise<void>
  chooseDirectory: () => Promise<string | null>
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: TaurineApi
  }
}
