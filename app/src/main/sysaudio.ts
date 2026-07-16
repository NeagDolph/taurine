import { existsSync } from 'fs'
import { dirname, join } from 'path'
import { AudioTee, type AudioChunk } from 'audiotee'
import { app, ipcMain, type WebContents } from 'electron'

const MINIMUM_MACOS_VERSION = [14, 2] as const

class SystemAudioTap {
  private tap: AudioTee | null = null
  private owner: WebContents | null = null
  private stopping = false

  available(): boolean {
    return isSupportedMacOS() && existsSync(audioTeeHelperPath())
  }

  async start(owner: WebContents): Promise<void> {
    if (!this.available()) {
      throw new Error('System audio capture requires macOS 14.2 or later')
    }
    if (this.tap?.isActive()) {
      return
    }

    await this.stop()

    const tap = new AudioTee({ sampleRate: 48000 })
    tap.on('data', (chunk: AudioChunk) => {
      if (this.tap === tap && this.owner && !this.owner.isDestroyed()) {
        this.owner.send('sysaudio:chunk', chunk.data)
      }
    })
    tap.on('error', (error: Error) => {
      this.handleUnexpectedStop(tap, error.message)
    })
    tap.on('stop', () => {
      this.handleUnexpectedStop(tap, 'System audio capture stopped unexpectedly')
    })

    this.tap = tap
    this.setOwner(owner)

    try {
      await tap.start()
    } catch (error) {
      if (this.tap === tap) {
        this.sendError(errorMessage(error))
        await this.stop()
      }
      throw error
    }
  }

  async stop(): Promise<void> {
    const tap = this.tap
    if (!tap) {
      this.clearOwner()
      return
    }

    this.stopping = true
    try {
      await tap.stop()
    } finally {
      tap.removeAllListeners()
      if (this.tap === tap) {
        this.tap = null
      }
      this.clearOwner()
      this.stopping = false
    }
  }

  private readonly ownerGone = (): void => {
    void this.stop()
  }

  private setOwner(owner: WebContents): void {
    this.clearOwner()
    this.owner = owner
    owner.once('destroyed', this.ownerGone)
    owner.once('did-start-navigation', this.ownerGone)
  }

  private clearOwner(): void {
    if (this.owner && !this.owner.isDestroyed()) {
      this.owner.removeListener('destroyed', this.ownerGone)
      this.owner.removeListener('did-start-navigation', this.ownerGone)
    }
    this.owner = null
  }

  private handleUnexpectedStop(tap: AudioTee, message: string): void {
    if (this.tap !== tap || this.stopping) {
      return
    }
    this.sendError(message)
    void this.stop()
  }

  private sendError(message: string): void {
    if (this.owner && !this.owner.isDestroyed()) {
      this.owner.send('sysaudio:error', message)
    }
  }
}

export function registerSystemAudioIpc(): void {
  const systemAudio = new SystemAudioTap()

  ipcMain.handle('sysaudio:available', () => systemAudio.available())
  ipcMain.handle('sysaudio:start', (event) => systemAudio.start(event.sender))
  ipcMain.handle('sysaudio:stop', () => systemAudio.stop())

  app.on('will-quit', () => {
    void systemAudio.stop()
  })
}

function isSupportedMacOS(): boolean {
  if (process.platform !== 'darwin') {
    return false
  }

  const [major = 0, minor = 0] = process.getSystemVersion().split('.').map(Number)
  return (
    major > MINIMUM_MACOS_VERSION[0] ||
    (major === MINIMUM_MACOS_VERSION[0] && minor >= MINIMUM_MACOS_VERSION[1])
  )
}

function audioTeeHelperPath(): string {
  return join(dirname(require.resolve('audiotee')), '..', 'bin', 'audiotee')
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
