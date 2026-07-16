import { spawn, type ChildProcess, type ChildProcessByStdio } from 'child_process'
import { existsSync } from 'fs'
import { createInterface } from 'readline'
import { join } from 'path'
import type { Readable } from 'stream'
import { app, BrowserWindow } from 'electron'

const ACTIVATION_DEBOUNCE_MS = 1500
const MAX_RESPAWNS = 3
const RESPAWN_BASE_MS = 500

export function parseMicWatchLine(line: string): boolean | null {
  try {
    const value: unknown = JSON.parse(line)
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    const active = (value as { active?: unknown }).active
    return typeof active === 'boolean' ? active : null
  } catch {
    return null
  }
}

export class MicActivityDebouncer {
  private pendingActivation: ReturnType<typeof setTimeout> | null = null
  private inputState: boolean | null = null
  private emittedState: boolean | null = null

  constructor(
    private readonly emit: (active: boolean) => void,
    private readonly delayMs = ACTIVATION_DEBOUNCE_MS
  ) {}

  accept(active: boolean): void {
    if (active === this.inputState) return
    this.inputState = active

    if (!active) {
      this.cancelPendingActivation()
      this.emitEdge(false)
      return
    }

    this.pendingActivation = setTimeout(() => {
      this.pendingActivation = null
      if (this.inputState) this.emitEdge(true)
    }, this.delayMs)
  }

  dispose(): void {
    this.cancelPendingActivation()
  }

  private emitEdge(active: boolean): void {
    if (active === this.emittedState) return
    this.emittedState = active
    this.emit(active)
  }

  private cancelPendingActivation(): void {
    if (!this.pendingActivation) return
    clearTimeout(this.pendingActivation)
    this.pendingActivation = null
  }
}

class MicWatchProcess {
  private child: ChildProcess | null = null
  private restartTimer: ReturnType<typeof setTimeout> | null = null
  private respawns = 0
  private stopping = false
  private lastActive: boolean | null = null
  private readonly debouncer = new MicActivityDebouncer((active) => this.broadcast(active))
  private readonly windowLoaded = (window: BrowserWindow): void => {
    window.webContents.once('did-finish-load', () => {
      if (this.lastActive !== null && !window.webContents.isDestroyed()) {
        window.webContents.send('micwatch:active', this.lastActive)
      }
    })
  }

  start(): void {
    if (process.platform !== 'darwin') return
    const binary = micWatchBinaryPath()
    if (!existsSync(binary)) return

    app.on('browser-window-created', (_event, window) => this.windowLoaded(window))
    BrowserWindow.getAllWindows().forEach((window) => this.windowLoaded(window))
    this.spawn(binary)
  }

  stop(): void {
    this.stopping = true
    this.debouncer.dispose()
    if (this.restartTimer) {
      clearTimeout(this.restartTimer)
      this.restartTimer = null
    }
    this.child?.kill('SIGTERM')
    this.child = null
  }

  private spawn(binary: string): void {
    if (this.stopping) return

    let child: ChildProcessByStdio<null, Readable, null>
    try {
      child = spawn(binary, [], { stdio: ['ignore', 'pipe', 'ignore'] })
    } catch {
      this.scheduleRespawn(binary)
      return
    }
    this.child = child

    const lines = createInterface({ input: child.stdout })
    lines.on('line', (line) => {
      const active = parseMicWatchLine(line)
      if (active !== null) this.debouncer.accept(active)
    })

    let settled = false
    const handleExit = (): void => {
      if (settled) return
      settled = true
      lines.close()
      if (this.child === child) this.child = null
      if (!this.stopping) this.scheduleRespawn(binary)
    }
    child.once('error', handleExit)
    child.once('close', handleExit)
  }

  private scheduleRespawn(binary: string): void {
    if (this.stopping || this.respawns >= MAX_RESPAWNS || this.restartTimer) return
    const delay = RESPAWN_BASE_MS * 2 ** this.respawns
    this.respawns += 1
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null
      this.spawn(binary)
    }, delay)
  }

  private broadcast(active: boolean): void {
    this.lastActive = active
    BrowserWindow.getAllWindows().forEach((window) => {
      if (!window.webContents.isDestroyed()) {
        window.webContents.send('micwatch:active', active)
      }
    })
  }
}

export function startMicWatch(): () => void {
  const watcher = new MicWatchProcess()
  watcher.start()
  return () => watcher.stop()
}

function micWatchBinaryPath(): string {
  return app.isPackaged
    ? join(process.resourcesPath, 'app.asar.unpacked', 'resources', 'native', 'micwatch')
    : join(app.getAppPath(), 'resources', 'native', 'micwatch')
}
