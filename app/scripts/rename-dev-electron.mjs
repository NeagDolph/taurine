/* eslint-disable @typescript-eslint/explicit-function-return-type -- plain-JS build script */
/**
 * Dev-mode dock label: `electron-vite dev` runs the stock Electron.app, whose
 * Info.plist names the process "Electron" in the dock and app switcher.
 * Rewrite its bundle name to Taurine and re-sign ad hoc (editing the plist
 * breaks the signature seal, and unsigned binaries won't launch on Apple
 * Silicon). Packaged builds are unaffected — electron-builder stamps
 * productName there. Runs from postinstall; safe to re-run.
 */
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const APP_NAME = 'Taurine'

if (process.platform !== 'darwin') process.exit(0)

const appDir = dirname(dirname(fileURLToPath(import.meta.url)))
const bundle = join(appDir, 'node_modules', 'electron', 'dist', 'Electron.app')
const plist = join(bundle, 'Contents', 'Info.plist')
if (!existsSync(plist)) process.exit(0)

const plistBuddy = '/usr/libexec/PlistBuddy'
const read = (key) => {
  try {
    return execFileSync(plistBuddy, ['-c', `Print :${key}`, plist])
      .toString()
      .trim()
  } catch {
    return null
  }
}
const write = (key, value) => {
  const command = read(key) === null ? `Add :${key} string ${value}` : `Set :${key} ${value}`
  execFileSync(plistBuddy, ['-c', command, plist])
}

if (read('CFBundleName') === APP_NAME && read('CFBundleDisplayName') === APP_NAME) {
  process.exit(0)
}

write('CFBundleName', APP_NAME)
write('CFBundleDisplayName', APP_NAME)
execFileSync('codesign', ['--force', '--deep', '--sign', '-', bundle], { stdio: 'ignore' })
console.log(`dev Electron bundle renamed to ${APP_NAME} and re-signed`)
