import { execFileSync } from 'node:child_process'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const swiftc = '/Library/Developer/CommandLineTools/usr/bin/swiftc'
const lipo = '/usr/bin/lipo'
const sdk = '/Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk'
const scriptDir = dirname(fileURLToPath(import.meta.url))
const appDir = dirname(scriptDir)
const source = join(appDir, 'resources', 'native', 'micwatch.swift')
const output = join(appDir, 'resources', 'native', 'micwatch')
const ifNeeded = process.argv.includes('--if-needed')

if (ifNeeded && existsSync(output)) process.exit(0)
if (!existsSync(swiftc)) {
  if (ifNeeded) process.exit(0)
  throw new Error(`swiftc not found at ${swiftc}`)
}

mkdirSync(dirname(output), { recursive: true })
const buildDir = mkdtempSync(join(tmpdir(), 'taurine-micwatch-'))

try {
  const arm64 = join(buildDir, 'micwatch-arm64')
  const x64 = join(buildDir, 'micwatch-x86_64')
  buildSlice('arm64-apple-macosx14.0', arm64)
  buildSlice('x86_64-apple-macosx14.0', x64)
  execFileSync(lipo, ['-create', arm64, x64, '-output', output], { stdio: 'inherit' })
  chmodSync(output, 0o755)
} finally {
  rmSync(buildDir, { recursive: true, force: true })
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
function buildSlice(target, sliceOutput) {
  execFileSync(
    swiftc,
    [
      '-O',
      '-sdk',
      sdk,
      '-module-cache-path',
      join(buildDir, 'module-cache'),
      '-target',
      target,
      source,
      '-framework',
      'CoreAudio',
      '-o',
      sliceOutput
    ],
    { stdio: 'inherit' }
  )
}
