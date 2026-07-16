import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SystemAudioCapture } from './sysaudio'

describe('SystemAudioCapture', () => {
  let emitChunk: ((chunk: ArrayBuffer) => void) | null
  let emitError: ((message: string) => void) | null
  const unsubscribeChunk = vi.fn()
  const unsubscribeError = vi.fn()
  const startTap = vi.fn<() => Promise<void>>()
  const stopTap = vi.fn<() => Promise<void>>()

  beforeEach(() => {
    emitChunk = null
    emitError = null
    unsubscribeChunk.mockReset()
    unsubscribeError.mockReset()
    startTap.mockReset().mockResolvedValue()
    stopTap.mockReset().mockResolvedValue()

    vi.stubGlobal('window', {
      api: {
        sysAudio: {
          available: vi.fn().mockResolvedValue(true),
          start: startTap,
          stop: stopTap,
          onChunk: (callback: (chunk: ArrayBuffer) => void): (() => void) => {
            emitChunk = callback
            return unsubscribeChunk
          },
          onError: (callback: (message: string) => void): (() => void) => {
            emitError = callback
            return unsubscribeError
          }
        }
      }
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('carries awkward chunk remainders into an exact 3:1 channel-1 frame', async () => {
    const frames: ArrayBuffer[] = []
    const capture = new SystemAudioCapture((frame) => frames.push(frame), vi.fn())
    await capture.start()

    const input = makeSinePcm(12000)
    const chunkLengths = [1000, 1001, 999, 1000, 1001, 999, 1000, 1001, 999, 1000, 1001, 999]
    let offset = 0
    for (const length of chunkLengths) {
      emitChunk?.(pcmChunk(input, offset, length))
      offset += length
      if (offset < input.length) expect(frames).toHaveLength(0)
    }

    expect(offset).toBe(input.length)
    expect(frames).toHaveLength(1)
    expect(frames[0].byteLength).toBe(1 + 4000 * Int16Array.BYTES_PER_ELEMENT)

    const bytes = new Uint8Array(frames[0])
    expect(bytes[0]).toBe(1)
    const output = new Int16Array(frames[0].slice(1))
    expect(output).toHaveLength(input.length / 3)
    for (let index = 0; index < output.length; index += 1) {
      const sourceIndex = index * 3
      expect(output[index]).toBe(
        Math.round((input[sourceIndex] + input[sourceIndex + 1] + input[sourceIndex + 2]) / 3)
      )
    }
  })

  it('forwards errors and cleans up subscriptions and the tap', async () => {
    const onError = vi.fn()
    const capture = new SystemAudioCapture(vi.fn(), onError)
    await capture.start()

    emitError?.('permission lost')
    expect(onError).toHaveBeenCalledWith('permission lost')

    capture.stop()
    expect(unsubscribeChunk).toHaveBeenCalledOnce()
    expect(unsubscribeError).toHaveBeenCalledOnce()
    expect(stopTap).toHaveBeenCalledOnce()
  })
})

function makeSinePcm(sampleCount: number): Int16Array {
  return Int16Array.from({ length: sampleCount }, (_, index) =>
    Math.round(Math.sin((index * 2 * Math.PI) / 97) * 24000)
  )
}

function pcmChunk(samples: Int16Array, offset: number, length: number): ArrayBuffer {
  const chunk = new Int16Array(length)
  chunk.set(samples.subarray(offset, offset + length))
  return chunk.buffer
}
