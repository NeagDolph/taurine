import { prefixChannel } from './audio'

const DECIMATION_FACTOR = 3
const FRAME_SAMPLE_COUNT = 4000
const SYSTEM_CHANNEL_ID = 1

export class SystemAudioCapture {
  private unsubscribeChunk: (() => void) | null = null
  private unsubscribeError: (() => void) | null = null
  private readonly remainder: number[] = []
  private readonly frame = new Int16Array(FRAME_SAMPLE_COUNT)
  private frameLength = 0

  constructor(
    private readonly onFrame: (frame: ArrayBuffer) => void,
    private readonly onError: (message: string) => void
  ) {}

  async start(): Promise<void> {
    if (this.unsubscribeChunk || this.unsubscribeError) {
      return
    }

    this.unsubscribeChunk = window.api.sysAudio.onChunk((chunk) => this.consume(chunk))
    this.unsubscribeError = window.api.sysAudio.onError(this.onError)

    try {
      await window.api.sysAudio.start()
    } catch (error) {
      this.unsubscribe()
      this.resetBuffers()
      throw error
    }
  }

  stop(): void {
    this.unsubscribe()
    this.resetBuffers()
    void window.api.sysAudio.stop().catch((error: unknown) => {
      this.onError(error instanceof Error ? error.message : String(error))
    })
  }

  private consume(chunk: ArrayBuffer): void {
    const input = new DataView(chunk)
    const sampleCount = Math.floor(input.byteLength / Int16Array.BYTES_PER_ELEMENT)
    let inputIndex = 0

    while (this.remainder.length < DECIMATION_FACTOR && inputIndex < sampleCount) {
      this.remainder.push(input.getInt16(inputIndex * Int16Array.BYTES_PER_ELEMENT, true))
      inputIndex += 1
    }
    if (this.remainder.length === DECIMATION_FACTOR) {
      this.pushSample(average(this.remainder[0], this.remainder[1], this.remainder[2]))
      this.remainder.length = 0
    }

    while (inputIndex + DECIMATION_FACTOR <= sampleCount) {
      this.pushSample(
        average(
          input.getInt16(inputIndex * Int16Array.BYTES_PER_ELEMENT, true),
          input.getInt16((inputIndex + 1) * Int16Array.BYTES_PER_ELEMENT, true),
          input.getInt16((inputIndex + 2) * Int16Array.BYTES_PER_ELEMENT, true)
        )
      )
      inputIndex += DECIMATION_FACTOR
    }

    while (inputIndex < sampleCount) {
      this.remainder.push(input.getInt16(inputIndex * Int16Array.BYTES_PER_ELEMENT, true))
      inputIndex += 1
    }
  }

  private pushSample(sample: number): void {
    this.frame[this.frameLength] = sample
    this.frameLength += 1
    if (this.frameLength !== FRAME_SAMPLE_COUNT) {
      return
    }

    const pcm = new ArrayBuffer(FRAME_SAMPLE_COUNT * Int16Array.BYTES_PER_ELEMENT)
    const view = new DataView(pcm)
    for (let index = 0; index < FRAME_SAMPLE_COUNT; index += 1) {
      view.setInt16(index * Int16Array.BYTES_PER_ELEMENT, this.frame[index], true)
    }
    this.onFrame(prefixChannel(pcm, SYSTEM_CHANNEL_ID))
    this.frameLength = 0
  }

  private unsubscribe(): void {
    this.unsubscribeChunk?.()
    this.unsubscribeError?.()
    this.unsubscribeChunk = null
    this.unsubscribeError = null
  }

  private resetBuffers(): void {
    this.remainder.length = 0
    this.frameLength = 0
  }
}

function average(first: number, second: number, third: number): number {
  return Math.round((first + second + third) / DECIMATION_FACTOR)
}
