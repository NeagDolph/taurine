const TARGET_SAMPLE_RATE = 16000
const FRAME_SAMPLES = TARGET_SAMPLE_RATE / 4

class PcmWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.inputOffset = 0
    this.downsampleRatio = sampleRate / TARGET_SAMPLE_RATE
    this.frame = new Int16Array(FRAME_SAMPLES)
    this.frameOffset = 0
  }

  process(inputs) {
    const input = inputs[0]?.[0]
    if (!input || input.length === 0) {
      return true
    }

    let cursor = this.inputOffset
    while (cursor < input.length) {
      const sample = this.interpolate(input, cursor)
      this.frame[this.frameOffset] = this.toInt16(sample)
      this.frameOffset += 1

      if (this.frameOffset === FRAME_SAMPLES) {
        const packet = this.frame.slice().buffer
        this.port.postMessage(packet, [packet])
        this.frameOffset = 0
      }

      cursor += this.downsampleRatio
    }

    this.inputOffset = cursor - input.length
    return true
  }

  interpolate(input, cursor) {
    const lowerIndex = Math.floor(cursor)
    const upperIndex = Math.min(input.length - 1, lowerIndex + 1)
    const blend = cursor - lowerIndex
    return input[lowerIndex] * (1 - blend) + input[upperIndex] * blend
  }

  toInt16(sample) {
    const clamped = Math.max(-1, Math.min(1, sample))
    return clamped < 0 ? clamped * 32768 : clamped * 32767
  }
}

registerProcessor('pcm-worklet', PcmWorkletProcessor)
