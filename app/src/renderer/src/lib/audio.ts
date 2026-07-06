const WORKLET_URL = '/pcm-worklet.js'
const MIC_CHANNEL_ID = 0

export type AudioFrameHandler = (frame: ArrayBuffer) => void

export class PcmAudioCapture {
  private stream: MediaStream | null = null
  private context: AudioContext | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private worklet: AudioWorkletNode | null = null

  constructor(private readonly onFrame: AudioFrameHandler) {}

  async start(): Promise<void> {
    if (this.stream || this.context) {
      return
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 48000,
        channelCount: 1
      }
    })

    const context = new AudioContext({ sampleRate: 48000 })
    await context.audioWorklet.addModule(WORKLET_URL)

    const source = context.createMediaStreamSource(stream)
    const worklet = new AudioWorkletNode(context, 'pcm-worklet', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1
    })

    worklet.port.onmessage = (event: MessageEvent<ArrayBuffer>): void => {
      if (!(event.data instanceof ArrayBuffer)) {
        return
      }

      this.onFrame(prefixChannel(event.data, MIC_CHANNEL_ID))
    }

    source.connect(worklet)

    this.stream = stream
    this.context = context
    this.source = source
    this.worklet = worklet
  }

  stop(): void {
    this.worklet?.port.close()
    this.worklet?.disconnect()
    this.source?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())

    void this.context?.close()

    this.worklet = null
    this.source = null
    this.context = null
    this.stream = null
  }
}

function prefixChannel(frame: ArrayBuffer, channelId: number): ArrayBuffer {
  const prefixed = new Uint8Array(frame.byteLength + 1)
  prefixed[0] = channelId
  prefixed.set(new Uint8Array(frame), 1)
  return prefixed.buffer
}
