#!/usr/bin/env python3
"""Production-path transcription test: replicate exactly what the wink server
does per call channel — 16k float32 -> SileroVadGate (the server's own module)
-> Taurine sherpa endpoint through Caddy — 1x-paced, both channels concurrently
like a real call. Prints per-channel segmented transcript + VAD stats."""
import asyncio, json, os, sys, time
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vadgate"))
from silero_vad_gate import SileroVadGate

WAV = sys.argv[1]
URL = sys.argv[2]
CHUNK_S = 0.56
CH_NAMES = {0: "mic(agent)", 1: "sys(customer)"}

x, sr = sf.read(WAV, dtype="float32")
if x.ndim == 1:
    x = x[:, None]
g = np.gcd(16000, sr)

NOGATE = os.environ.get("NOGATE") == "1"  # raw client: box-side vadgate does VAD

async def run_channel(ch):
    m = resample_poly(x[:, ch], 16000 // g, sr // g).astype("float32")
    gate = None if NOGATE else SileroVadGate()  # server defaults
    CHUNK = int(16000 * CHUNK_S)
    segs, seg_t = {}, {}
    in_e = out_e = 0.0
    kept = total = 0
    async with websockets.connect(URL, max_size=None, open_timeout=30) as ws:
        t0 = time.monotonic()

        async def recv():
            async for msg in ws:
                try:
                    d = json.loads(msg)
                except Exception:
                    continue
                if isinstance(d, dict) and d.get("text"):
                    k = int(d.get("segment", 0))
                    segs[k] = d["text"]
                    seg_t.setdefault(k, time.monotonic() - t0)

        rt = asyncio.create_task(recv())
        nxt = t0
        for i in range(0, len(m), CHUNK):
            raw = m[i:i + CHUNK]
            in_e += float(np.square(raw).sum())
            gated = raw if gate is None else gate.process(raw)
            if gated.size:
                out_e += float(np.square(gated).sum())
                kept += int(np.count_nonzero(gated))
                total += gated.size
                await ws.send(gated.tobytes())
            nxt += CHUNK_S
            d = nxt - time.monotonic()
            if d > 0:
                await asyncio.sleep(d)
        tail = np.zeros(0, dtype=np.float32) if gate is None else gate.flush()
        if tail.size:
            out_e += float(np.square(tail).sum())
            kept += int(np.count_nonzero(tail))
            total += tail.size
            await ws.send(tail.tobytes())
        await ws.send("Done")
        try:
            await asyncio.wait_for(rt, timeout=12)
        except asyncio.TimeoutError:
            rt.cancel()
    return ch, segs, seg_t, in_e, out_e, kept, total

async def main():
    print(f"file={WAV.split('/')[-1]} dur={len(x)/sr/60:.1f}min | production path: "
          f"SileroVadGate(server module, defaults) -> taurine via Caddy, 1x-paced, "
          f"both channels concurrent", flush=True)
    results = await asyncio.gather(*[run_channel(c) for c in range(x.shape[1])])
    for ch, segs, seg_t, in_e, out_e, kept, total in results:
        print(f"\n===== channel {ch} [{CH_NAMES.get(ch, ch)}] =====")
        print(f"VAD: speech-energy kept={out_e/max(in_e,1e-9)*100:.1f}% | "
              f"timeline non-zeroed={kept/max(total,1)*100:.1f}% "
              f"(zeroed {100-kept/max(total,1)*100:.1f}%)")
        print(f"segments={len(segs)}")
        for k in sorted(segs):
            print(f"  [{seg_t.get(k,0)/60:5.2f}m seg{k:3d}] {segs[k]}")
    print("\nPROD_PATH_DONE", flush=True)

asyncio.run(main())
