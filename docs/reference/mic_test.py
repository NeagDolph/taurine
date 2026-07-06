#!/usr/bin/env python3
"""Live mic -> Taurine (via vadgate) test client.

Speaks the CURRENT production wire protocol (the box vadgate proxy), which is
NOT the raw sherpa protocol the older probes used:

  * send  : binary frames of 16 kHz mono float32 PCM (little-endian)
  * end   : one text frame ("Done") -> flushes the decoder tail
  * recv  : JSON per decode tick
              {"text": <interim>, "segment": k}                 interim
              {"text": <final>,   "segment": k, "is_final": 1}  committed

We send RAW mic audio (no client-side VAD) on purpose: the box vadgate owns the
Silero VAD + segmentation, so a second gate here would just double-clip. This is
exactly `prod_path_test.py`'s NOGATE mode, but live from your microphone — the
cleanest way to hear what the deployed encoder (currently 560 ms lookahead) +
vadgate actually do to your speech.

Setup (already done if the sibling .venv-mic exists):
    python3 -m venv .venv-mic
    .venv-mic/bin/pip install sounddevice websockets numpy

Run (from this scripts/ dir):
    .venv-mic/bin/python mic_test.py                 # default endpoint + token from ../Caddyfile
    .venv-mic/bin/python mic_test.py --list-devices  # pick a mic
    .venv-mic/bin/python mic_test.py --device 2
    TAURINE_TOKEN=... .venv-mic/bin/python mic_test.py --url wss://host/taurine

Ctrl-C to stop (it flushes the final segment before exiting).
"""
import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time

import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

RATE = 16000            # vadgate/Silero/sherpa all run at 16 kHz
BLOCK_MS = 100          # mic block size; small = snappier interims
DEFAULT_URL = "wss://transcription.trywink.io/taurine"

# ANSI (skip if not a tty)
_TTY = sys.stdout.isatty()
DIM = "\033[2m" if _TTY else ""
GREEN = "\033[32m" if _TTY else ""
CYAN = "\033[36m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""
CLEAR = "\r\033[K" if _TTY else "\r"


def read_token_from_caddyfile():
    """The token lives in the repo Caddyfile as `query token=<hex>` — read it
    from there instead of hardcoding a secret into this script."""
    caddy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Caddyfile")
    try:
        with open(caddy) as f:
            m = re.search(r"token=([0-9a-fA-F]{16,})", f.read())
            return m.group(1) if m else None
    except OSError:
        return None


def resolve_token(arg_token):
    return arg_token or os.environ.get("TAURINE_TOKEN") or read_token_from_caddyfile()


async def receiver(ws, state):
    """Read vadgate JSON; live-update the interim line, print committed finals."""
    async for msg in ws:
        if isinstance(msg, (bytes, bytearray)):
            continue  # vadgate only sends text/JSON downstream
        try:
            d = json.loads(msg)
        except (ValueError, TypeError):
            continue
        text = (d.get("text") or "").strip()
        if not text:
            continue
        seg = int(d.get("segment", 0))
        if d.get("is_final"):
            dt = time.monotonic() - state["t0"]
            sys.stdout.write(f"{CLEAR}{GREEN}[{dt:6.1f}s seg{seg:>3}] {text}{RESET}\n")
            sys.stdout.flush()
            state["finals"] += 1
            state["interim"] = ""
        else:
            state["interim"] = text
            sys.stdout.write(f"{CLEAR}{DIM}… seg{seg:>3} {text}{RESET}")
            sys.stdout.flush()


async def sender(ws, audio_q, stop):
    """Drain mic blocks from the callback queue and ship them as binary frames."""
    while not stop.is_set():
        try:
            chunk = await asyncio.wait_for(audio_q.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        try:
            await ws.send(chunk)
        except ConnectionClosed:
            return


async def run(url, device):
    loop = asyncio.get_running_loop()
    audio_q = asyncio.Queue()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass  # e.g. non-Unix; falls back to KeyboardInterrupt

    def on_audio(indata, frames, time_info, status):
        if status:
            print(f"{DIM}(audio status: {status}){RESET}", file=sys.stderr)
        loop.call_soon_threadsafe(audio_q.put_nowait, indata[:, 0].copy().tobytes())

    state = {"t0": time.monotonic(), "finals": 0, "interim": ""}
    print(f"{CYAN}connecting{RESET} {url.split('?')[0]}")
    try:
        ws = await websockets.connect(url, max_size=None, open_timeout=15)
    except InvalidStatus as e:
        code = getattr(getattr(e, "response", None), "status_code", "?")
        msg = {503: "box at capacity (503) — all Taurine slots busy; retry shortly.",
               401: "unauthorized (bad/missing token). Pass --token or set TAURINE_TOKEN.",
               403: "forbidden (bad token)."}.get(code, f"handshake rejected: HTTP {code}")
        print(msg)
        return
    except OSError as e:
        print(f"could not reach endpoint: {e}")
        return

    recv_task = asyncio.create_task(receiver(ws, state))
    send_task = asyncio.create_task(sender(ws, audio_q, stop))
    stream = sd.InputStream(
        samplerate=RATE, channels=1, dtype="float32",
        blocksize=int(RATE * BLOCK_MS / 1000), device=device, callback=on_audio,
    )
    try:
        stream.start()
        print(f"{GREEN}● listening{RESET} — speak into the mic. Ctrl-C to stop.\n")
        await stop.wait()
    finally:
        stream.stop()
        stream.close()
        # tell vadgate we're done so it commits the final tail, then let the
        # decoder drain before teardown
        try:
            await ws.send("Done")
        except ConnectionClosed:
            pass
        try:
            await asyncio.wait_for(recv_task, timeout=8)
        except asyncio.TimeoutError:
            recv_task.cancel()
        send_task.cancel()
        await ws.close()

    if state["interim"]:
        sys.stdout.write(CLEAR)
    print(f"\n{CYAN}done{RESET} — {state['finals']} segment(s) committed.")


def main():
    ap = argparse.ArgumentParser(description="Live mic test for the Taurine ASR endpoint.")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"base WS url (default {DEFAULT_URL})")
    ap.add_argument("--token", default=None,
                    help="auth token (default: TAURINE_TOKEN env or ../Caddyfile)")
    ap.add_argument("--device", default=None, help="input device index or name (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    token = resolve_token(args.token)
    if not token:
        sys.exit("no token found. Pass --token, set TAURINE_TOKEN, or run from the repo "
                 "(reads ../Caddyfile).")
    url = f"{args.url}?token={token}"

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    try:
        asyncio.run(run(url, device))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
