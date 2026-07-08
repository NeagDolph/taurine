from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable, Iterable
from typing import TextIO

from taurine_core.schemas import Chunk
from taurine_core.server import Runtime, build_runtime


async def run_devfeed(
    lines: Iterable[str],
    *,
    runtime: Runtime | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
    now: Callable[[], float] = time.time,
) -> None:
    active_runtime = runtime or build_runtime()
    logger = active_runtime.make_session_logger()
    if logger.path is not None:
        print(f"session log: {logger.path}", file=error_output, flush=True)
    logger.log("session_start", channels=[{"id": 0, "label": "devfeed"}])

    async def emit(event) -> None:
        print(event.model_dump_json(), file=output, flush=True)

    session = active_runtime.make_live_session(emit, logger)
    try:
        for raw_line in lines:
            text = raw_line.strip()
            if not text:
                continue
            await session.on_chunk(Chunk(channel=0, text=text, ts=now(), final=True))
        # an accepted summarizer pass may still be in flight; wait for it so
        # its events and session-log records land before the logger closes
        summary_task = getattr(session, "_summary_task", None)
        if summary_task is not None:
            await summary_task
    finally:
        logger.close()


def main() -> int:
    asyncio.run(run_devfeed(sys.stdin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
