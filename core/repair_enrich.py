"""One-off repair: re-enrich notes that got fallback enrichment, then rebuild edges."""
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv("/Users/neil/taurine/core/.env")

from taurine_core.indexer.enrich import enrich_note_with_status
from taurine_core.indexer.store import Index
from taurine_core.indexer.sync import _embedding_units
from taurine_core.llm import OpenRouterClient


async def main() -> int:
    index = Index("taurine.db")
    index.load()
    llm = OpenRouterClient(os.environ["OPEN_ROUTER_API_KEY"])

    targets = []
    for note in index.notes():
        enr = index.cached_enrichment(note.note_id, note.content_hash)
        if enr is None or (not enr.utterances and not enr.topics):
            targets.append(note)
    print(f"repairing {len(targets)} notes", flush=True)

    sem = asyncio.Semaphore(4)
    failures = 0

    async def one(note):
        nonlocal failures
        async with sem:
            enrichment, failed = await enrich_note_with_status(note, llm)
        if failed:
            failures += 1
            return
        index.cache_enrichment(note.note_id, note.content_hash, enrichment)
        index.upsert_note(note, enrichment, _embedding_units(note, enrichment))
        index.rebuild_for(note.note_id)

    for i in range(0, len(targets), 20):
        batch = targets[i : i + 20]
        await asyncio.gather(*(one(n) for n in batch))
        print(f"{min(i + 20, len(targets))}/{len(targets)} done, failures={failures}", flush=True)

    index.load()
    print(f"REPAIR-DONE failures={failures}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
