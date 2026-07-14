import os

# Must be set before torch is first imported anywhere: lets MPS fall back to
# CPU for individual unsupported ops instead of crashing local inference.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Cap the Metal caching allocator. PyTorch's default high watermark (1.7x the
# recommended working set — ~60 GB on a 36 GB machine) lets the MPS allocator
# retain cached blocks across the many small encode() calls of a full index
# build, ballooning RSS to tens of GB during embedding/extraction. The original
# 0.5/0.4 pair was sized when local embedding shared MPS with extraction; with
# embeddings on the OpenRouter API (2026-07-14) extraction is the only heavy
# MPS tenant, and 0.5 starved Relex on ~2048-token year-long transcripts into
# its CPU fallback (10x slower full-corpus passes). 0.75/0.6 keeps eviction
# honest while letting long-context DeBERTa attention peak on-device. Override
# with real env vars if a machine needs a different bound. See indexer/build.py's
# build_memory_watchdog for the fail-safe abort if a path climbs past the ceiling.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
