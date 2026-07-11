import os

# Must be set before torch is first imported anywhere: lets MPS fall back to
# CPU for individual unsupported ops instead of crashing local inference.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Cap the Metal caching allocator. PyTorch's default high watermark (1.7x the
# recommended working set — ~60 GB on a 36 GB machine) lets the MPS allocator
# retain cached blocks across the many small encode() calls of a full index
# build, ballooning RSS to tens of GB during embedding/extraction. A 0.5/0.4
# high/low pair forces aggressive eviction while leaving ample room for the
# 340M GLiNER and 0.6B Qwen models. Override with real env vars if a machine
# needs a different bound. See indexer/build.py's build_memory_watchdog for the
# fail-safe abort if a future path still climbs past the configured ceiling.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
