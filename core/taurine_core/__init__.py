import os

# Must be set before torch is first imported anywhere: lets MPS fall back to
# CPU for individual unsupported ops instead of crashing local inference.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
