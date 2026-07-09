"""Repo-local model storage.

Models live under core/models/<repo-name>/ (gitignored). When present they
load directly from disk with zero HuggingFace Hub traffic — no etag checks,
no "unauthenticated requests" warnings, works offline. Populate with:

    uv run python -m taurine_core.download_models
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from taurine_core import config

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_PASSAGE_TOKENIZER: Any | None = None


def local_model_path(repo_id: str) -> str | None:
    """Filesystem path for a bundled model, or None when not downloaded.

    Accepts both layouts: models/<name> (canonical, what download_models
    writes) and models/<org>/<name>.
    """
    for candidate in (MODELS_DIR / repo_id.split("/")[-1], MODELS_DIR / repo_id):
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)
    return None


def model_ref(repo_id: str) -> str:
    """Local path when bundled, else the hub repo id."""
    return local_model_path(repo_id) or repo_id


def passage_tokenizer() -> Any:
    """Return the process-cached Qwen tokenizer used for passage boundaries."""
    global _PASSAGE_TOKENIZER
    if _PASSAGE_TOKENIZER is None:
        ref = model_ref(config.EMBED_MODEL)
        _PASSAGE_TOKENIZER = AutoTokenizer.from_pretrained(
            ref,
            local_files_only=ref != config.EMBED_MODEL,
            use_fast=True,
        )
    return _PASSAGE_TOKENIZER
