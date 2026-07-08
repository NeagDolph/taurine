"""Repo-local model storage.

Models live under core/models/<repo-name>/ (gitignored). When present they
load directly from disk with zero HuggingFace Hub traffic — no etag checks,
no "unauthenticated requests" warnings, works offline. Populate with:

    uv run python -m taurine_core.download_models
"""
from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


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
