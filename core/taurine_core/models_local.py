"""Repo-local model storage.

Models live under core/models/<repo-name>/ (gitignored). When present they
load directly from disk with zero HuggingFace Hub traffic — no etag checks,
no "unauthenticated requests" warnings, works offline. Populate with:

    uv run python -m taurine_core.download_models
"""
from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any, Protocol, Sequence

import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from taurine_core import config

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_PASSAGE_TOKENIZER: Any | None = None
_PAIR_RERANKERS: dict[str, "PairScorer"] = {}
_NLI_SCORERS: dict[str, "EntailmentScorer"] = {}


def _inference_device() -> str:
    """Prefer Apple-GPU inference; local models on CPU peg every core."""
    import torch

    _configure_torch_threads(torch)
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_TORCH_THREADS_SET = False


def _configure_torch_threads(torch: Any) -> None:
    """Leave headroom for the event loop and UI when inference runs on CPU."""
    global _TORCH_THREADS_SET
    if _TORCH_THREADS_SET:
        return
    _TORCH_THREADS_SET = True
    try:
        import os as os_module

        torch.set_num_threads(max(2, (os_module.cpu_count() or 8) // 2))
    except Exception:
        pass


def _to_inference_device(model: Any) -> Any:
    device = _inference_device()
    if device == "cpu":
        return model
    try:
        return model.to(device)
    except Exception:
        return model


class LocalModelUnavailable(RuntimeError):
    """A required bundled model is absent or cannot be loaded offline."""


class ModelDeadlineExceeded(TimeoutError):
    """A local inference request exceeded its caller-owned deadline."""


class PairScorer(Protocol):
    """Small fake-friendly seam shared by triple and passage reranking."""

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> tuple[float, ...]: ...


class EntailmentScorer(Protocol):
    """Offline NLI seam used by sentence-level evidence validation."""

    def score(self, premise: str, hypothesis: str) -> float: ...


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


class _QwenYesNoPairScorer:
    """Offline-only Qwen3 pair scorer using its calibrated yes/no logits."""

    _BATCH_SIZE = 8

    def __init__(self, model_path: str) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                use_fast=True,
                padding_side="left",
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype="auto",
            )
            self._model.eval()
            self._model = _to_inference_device(self._model)
            self._yes_id = self._single_token_id("yes")
            self._no_id = self._single_token_id("no")
        except Exception as exc:
            raise LocalModelUnavailable(
                f"could not load bundled reranker at {model_path}"
            ) from exc

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str,
        deadline_at: float | None = None,
    ) -> tuple[float, ...]:
        rows = tuple(documents)
        if not rows:
            return ()
        results: list[float] = []
        for start in range(0, len(rows), self._BATCH_SIZE):
            _check_deadline(deadline_at)
            batch = rows[start : start + self._BATCH_SIZE]
            results.extend(self._score_batch(instruction, query, batch))
            _check_deadline(deadline_at)
        return tuple(results)

    def _score_batch(
        self,
        instruction: str,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[float, ...]:
        prompts = [self._prompt(instruction, query, document) for document in documents]
        try:
            import torch

            encoded = self._tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits[:, -1, :]
                pair_logits = logits[:, [self._no_id, self._yes_id]].float()
                scores = torch.softmax(pair_logits, dim=-1)[:, 1]
            return tuple(
                float(value)
                for value in np.asarray(scores.detach().cpu(), dtype=np.float64)
            )
        except (LocalModelUnavailable, ModelDeadlineExceeded):
            raise
        except Exception as exc:
            raise LocalModelUnavailable("bundled reranker inference failed") from exc

    def _single_token_id(self, answer: str) -> int:
        token_ids = self._tokenizer.encode(answer, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"reranker answer {answer!r} is not one token")
        return int(token_ids[0])

    @classmethod
    def _prompt(cls, instruction: str, query: str, document: str) -> str:
        return (
            "<|im_start|>system\n"
            f"{instruction} Answer only yes or no.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {instruction}\n<Query>: {query}\n"
            f"<Document>: {document}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def local_pair_reranker(repo_id: str = config.RERANK_MODEL) -> PairScorer:
    """Return a cached bundled reranker without any Hugging Face network fallback."""

    cached = _PAIR_RERANKERS.get(repo_id)
    if cached is not None:
        return cached
    path = local_model_path(repo_id)
    if path is None:
        raise LocalModelUnavailable(
            f"bundled local model {repo_id!r} is unavailable under {MODELS_DIR}"
        )
    scorer = _QwenYesNoPairScorer(path)
    _PAIR_RERANKERS[repo_id] = scorer
    return scorer


class _DebertaEntailmentScorer:
    """Offline-only DeBERTa cross-encoder entailment probability scorer."""

    def __init__(self, model_path: str) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                use_fast=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._model.eval()
            self._model = _to_inference_device(self._model)
            self._entailment_id = _entailment_label_id(self._model.config)
        except LocalModelUnavailable:
            raise
        except Exception as exc:
            raise LocalModelUnavailable(
                f"could not load bundled NLI model at {model_path}"
            ) from exc

    def score(self, premise: str, hypothesis: str) -> float:
        if not premise.strip() or not hypothesis.strip():
            return 0.0
        try:
            import torch

            encoded = self._tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits[0].float()
                probability = torch.softmax(logits, dim=-1)[self._entailment_id]
            return float(probability.detach().cpu().item())
        except LocalModelUnavailable:
            raise
        except Exception as exc:
            raise LocalModelUnavailable("bundled NLI inference failed") from exc


def local_nli_scorer(
    repo_id: str = config.EVIDENCE_NLI_MODEL,
) -> EntailmentScorer:
    """Return the cached bundled NLI scorer with no network fallback."""

    cached = _NLI_SCORERS.get(repo_id)
    if cached is not None:
        return cached
    path = local_model_path(repo_id)
    if path is None:
        raise LocalModelUnavailable(
            f"bundled local model {repo_id!r} is unavailable under {MODELS_DIR}"
        )
    scorer = _DebertaEntailmentScorer(path)
    _NLI_SCORERS[repo_id] = scorer
    return scorer


def _entailment_label_id(model_config: Any) -> int:
    for label_id, label in getattr(model_config, "id2label", {}).items():
        if "entail" in str(label).casefold():
            return int(label_id)
    for label, label_id in getattr(model_config, "label2id", {}).items():
        if "entail" in str(label).casefold():
            return int(label_id)
    raise LocalModelUnavailable("bundled NLI model has no entailment label")


def _check_deadline(deadline_at: float | None) -> None:
    if deadline_at is not None and monotonic() >= deadline_at:
        raise ModelDeadlineExceeded("local model inference deadline exceeded")
