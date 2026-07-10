"""Download all configured offline retrieval models into core/models/."""
from __future__ import annotations

from huggingface_hub import snapshot_download

from taurine_core import config
from taurine_core.models_local import MODELS_DIR


def main() -> int:
    for repo_id in (
        config.EMBED_MODEL,
        config.GLINER2_MODEL,
        config.RERANK_MODEL,
    ):
        target = MODELS_DIR / repo_id.split("/")[-1]
        print(f"downloading {repo_id} -> {target}", flush=True)
        snapshot_download(
            repo_id,
            local_dir=str(target),
            ignore_patterns=["onnx/*", "openvino/*", "*.onnx", "pytorch_model.bin", "tf_model.h5"],
        )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
