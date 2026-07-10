from __future__ import annotations

import os
import stat
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SCRIPT = REPO_ROOT / "run.sh"


def test_root_run_script_launches_sidecar_and_electron_app() -> None:
    assert RUN_SCRIPT.exists()
    assert RUN_SCRIPT.stat().st_mode & stat.S_IXUSR

    script = RUN_SCRIPT.read_text()

    assert 'cd "$ROOT_DIR/core"' in script
    assert "uv run uvicorn taurine_core.server:app --host 127.0.0.1 --port 8930" in script
    assert 'cd "$ROOT_DIR/app"' in script
    assert "npm run dev" in script
    assert "trap cleanup" in script
    assert "wait \"$APP_PID\"" in script
    assert os.linesep in script
