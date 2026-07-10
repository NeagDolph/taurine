#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_PID=""
APP_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi

  if [[ -n "$CORE_PID" ]] && kill -0 "$CORE_PID" 2>/dev/null; then
    kill "$CORE_PID" 2>/dev/null || true
    wait "$CORE_PID" 2>/dev/null || true
  fi

  exit "$status"
}

trap cleanup EXIT INT TERM

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to run the Taurine sidecar." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to run the Taurine app." >&2
  exit 1
fi

echo "Starting Taurine sidecar on http://127.0.0.1:8930..."
(
  cd "$ROOT_DIR/core"
  exec uv run uvicorn taurine_core.server:app --host 127.0.0.1 --port 8930
) &
CORE_PID=$!

sleep 1
if ! kill -0 "$CORE_PID" 2>/dev/null; then
  echo "Taurine sidecar exited before the app could start." >&2
  wait "$CORE_PID"
fi

echo "Starting Taurine app..."
(
  cd "$ROOT_DIR/app"
  exec npm run dev
) &
APP_PID=$!

if wait "$APP_PID"; then
  APP_PID=""
  exit 0
else
  status=$?
  APP_PID=""
  exit "$status"
fi
