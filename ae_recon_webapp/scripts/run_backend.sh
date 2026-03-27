#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/ae_recon_webapp/.env" ]]; then
  set -a
  source "$ROOT/ae_recon_webapp/.env"
  set +a
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PY_BIN="python3"
if [[ -x "$ROOT/ae_recon_webapp/.venv/bin/python" ]]; then
  PY_BIN="$ROOT/ae_recon_webapp/.venv/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY_BIN="$ROOT/.venv/bin/python"
fi

"$PY_BIN" -m uvicorn ae_recon_webapp.backend.app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
