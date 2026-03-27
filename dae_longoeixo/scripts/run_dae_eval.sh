#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${ROOT_DIR}/.venv/bin/python"
CFG="${ROOT_DIR}/dae_longoeixo/dae_train_config.json"
RUN_NAME="${1:-}"
SPLIT="${2:-val}"

CMD=("$PY" dae_longoeixo/eval_dae.py --config "$CFG" --split "$SPLIT")

if [ -n "$RUN_NAME" ]; then
  CMD+=(--run-name "$RUN_NAME")
fi

"${CMD[@]}"
