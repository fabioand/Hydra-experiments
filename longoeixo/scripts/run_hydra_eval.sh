#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${ROOT_DIR}/.venv/bin/python"
CFG="${ROOT_DIR}/hydra_train_config.json"
RUN_NAME="${1:-}"

if [ -n "$RUN_NAME" ]; then
  "$PY" eval.py --config "$CFG" --split val --run-name "$RUN_NAME"
else
  "$PY" eval.py --config "$CFG" --split val
fi
