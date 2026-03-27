#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${ROOT_DIR}/.venv/bin/python"
CFG="${ROOT_DIR}/dae_longoeixo/dae_train_config.json"
RUN_NAME="${1:-dae_smoke}"

"$PY" dae_longoeixo/train_dae.py --config "$CFG" --run-name "$RUN_NAME" --smoke
"$PY" dae_longoeixo/eval_dae.py --config "$CFG" --run-name "$RUN_NAME" --split val --num-knockout-passes 2 --smoke
