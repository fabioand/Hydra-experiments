#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${ROOT_DIR}/.venv/bin/python"
CFG="${ROOT_DIR}/hydra_train_config.json"
RUN_NAME="${1:-smoke_$(date +%Y-%m-%d_%H-%M-%S)}"

"$PY" train.py --config "$CFG" --smoke --force-regenerate-split --run-name "$RUN_NAME"
"$PY" eval.py --config "$CFG" --smoke --split val --run-name "$RUN_NAME"
