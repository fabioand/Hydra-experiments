#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" ae_radiograph_filter/scripts/run_filter.py \
  --ckpt ae_radiograph_filter/models/ae_identity_bestE21.ckpt \
  --images-dir longoeixo/imgs \
  --limit 300 \
  --batch-size 16 \
  --percentile 95 \
  --top-k-panels 20 \
  --run-name AE_FILTER_LOCAL300_BESTE21 \
  "$@"

