#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" ae_radiograph_filter/scripts/run_batch_enhance_and_html.py \
  --images-dir longoeixo/imgs \
  --ckpt ae_radiograph_filter/models/ae_identity_bestE21.ckpt \
  --limit 100 \
  --fs 0.3 \
  --fa 0.3 \
  --run-name AE_ENHANCE_LOCAL100_FS03_FA03 \
  "$@"

