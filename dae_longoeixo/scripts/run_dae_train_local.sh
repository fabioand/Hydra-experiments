#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${ROOT_DIR}/.venv/bin/python"
CFG="${ROOT_DIR}/dae_longoeixo/dae_train_config.json"
RUN_NAME="${1:-}"
MAX_SAMPLES="${2:-}"

CMD=("$PY" dae_longoeixo/train_dae.py --config "$CFG")

if [ -n "$RUN_NAME" ]; then
  CMD+=(--run-name "$RUN_NAME")
fi

if [ -n "$MAX_SAMPLES" ]; then
  if ! [[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Erro: max_samples precisa ser inteiro >= 0"
    echo "Uso: $0 [run_name] [max_samples]"
    exit 1
  fi
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

"${CMD[@]}"
