#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-longoeixo/gaussian_maps_stack64}"
NUM_SAMPLES="${2:-}"
SIGMA="${SIGMA:-7.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CMD=(
  .venv/bin/python longoeixo/scripts/generate_gaussian_point_maps.py
  --imgs-dir longoeixo/imgs
  --json-dir longoeixo/data_longoeixo
  --out-dir "$OUT_DIR"
  --output-mode stack64
  --sigma "$SIGMA"
)

if [[ -n "$NUM_SAMPLES" ]]; then
  CMD+=(--num-samples "$NUM_SAMPLES")
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  CMD+=(--skip-existing)
fi

"${CMD[@]}"

echo "Concluido. Saida em: $OUT_DIR"
