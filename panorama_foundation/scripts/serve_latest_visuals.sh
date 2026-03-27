#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNS_DIR="panorama_foundation/experiments/ae_visual_smoke/runs"
if [[ ! -d "$RUNS_DIR" ]]; then
  echo "Runs directory not found: $RUNS_DIR"
  exit 1
fi

LATEST_RUN="$(ls -1 "$RUNS_DIR" | tail -n 1)"
if [[ -z "$LATEST_RUN" ]]; then
  echo "No runs found in: $RUNS_DIR"
  exit 1
fi

VIS_DIR="$RUNS_DIR/$LATEST_RUN/train_visuals"
if [[ ! -d "$VIS_DIR" ]]; then
  echo "Visuals directory not found: $VIS_DIR"
  exit 1
fi

PORT="${1:-8080}"
echo "Serving visuals from: $VIS_DIR"
echo "Open: http://localhost:${PORT}/index.html"

cd "$VIS_DIR"
python3 -m http.server "$PORT"

