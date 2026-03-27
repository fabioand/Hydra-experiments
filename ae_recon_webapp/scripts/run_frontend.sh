#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONT="$ROOT/ae_recon_webapp/frontend"
cd "$FRONT"

if [[ -f "$ROOT/ae_recon_webapp/.env" ]]; then
  export VITE_API_BASE="$(grep '^VITE_API_BASE=' "$ROOT/ae_recon_webapp/.env" | cut -d '=' -f2-)"
fi

npm install
npm run dev -- --host 0.0.0.0 --port 5173
