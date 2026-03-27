#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r longoeixo/requirements.txt

# Em hosts com GPU NVIDIA (ex.: EC2 g4dn), garantir wheel CUDA do PyTorch.
# Pode ser sobrescrito externamente:
#   TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[SETUP] GPU NVIDIA detectada; garantindo PyTorch com CUDA via ${TORCH_CUDA_INDEX_URL}"
  .venv/bin/pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_CUDA_INDEX_URL}"
fi

echo "[SETUP] validando PyTorch/CUDA..."
.venv/bin/python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

echo "Ambiente pronto em: $ROOT_DIR/.venv"
