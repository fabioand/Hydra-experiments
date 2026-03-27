#!/usr/bin/env bash
set -euo pipefail

# Sync do repositorio local para EC2 sem dataset e sem artefatos de treino.
#
# Uso:
#   longoeixo/scripts/ec2_sync_repo.sh <usuario@ec2-host>
#
# Variaveis opcionais:
#   SSH_KEY=/Users/you/.ssh/key.pem
#   REMOTE_DIR=/dataset/hydra

if [ "${1:-}" = "" ]; then
  echo "Uso: $0 <usuario@ec2-host>"
  exit 1
fi

REMOTE_HOST="$1"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/fabio/pem}"
REMOTE_DIR="${REMOTE_DIR:-/dataset/hydra}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SSH_CMD=(ssh -i "$SSH_KEY" -o IdentitiesOnly=yes)

if [ ! -f "$SSH_KEY" ]; then
  echo "Erro: chave SSH nao encontrada: $SSH_KEY"
  exit 1
fi

echo "[EC2 SYNC] destino: ${REMOTE_HOST}:${REMOTE_DIR}"
"${SSH_CMD[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"

rsync -az --progress \
  -e "ssh -i '$SSH_KEY' -o IdentitiesOnly=yes" \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "*.pyo" \
  --exclude "*.pt" \
  --exclude "*.pth" \
  --exclude "*.ckpt" \
  --exclude "*.npy" \
  --exclude ".env" \
  --exclude ".env.*" \
  --exclude "*.pem" \
  --exclude "*.key" \
  --exclude "*.crt" \
  --exclude "*.p12" \
  --exclude "*.pfx" \
  --exclude "longoeixo/imgs/" \
  --exclude "longoeixo/data_longoeixo/" \
  --exclude "longoeixo/gaussian_maps*/" \
  --exclude "longoeixo/experiments/" \
  --exclude "longoeixo/train_visuals/" \
  --exclude "longoeixo/onthefly_preview/" \
  --exclude "longoeixo/aug_inspection*/" \
  ./ "${REMOTE_HOST}:${REMOTE_DIR}/"

echo
echo "[EC2 SYNC] concluido."
echo "[PROXIMO PASSO NA EC2]"
echo "  cd ${REMOTE_DIR}"
echo "  bash longoeixo/scripts/ec2_configure_paths.sh"
