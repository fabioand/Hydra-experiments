#!/usr/bin/env bash
set -euo pipefail

# Copia parcial do dataset longoeixo da EC2 para o repo local.
# Dry-run por padrao (-n): apenas mostra o que seria transferido.

KEY_PATH="${HOME}/.ssh/fabio.pem"
REMOTE_HOST="ubuntu@35.92.136.175"
REMOTE_BASE="/dataminer/rmdatasets/data/longoeixo"
LOCAL_DEST="/Users/fabioandrade/hydra/longoeixo/"
MAX_PAIRS="200"
APPLY_MODE="0"

for arg in "$@"; do
  case "$arg" in
    --apply)
      APPLY_MODE="1"
      ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        MAX_PAIRS="$arg"
      else
        echo "Parametro invalido: $arg" >&2
        echo "Uso: $0 [num_pares] [--apply]" >&2
        exit 2
      fi
      ;;
  esac
done

MAX_LINES="$((MAX_PAIRS * 2))"

if [ ! -f "$KEY_PATH" ]; then
  echo "Chave SSH nao encontrada: $KEY_PATH" >&2
  exit 1
fi

RSYNC_MODE="-navz"
if [ "$APPLY_MODE" = "1" ]; then
  RSYNC_MODE="-avz"
fi

rsync "$RSYNC_MODE" -e "ssh -i $KEY_PATH" --progress --prune-empty-dirs \
  --files-from=<(ssh -i "$KEY_PATH" "$REMOTE_HOST" "
    cd '$REMOTE_BASE' &&
    find imgs -maxdepth 1 -type f -name '*.jpg' | sed 's|^imgs/||' | sed 's|\\.jpg$||' | sort |
    while read -r b; do
      [ -f \"data_longoeixo/\$b.json\" ] && {
        echo \"imgs/\$b.jpg\"
        echo \"data_longoeixo/\$b.json\"
      }
    done | head -n '$MAX_LINES'
  ") \
  "$REMOTE_HOST:$REMOTE_BASE/" \
  "$LOCAL_DEST"

if [ "$APPLY_MODE" = "1" ]; then
  echo
  echo "Copia finalizada em modo real (--apply)."
else
  cat <<EOF

Dry-run finalizado.
Nenhum arquivo foi copiado porque o comando usa -n.
Para executar de verdade, use: $0 [num_pares] --apply
EOF
fi
