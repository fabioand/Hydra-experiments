#!/usr/bin/env bash
set -euo pipefail

# Abre tuneis SSH locais para monitorar EC2 no browser:
# - TensorBoard: http://localhost:6006
# - Viewer HTML: http://localhost:8080
#
# Uso:
#   longoeixo/scripts/ec2_open_tunnels.sh <usuario@ec2-host>

if [ "${1:-}" = "" ]; then
  echo "Uso: $0 <usuario@ec2-host>"
  exit 1
fi

REMOTE_HOST="$1"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/fabio/pem}"

exec ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -N \
  -L 6006:127.0.0.1:6006 \
  -L 8080:127.0.0.1:8080 \
  "$REMOTE_HOST"
