#!/usr/bin/env bash
set -euo pipefail

# Executa treino na EC2 com config preparada para caminhos absolutos.
#
# Uso:
#   bash longoeixo/scripts/ec2_run_train.sh [run_name] [config]
#
# Exemplo:
#   bash longoeixo/scripts/ec2_run_train.sh FifthTest999_absentHM0 hydra_train_config_fifth999_absenthm0_ec2.json

HYDRA_ROOT="${HYDRA_ROOT:-/dataset/hydra}"
RUN_NAME="${1:-FifthTest999_absentHM0}"
CFG="${2:-hydra_train_config_fifth999_absenthm0_ec2.json}"

cd "$HYDRA_ROOT"

if [ ! -f ".venv/bin/python" ]; then
  echo "Erro: .venv nao encontrado em $HYDRA_ROOT"
  echo "Execute: bash longoeixo/setup_env.sh"
  exit 1
fi

if [ ! -f "$CFG" ]; then
  echo "Erro: config nao encontrada: $CFG"
  echo "Execute antes: bash longoeixo/scripts/ec2_configure_paths.sh"
  exit 1
fi

exec .venv/bin/python train.py --config "$CFG" --run-name "$RUN_NAME"
