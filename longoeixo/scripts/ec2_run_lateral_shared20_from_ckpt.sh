#!/usr/bin/env bash
set -euo pipefail

# Treino da lateral shared20 (no_pres) na EC2 iniciando de checkpoint.
#
# Uso:
#   bash longoeixo/scripts/ec2_run_lateral_shared20_from_ckpt.sh [run_name] [init_ckpt] [config]
#
# Exemplo:
#   bash longoeixo/scripts/ec2_run_lateral_shared20_from_ckpt.sh \
#     lateral20_v1_fixedorient_nopres_absenthm1_16k_ft \
#     /dataset/hydra/longoeixo/checkpoints/roi_lateral_shared20/lateral20_v1_fixedorient_nopres_absenthm1_full_mps_best.ckpt \
#     hydra_train_config_roi_lateral_shared20_v1_nopres_absenthm1_ec2.json

HYDRA_ROOT="${HYDRA_ROOT:-/dataset/hydra}"
RUN_NAME="${1:-lateral20_v1_fixedorient_nopres_absenthm1_16k_ft}"
INIT_CKPT="${2:-/dataset/hydra/longoeixo/checkpoints/roi_lateral_shared20/lateral20_v1_fixedorient_nopres_absenthm1_full_mps_best.ckpt}"
CFG="${3:-hydra_train_config_roi_lateral_shared20_v1_nopres_absenthm1_ec2.json}"

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

if [ ! -f "$INIT_CKPT" ]; then
  echo "Erro: init checkpoint nao encontrado: $INIT_CKPT"
  exit 1
fi

exec .venv/bin/python train.py \
  --config "$CFG" \
  --run-name "$RUN_NAME" \
  --init-ckpt "$INIT_CKPT"
