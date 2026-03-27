#!/usr/bin/env bash
set -euo pipefail

# Gera configs *_ec2.json apontando dataset para:
#   /dataminer/rmdatasets/data/longoeixo
#
# Uso (na EC2):
#   bash longoeixo/scripts/ec2_configure_paths.sh
#
# Variaveis opcionais:
#   HYDRA_ROOT=/dataset/hydra
#   DATASET_BASE=/dataminer/rmdatasets/data/longoeixo

HYDRA_ROOT="${HYDRA_ROOT:-/dataset/hydra}"
DATASET_BASE="${DATASET_BASE:-/dataminer/rmdatasets/data/longoeixo}"

cd "$HYDRA_ROOT"

if [ ! -d "$DATASET_BASE/imgs" ]; then
  echo "Erro: pasta de imagens nao encontrada: $DATASET_BASE/imgs"
  exit 1
fi
if [ ! -d "$DATASET_BASE/data_longoeixo" ]; then
  echo "Erro: pasta de JSON nao encontrada: $DATASET_BASE/data_longoeixo"
  exit 1
fi

python3 - "$HYDRA_ROOT" "$DATASET_BASE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
dataset_base = Path(sys.argv[2]).resolve()

targets = [
    "hydra_train_config.json",
    "hydra_train_config_secondtest100.json",
    "hydra_train_config_fourth999_sigma52.json",
    "hydra_train_config_fifth999_absenthm0.json",
    "hydra_train_config_full70_15_15.json",
    "hydra_train_config_full70_15_15_no_presence_hmfull.json",
    "hydra_train_config_roi_center24_v1.json",
    "hydra_train_config_roi_center24_v1_nopres_absenthm1.json",
    "hydra_train_config_roi_center24_sharedflip_v1.json",
    "hydra_train_config_roi_center24_sharedflip_v1_nopres_absenthm1.json",
    "hydra_train_config_roi_lateral_shared20_v1.json",
    "hydra_train_config_roi_lateral_shared20_v1_nopres_absenthm1.json",
    "hydra_train_config_roi_lateral_shared20_v1_nopres_absenthm1_stable.json",
]

def to_abs(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (root / p))

for name in targets:
    src = root / name
    if not src.exists():
        continue

    cfg = json.loads(src.read_text(encoding="utf-8"))
    paths = cfg.setdefault("paths", {})
    paths["imgs_dir"] = str(dataset_base / "imgs")
    paths["json_dir"] = str(dataset_base / "data_longoeixo")

    for key in ("splits_path", "preset_path", "output_dir", "masks_dir"):
        if key in paths and isinstance(paths[key], str):
            paths[key] = to_abs(paths[key])

    smoke = cfg.get("smoke_test")
    if isinstance(smoke, dict):
        if isinstance(smoke.get("output_dir"), str):
            smoke["output_dir"] = to_abs(smoke["output_dir"])
        if isinstance(smoke.get("masks_dir"), str):
            smoke["masks_dir"] = to_abs(smoke["masks_dir"])

    out = src.with_name(src.stem + "_ec2.json")
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {out}")
PY

echo
echo "[EC2 CONFIG] arquivos gerados com sufixo _ec2.json."
echo "[EXEMPLO TREINO]"
echo "  .venv/bin/python train.py --config hydra_train_config_fifth999_absenthm0_ec2.json --run-name FifthTest999_absentHM0"
