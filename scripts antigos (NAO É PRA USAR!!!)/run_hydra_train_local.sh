#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${ROOT_DIR}/.venv/bin/python"
CFG="${ROOT_DIR}/hydra_train_config.json"
RUN_NAME="${1:-}"
NUM_SAMPLES="${2:-}"

run_train() {
  local cfg_path="$1"
  if [ -n "$RUN_NAME" ]; then
    "$PY" train.py --config "$cfg_path" --run-name "$RUN_NAME"
  else
    "$PY" train.py --config "$cfg_path"
  fi
}

if [ -n "$NUM_SAMPLES" ]; then
  if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Erro: num_samples precisa ser inteiro positivo."
    echo "Uso: $0 [run_name] [num_samples]"
    exit 1
  fi
  if [ "$NUM_SAMPLES" -lt 2 ]; then
    echo "Erro: num_samples deve ser >= 2."
    exit 1
  fi

  SUB_DIR="${ROOT_DIR}/longoeixo/subsets/N${NUM_SAMPLES}_seed123"
  SUB_IMGS_DIR="${SUB_DIR}/imgs"
  SUB_JSON_DIR="${SUB_DIR}/data_longoeixo"
  SUB_SPLIT_PATH="${SUB_DIR}/splits.json"
  SUB_CFG_PATH="${SUB_DIR}/hydra_train_config.json"
  mkdir -p "$SUB_IMGS_DIR" "$SUB_JSON_DIR"

  "$PY" - <<'PY' "$ROOT_DIR" "$CFG" "$NUM_SAMPLES" "$SUB_IMGS_DIR" "$SUB_JSON_DIR" "$SUB_SPLIT_PATH" "$SUB_CFG_PATH"
import json
import random
import sys
from pathlib import Path

root = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
num_samples = int(sys.argv[3])
sub_imgs_dir = Path(sys.argv[4])
sub_json_dir = Path(sys.argv[5])
sub_split_path = Path(sys.argv[6])
sub_cfg_path = Path(sys.argv[7])

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
seed = int(cfg.get("split", {}).get("seed", cfg.get("seed", 123)))
val_ratio = float(cfg.get("split", {}).get("val_ratio", 0.2))

imgs_dir = root / cfg["paths"]["imgs_dir"]
json_dir = root / cfg["paths"]["json_dir"]

img_stems = {p.stem for p in imgs_dir.glob("*.jpg")}
json_stems = {p.stem for p in json_dir.glob("*.json")}
common = sorted(img_stems & json_stems)
if not common:
    raise SystemExit(f"Nenhum par JPG+JSON encontrado em {imgs_dir} e {json_dir}")
if num_samples > len(common):
    raise SystemExit(f"num_samples={num_samples} maior que total disponivel={len(common)}")

rng = random.Random(seed)
rng.shuffle(common)
selected = common[:num_samples]

for stem in selected:
    src_img = imgs_dir / f"{stem}.jpg"
    src_json = json_dir / f"{stem}.json"
    dst_img = sub_imgs_dir / src_img.name
    dst_json = sub_json_dir / src_json.name
    if not dst_img.exists():
        dst_img.symlink_to(src_img.resolve())
    if not dst_json.exists():
        dst_json.symlink_to(src_json.resolve())

n_val = max(1, int(round(num_samples * val_ratio))) if num_samples > 1 else 0
val = selected[:n_val]
train = selected[n_val:]
if not train or not val:
    raise SystemExit("Split invalido para subset; ajuste num_samples/val_ratio.")

split_payload = {
    "seed": seed,
    "val_ratio": val_ratio,
    "num_samples": num_samples,
    "train": train,
    "val": val,
}
sub_split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

cfg["paths"]["imgs_dir"] = str(sub_imgs_dir.relative_to(root))
cfg["paths"]["json_dir"] = str(sub_json_dir.relative_to(root))
cfg["paths"]["splits_path"] = str(sub_split_path.relative_to(root))
sub_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[SUBSET] total={num_samples} train={len(train)} val={len(val)} seed={seed}")
print(f"[SUBSET] config={sub_cfg_path}")
PY

  echo "[RUN] usando subset de ${NUM_SAMPLES} amostras"
  run_train "$SUB_CFG_PATH"
  exit 0
fi

run_train "$CFG"
