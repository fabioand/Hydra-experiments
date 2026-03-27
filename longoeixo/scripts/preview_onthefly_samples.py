#!/usr/bin/env python3
"""Preview visual do pipeline on-the-fly usando as mesmas rotinas do treino.

Gera paineis lado a lado:
- Before: x_before + max(y_before)
- After:  x + max(y_heatmap) (apos augmentacao, se habilitada)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra_data import HydraSample, HydraTeethDataset, discover_samples, load_json


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def make_overlay_rgb(img01: np.ndarray, mask64: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    base = np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    m = np.max(mask64, axis=0)
    red = np.zeros_like(base_bgr, dtype=np.uint8)
    red[:, :, 2] = np.clip(m * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.addWeighted(base_bgr, 1.0, red, alpha, 0.0)


def save_panel(before: np.ndarray, after: np.ndarray, out_path: Path) -> None:
    h = before.shape[0]
    bar_h = 28
    panel = np.zeros((h + bar_h, before.shape[1] + after.shape[1], 3), dtype=np.uint8)
    panel[bar_h:, : before.shape[1]] = before
    panel[bar_h:, before.shape[1] :] = after

    cv2.putText(panel, "Before (img+max(mask64))", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "After (img+max(mask64))",
        (before.shape[1] + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def choose_samples(samples: List[HydraSample], num_samples: int, seed: int) -> List[HydraSample]:
    if num_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    return [samples[i] for i in idx[:num_samples]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview on-the-fly samples (mesmo pipeline do treino)")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("longoeixo/onthefly_preview"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--no-augment", action="store_true", help="Desativa augmentacao para comparar apenas geracao base")
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples deve ser >= 1")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha deve estar em [0,1]")

    repo_root = REPO_ROOT
    cfg: Dict = load_json(_resolve_path(repo_root, str(args.config)))
    preset = load_json(_resolve_path(repo_root, cfg["paths"]["preset_path"]))

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    masks_cfg = cfg["paths"].get("masks_dir")
    masks_dir = _resolve_path(repo_root, masks_cfg) if masks_cfg else None
    source_mode = str(cfg.get("data", {}).get("source_mode", "on_the_fly"))

    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        masks_dir=masks_dir,
        source_mode=source_mode,
    )
    if not samples:
        raise FileNotFoundError("Nenhuma amostra encontrada para preview.")

    selected = choose_samples(samples, num_samples=args.num_samples, seed=args.seed)
    ds = HydraTeethDataset(
        samples=selected,
        preset=preset,
        augment=not args.no_augment,
        source_mode=source_mode,
        seed=args.seed,
    )

    out_dir = _resolve_path(repo_root, str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(ds)):
        item = ds[i]
        stem = str(item["stem"])
        x_before = item["x_before"].numpy()[0]
        y_before = item["y_before"].numpy()
        x_after = item["x"].numpy()[0]
        y_after = item["y_heatmap"].numpy()

        over_before = make_overlay_rgb(x_before, y_before, alpha=args.alpha)
        over_after = make_overlay_rgb(x_after, y_after, alpha=args.alpha)
        save_panel(over_before, over_after, out_dir / f"{i:02d}_{stem}.png")

    print(f"[PREVIEW] source_mode={source_mode} samples={len(ds)} augment={'off' if args.no_augment else 'on'}")
    print(f"[PREVIEW] output={out_dir}")


if __name__ == "__main__":
    main()
