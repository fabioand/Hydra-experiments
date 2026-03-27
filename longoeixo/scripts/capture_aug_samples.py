#!/usr/bin/env python3
"""Captura amostras antes/depois da augmentacao para inspeção visual.

Gera painéis com:
- esquerda: overlay ANTES (imagem 256 + max das 64 gaussianas em vermelho)
- direita: overlay DEPOIS da augmentacao
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np


def load_preset(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_pairs(imgs_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    img_map = {p.stem: p for p in imgs_dir.glob("*.jpg")}
    mask_map = {p.stem: p for p in masks_dir.glob("*.npy")}
    common = sorted(set(img_map).intersection(mask_map))
    return [(img_map[s], mask_map[s]) for s in common]


def load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return img


def load_stack64(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[0] != 64:
        raise ValueError(f"Esperado stack64 (64,H,W). Recebido: {arr.shape} em {path}")
    return arr.astype(np.float32, copy=False)


def preprocess_unet256(img_gray: np.ndarray, mask64: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # imagem: INTER_AREA para downscale
    x = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    # mascara: linear para manter smooth gaussian
    ys = [cv2.resize(mask64[c], (256, 256), interpolation=cv2.INTER_LINEAR) for c in range(64)]
    y = np.stack(ys, axis=0).astype(np.float32)
    np.clip(y, 0.0, 1.0, out=y)
    return x, y


def make_overlay_rgb(img01: np.ndarray, mask64: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    base = np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    m = np.max(mask64, axis=0)
    red = np.zeros_like(base_bgr, dtype=np.uint8)
    red[:, :, 2] = np.clip(m * 255.0, 0.0, 255.0).astype(np.uint8)

    over = cv2.addWeighted(base_bgr, 1.0, red, alpha, 0.0)
    return over


def build_geo_aug(preset: Dict) -> A.Compose:
    aug = preset.get("augmentation", {})
    rot = float(aug.get("rotation_deg", 7.0))
    scale_range = aug.get("scale_range", [0.95, 1.05])
    trans = float(aug.get("translate_frac", 0.03))

    scale_low = float(scale_range[0])
    scale_high = float(scale_range[1])
    scale_limit = (scale_low - 1.0, scale_high - 1.0)

    # Sem flips por decisão do projeto.
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=trans,
                scale_limit=scale_limit,
                rotate_limit=rot,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=1.0,
            )
        ]
    )


def apply_intensity_and_noise(img01: np.ndarray, preset: Dict, rng: np.random.Generator) -> np.ndarray:
    out = img01.copy()
    aug = preset.get("augmentation", {})

    # Brightness/contrast leve apenas na imagem.
    intensity = aug.get("intensity", {})
    bdelta = float(intensity.get("brightness_delta", 0.1))
    cmin, cmax = intensity.get("contrast_range", [0.9, 1.1])
    contrast = float(rng.uniform(float(cmin), float(cmax)))
    bright = float(rng.uniform(-bdelta, bdelta))
    out = out * contrast + bright
    out = np.clip(out, 0.0, 1.0)

    noise_cfg = aug.get("noise", {})
    if not noise_cfg.get("enabled", True):
        return out

    overall_p = float(noise_cfg.get("overall_p", 0.35))
    if rng.random() > overall_p:
        return out

    patterns = noise_cfg.get("patterns", [])
    if not patterns:
        return out

    probs = np.array([float(p.get("p", 1.0)) for p in patterns], dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(patterns), p=probs))
    spec = patterns[idx]
    ntype = spec.get("type", "gaussian_additive")

    if ntype == "gaussian_additive":
        s0, s1 = spec.get("std_range", [0.005, 0.02])
        std = float(rng.uniform(float(s0), float(s1)))
        out = out + rng.normal(0.0, std, size=out.shape).astype(np.float32)

    elif ntype == "poisson":
        # aproximação leve de ruído Poisson em escala [0,1]
        r0, r1 = spec.get("scale_range", [0.9, 1.1])
        scale = float(rng.uniform(float(r0), float(r1)))
        lam = max(10.0, 50.0 * scale)
        out = rng.poisson(np.clip(out, 0.0, 1.0) * lam).astype(np.float32) / lam

    elif ntype == "speckle_multiplicative":
        s0, s1 = spec.get("std_range", [0.003, 0.012])
        std = float(rng.uniform(float(s0), float(s1)))
        out = out + out * rng.normal(0.0, std, size=out.shape).astype(np.float32)

    out = np.clip(out, 0.0, 1.0)
    return out


def save_panel(before: np.ndarray, after: np.ndarray, title_left: str, title_right: str, out_path: Path) -> None:
    h = before.shape[0]
    bar_h = 28
    panel = np.zeros((h + bar_h, before.shape[1] + after.shape[1], 3), dtype=np.uint8)
    panel[bar_h:, : before.shape[1]] = before
    panel[bar_h:, before.shape[1] :] = after

    cv2.putText(panel, title_left, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        title_right,
        (before.shape[1] + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura exemplos antes/depois da augmentacao.")
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--masks-dir", type=Path, default=Path("longoeixo/gaussian_maps_stack64"))
    parser.add_argument("--preset", type=Path, default=Path("longoeixo/presets/unet256_stack64_preset.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("longoeixo/aug_inspection"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overlay-alpha", type=float, default=0.6)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples deve ser >= 1")

    preset = load_preset(args.preset)
    geo_aug = build_geo_aug(preset)
    rng = np.random.default_rng(args.seed)

    pairs = list_pairs(args.imgs_dir, args.masks_dir)
    if not pairs:
        raise FileNotFoundError("Nenhum par imagem+mask encontrado.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pairs[: args.num_samples]

    for i, (img_path, mask_path) in enumerate(pairs, start=1):
        img_gray = load_gray_image(img_path)
        mask64 = load_stack64(mask_path)

        x, y = preprocess_unet256(img_gray, mask64)

        # Albumentations trabalha com máscara HWC
        y_hwc = np.transpose(y, (1, 2, 0))
        aug_out = geo_aug(image=x.astype(np.float32), mask=y_hwc.astype(np.float32))
        x_aug = aug_out["image"].astype(np.float32)
        y_aug = np.transpose(aug_out["mask"], (2, 0, 1)).astype(np.float32)

        x_aug = apply_intensity_and_noise(x_aug, preset, rng)
        np.clip(y_aug, 0.0, 1.0, out=y_aug)

        over_before = make_overlay_rgb(x, y, alpha=args.overlay_alpha)
        over_after = make_overlay_rgb(x_aug, y_aug, alpha=args.overlay_alpha)

        stem = img_path.stem
        save_panel(
            over_before,
            over_after,
            "Before Aug (img + max(mask64))",
            "After Aug (img + max(mask64))",
            out_dir / f"{i:02d}_{stem}.png",
        )

    print(f"Painéis salvos: {len(pairs)}")
    print(f"Saída: {out_dir}")


if __name__ == "__main__":
    main()
