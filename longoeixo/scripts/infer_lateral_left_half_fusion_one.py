#!/usr/bin/env python3
"""Inferencia da rede lateral no ROI LEFT de 1 radiografia e fusao com max-heatmap.

Saida: imagem da metade esquerda (ROI LEFT) fusionada com o mapa composto
(maximo dos canais da rede lateral).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from hydra_data import discover_samples, fixed_three_windows
from hydra_multitask_model import HydraUNetMultiTask


DEFAULT_LATERAL_CKPT = Path(
    "longoeixo/checkpoints/ec2_lateral_shared20/"
    "lateral20_v1_fixedorient_nopres_absenthm1_16k_ft_best_ep29.ckpt"
)
DEFAULT_OUTPUT = Path("/tmp/lateral_left_half_fusion_one.png")
TARGET_HW = (256, 256)


def _auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_lateral_model(ckpt_path: Path, device: torch.device) -> HydraUNetMultiTask:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("model_config", {})
    model = HydraUNetMultiTask(
        in_channels=1,
        heatmap_out_channels=20,
        presence_out_channels=10,
        enable_presence_head=bool(cfg.get("enable_presence_head", False)),
        backbone=cfg.get("backbone", "resnet34"),
        presence_dropout=float(cfg.get("presence_dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _infer_maps(
    model: HydraUNetMultiTask, crop_gray: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tgt_h, tgt_w = TARGET_HW
    x = cv2.resize(crop_gray, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(x[None, None, ...]).to(device)
    with torch.no_grad():
        out = model(x)
        logits = out["heatmap_logits"][0].detach().cpu().numpy()
        probs = torch.sigmoid(out["heatmap_logits"])[0].detach().cpu().numpy()
    return logits, probs


def _pseudocolor_heatmap(hm: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    # Escala por percentis para preservar gradientes do mapa (incluindo fundo),
    # evitando blobs chapados quando os picos saturam perto de 1.0.
    hm_f = hm.astype(np.float32, copy=False)
    lo = float(np.percentile(hm_f, p_low))
    hi = float(np.percentile(hm_f, p_high))
    if hi <= lo + 1e-8:
        lo = float(np.min(hm_f))
        hi = float(np.max(hm_f))
    if hi <= lo + 1e-8:
        hm_n = np.zeros_like(hm_f, dtype=np.float32)
    else:
        hm_n = (hm_f - lo) / (hi - lo)
    hm_u8 = np.clip(hm_n * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)


def _fuse_heatmap(image_gray: np.ndarray, hm: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    color = _pseudocolor_heatmap(hm)
    return cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roda rede lateral em 1 radiografia (ROI LEFT) e salva fusao com max-heatmap."
    )
    parser.add_argument("--lateral-ckpt", type=Path, default=DEFAULT_LATERAL_CKPT)
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--json-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Opcional: caminho direto de uma radiografia .jpg. Se omitido, escolhe 1 amostra pelo seed.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument(
        "--map-source",
        type=str,
        default="logits",
        choices=["logits", "probs"],
        help="Fonte do mapa composto: logits crus (mais tons) ou probs sigmoid.",
    )
    parser.add_argument(
        "--pure-map",
        action="store_true",
        help="Salva somente o mapa pseudocolorizado (sem fusao com radiografia).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = args.lateral_ckpt if args.lateral_ckpt.is_absolute() else (repo_root / args.lateral_ckpt)
    imgs_dir = args.imgs_dir if args.imgs_dir.is_absolute() else (repo_root / args.imgs_dir)
    json_dir = args.json_dir if args.json_dir.is_absolute() else (repo_root / args.json_dir)
    out_path = args.out if args.out.is_absolute() else (repo_root / args.out)

    if args.image is not None:
        image_path = args.image if args.image.is_absolute() else (repo_root / args.image)
    else:
        samples = discover_samples(imgs_dir=imgs_dir, json_dir=json_dir, masks_dir=None, source_mode="on_the_fly")
        if not samples:
            raise RuntimeError("Nenhuma amostra encontrada.")
        pool = sorted(samples, key=lambda s: s.stem)
        image_path = random.Random(args.seed).choice(pool).image_path

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {image_path}")

    h, w = img.shape
    rect_left = fixed_three_windows(w, h)["LEFT"]
    x1, y1, x2, y2 = rect_left
    crop_left = img[y1:y2, x1:x2]

    device = _auto_device()
    model = _load_lateral_model(ckpt_path, device)
    logits, probs = _infer_maps(model, crop_left, device)  # (20,256,256), (20,256,256)
    hm_src = logits if args.map_source == "logits" else probs
    hm_max_256 = np.max(hm_src, axis=0).astype(np.float32)
    hm_max_left = cv2.resize(hm_max_256, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    if args.pure_map:
        fused = _pseudocolor_heatmap(hm_max_left)
    else:
        fused = _fuse_heatmap(crop_left, hm_max_left, alpha=float(args.alpha))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fused)

    print(f"[INFO] device={device}")
    print(f"[INFO] image={image_path}")
    print(f"[INFO] roi_left={rect_left}")
    print(f"[INFO] map_source={args.map_source}")
    print(f"[DONE] out={out_path}")


if __name__ == "__main__":
    main()
