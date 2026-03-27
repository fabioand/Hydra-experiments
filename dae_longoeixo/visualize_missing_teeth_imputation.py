#!/usr/bin/env python3
"""Visualiza imputacao do DAE em exames com dentes ausentes reais.

Gera overlays com:
- dentes anotados (presentes) em verde
- dentes faltantes reconstruidos pelo DAE em vermelho

E monta um HTML para inspeção visual em lote.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from dae_longoeixo.dae_data import build_noisy_input, load_json
from dae_longoeixo.dae_model import CoordinateDenoisingAutoencoder
from hydra_multitask_model import CANONICAL_TEETH_32


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_run_name(output_base_dir: Path) -> str | None:
    latest_file = output_base_dir / "latest_run.txt"
    if latest_file.exists():
        name = latest_file.read_text(encoding="utf-8").strip()
        if name:
            return name

    runs_dir = output_base_dir / "runs"
    if not runs_dir.exists():
        return None

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def _load_points_by_label(json_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    data = load_json(json_path)
    points_by_label: Dict[str, List[Tuple[float, float]]] = {}

    for ann in data:
        label = str(ann.get("label", ""))
        pts = ann.get("pts", [])
        valid_pts: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid_pts.append((float(x), float(y)))
        if label not in points_by_label and valid_pts:
            points_by_label[label] = valid_pts

    return points_by_label


def _normalize_xy(x: float, y: float, src_hw: Tuple[int, int]) -> Tuple[float, float]:
    h, w = src_hw
    if h <= 1 or w <= 1:
        return 0.0, 0.0
    return x / float(w - 1), y / float(h - 1)


def _denormalize_xy(xn: float, yn: float, src_hw: Tuple[int, int]) -> Tuple[float, float]:
    h, w = src_hw
    x = float(np.clip(xn, 0.0, 1.0)) * float(w - 1)
    y = float(np.clip(yn, 0.0, 1.0)) * float(h - 1)
    return x, y


def _build_partial_coords_and_missing(
    points_by_label: Dict[str, List[Tuple[float, float]]],
    src_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    coords = np.zeros((128,), dtype=np.float32)
    knocked = np.zeros((32,), dtype=np.float32)
    present_teeth: List[str] = []
    missing_teeth: List[str] = []

    for i, tooth in enumerate(CANONICAL_TEETH_32):
        pts = points_by_label.get(tooth)
        if pts is not None and len(pts) >= 2:
            present_teeth.append(tooth)
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1n, y1n = _normalize_xy(x1, y1, src_hw)
            x2n, y2n = _normalize_xy(x2, y2, src_hw)
            base = 4 * i
            coords[base + 0] = float(np.clip(x1n, 0.0, 1.0))
            coords[base + 1] = float(np.clip(y1n, 0.0, 1.0))
            coords[base + 2] = float(np.clip(x2n, 0.0, 1.0))
            coords[base + 3] = float(np.clip(y2n, 0.0, 1.0))
        else:
            missing_teeth.append(tooth)
            knocked[i] = 1.0

    return coords, knocked, present_teeth, missing_teeth


def _load_alveolar_curves_flat(curves_json_path: Path, n_curve_points: int, src_hw: Tuple[int, int]) -> np.ndarray | None:
    try:
        payload = load_json(curves_json_path)
    except Exception:
        return None
    sup = payload.get("RebAlvSup")
    inf = payload.get("RebAlvInf")
    if not isinstance(sup, list) or not isinstance(inf, list):
        return None
    sup_arr = np.asarray(sup, dtype=np.float32)
    inf_arr = np.asarray(inf, dtype=np.float32)
    if tuple(sup_arr.shape) != (n_curve_points, 2) or tuple(inf_arr.shape) != (n_curve_points, 2):
        return None
    if float(np.max(np.abs(sup_arr))) > 1.5 or float(np.max(np.abs(inf_arr))) > 1.5:
        h, w = src_hw
        if h > 1 and w > 1:
            sup_arr[:, 0] = np.clip(sup_arr[:, 0] / float(w - 1), 0.0, 1.0)
            sup_arr[:, 1] = np.clip(sup_arr[:, 1] / float(h - 1), 0.0, 1.0)
            inf_arr[:, 0] = np.clip(inf_arr[:, 0] / float(w - 1), 0.0, 1.0)
            inf_arr[:, 1] = np.clip(inf_arr[:, 1] / float(h - 1), 0.0, 1.0)
    return np.concatenate([sup_arr.reshape(-1), inf_arr.reshape(-1)], axis=0).astype(np.float32, copy=False)


def _draw_overlay(
    img_gray: np.ndarray,
    points_by_label: Dict[str, List[Tuple[float, float]]],
    pred_coords_128: np.ndarray,
    missing_teeth: List[str],
) -> np.ndarray:
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Presentes anotados (verde)
    for tooth in CANONICAL_TEETH_32:
        pts = points_by_label.get(tooth)
        if pts is None or len(pts) < 2:
            continue

        x1, y1 = int(round(pts[0][0])), int(round(pts[0][1]))
        x2, y2 = int(round(pts[1][0])), int(round(pts[1][1]))
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 220, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, (0, 220, 0), -1, cv2.LINE_AA)

    # Faltantes reconstruidos (vermelho)
    h, w = img_gray.shape[:2]
    pred = np.clip(pred_coords_128.reshape(32, 4), 0.0, 1.0)
    missing_set = set(missing_teeth)
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        if tooth not in missing_set:
            continue

        x1, y1 = _denormalize_xy(pred[i, 0], pred[i, 1], (h, w))
        x2, y2 = _denormalize_xy(pred[i, 2], pred[i, 3], (h, w))
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        cv2.line(canvas, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, (0, 0, 255), -1, cv2.LINE_AA)
        lx = int(round((p1[0] + p2[0]) * 0.5 + 2))
        ly = int(round((p1[1] + p2[1]) * 0.5 - 2))
        cv2.putText(canvas, tooth, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1, cv2.LINE_AA)

    return canvas


def _write_html(out_dir: Path, run_name: str, cards: List[str], total: int) -> Path:
    html_path = out_dir / "index.html"
    body = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Missing Teeth Imputation - {html.escape(run_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1115; color: #e8edf2; }}
    .header {{ margin-bottom: 14px; }}
    .legend {{ font-size: 13px; opacity: 0.9; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }}
    .card {{ background: #171b22; border: 1px solid #2a3340; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; }}
    .meta {{ padding: 8px 10px; font-size: 12px; line-height: 1.45; }}
    a {{ color: #7eb7ff; }}
  </style>
</head>
<body>
  <div class='header'>
    <h2>Missing Teeth Imputation</h2>
    <div class='legend'>Anotado presente = <span style='color:#40d16f;'>verde</span> | Faltante reconstruido = <span style='color:#ff4d4d;'>vermelho</span></div>
    <div class='legend'>run={html.escape(run_name)} | cases={total}</div>
  </div>
  <div class='grid'>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar imputacao em exames com dentes ausentes")
    parser.add_argument("--config", type=Path, default=Path("dae_longoeixo/dae_train_config.json"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    curves_json_dir_cfg = cfg["paths"].get("curves_json_dir")
    curves_json_dir = _resolve_path(repo_root, curves_json_dir_cfg) if curves_json_dir_cfg else None
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])
    preset_path = _resolve_path(repo_root, cfg["paths"]["preset_path"])
    preset = load_json(preset_path)

    if args.checkpoint is not None:
        ckpt_path = _resolve_path(repo_root, str(args.checkpoint))
        run_name = args.run_name or ckpt_path.parent.name
        run_dir = ckpt_path.parent
    else:
        run_name = args.run_name or _latest_run_name(output_base_dir)
        if not run_name:
            raise FileNotFoundError(f"Nenhuma run encontrada em {output_base_dir / 'runs'}")
        run_dir = output_base_dir / "runs" / run_name
        ckpt_path = run_dir / "best.ckpt"

    if args.out_dir is not None:
        out_dir = _resolve_path(repo_root, str(args.out_dir))
    else:
        out_dir = run_dir / "imputation_on_real_missing"

    img_map = {p.stem: p for p in imgs_dir.glob("*.jpg")}
    json_map = {p.stem: p for p in json_dir.glob("*.json")}
    stems = sorted(set(img_map).intersection(json_map))

    device_name = str(cfg["training"].get("device", "auto"))
    device = _auto_device() if device_name == "auto" else torch.device(device_name)

    model_cfg = preset.get("model", {})
    include_mask = bool(model_cfg.get("include_point_mask_in_input", True))
    include_curves = bool(model_cfg.get("include_alveolar_curves_in_input", False))
    n_curve_points = int(model_cfg.get("n_curve_points", 128))
    curves_dim = 4 * n_curve_points
    reconstruct_target = str(model_cfg.get("reconstruct_target", "teeth_only"))
    output_dim = 128 + (curves_dim if reconstruct_target == "teeth_plus_curves" else 0)
    input_dim = 128 + (64 if include_mask else 0) + (curves_dim if include_curves else 0)

    model = CoordinateDenoisingAutoencoder(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(int(v) for v in model_cfg.get("hidden_dims", [512, 256])),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_activation=str(model_cfg.get("output_activation", "sigmoid")),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    cards: List[str] = []
    used = 0

    for stem in stems:
        if used >= args.max_cases:
            break

        img_path = img_map[stem]
        json_path = json_map[stem]

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        pb = _load_points_by_label(json_path)
        partial128, missing_mask32, present_teeth, missing_teeth = _build_partial_coords_and_missing(pb, img_gray.shape)

        if not missing_teeth:
            continue

        x_input, _, _ = build_noisy_input(
            clean_coords_128=partial128,
            knocked_teeth_mask_32=missing_mask32,
            include_point_mask_in_input=include_mask,
        )

        if include_curves:
            curves_flat = None
            if curves_json_dir is not None:
                curves_path = curves_json_dir / f"{stem}.json"
                if curves_path.exists():
                    curves_flat = _load_alveolar_curves_flat(curves_path, n_curve_points=n_curve_points, src_hw=img_gray.shape)
            if curves_flat is None:
                curves_flat = np.zeros((curves_dim,), dtype=np.float32)
            x_input = np.concatenate([x_input, curves_flat], axis=0).astype(np.float32, copy=False)

        xt = torch.from_numpy(x_input[None, ...]).to(device)
        with torch.no_grad():
            pred = model(xt)["coords_pred"][0].detach().cpu().numpy()

        overlay = _draw_overlay(
            img_gray=img_gray,
            points_by_label=pb,
            pred_coords_128=pred,
            missing_teeth=missing_teeth,
        )

        out_name = f"{used:04d}_{stem}.png"
        out_path = out_images / out_name
        cv2.imwrite(str(out_path), overlay)

        cards.append(
            (
                "<div class='card'>"
                f"<img src='images/{html.escape(out_name)}' loading='lazy'/>"
                "<div class='meta'>"
                f"<div><b>{html.escape(stem)}</b></div>"
                f"<div>present={len(present_teeth)} | missing={len(missing_teeth)}</div>"
                f"<div>missing_teeth={html.escape(','.join(missing_teeth))}</div>"
                "</div></div>"
            )
        )

        used += 1

    html_path = _write_html(out_dir=out_dir, run_name=run_name, cards=cards, total=used)

    print(f"[RUN] name={run_name}")
    print(f"[RUN] checkpoint={ckpt_path}")
    print(f"[OUT] cases={used}")
    print(f"[OUT] html={html_path}")
    print(f"[OUT] images_dir={out_images}")


if __name__ == "__main__":
    main()
