#!/usr/bin/env python3
"""Gera mosaico HTML com overlays de GT (verde) e predicao (vermelho) para top erros de presenca."""

from __future__ import annotations

import argparse
import csv
import html
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from hydra_data import discover_samples, load_json
from hydra_multitask_model import CANONICAL_TEETH_32, HydraUNetMultiTask


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


def _load_gt_lines(json_path: Path) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    data = load_json(json_path)
    out: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for ann in data:
        label = str(ann.get("label", ""))
        if label not in CANONICAL_TEETH_32:
            continue
        pts = ann.get("pts", [])
        valid: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid.append((float(x), float(y)))
        if len(valid) >= 2 and label not in out:
            out[label] = (valid[0], valid[1])
    return out


def _preprocess_image(img_gray: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    x = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return x[None, None, ...]


def _target_to_original(x_t: float, y_t: float, src_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> Tuple[float, float]:
    src_h, src_w = src_hw
    tgt_h, tgt_w = target_hw
    if tgt_h <= 1 or tgt_w <= 1:
        return 0.0, 0.0
    x = (x_t / float(tgt_w - 1)) * float(src_w - 1)
    y = (y_t / float(tgt_h - 1)) * float(src_h - 1)
    return x, y


def _predict_lines(
    model: HydraUNetMultiTask,
    device: torch.device,
    img_gray: np.ndarray,
    target_hw: Tuple[int, int],
    threshold: float,
    draw_only_pred_present: bool,
) -> Tuple[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]], np.ndarray]:
    x = _preprocess_image(img_gray, target_hw=target_hw)
    xt = torch.from_numpy(x).to(device)

    with torch.no_grad():
        pred = model(xt)
        p_score = torch.sigmoid(pred["presence_logits"])[0].detach().cpu().numpy()
        h_score = torch.sigmoid(pred["heatmap_logits"])[0].detach().cpu().numpy()  # (64,H,W)

    tgt_h, tgt_w = target_hw
    src_hw = img_gray.shape[:2]
    out: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        if draw_only_pred_present and p_score[i] < threshold:
            continue
        c0 = 2 * i
        c1 = c0 + 1
        idx0 = int(np.argmax(h_score[c0]))
        idx1 = int(np.argmax(h_score[c1]))
        y0_t, x0_t = divmod(idx0, tgt_w)
        y1_t, x1_t = divmod(idx1, tgt_w)

        x0, y0 = _target_to_original(float(x0_t), float(y0_t), src_hw=src_hw, target_hw=target_hw)
        x1, y1 = _target_to_original(float(x1_t), float(y1_t), src_hw=src_hw, target_hw=target_hw)
        out[tooth] = ((x0, y0), (x1, y1))

    return out, p_score


def _draw_overlay(
    img_gray: np.ndarray,
    gt_lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    pred_lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
) -> np.ndarray:
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # GT em verde.
    for tooth, (p1, p2) in gt_lines.items():
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, (0, 255, 0), -1, cv2.LINE_AA)
        lx = int(round((x1 + x2) * 0.5 + 2))
        ly = int(round((y1 + y2) * 0.5 - 2))
        cv2.putText(canvas, tooth, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 0), 1, cv2.LINE_AA)

    # Predicao em vermelho.
    for tooth, (p1, p2) in pred_lines.items():
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, (0, 0, 255), -1, cv2.LINE_AA)
        lx = int(round((x1 + x2) * 0.5 + 2))
        ly = int(round((y1 + y2) * 0.5 + 10))
        cv2.putText(canvas, tooth, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar top erros de presenca com overlay de longoeixos")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ranking-csv", type=Path, default=None, help="CSV top errors do eval_presence_top_errors.py")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=None, help="Threshold para considerar dente predito presente")
    parser.add_argument("--draw-all-pred", action="store_true", help="Se ativo, desenha todos os 32 longoeixos preditos")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])

    run_name = args.run_name or _latest_run_name(output_base_dir)
    if not run_name:
        raise FileNotFoundError("Nao foi possivel determinar run_name; passe --run-name.")
    run_dir = output_base_dir / "runs" / run_name

    ckpt_path = _resolve_path(repo_root, str(args.checkpoint)) if args.checkpoint else (run_dir / "best.ckpt")
    ranking_csv = (
        _resolve_path(repo_root, str(args.ranking_csv))
        if args.ranking_csv
        else (run_dir / "eval_presence" / f"presence_top_errors_top{int(args.top_k)}.csv")
    )
    if not ranking_csv.exists():
        raise FileNotFoundError(f"Ranking CSV nao encontrado: {ranking_csv}")

    target_size = cfg.get("input", {}).get("size")
    if target_size is None:
        preset_path = _resolve_path(repo_root, cfg["paths"]["preset_path"])
        preset = load_json(preset_path)
        target_size = preset.get("input", {}).get("size", [256, 256])
    target_hw = (int(target_size[0]), int(target_size[1]))

    threshold = float(args.threshold) if args.threshold is not None else float(cfg.get("evaluation", {}).get("threshold", 0.5))
    draw_only_pred_present = not bool(args.draw_all_pred)

    samples = discover_samples(imgs_dir=imgs_dir, json_dir=json_dir, masks_dir=None, source_mode="on_the_fly")
    by_stem = {s.stem: s for s in samples}

    device_name = str(cfg["training"].get("device", "auto"))
    device = _auto_device() if device_name == "auto" else torch.device(device_name)
    print(f"[DEVICE] using {device}")

    model = HydraUNetMultiTask(
        in_channels=1,
        heatmap_out_channels=64,
        presence_out_channels=32,
        backbone=cfg["model"].get("backbone", "resnet34"),
        presence_dropout=float(cfg["model"].get("presence_dropout", 0.1)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = run_dir / "eval_presence_overlay"
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    with ranking_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ranking_rows = list(reader)
    ranking_rows = ranking_rows[: max(1, int(args.top_k))]

    cards: List[str] = []
    t0 = time.time()
    n = len(ranking_rows)
    print(f"[OVERLAY] run={run_name} total_cases={n} threshold={threshold:.3f}")

    for i, row in enumerate(ranking_rows, start=1):
        stem = row.get("stem", "")
        sample = by_stem.get(stem)
        if sample is None:
            continue

        img_gray = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        gt_lines = _load_gt_lines(sample.json_path)
        pred_lines, _ = _predict_lines(
            model=model,
            device=device,
            img_gray=img_gray,
            target_hw=target_hw,
            threshold=threshold,
            draw_only_pred_present=draw_only_pred_present,
        )
        overlay = _draw_overlay(img_gray, gt_lines=gt_lines, pred_lines=pred_lines)

        out_name = f"{i:04d}_{stem}.png"
        out_path = img_dir / out_name
        cv2.imwrite(str(out_path), overlay)

        subtitle = (
            f"score={row.get('suspect_score','')} | FN={row.get('fn_count','')} | "
            f"FP={row.get('fp_count','')} | err={row.get('presence_error_count','')}"
        )
        cards.append(
            (
                "<div class='card'>"
                f"<img src='images/{html.escape(out_name)}' loading='lazy'/>"
                "<div class='meta'>"
                f"<div class='stem'>{html.escape(stem)}</div>"
                f"<div class='sub'>{html.escape(subtitle)}</div>"
                "</div></div>"
            )
        )

        if i == 1 or i % 20 == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-8)
            eta = (n - i) / max(rate, 1e-8)
            print(f"[OVERLAY_PROGRESS] {i}/{n} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    html_path = out_dir / "index.html"
    html_body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Presence Top Errors Overlay - {html.escape(run_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1115; color: #e8edf2; }}
    .header {{ margin-bottom: 14px; }}
    .legend {{ font-size: 13px; opacity: 0.9; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
    .card {{ background: #171b22; border: 1px solid #2a3340; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; }}
    .meta {{ padding: 8px 10px; }}
    .stem {{ font-size: 13px; font-weight: 600; word-break: break-all; }}
    .sub {{ font-size: 12px; opacity: 0.85; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="header">
    <h2>Presence Top Errors Overlay</h2>
    <div class="legend">GT long-axis = <span style="color:#40d16f;">green</span> | Predicted long-axis = <span style="color:#ff4d4d;">red</span></div>
    <div class="legend">run={html.escape(run_name)} | top_k={len(cards)} | draw_only_pred_present={str(draw_only_pred_present)} | threshold={threshold:.3f}</div>
  </div>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    html_path.write_text(html_body, encoding="utf-8")

    print(f"[OVERLAY] html={html_path}")
    print(f"[OVERLAY] images_dir={img_dir}")


if __name__ == "__main__":
    main()
