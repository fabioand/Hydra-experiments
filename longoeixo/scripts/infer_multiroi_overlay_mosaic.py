#!/usr/bin/env python3
"""Inferencia qualitativa Multi-ROI (center + lateral compartilhada) com mosaico HTML.

Pipeline:
- Seleciona N radiografias do dataset.
- Roda modelo center24 no ROI CENTER.
- Roda modelo lateral_shared20 no ROI LEFT (direto) e ROI RIGHT (flip horizontal).
- Decodifica argmax por canal, aplica threshold de presenca (score=min(peak_p1,peak_p2)).
- Mapeia coordenadas de ROI->global corretamente.
- Desenha longo-eixos (linha p1-p2) em vermelho e salva overlay.
- Gera index.html em mosaico para inspeção visual.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from hydra_data import discover_samples, fixed_three_windows, load_json
from hydra_multitask_model import HydraUNetMultiTask
from longoeixo.scripts.roi_lateral_shared_config import (
    CENTER_TEETH,
    LATERAL_RIGHT_TEETH,
    lateral_restore_left_inference,
    lateral_restore_right_inference,
    remap_right_to_left,
)


DEFAULT_CENTER_CKPT = Path(
    "longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1/runs/"
    "center24_sharedflip_v1_nopres_absenthm1_full_mps/best.ckpt"
)
DEFAULT_LATERAL_CKPT = Path(
    "longoeixo/checkpoints/ec2_lateral_shared20/"
    "lateral20_v1_fixedorient_nopres_absenthm1_16k_ft_best_ep29.ckpt"
)
DEFAULT_OUTPUT_DIR = Path(
    "longoeixo/experiments/hydra_roi_fixed_shared_lateral/"
    "qualitative_multiroi_100_centerlocal_lateralec2_ep29"
)
TARGET_HW = (256, 256)


@dataclass
class ToothPred:
    tooth: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    score: float
    peak_p1: float
    peak_p2: float
    source: str


@dataclass
class ChannelDebugItem:
    label: str
    hm_global: np.ndarray
    point_global: Tuple[float, float]


def _resolve_path(root: Path, value: Path | str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _latest_best_ckpt(output_base_dir: Path) -> Path:
    latest_file = output_base_dir / "latest_run.txt"
    if latest_file.exists():
        run_name = latest_file.read_text(encoding="utf-8").strip()
        if run_name:
            cand = output_base_dir / "runs" / run_name / "best.ckpt"
            if cand.exists():
                return cand
    runs_dir = output_base_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir nao encontrado: {runs_dir}")
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for rd in run_dirs:
        cand = rd / "best.ckpt"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"best.ckpt nao encontrado em {runs_dir}")


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(
    ckpt_path: Path,
    heatmap_channels: int,
    presence_channels: int,
    default_presence_head: bool,
    device: torch.device,
) -> HydraUNetMultiTask:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    backbone = str(model_cfg.get("backbone", "resnet34"))
    presence_dropout = float(model_cfg.get("presence_dropout", 0.1))
    enable_presence_head = bool(train_cfg.get("use_presence_head", default_presence_head))

    model = HydraUNetMultiTask(
        in_channels=1,
        heatmap_out_channels=heatmap_channels,
        presence_out_channels=presence_channels,
        enable_presence_head=enable_presence_head,
        backbone=backbone,
        presence_dropout=presence_dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _preprocess_crop(crop_gray: np.ndarray) -> torch.Tensor:
    tgt_h, tgt_w = TARGET_HW
    x = cv2.resize(crop_gray, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = x[None, None, ...]  # (1,1,H,W)
    return torch.from_numpy(x)


def _infer_heatmaps(model: HydraUNetMultiTask, crop_gray: np.ndarray, device: torch.device) -> np.ndarray:
    x = _preprocess_crop(crop_gray).to(device)
    with torch.no_grad():
        out = model(x)
        logits = out["heatmap_logits"][0].detach().cpu().numpy().astype(np.float32)
    return logits


def _argmax_xy_and_peak(hm: np.ndarray) -> Tuple[float, float, float]:
    idx = int(np.argmax(hm))
    h, w = hm.shape
    y = float(idx // w)
    x = float(idx % w)
    peak = float(hm.reshape(-1)[idx])
    return x, y, peak


def _subpixel_xy_and_peak(hm: np.ndarray, radius: int = 1) -> Tuple[float, float, float]:
    """Refina coordenada do pico por centro de massa local (subpixel)."""
    x0, y0, peak = _argmax_xy_and_peak(hm)
    h, w = hm.shape
    xi = int(round(x0))
    yi = int(round(y0))
    x1 = max(0, xi - radius)
    y1 = max(0, yi - radius)
    x2 = min(w, xi + radius + 1)
    y2 = min(h, yi + radius + 1)
    patch = hm[y1:y2, x1:x2].astype(np.float64, copy=False)
    s = float(patch.sum())
    if s <= 1e-12:
        return x0, y0, peak
    yy, xx = np.mgrid[y1:y2, x1:x2]
    x_ref = float((patch * xx).sum() / s)
    y_ref = float((patch * yy).sum() / s)
    return x_ref, y_ref, peak


def _map_xy_256_to_roi(x_256: float, y_256: float, roi_w: int, roi_h: int) -> Tuple[float, float]:
    # Usa a mesma convencao de centro de pixel do cv2.resize (half-pixel),
    # para manter alinhamento entre ponto (argmax) e heatmap reprojetado.
    mw, mh = TARGET_HW[1], TARGET_HW[0]
    x_local = ((x_256 + 0.5) * (float(max(1, roi_w)) / float(mw))) - 0.5
    y_local = ((y_256 + 0.5) * (float(max(1, roi_h)) / float(mh))) - 0.5
    return x_local, y_local


def _decode_roi_predictions(
    probs: np.ndarray,
    teeth_order: List[str],
    roi_w: int,
    roi_h: int,
    threshold: float,
    source: str,
    use_subpixel: bool = True,
) -> List[ToothPred]:
    preds: List[ToothPred] = []
    for i, tooth in enumerate(teeth_order):
        c0, c1 = 2 * i, 2 * i + 1
        if use_subpixel:
            x0_256, y0_256, p0 = _subpixel_xy_and_peak(probs[c0], radius=1)
            x1_256, y1_256, p1 = _subpixel_xy_and_peak(probs[c1], radius=1)
        else:
            x0_256, y0_256, p0 = _argmax_xy_and_peak(probs[c0])
            x1_256, y1_256, p1 = _argmax_xy_and_peak(probs[c1])
        score = min(p0, p1)
        if score < threshold:
            continue
        p0_local = _map_xy_256_to_roi(x0_256, y0_256, roi_w, roi_h)
        p1_local = _map_xy_256_to_roi(x1_256, y1_256, roi_w, roi_h)
        preds.append(
            ToothPred(
                tooth=tooth,
                p1=p0_local,
                p2=p1_local,
                score=float(score),
                peak_p1=float(p0),
                peak_p2=float(p1),
                source=source,
            )
        )
    return preds


def _draw_overlay(image_gray: np.ndarray, preds: List[ToothPred]) -> np.ndarray:
    out = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    return _draw_axes_on_bgr(out, preds)


def _draw_axes_on_bgr(image_bgr: np.ndarray, preds: List[ToothPred]) -> np.ndarray:
    out = image_bgr.copy()
    red = (0, 0, 255)
    for pred in preds:
        x1, y1 = int(round(pred.p1[0])), int(round(pred.p1[1]))
        x2, y2 = int(round(pred.p2[0])), int(round(pred.p2[1]))
        cv2.line(out, (x1, y1), (x2, y2), red, 2, cv2.LINE_AA)
        cv2.circle(out, (x1, y1), 3, red, -1, cv2.LINE_AA)
        cv2.circle(out, (x2, y2), 3, red, -1, cv2.LINE_AA)
        xm = int(round((x1 + x2) * 0.5))
        ym = int(round((y1 + y2) * 0.5))
        cv2.putText(out, pred.tooth, (xm + 2, ym - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, red, 1, cv2.LINE_AA)
    return out


def _max_heatmap_to_global(hm_stack: np.ndarray, rect_xyxy: List[int], full_hw: Tuple[int, int]) -> np.ndarray:
    """Projeta max dos canais de um ROI para grade global da imagem."""
    H, W = full_hw
    x1, y1, x2, y2 = rect_xyxy
    roi_h = max(1, y2 - y1)
    roi_w = max(1, x2 - x1)
    hm_max = np.max(hm_stack, axis=0).astype(np.float32)  # (256,256)
    hm_roi = cv2.resize(hm_max, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((H, W), dtype=np.float32)
    out[y1:y2, x1:x2] = hm_roi
    return out


def _single_heatmap_to_global(hm_2d: np.ndarray, rect_xyxy: List[int], full_hw: Tuple[int, int]) -> np.ndarray:
    H, W = full_hw
    x1, y1, x2, y2 = rect_xyxy
    roi_h = max(1, y2 - y1)
    roi_w = max(1, x2 - x1)
    hm_roi = cv2.resize(hm_2d.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((H, W), dtype=np.float32)
    out[y1:y2, x1:x2] = hm_roi
    return out


def _peak_point_from_global_heatmap(hm_global: np.ndarray) -> Tuple[float, float]:
    x, y, _ = _subpixel_xy_and_peak(hm_global.astype(np.float32), radius=1)
    return x, y


def _argmax_point_from_global_heatmap(hm_global: np.ndarray) -> Tuple[float, float]:
    x, y, _ = _argmax_xy_and_peak(hm_global.astype(np.float32))
    return x, y


def _draw_heatmap_fusion(image_gray: np.ndarray, hm_global_max: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    hm = hm_global_max.astype(np.float32, copy=False)
    lo = float(np.percentile(hm, 2.0))
    hi = float(np.percentile(hm, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    hm01 = np.clip((hm - lo) / (hi - lo), 0.0, 1.0)
    hm_u8 = np.clip(hm01 * 255.0, 0.0, 255.0).astype(np.uint8)
    color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    fused = cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0.0)
    return fused


def _make_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    w = left_bgr.shape[1] + gap + right_bgr.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = (12, 17, 26)
    out[0:left_bgr.shape[0], 0:left_bgr.shape[1]] = left_bgr
    out[0:right_bgr.shape[0], left_bgr.shape[1] + gap:left_bgr.shape[1] + gap + right_bgr.shape[1]] = right_bgr
    return out


def _make_horizontal_panels(panels: List[np.ndarray], gap: int = 8) -> np.ndarray:
    if not panels:
        raise ValueError("panels vazio")
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = (12, 17, 26)
    x = 0
    for i, p in enumerate(panels):
        out[0 : p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1]
        if i < len(panels) - 1:
            x += gap
    return out


def _render_channel_debug_grid(
    image_gray: np.ndarray,
    items: List[ChannelDebugItem],
    cols: int = 6,
    tile_w: int = 420,
) -> np.ndarray:
    if not items:
        base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return base

    h0, w0 = image_gray.shape
    scale = float(tile_w) / float(max(1, w0))
    tile_h = max(64, int(round(h0 * scale)))
    header_h = 22
    tile_h_total = tile_h + header_h
    rows = (len(items) + cols - 1) // cols

    canvas = np.zeros((rows * tile_h_total, cols * tile_w, 3), dtype=np.uint8)
    canvas[:, :] = (10, 14, 22)

    for i, it in enumerate(items):
        r = i // cols
        c = i % cols
        y0 = r * tile_h_total
        x0 = c * tile_w

        fused = _draw_heatmap_fusion(image_gray, it.hm_global, alpha=0.45)
        px, py = int(round(it.point_global[0])), int(round(it.point_global[1]))
        cv2.circle(fused, (px, py), 4, (20, 255, 20), -1, cv2.LINE_AA)
        cv2.circle(fused, (px, py), 8, (10, 80, 10), 1, cv2.LINE_AA)

        tile = cv2.resize(fused, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        canvas[y0 + header_h : y0 + header_h + tile_h, x0 : x0 + tile_w] = tile
        cv2.putText(
            canvas,
            it.label,
            (x0 + 6, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 240, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _to_global(pred: ToothPred, rect: List[int]) -> ToothPred:
    x1, y1, _, _ = rect
    return ToothPred(
        tooth=pred.tooth,
        p1=(pred.p1[0] + x1, pred.p1[1] + y1),
        p2=(pred.p2[0] + x1, pred.p2[1] + y1),
        score=pred.score,
        peak_p1=pred.peak_p1,
        peak_p2=pred.peak_p2,
        source=pred.source,
    )


def _infer_one_image(
    image_path: Path,
    center_model: HydraUNetMultiTask,
    lateral_model: HydraUNetMultiTask,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, List[ToothPred]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {image_path}")
    h, w = img.shape
    rects = fixed_three_windows(w, h)

    debug_items: List[ChannelDebugItem] = []

    # CENTER (incisivos+caninos)
    cx1, cy1, cx2, cy2 = rects["CENTER"]
    crop_center = img[cy1:cy2, cx1:cx2]
    hm_center = _infer_heatmaps(center_model, crop_center, device)
    preds_center: List[ToothPred] = []
    center_w = max(1, cx2 - cx1)
    center_h = max(1, cy2 - cy1)
    for i, tooth in enumerate(CENTER_TEETH):
        c0, c1 = 2 * i, 2 * i + 1
        x0_raw, y0_raw, p0 = _argmax_xy_and_peak(hm_center[c0])
        x1_raw, y1_raw, p1 = _argmax_xy_and_peak(hm_center[c1])
        score = min(p0, p1)
        if score < threshold:
            continue
        x0l, y0l = _map_xy_256_to_roi(x0_raw, y0_raw, center_w, center_h)
        x1l, y1l = _map_xy_256_to_roi(x1_raw, y1_raw, center_w, center_h)
        x0g, y0g = (x0l + cx1, y0l + cy1)
        x1g, y1g = (x1l + cx1, y1l + cy1)
        preds_center.append(
            ToothPred(
                tooth=tooth,
                p1=(x0g, y0g),
                p2=(x1g, y1g),
                score=float(score),
                peak_p1=float(p0),
                peak_p2=float(p1),
                source="center",
            )
        )
    # canais individuais center
    for i, tooth in enumerate(CENTER_TEETH):
        for pi, suffix in [(0, "p1"), (1, "p2")]:
            cidx = 2 * i + pi
            hm_g = _single_heatmap_to_global(hm_center[cidx], rects["CENTER"], (h, w))
            xg, yg = _peak_point_from_global_heatmap(hm_g)
            debug_items.append(
                ChannelDebugItem(label=f"C:{tooth}_{suffix}", hm_global=hm_g, point_global=(xg, yg))
            )

    # LATERAL - ramo canônico (direito anatômico no lado LEFT da imagem)
    lx1, ly1, lx2, ly2 = rects["LEFT"]
    crop_left = img[ly1:ly2, lx1:lx2]
    hm_left = _infer_heatmaps(lateral_model, crop_left, device)
    preds_right_global: List[ToothPred] = []
    left_w = max(1, lx2 - lx1)
    left_h = max(1, ly2 - ly1)
    for i, tooth_r in enumerate(LATERAL_RIGHT_TEETH):
        c0, c1 = 2 * i, 2 * i + 1
        x0_raw, y0_raw, p0 = _argmax_xy_and_peak(hm_left[c0])
        x1_raw, y1_raw, p1 = _argmax_xy_and_peak(hm_left[c1])
        score = min(p0, p1)
        if score < threshold:
            continue
        x0l, y0l = _map_xy_256_to_roi(x0_raw, y0_raw, left_w, left_h)
        x1l, y1l = _map_xy_256_to_roi(x1_raw, y1_raw, left_w, left_h)
        x0g, y0g = (x0l + lx1, y0l + ly1)
        x1g, y1g = (x1l + lx1, y1l + ly1)
        preds_right_global.append(
            ToothPred(
                tooth=tooth_r,
                p1=(x0g, y0g),
                p2=(x1g, y1g),
                score=float(score),
                peak_p1=float(p0),
                peak_p2=float(p1),
                source="lateral_left_direct",
            )
        )
    # canais individuais lateral canônico (LEFT)
    for i, tooth in enumerate(LATERAL_RIGHT_TEETH):
        for pi, suffix in [(0, "p1"), (1, "p2")]:
            cidx = 2 * i + pi
            hm_g = _single_heatmap_to_global(hm_left[cidx], rects["LEFT"], (h, w))
            xg, yg = _peak_point_from_global_heatmap(hm_g)
            debug_items.append(
                ChannelDebugItem(label=f"LR:{tooth}_{suffix}", hm_global=hm_g, point_global=(xg, yg))
            )

    # LATERAL - ramo espelhado (lado RIGHT flipado -> dentes esquerdos)
    rx1, ry1, rx2, ry2 = rects["RIGHT"]
    crop_right = img[ry1:ry2, rx1:rx2]
    crop_right_flip = np.ascontiguousarray(np.fliplr(crop_right))
    hm_right_flip = _infer_heatmaps(lateral_model, crop_right_flip, device)
    preds_left_global: List[ToothPred] = []
    right_w = max(1, rx2 - rx1)
    right_h = max(1, ry2 - ry1)
    for i, tooth_r in enumerate(LATERAL_RIGHT_TEETH):
        c0, c1 = 2 * i, 2 * i + 1
        x0_flip_raw, y0_raw, p0 = _argmax_xy_and_peak(hm_right_flip[c0])
        x1_flip_raw, y1_raw, p1 = _argmax_xy_and_peak(hm_right_flip[c1])
        score = min(p0, p1)
        if score < threshold:
            continue
        tooth_l = remap_right_to_left(tooth_r)
        x0_raw = float(TARGET_HW[1] - 1) - x0_flip_raw
        x1_raw = float(TARGET_HW[1] - 1) - x1_flip_raw
        x0l, y0l = _map_xy_256_to_roi(x0_raw, y0_raw, right_w, right_h)
        x1l, y1l = _map_xy_256_to_roi(x1_raw, y1_raw, right_w, right_h)
        x0g, y0g = (x0l + rx1, y0l + ry1)
        x1g, y1g = (x1l + rx1, y1l + ry1)
        preds_left_global.append(
            ToothPred(
                tooth=tooth_l,
                p1=(x0g, y0g),
                p2=(x1g, y1g),
                score=float(score),
                peak_p1=float(p0),
                peak_p2=float(p1),
                source="lateral_right_flipped_restore",
            )
        )
    # canais individuais lateral espelhado (RIGHT flip -> unflip -> LEFT anatomy)
    for i, tooth_r in enumerate(LATERAL_RIGHT_TEETH):
        for pi, suffix in [(0, "p1"), (1, "p2")]:
            cidx = 2 * i + pi
            tooth_l = remap_right_to_left(tooth_r)
            hm_unflip = np.flip(hm_right_flip[cidx], axis=1)
            hm_g = _single_heatmap_to_global(hm_unflip, rects["RIGHT"], (h, w))
            xg, yg = _peak_point_from_global_heatmap(hm_g)
            debug_items.append(
                ChannelDebugItem(label=f"LL:{tooth_l}_{suffix}", hm_global=hm_g, point_global=(xg, yg))
            )

    preds_all = preds_center + preds_right_global + preds_left_global
    overlay_axes = _draw_overlay(img, preds_all)

    # Projeta max-map global de cada ramo e combina por max.
    hm_center_global = _max_heatmap_to_global(hm_center, rects["CENTER"], (h, w))
    hm_left_global = _max_heatmap_to_global(hm_left, rects["LEFT"], (h, w))
    # Importante: hm_right_flip está no espaço do input flipado do ROI RIGHT.
    # Para projetar corretamente na imagem original, desfazemos o flip no eixo X.
    hm_right_unflip = np.flip(hm_right_flip, axis=2)
    hm_right_global = _max_heatmap_to_global(hm_right_unflip, rects["RIGHT"], (h, w))
    hm_global = np.maximum.reduce([hm_center_global, hm_left_global, hm_right_global])
    overlay_hm = _draw_heatmap_fusion(img, hm_global, alpha=0.50)
    overlay_hm_axes = _draw_axes_on_bgr(overlay_hm, preds_all)

    panel = _make_horizontal_panels([overlay_axes, overlay_hm_axes], gap=10)
    debug_grid = _render_channel_debug_grid(img, debug_items, cols=6, tile_w=420)
    return panel, preds_all, debug_grid


def _write_html(output_dir: Path, records: List[Dict]) -> Path:
    html_path = output_dir / "index.html"
    items = []
    for rec in records:
        rel = rec["overlay_file"]
        stem = rec["stem"]
        n = rec["num_predicted_teeth"]
        debug_link = rec.get("channel_debug_file", "")
        debug_html = f'<br/><a href="{debug_link}" target="_blank">channel debug</a>' if debug_link else ""
        items.append(
            f"""
            <div class="card">
              <img src="{rel}" alt="{stem}" loading="lazy" />
              <div class="meta"><b>{stem}</b><br/>pred teeth: {n}{debug_html}</div>
            </div>
            """
        )
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Hydra Multi-ROI Qualitativo</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0f1115; color:#e7e9ee; margin:16px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:12px; }}
    .card {{ background:#1a1f2b; border:1px solid #2a3142; border-radius:8px; overflow:hidden; }}
    img {{ width:100%; height:auto; display:block; background:#000; }}
    .meta {{ padding:8px 10px; font-size:12px; color:#c8cfdd; }}
  </style>
</head>
<body>
  <h2>Hydra Multi-ROI - 100 amostras</h2>
  <p>[1] radiografia+eixos | [2] heatmap global (logits crus)+eixos | threshold={records[0]["threshold"] if records else "n/a"}</p>
  <div class="grid">
    {''.join(items)}
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia qualitativa Multi-ROI (center+lateral) com mosaico HTML")
    parser.add_argument("--center-ckpt", type=Path, default=DEFAULT_CENTER_CKPT)
    parser.add_argument("--lateral-ckpt", type=Path, default=DEFAULT_LATERAL_CKPT)
    parser.add_argument(
        "--center-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1"),
        help="Output dir base da center para resolver automaticamente latest best.ckpt.",
    )
    parser.add_argument(
        "--lateral-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/lateral_shared20_nopres_absenthm1"),
        help="Output dir base da lateral para resolver automaticamente latest best.ckpt.",
    )
    parser.add_argument(
        "--use-latest-from-output-dirs",
        action="store_true",
        help="Se ligado, ignora --center-ckpt/--lateral-ckpt e usa latest_run.txt + best.ckpt.",
    )
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--json-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--export-channel-debug",
        action="store_true",
        help="Exporta painel por canal (canal fundido + ponto) para auditoria fina.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="JSON de split (ex.: longoeixo/splits_70_15_15_seed123.json). Se informado, usa --split-name.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Grupo do split a processar quando --split-path for informado.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.use_latest_from_output_dirs:
        center_ckpt = _latest_best_ckpt(_resolve_path(repo_root, args.center_output_dir))
        lateral_ckpt = _latest_best_ckpt(_resolve_path(repo_root, args.lateral_output_dir))
    else:
        center_ckpt = _resolve_path(repo_root, args.center_ckpt)
        lateral_ckpt = _resolve_path(repo_root, args.lateral_ckpt)
    imgs_dir = _resolve_path(repo_root, args.imgs_dir)
    json_dir = _resolve_path(repo_root, args.json_dir)
    output_dir = _resolve_path(repo_root, args.output_dir)
    overlays_dir = output_dir / "overlays"
    preds_dir = output_dir / "predictions_json"
    debug_dir = output_dir / "channel_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    if args.export_channel_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    device = _auto_device()
    print(f"[INFO] device={device}")
    print(f"[INFO] center_ckpt={center_ckpt}")
    print(f"[INFO] lateral_ckpt={lateral_ckpt}")

    center_model = _load_model(
        ckpt_path=center_ckpt,
        heatmap_channels=24,
        presence_channels=12,
        default_presence_head=True,
        device=device,
    )
    lateral_model = _load_model(
        ckpt_path=lateral_ckpt,
        heatmap_channels=20,
        presence_channels=10,
        default_presence_head=False,
        device=device,
    )

    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        masks_dir=None,
        source_mode="on_the_fly",
    )
    if not samples:
        raise RuntimeError("Nenhuma amostra encontrada para inferencia.")

    pool = sorted(samples, key=lambda s: s.stem)
    by_stem = {s.stem: s for s in pool}

    if args.split_path is not None:
        split_path = _resolve_path(repo_root, args.split_path)
        split_obj = load_json(split_path)
        stems = list(split_obj.get(args.split_name, []))
        chosen = [by_stem[s] for s in stems if s in by_stem]
        chosen = sorted(chosen, key=lambda s: s.stem)
        print(
            f"[INFO] split={args.split_name} from {split_path} | "
            f"selected_images={len(chosen)} / listed={len(stems)} / total={len(pool)}"
        )
    else:
        rng = random.Random(args.seed)
        n = min(int(args.num_images), len(pool))
        chosen = rng.sample(pool, k=n) if n < len(pool) else pool
        chosen = sorted(chosen, key=lambda s: s.stem)
        print(f"[INFO] selected_images={len(chosen)} / total={len(pool)}")

    records: List[Dict] = []
    for i, sample in enumerate(chosen, start=1):
        overlay, preds, debug_grid = _infer_one_image(
            image_path=sample.image_path,
            center_model=center_model,
            lateral_model=lateral_model,
            device=device,
            threshold=float(args.threshold),
        )
        overlay_name = f"{sample.stem}_overlay.png"
        overlay_path = overlays_dir / overlay_name
        cv2.imwrite(str(overlay_path), overlay)

        debug_file_rel = ""
        if args.export_channel_debug:
            dbg_name = f"{sample.stem}_channels.png"
            dbg_path = debug_dir / dbg_name
            cv2.imwrite(str(dbg_path), debug_grid)
            debug_file_rel = f"channel_debug/{dbg_name}"

        pred_payload = {
            "stem": sample.stem,
            "image_path": str(sample.image_path),
            "threshold": float(args.threshold),
            "predictions": [
                {
                    "tooth": p.tooth,
                    "p1": [float(p.p1[0]), float(p.p1[1])],
                    "p2": [float(p.p2[0]), float(p.p2[1])],
                    "score": float(p.score),
                    "peak_p1": float(p.peak_p1),
                    "peak_p2": float(p.peak_p2),
                    "source": p.source,
                }
                for p in preds
            ],
        }
        (preds_dir / f"{sample.stem}.json").write_text(
            json.dumps(pred_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        records.append(
            {
                "stem": sample.stem,
                "overlay_file": f"overlays/{overlay_name}",
                "channel_debug_file": debug_file_rel,
                "num_predicted_teeth": len(preds),
                "threshold": float(args.threshold),
            }
        )
        if i % 10 == 0 or i == len(chosen):
            print(f"[INFO] processed {i}/{len(chosen)}")

    html_path = _write_html(output_dir, records)
    summary = {
        "num_selected_images": len(chosen),
        "threshold": float(args.threshold),
        "center_ckpt": str(center_ckpt),
        "lateral_ckpt": str(lateral_ckpt),
        "output_dir": str(output_dir),
        "html": str(html_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] overlays_dir={overlays_dir}")
    print(f"[DONE] html={html_path}")


if __name__ == "__main__":
    main()
