#!/usr/bin/env python3
"""Biblioteca de inferencia composta Multi-ROI para longo-eixo.

Encapsula o pipeline center + lateral compartilhada:
- split fixo da panoramica em 3 ROIs (LEFT/CENTER/RIGHT)
- inferencia center no ROI CENTER
- inferencia lateral no ROI LEFT (direto) e ROI RIGHT (flip horizontal)
- mapeamento correto de coordenadas (argmax em logits crus) para a panoramica
- composicao de heatmap global por maximo entre ramos
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from hydra_data import fixed_three_windows
from hydra_multitask_model import HydraUNetMultiTask
from longoeixo.scripts.roi_lateral_shared_config import (
    CENTER_TEETH,
    LATERAL_RIGHT_TEETH,
    remap_right_to_left,
)


DEFAULT_CENTER_CKPT = Path(
    "hydra-checkpoints/multiROI/center/center20shared_16k_stable3.ckpt"
)
DEFAULT_LATERAL_CKPT = Path(
    "hydra-checkpoints/multiROI/lateral/lateral20_v1_fixedorient_nopres_absenthm1_16k_ft_stable_ep60_best.ckpt"
)

TARGET_HW: Tuple[int, int] = (256, 256)
CENTER_ROI_KEY = "CENTER"
LEFT_ROI_KEY = "LEFT"
RIGHT_ROI_KEY = "RIGHT"

LATERAL_LEFT_TEETH: Tuple[str, ...] = tuple(remap_right_to_left(t) for t in LATERAL_RIGHT_TEETH)
TEETH_BY_BRANCH: Dict[str, Tuple[str, ...]] = {
    "center": tuple(CENTER_TEETH),
    "lateral_left_direct": tuple(LATERAL_RIGHT_TEETH),
    "lateral_right_flipped_restore": LATERAL_LEFT_TEETH,
}
ROI_BY_BRANCH: Dict[str, str] = {
    "center": CENTER_ROI_KEY,
    "lateral_left_direct": LEFT_ROI_KEY,
    "lateral_right_flipped_restore": RIGHT_ROI_KEY,
}


@dataclass
class ToothPrediction:
    tooth: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    score: float
    peak_p1: float
    peak_p2: float
    source: str


@dataclass
class MultiROIHeatmaps:
    center_logits: np.ndarray
    left_logits: np.ndarray
    right_logits_flip: np.ndarray
    right_logits_unflip: np.ndarray
    center_global_max: np.ndarray
    left_global_max: np.ndarray
    right_global_max: np.ndarray
    global_max: np.ndarray


@dataclass
class MultiROIInferenceResult:
    image_hw: Tuple[int, int]
    rects: Dict[str, List[int]]
    predictions: List[ToothPrediction]
    heatmaps: MultiROIHeatmaps


@dataclass
class MultiROIModels:
    center: HydraUNetMultiTask
    lateral: HydraUNetMultiTask
    device: torch.device


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(root: Path, value: Path | str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def latest_best_ckpt(output_base_dir: Path) -> Path:
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


def load_multiroi_models(
    center_ckpt: Path,
    lateral_ckpt: Path,
    device: torch.device | None = None,
) -> MultiROIModels:
    dev = auto_device() if device is None else device
    center_model = _load_model(
        ckpt_path=center_ckpt,
        heatmap_channels=24,
        presence_channels=12,
        default_presence_head=True,
        device=dev,
    )
    lateral_model = _load_model(
        ckpt_path=lateral_ckpt,
        heatmap_channels=20,
        presence_channels=10,
        default_presence_head=False,
        device=dev,
    )
    return MultiROIModels(center=center_model, lateral=lateral_model, device=dev)


def _preprocess_crop(crop_gray: np.ndarray) -> torch.Tensor:
    tgt_h, tgt_w = TARGET_HW
    x = cv2.resize(crop_gray, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = x[None, None, ...]
    return torch.from_numpy(x)


def _infer_heatmap_logits(model: HydraUNetMultiTask, crop_gray: np.ndarray, device: torch.device) -> np.ndarray:
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


def _centroid_xy(hm: np.ndarray) -> Tuple[float, float]:
    hm_f = hm.astype(np.float64, copy=False)
    mass = float(hm_f.sum())
    if mass <= 1e-12:
        x, y, _ = _argmax_xy_and_peak(hm_f.astype(np.float32))
        return x, y
    h, w = hm_f.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = float((hm_f * xx).sum() / mass)
    y = float((hm_f * yy).sum() / mass)
    return x, y


def _argmax_xy_and_peak_sigmoid_centroid(hm_logits: np.ndarray) -> Tuple[float, float, float]:
    logits_f = hm_logits.astype(np.float32, copy=False)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_f, -60.0, 60.0)))
    x, y = _centroid_xy(probs)
    peak = float(np.max(probs))
    return x, y, peak


def centroid_xy_and_peak_from_sigmoid_logits(hm_logits: np.ndarray) -> Tuple[float, float, float]:
    """Alternativa de ponto central: centróide em sigmoid(logits)."""
    return _argmax_xy_and_peak_sigmoid_centroid(hm_logits)


def _map_xy_256_to_roi(x_256: float, y_256: float, roi_w: int, roi_h: int) -> Tuple[float, float]:
    mw, mh = TARGET_HW[1], TARGET_HW[0]
    x_local = ((x_256 + 0.5) * (float(max(1, roi_w)) / float(mw))) - 0.5
    y_local = ((y_256 + 0.5) * (float(max(1, roi_h)) / float(mh))) - 0.5
    return x_local, y_local


def _max_heatmap_to_global(hm_stack: np.ndarray, rect_xyxy: List[int], full_hw: Tuple[int, int]) -> np.ndarray:
    h_img, w_img = full_hw
    x1, y1, x2, y2 = rect_xyxy
    roi_h = max(1, y2 - y1)
    roi_w = max(1, x2 - x1)
    hm_max = np.max(hm_stack, axis=0).astype(np.float32)
    hm_roi = cv2.resize(hm_max, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h_img, w_img), dtype=np.float32)
    out[y1:y2, x1:x2] = hm_roi
    return out


def infer_multiroi_from_image(
    image_gray: np.ndarray,
    models: MultiROIModels,
    threshold: float = 0.1,
) -> MultiROIInferenceResult:
    if image_gray.ndim != 2:
        raise ValueError(f"Esperado imagem grayscale 2D, recebido shape={tuple(image_gray.shape)}")

    h, w = image_gray.shape
    rects = fixed_three_windows(w, h)

    cx1, cy1, cx2, cy2 = rects[CENTER_ROI_KEY]
    lx1, ly1, lx2, ly2 = rects[LEFT_ROI_KEY]
    rx1, ry1, rx2, ry2 = rects[RIGHT_ROI_KEY]

    crop_center = image_gray[cy1:cy2, cx1:cx2]
    crop_left = image_gray[ly1:ly2, lx1:lx2]
    crop_right = image_gray[ry1:ry2, rx1:rx2]
    crop_right_flip = np.ascontiguousarray(np.fliplr(crop_right))

    center_logits = _infer_heatmap_logits(models.center, crop_center, models.device)
    left_logits = _infer_heatmap_logits(models.lateral, crop_left, models.device)
    right_logits_flip = _infer_heatmap_logits(models.lateral, crop_right_flip, models.device)

    preds_center: List[ToothPrediction] = []
    center_w = max(1, cx2 - cx1)
    center_h = max(1, cy2 - cy1)
    for i, tooth in enumerate(CENTER_TEETH):
        c0, c1 = 2 * i, 2 * i + 1
        x0_raw, y0_raw, p0 = _argmax_xy_and_peak(center_logits[c0])
        x1_raw, y1_raw, p1 = _argmax_xy_and_peak(center_logits[c1])
        score = min(p0, p1)
        if score < threshold:
            continue
        x0l, y0l = _map_xy_256_to_roi(x0_raw, y0_raw, center_w, center_h)
        x1l, y1l = _map_xy_256_to_roi(x1_raw, y1_raw, center_w, center_h)
        preds_center.append(
            ToothPrediction(
                tooth=tooth,
                p1=(x0l + cx1, y0l + cy1),
                p2=(x1l + cx1, y1l + cy1),
                score=float(score),
                peak_p1=float(p0),
                peak_p2=float(p1),
                source="center",
            )
        )

    preds_right: List[ToothPrediction] = []
    left_w = max(1, lx2 - lx1)
    left_h = max(1, ly2 - ly1)
    for i, tooth_r in enumerate(LATERAL_RIGHT_TEETH):
        c0, c1 = 2 * i, 2 * i + 1
        x0_raw, y0_raw, p0 = _argmax_xy_and_peak(left_logits[c0])
        x1_raw, y1_raw, p1 = _argmax_xy_and_peak(left_logits[c1])
        score = min(p0, p1)
        if score < threshold:
            continue
        x0l, y0l = _map_xy_256_to_roi(x0_raw, y0_raw, left_w, left_h)
        x1l, y1l = _map_xy_256_to_roi(x1_raw, y1_raw, left_w, left_h)
        preds_right.append(
            ToothPrediction(
                tooth=tooth_r,
                p1=(x0l + lx1, y0l + ly1),
                p2=(x1l + lx1, y1l + ly1),
                score=float(score),
                peak_p1=float(p0),
                peak_p2=float(p1),
                source="lateral_left_direct",
            )
        )

    preds_left: List[ToothPrediction] = []
    right_w = max(1, rx2 - rx1)
    right_h = max(1, ry2 - ry1)
    for i, tooth_r in enumerate(LATERAL_RIGHT_TEETH):
        c0, c1 = 2 * i, 2 * i + 1
        x0_flip_raw, y0_raw, p0 = _argmax_xy_and_peak(right_logits_flip[c0])
        x1_flip_raw, y1_raw, p1 = _argmax_xy_and_peak(right_logits_flip[c1])
        score = min(p0, p1)
        if score < threshold:
            continue
        tooth_l = remap_right_to_left(tooth_r)
        x0_raw = float(TARGET_HW[1] - 1) - x0_flip_raw
        x1_raw = float(TARGET_HW[1] - 1) - x1_flip_raw
        x0l, y0l = _map_xy_256_to_roi(x0_raw, y0_raw, right_w, right_h)
        x1l, y1l = _map_xy_256_to_roi(x1_raw, y1_raw, right_w, right_h)
        preds_left.append(
            ToothPrediction(
                tooth=tooth_l,
                p1=(x0l + rx1, y0l + ry1),
                p2=(x1l + rx1, y1l + ry1),
                score=float(score),
                peak_p1=float(p0),
                peak_p2=float(p1),
                source="lateral_right_flipped_restore",
            )
        )

    right_logits_unflip = np.flip(right_logits_flip, axis=2)
    center_global_max = _max_heatmap_to_global(center_logits, rects[CENTER_ROI_KEY], (h, w))
    left_global_max = _max_heatmap_to_global(left_logits, rects[LEFT_ROI_KEY], (h, w))
    right_global_max = _max_heatmap_to_global(right_logits_unflip, rects[RIGHT_ROI_KEY], (h, w))
    global_max = np.maximum.reduce([center_global_max, left_global_max, right_global_max])

    heatmaps = MultiROIHeatmaps(
        center_logits=center_logits,
        left_logits=left_logits,
        right_logits_flip=right_logits_flip,
        right_logits_unflip=right_logits_unflip,
        center_global_max=center_global_max,
        left_global_max=left_global_max,
        right_global_max=right_global_max,
        global_max=global_max,
    )
    preds_all = preds_center + preds_right + preds_left
    return MultiROIInferenceResult(
        image_hw=(h, w),
        rects=rects,
        predictions=preds_all,
        heatmaps=heatmaps,
    )


def infer_multiroi_from_path(
    image_path: Path,
    models: MultiROIModels,
    threshold: float = 0.1,
) -> MultiROIInferenceResult:
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {image_path}")
    return infer_multiroi_from_image(image_gray=image_gray, models=models, threshold=threshold)


__all__ = [
    "CENTER_ROI_KEY",
    "DEFAULT_CENTER_CKPT",
    "DEFAULT_LATERAL_CKPT",
    "LATERAL_LEFT_TEETH",
    "ROI_BY_BRANCH",
    "TARGET_HW",
    "TEETH_BY_BRANCH",
    "ToothPrediction",
    "MultiROIHeatmaps",
    "MultiROIInferenceResult",
    "MultiROIModels",
    "auto_device",
    "centroid_xy_and_peak_from_sigmoid_logits",
    "infer_multiroi_from_image",
    "infer_multiroi_from_path",
    "latest_best_ckpt",
    "load_multiroi_models",
    "resolve_path",
]
