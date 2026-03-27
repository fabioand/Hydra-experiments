#!/usr/bin/env python3
"""Avaliacao do Hydra U-Net MultiTask."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from hydra_data import (
    HydraTeethDataset,
    discover_samples,
    load_json,
    make_or_load_split,
    quadrant_for_tooth,
)
from hydra_multitask_model import CANONICAL_TEETH_32, HydraUNetMultiTask
from dashboard_registry import rel_to_experiment, register_record
from longoeixo.scripts.multiroi_composed_inference import (
    DEFAULT_CENTER_CKPT,
    DEFAULT_LATERAL_CKPT,
    infer_multiroi_from_image,
    latest_best_ckpt,
    load_multiroi_models,
    resolve_path as resolve_multiroi_path,
)


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


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


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]

    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j

    unsorted_ranks = np.empty_like(ranks)
    unsorted_ranks[order] = ranks

    sum_ranks_pos = float(unsorted_ranks[y_true == 1].sum())
    auc = (sum_ranks_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(auc)


def _tooth_f1(yt: np.ndarray, ys: np.ndarray, thr: float) -> float:
    yp = (ys >= thr).astype(np.int32)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _calibrate_presence_thresholds_per_tooth(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    default_thr: float,
) -> Tuple[Dict[str, float], List[Dict]]:
    if thr_step <= 0:
        raise ValueError("thr_step deve ser > 0")
    grid = np.arange(thr_min, thr_max + 1e-12, thr_step, dtype=np.float64)
    if grid.size == 0:
        raise ValueError("grid de thresholds vazio")

    thresholds: Dict[str, float] = {}
    rows: List[Dict] = []
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        yt = y_true[:, i].astype(np.int32)
        ys = y_score[:, i].astype(np.float64)

        if np.unique(yt).size < 2:
            best_thr = float(default_thr)
            best_f1 = float("nan")
        else:
            best_thr = float(default_thr)
            best_f1 = -1.0
            for thr in grid:
                f1 = _tooth_f1(yt, ys, float(thr))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)

        thresholds[tooth] = best_thr
        rows.append(
            {
                "tooth": tooth,
                "threshold": best_thr,
                "best_f1_on_calibration_split": float(best_f1),
                "gt_pos": int((yt == 1).sum()),
                "gt_neg": int((yt == 0).sum()),
            }
        )
    return thresholds, rows


def _presence_scores_from_heatmap(y_pred_probs: torch.Tensor) -> np.ndarray:
    # score por dente = media dos picos dos 2 canais (0..1)
    b = y_pred_probs.shape[0]
    peak = y_pred_probs.view(b, 64, -1).amax(dim=-1)
    score = 0.5 * (peak[:, 0::2] + peak[:, 1::2])
    return score.detach().cpu().numpy()


def _presence_scores_from_heatmap_composite(
    y_pred_probs: torch.Tensor,
    local_window: int,
    w_peak: float,
    w_mass: float,
    w_sharp: float,
    w_balance: float,
    w_dist: float,
    sharpness_scale: float,
    dist_min: float,
    dist_max: float,
    dist_sigma: float,
) -> np.ndarray:
    """Score composto por dente (0..1), derivado apenas do heatmap.

    Componentes:
    - peak: media dos picos dos dois canais do dente
    - mass: energia local media em torno do pico (vizinhanca)
    - sharp: nitidez do pico vs media global do canal
    - balance: consistencia entre os dois canais (evita sparks isolados)
    - dist: plausibilidade do comprimento (distancia entre os dois argmax)
    """
    if local_window < 1 or local_window % 2 == 0:
        raise ValueError("local_window deve ser inteiro impar >= 1")

    b, c, h, w = y_pred_probs.shape
    if c != 64:
        raise ValueError(f"Esperado 64 canais de heatmap, recebido {c}")

    flat = y_pred_probs.view(b, c, -1)
    peak, idx = flat.max(dim=-1)  # (B,64)

    # Energia local em torno do pico.
    local_avg = F.avg_pool2d(y_pred_probs, kernel_size=local_window, stride=1, padding=local_window // 2)
    local_flat = local_avg.view(b, c, -1)
    local_at_peak = torch.gather(local_flat, dim=2, index=idx.unsqueeze(-1)).squeeze(-1)  # (B,64)

    # Nitidez relativa: pico comparado com media global do canal.
    global_mean = y_pred_probs.mean(dim=(2, 3))
    sharp_raw = peak / (global_mean + 1e-6)
    sharp = 1.0 - torch.exp(-sharp_raw / max(sharpness_scale, 1e-6))
    sharp = torch.clamp(sharp, 0.0, 1.0)

    # Coordenadas dos argmax para plausibilidade de distancia entre p1/p2.
    py = (idx // w).float()
    px = (idx % w).float()
    p0x, p1x = px[:, 0::2], px[:, 1::2]
    p0y, p1y = py[:, 0::2], py[:, 1::2]
    dist = torch.sqrt((p0x - p1x) ** 2 + (p0y - p1y) ** 2)

    low_violation = torch.relu(dist_min - dist)
    high_violation = torch.relu(dist - dist_max)
    dist_score = torch.exp(-((low_violation / max(dist_sigma, 1e-6)) ** 2)) * torch.exp(
        -((high_violation / max(dist_sigma, 1e-6)) ** 2)
    )
    dist_score = torch.clamp(dist_score, 0.0, 1.0)

    peak0, peak1 = peak[:, 0::2], peak[:, 1::2]
    mass0, mass1 = local_at_peak[:, 0::2], local_at_peak[:, 1::2]
    sharp0, sharp1 = sharp[:, 0::2], sharp[:, 1::2]

    peak_mean = 0.5 * (peak0 + peak1)
    mass_mean = 0.5 * (mass0 + mass1)
    sharp_mean = 0.5 * (sharp0 + sharp1)
    balance = 1.0 - (torch.abs(peak0 - peak1) / (peak0 + peak1 + 1e-6))
    balance = torch.clamp(balance, 0.0, 1.0)

    weights_sum = max(w_peak + w_mass + w_sharp + w_balance + w_dist, 1e-8)
    score = (
        w_peak * peak_mean
        + w_mass * mass_mean
        + w_sharp * sharp_mean
        + w_balance * balance
        + w_dist * dist_score
    ) / weights_sum
    return torch.clamp(score, 0.0, 1.0).detach().cpu().numpy()


def _presence_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    thresholds_per_tooth: Dict[str, float] | None = None,
) -> Dict:
    if thresholds_per_tooth is None:
        y_pred = (y_score >= threshold).astype(np.int32)
    else:
        thr_vec = np.array([float(thresholds_per_tooth[t]) for t in CANONICAL_TEETH_32], dtype=np.float64).reshape(1, 32)
        y_pred = (y_score >= thr_vec).astype(np.int32)

    aucs: List[float] = []
    f1s: List[float] = []
    ps: List[float] = []
    rs: List[float] = []
    accs: List[float] = []

    per_tooth = {}
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        yt = y_true[:, i]
        ys = y_score[:, i]
        yp = y_pred[:, i]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / max(1, len(yt))
        auc = _binary_auc(yt, ys)

        aucs.append(auc)
        f1s.append(f1)
        ps.append(precision)
        rs.append(recall)
        accs.append(acc)

        per_tooth[tooth] = {
            "auc": auc,
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(acc),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    auc_macro = float(np.nanmean(np.array(aucs, dtype=np.float64))) if aucs else float("nan")

    return {
        "auc_macro": auc_macro,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "precision_macro": float(np.mean(ps)) if ps else 0.0,
        "recall_macro": float(np.mean(rs)) if rs else 0.0,
        "per_tooth": per_tooth,
        "threshold": threshold,
        "thresholds_per_tooth": thresholds_per_tooth,
        "accuracy_macro": float(np.mean(accs)) if accs else 0.0,
        "y_pred": y_pred,
    }


def _overlay_pred_gt_panel(x01: np.ndarray, y_gt: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    base = np.clip(x01 * 255.0, 0.0, 255.0).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    gt_max = np.max(y_gt, axis=0)
    pred_max = np.max(y_pred, axis=0)

    gt_overlay = base_bgr.copy()
    gt_overlay[:, :, 2] = np.clip(gt_overlay[:, :, 2].astype(np.float32) + gt_max * 180.0, 0, 255).astype(np.uint8)

    pred_overlay = base_bgr.copy()
    pred_overlay[:, :, 1] = np.clip(pred_overlay[:, :, 1].astype(np.float32) + pred_max * 180.0, 0, 255).astype(np.uint8)

    both = base_bgr.copy()
    both[:, :, 2] = np.clip(both[:, :, 2].astype(np.float32) + gt_max * 160.0, 0, 255).astype(np.uint8)
    both[:, :, 1] = np.clip(both[:, :, 1].astype(np.float32) + pred_max * 160.0, 0, 255).astype(np.uint8)

    h, w = base_bgr.shape[:2]
    bar_h = 28
    panel = np.zeros((h + bar_h, w * 3, 3), dtype=np.uint8)
    panel[bar_h:, 0:w] = gt_overlay
    panel[bar_h:, w : 2 * w] = pred_overlay
    panel[bar_h:, 2 * w : 3 * w] = both

    cv2.putText(panel, "GT overlay (red)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "Pred overlay (green)",
        (w + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "GT vs Pred",
        (2 * w + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def _safe_stat(arr: np.ndarray, fn: str) -> float:
    if arr.size == 0:
        return float("nan")
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "median":
        return float(np.median(arr))
    if fn == "p90":
        return float(np.percentile(arr, 90))
    raise ValueError(f"fn invalida: {fn}")


def _resolve_stems_for_split(split: Dict[str, List[str]], split_name: str) -> List[str]:
    if split_name == "train":
        return list(split["train"])
    if split_name == "val":
        return list(split["val"])
    if split_name == "test":
        if "test" not in split:
            raise RuntimeError("Split 'test' indisponivel no arquivo de split atual.")
        return list(split["test"])
    return list(split["train"] + split["val"] + split.get("test", []))


def _gt_presence_from_json(json_path: Path) -> np.ndarray:
    arr = np.zeros((32,), dtype=np.int32)
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}
    data = load_json(json_path)
    for ann in data:
        label = str(ann.get("label", ""))
        idx = idx_map.get(label)
        if idx is None:
            continue
        pts = ann.get("pts", [])
        if isinstance(pts, list) and len(pts) > 0:
            arr[idx] = 1
    return arr


def _gt_points_and_presence_from_json(json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai presença + dois pontos por dente no espaço global da panorâmica.

    Retorna:
    - presence: (32,) int32
    - p1: (32,2) float64
    - p2: (32,2) float64
    """
    presence = np.zeros((32,), dtype=np.int32)
    p1 = np.full((32, 2), np.nan, dtype=np.float64)
    p2 = np.full((32, 2), np.nan, dtype=np.float64)
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}

    data = load_json(json_path)
    for ann in data:
        label = str(ann.get("label", ""))
        idx = idx_map.get(label)
        if idx is None:
            continue
        pts = ann.get("pts", [])
        valid_pts: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid_pts.append((float(x), float(y)))
        if not valid_pts:
            continue
        presence[idx] = 1
        p1[idx, 0], p1[idx, 1] = valid_pts[0]
        if len(valid_pts) >= 2:
            p2[idx, 0], p2[idx, 1] = valid_pts[1]
        else:
            # Fallback raro: quando só existe 1 ponto anotado, reutiliza p1.
            p2[idx, 0], p2[idx, 1] = valid_pts[0]
    return presence, p1, p2


def _image_diag_and_gt_tooth_length(
    image_path: Path,
    gt_presence: np.ndarray,
    gt_p1: np.ndarray,
    gt_p2: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Retorna diagonal da imagem e comprimento GT por dente (32,)."""
    with Image.open(image_path) as img:
        w, h = img.size
    diag = float(np.hypot(float(w), float(h)))
    gt_len = np.full((32,), np.nan, dtype=np.float64)
    for i in range(32):
        if gt_presence[i] <= 0:
            continue
        x1, y1 = gt_p1[i]
        x2, y2 = gt_p2[i]
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue
        d = float(np.hypot(x2 - x1, y2 - y1))
        if d > 1e-8:
            gt_len[i] = d
    return diag, gt_len


def _multiroi_scores_and_points_for_image(
    image_path: Path,
    models,
    infer_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {image_path}")

    result = infer_multiroi_from_image(
        image_gray=image_gray,
        models=models,
        threshold=float(infer_threshold),
    )

    score_floor = -1e6
    scores = np.full((32,), score_floor, dtype=np.float64)
    p1 = np.full((32, 2), np.nan, dtype=np.float64)
    p2 = np.full((32, 2), np.nan, dtype=np.float64)
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}

    for pred in result.predictions:
        idx = idx_map.get(pred.tooth)
        if idx is None:
            continue
        sc = float(pred.score)
        if sc >= scores[idx]:
            scores[idx] = sc
            p1[idx, 0], p1[idx, 1] = float(pred.p1[0]), float(pred.p1[1])
            p2[idx, 0], p2[idx, 1] = float(pred.p2[0]), float(pred.p2[1])

    return scores, p1, p2


def _draw_axes(image_bgr: np.ndarray, p1: np.ndarray, p2: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = image_bgr.copy()
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        if int(mask[i]) <= 0:
            continue
        x1, y1 = p1[i]
        x2, y2 = p2[i]
        if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
            continue
        a = (int(round(x1)), int(round(y1)))
        b = (int(round(x2)), int(round(y2)))
        cv2.line(out, a, b, color, 2, cv2.LINE_AA)
        cv2.circle(out, a, 3, color, -1, cv2.LINE_AA)
        cv2.circle(out, b, 3, color, -1, cv2.LINE_AA)
        xm = int(round((a[0] + b[0]) * 0.5))
        ym = int(round((a[1] + b[1]) * 0.5))
        cv2.putText(out, tooth, (xm + 2, ym - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def _overlay_pred_gt_axes_panel(
    image_gray: np.ndarray,
    gt_p1: np.ndarray,
    gt_p2: np.ndarray,
    gt_mask: np.ndarray,
    pred_p1: np.ndarray,
    pred_p2: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    gt_overlay = _draw_axes(base, gt_p1, gt_p2, gt_mask, color=(0, 0, 255))
    pred_overlay = _draw_axes(base, pred_p1, pred_p2, pred_mask, color=(0, 255, 0))
    both = _draw_axes(gt_overlay, pred_p1, pred_p2, pred_mask, color=(0, 255, 0))

    h, w = base.shape[:2]
    bar_h = 28
    panel = np.zeros((h + bar_h, w * 3, 3), dtype=np.uint8)
    panel[bar_h:, 0:w] = gt_overlay
    panel[bar_h:, w : 2 * w] = pred_overlay
    panel[bar_h:, 2 * w : 3 * w] = both

    cv2.putText(panel, "GT axes (red)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "Pred axes (green)",
        (w + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "GT vs Pred",
        (2 * w + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def _rm_api_auth_headers(base: str, username: str, password: str, timeout: int) -> Dict[str, str]:
    auth_url = f"{base.rstrip('/')}/v1/auth/token"
    body = (
        f"grant_type=&username={username}&password={password}"
        "&scope=&client_id=&client_secret="
    )
    response = requests.post(
        auth_url,
        headers={"Content-type": "application/x-www-form-urlencoded", "accept": "application/json"},
        data=body,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    token_type = payload.get("token_type")
    access_token = payload.get("access_token")
    if not token_type or not access_token:
        raise RuntimeError(f"Falha de autenticacao RM API: {payload}")
    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def _parse_longaxis_entities_from_response(resp: requests.Response) -> List[Dict]:
    text = resp.text or ""
    try:
        obj = resp.json()
    except Exception:
        obj = None

    if isinstance(obj, dict) and isinstance(obj.get("entities"), list):
        return obj["entities"]

    entities: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            one = json.loads(line)
        except Exception:
            continue
        if str(one.get("model_name") or "").lower() == "longaxis" and isinstance(one.get("entities"), list):
            entities.extend(one["entities"])
    return entities


def _rm_api_longaxis_scores_for_image(
    image_path: Path,
    base: str,
    headers: Dict[str, str],
    timeout: int,
    use_cache: bool,
) -> np.ndarray:
    # Mantem exatamente o protocolo usado nos scripts de probe:
    # abre imagem, converte RGB e re-serializa em JPEG antes do base64.
    with Image.open(image_path).convert("RGB") as img:
        import io
        buff = io.BytesIO()
        img.save(buff, format="JPEG")
        image_b64 = base64.b64encode(buff.getvalue()).decode("ascii")

    body = {
        "base64_image": image_b64,
        "output_width": 0,
        "output_height": 0,
        "threshold": 0.0,
        "resource": "describe",
        "lang": "pt-br",
        "use_cache": bool(use_cache),
    }
    url = f"{base.rstrip('/')}/v1/panoramics/longaxis"
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()

    entities = _parse_longaxis_entities_from_response(resp)
    scores = np.zeros((32,), dtype=np.float64)
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}
    for ent in entities:
        label = str(ent.get("class_name") or ent.get("tooth") or "")
        idx = idx_map.get(label)
        if idx is None:
            continue
        score = float(ent.get("score") or 0.0)
        if score > scores[idx]:
            scores[idx] = score
    return scores


def _presence_rows_only(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        gt_t = y_true[:, i]
        pred_t = y_pred[:, i]
        rows.append(
            {
                "tooth": tooth,
                "quadrant": quadrant_for_tooth(tooth),
                "presence_auc": metrics["per_tooth"][tooth]["auc"],
                "presence_f1": metrics["per_tooth"][tooth]["f1"],
                "presence_precision": metrics["per_tooth"][tooth]["precision"],
                "presence_recall": metrics["per_tooth"][tooth]["recall"],
                "presence_accuracy": metrics["per_tooth"][tooth]["accuracy"],
                "pred_presence_pos_count": int((pred_t > 0).sum()),
                "gt_present_count": int((gt_t > 0).sum()),
                "gt_absent_count": int((gt_t == 0).sum()),
            }
        )
    return rows


def _run_multiroi_model_eval(
    args: argparse.Namespace,
    repo_root: Path,
    output_base_dir: Path,
    split: Dict[str, List[str]],
    by_stem: Dict[str, object],
) -> None:
    if bool(args.multiroi_use_latest_from_output_dirs):
        center_ckpt = latest_best_ckpt(resolve_multiroi_path(repo_root, args.multiroi_center_output_dir))
        lateral_ckpt = latest_best_ckpt(resolve_multiroi_path(repo_root, args.multiroi_lateral_output_dir))
    else:
        center_ckpt = resolve_multiroi_path(repo_root, args.multiroi_center_ckpt)
        lateral_ckpt = resolve_multiroi_path(repo_root, args.multiroi_lateral_ckpt)

    run_name = args.run_name or _latest_run_name(output_base_dir) or "MULTIROI_MODEL"
    run_dir = output_base_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / "eval_multiroi_model"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stems_eval = _resolve_stems_for_split(split, args.split)
    if args.max_samples is not None:
        stems_eval = stems_eval[: max(0, int(args.max_samples))]
    eval_samples = [by_stem[s] for s in stems_eval if s in by_stem]
    if not eval_samples:
        raise RuntimeError("Nenhuma amostra para avaliacao MultiROI no split selecionado.")

    models = load_multiroi_models(center_ckpt=center_ckpt, lateral_ckpt=lateral_ckpt)
    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")
    print(f"[DATA] source=multiroi_model split={args.split} num_samples={len(eval_samples)}")
    print(f"[MULTIROI] center_ckpt={center_ckpt}")
    print(f"[MULTIROI] lateral_ckpt={lateral_ckpt}")
    print(f"[MULTIROI] device={models.device}")

    visual_out = eval_dir / "pred_vs_gt_samples"
    visual_out.mkdir(parents=True, exist_ok=True)
    max_visuals = int(args.max_visual_samples_multiroi)
    vis_saved = 0

    failed_samples: List[Dict[str, str]] = []

    def _collect_for_samples(
        sample_list: List[object],
        tag: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonlocal vis_saved
        y_true_rows: List[np.ndarray] = []
        y_score_rows: List[np.ndarray] = []
        tooth_dist_rows: List[np.ndarray] = []
        tooth_dist_norm_diag_rows: List[np.ndarray] = []
        tooth_dist_norm_gtlen_rows: List[np.ndarray] = []
        t0 = time.time()
        total = len(sample_list)
        for i, sample in enumerate(sample_list, start=1):
            y_true_presence, gt_p1, gt_p2 = _gt_points_and_presence_from_json(sample.json_path)
            y_true_rows.append(y_true_presence)
            try:
                scores, pred_p1, pred_p2 = _multiroi_scores_and_points_for_image(
                    image_path=sample.image_path,
                    models=models,
                    infer_threshold=float(args.multiroi_infer_threshold),
                )
            except Exception as e:
                failed_samples.append(
                    {
                        "tag": tag,
                        "stem": sample.stem,
                        "image_path": str(sample.image_path),
                        "error": str(e),
                    }
                )
                if not bool(args.multiroi_skip_errors):
                    raise
                scores = np.full((32,), -1e6, dtype=np.float64)
                pred_p1 = np.full((32, 2), np.nan, dtype=np.float64)
                pred_p2 = np.full((32, 2), np.nan, dtype=np.float64)
                print(f"[WARN] multiroi sample_failed tag={tag} stem={sample.stem} error={e}")

            y_score_rows.append(scores)

            tooth_dist = np.full((32,), np.nan, dtype=np.float64)
            for ti in range(32):
                if y_true_presence[ti] <= 0:
                    continue
                xg1, yg1 = gt_p1[ti]
                xg2, yg2 = gt_p2[ti]
                xp1, yp1 = pred_p1[ti]
                xp2, yp2 = pred_p2[ti]
                d1 = np.hypot(xp1 - xg1, yp1 - yg1) if np.isfinite([xp1, yp1, xg1, yg1]).all() else np.nan
                d2 = np.hypot(xp2 - xg2, yp2 - yg2) if np.isfinite([xp2, yp2, xg2, yg2]).all() else np.nan
                tooth_dist[ti] = 0.5 * (d1 + d2) if np.isfinite([d1, d2]).all() else np.nan
            tooth_dist_rows.append(tooth_dist)

            diag_img, gt_tooth_len = _image_diag_and_gt_tooth_length(
                image_path=sample.image_path,
                gt_presence=y_true_presence,
                gt_p1=gt_p1,
                gt_p2=gt_p2,
            )
            tooth_dist_norm_diag = np.full((32,), np.nan, dtype=np.float64)
            tooth_dist_norm_gtlen = np.full((32,), np.nan, dtype=np.float64)
            for ti in range(32):
                d = tooth_dist[ti]
                if np.isfinite(d) and diag_img > 1e-8:
                    tooth_dist_norm_diag[ti] = float(d) / float(diag_img)
                gl = gt_tooth_len[ti]
                if np.isfinite(d) and np.isfinite(gl) and float(gl) > 1e-8:
                    tooth_dist_norm_gtlen[ti] = float(d) / float(gl)
            tooth_dist_norm_diag_rows.append(tooth_dist_norm_diag)
            tooth_dist_norm_gtlen_rows.append(tooth_dist_norm_gtlen)

            if vis_saved < max_visuals and tag == args.split:
                image_gray = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
                if image_gray is not None:
                    panel = _overlay_pred_gt_axes_panel(
                        image_gray=image_gray,
                        gt_p1=gt_p1,
                        gt_p2=gt_p2,
                        gt_mask=y_true_presence,
                        pred_p1=pred_p1,
                        pred_p2=pred_p2,
                        pred_mask=np.ones((32,), dtype=np.int32),
                    )
                    cv2.imwrite(str(visual_out / f"{vis_saved:02d}_{sample.stem}.png"), panel)
                    vis_saved += 1

            if i == 1 or i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-8)
                eta = (total - i) / max(rate, 1e-8)
                print(
                    f"[EVAL_PROGRESS] source=multiroi tag={tag} sample={i}/{total} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                )

        return (
            np.stack(y_true_rows, axis=0),
            np.stack(y_score_rows, axis=0),
            np.stack(tooth_dist_rows, axis=0),
            np.stack(tooth_dist_norm_diag_rows, axis=0),
            np.stack(tooth_dist_norm_gtlen_rows, axis=0),
        )

    (
        y_true_eval,
        y_score_eval,
        tooth_dist_eval,
        tooth_dist_norm_diag_eval,
        tooth_dist_norm_gtlen_eval,
    ) = _collect_for_samples(eval_samples, tag=args.split)

    thresholds_per_tooth: Dict[str, float] | None = None
    calibration_report_rows: List[Dict] = []
    calibrated_thresholds_out_path: Path | None = None
    calibration_report_path: Path | None = None

    if args.presence_thresholds_json is not None:
        thr_path = _resolve_path(repo_root, str(args.presence_thresholds_json))
        thr_data = load_json(thr_path)
        thr_map = thr_data.get("thresholds_per_tooth", thr_data) if isinstance(thr_data, dict) else thr_data
        missing = [t for t in CANONICAL_TEETH_32 if t not in thr_map]
        if missing:
            raise ValueError(f"presence-thresholds-json sem chaves: {missing}")
        thresholds_per_tooth = {t: float(thr_map[t]) for t in CANONICAL_TEETH_32}

    if args.calibrate_presence_thresholds:
        calibration_split = args.calibration_split or args.split
        stems_cal = _resolve_stems_for_split(split, calibration_split)
        if args.max_samples is not None:
            stems_cal = stems_cal[: max(0, int(args.max_samples))]
        cal_samples = [by_stem[s] for s in stems_cal if s in by_stem]
        if not cal_samples:
            raise RuntimeError(f"Nenhuma amostra para calibracao no split={calibration_split}.")

        y_true_cal, y_score_cal, _, _, _ = _collect_for_samples(cal_samples, tag=f"cal:{calibration_split}")
        thresholds_per_tooth, calibration_report_rows = _calibrate_presence_thresholds_per_tooth(
            y_true=y_true_cal,
            y_score=y_score_cal,
            thr_min=float(args.calibration_threshold_min),
            thr_max=float(args.calibration_threshold_max),
            thr_step=float(args.calibration_threshold_step),
            default_thr=float(args.presence_threshold),
        )

        calib_out = args.calibrated_thresholds_out
        if calib_out is None:
            calib_out = eval_dir / f"presence_thresholds_multiroi_score_calibrated_{calibration_split}.json"
        else:
            calib_out = _resolve_path(repo_root, str(calib_out))
        calib_out.parent.mkdir(parents=True, exist_ok=True)
        calib_payload = {
            "presence_source": "multiroi_score",
            "split_used_for_calibration": calibration_split,
            "thresholds_per_tooth": thresholds_per_tooth,
            "calibration_grid": {
                "min": float(args.calibration_threshold_min),
                "max": float(args.calibration_threshold_max),
                "step": float(args.calibration_threshold_step),
            },
            "multiroi": {
                "center_ckpt": str(center_ckpt),
                "lateral_ckpt": str(lateral_ckpt),
                "lib_infer_threshold": float(args.multiroi_infer_threshold),
            },
        }
        calib_out.write_text(json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        calibrated_thresholds_out_path = calib_out

        calib_report_path = eval_dir / f"presence_thresholds_multiroi_score_calibration_report_{calibration_split}.csv"
        with calib_report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(calibration_report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibration_report_rows)
        calibration_report_path = calib_report_path
        print(f"[EVAL] calibrated_thresholds_json={calib_out}")
        print(f"[EVAL] calibrated_thresholds_report={calib_report_path}")

    presence = _presence_metrics(
        y_true=y_true_eval,
        y_score=y_score_eval,
        threshold=float(args.presence_threshold),
        thresholds_per_tooth=thresholds_per_tooth,
    )
    y_pred_presence = presence.pop("y_pred")

    thresholds = [3.0, 5.0, 10.0]
    present_mask = y_true_eval > 0.5
    pred_pos_mask = y_pred_presence > 0
    tooth_dist_present = tooth_dist_eval[present_mask]
    tooth_dist_pred_pos_present = tooth_dist_eval[present_mask & pred_pos_mask]

    global_loc = {
        "point_error_mean_px": _safe_stat(tooth_dist_present, "mean"),
        "point_error_median_px": _safe_stat(tooth_dist_present, "median"),
        "point_error_p90_px": _safe_stat(tooth_dist_present, "p90"),
    }
    for thr in thresholds:
        valid = np.isfinite(tooth_dist_present)
        arr = tooth_dist_present[valid]
        global_loc[f"point_within_{int(thr)}px_rate"] = float(np.mean(arr <= thr)) if arr.size > 0 else float("nan")

    global_loc_operational = {
        "point_error_mean_px": _safe_stat(tooth_dist_pred_pos_present, "mean"),
        "point_error_median_px": _safe_stat(tooth_dist_pred_pos_present, "median"),
        "point_error_p90_px": _safe_stat(tooth_dist_pred_pos_present, "p90"),
    }
    for thr in thresholds:
        valid = np.isfinite(tooth_dist_pred_pos_present)
        arr = tooth_dist_pred_pos_present[valid]
        global_loc_operational[f"point_within_{int(thr)}px_rate"] = (
            float(np.mean(arr <= thr)) if arr.size > 0 else float("nan")
        )

    global_loc_operational_alias = {
        "point_error_mean_px_when_pred_presence_pos": global_loc_operational["point_error_mean_px"],
        "point_error_median_px_when_pred_presence_pos": global_loc_operational["point_error_median_px"],
        "point_error_p90_px_when_pred_presence_pos": global_loc_operational["point_error_p90_px"],
        "point_within_3px_rate_when_pred_presence_pos": global_loc_operational["point_within_3px_rate"],
        "point_within_5px_rate_when_pred_presence_pos": global_loc_operational["point_within_5px_rate"],
        "point_within_10px_rate_when_pred_presence_pos": global_loc_operational["point_within_10px_rate"],
    }

    per_tooth_rows: List[Dict] = []
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        gt_t = y_true_eval[:, i]
        pred_t = y_pred_presence[:, i]
        dist_t = tooth_dist_eval[:, i]
        dist_diag_t = tooth_dist_norm_diag_eval[:, i]
        dist_gtlen_t = tooth_dist_norm_gtlen_eval[:, i]

        present_mask_t = gt_t > 0.5
        absent_mask_t = ~present_mask_t
        dist_present_t = dist_t[present_mask_t]
        dist_present_t = dist_present_t[np.isfinite(dist_present_t)]
        dist_present_diag_t = dist_diag_t[present_mask_t]
        dist_present_diag_t = dist_present_diag_t[np.isfinite(dist_present_diag_t)]
        dist_present_gtlen_t = dist_gtlen_t[present_mask_t]
        dist_present_gtlen_t = dist_present_gtlen_t[np.isfinite(dist_present_gtlen_t)]

        false_point_absent_rate = float(np.mean(pred_t[absent_mask_t] > 0)) if absent_mask_t.any() else float("nan")

        pred_pos = pred_t > 0
        valid_point_when_pred_pos = (
            float(np.mean((gt_t[pred_pos] > 0.5) & np.isfinite(dist_t[pred_pos]) & (dist_t[pred_pos] <= 5.0)))
            if pred_pos.any()
            else float("nan")
        )

        row = {
            "tooth": tooth,
            "quadrant": quadrant_for_tooth(tooth),
            "presence_auc": presence["per_tooth"][tooth]["auc"],
            "presence_f1": presence["per_tooth"][tooth]["f1"],
            "presence_precision": presence["per_tooth"][tooth]["precision"],
            "presence_recall": presence["per_tooth"][tooth]["recall"],
            "presence_accuracy": presence["per_tooth"][tooth]["accuracy"],
            "point_error_mean_px": _safe_stat(dist_present_t, "mean"),
            "point_error_median_px": _safe_stat(dist_present_t, "median"),
            "point_error_p90_px": _safe_stat(dist_present_t, "p90"),
            "point_error_mean_over_image_diag": _safe_stat(dist_present_diag_t, "mean"),
            "point_error_median_over_image_diag": _safe_stat(dist_present_diag_t, "median"),
            "point_error_p90_over_image_diag": _safe_stat(dist_present_diag_t, "p90"),
            "point_error_mean_over_gt_tooth_len": _safe_stat(dist_present_gtlen_t, "mean"),
            "point_error_median_over_gt_tooth_len": _safe_stat(dist_present_gtlen_t, "median"),
            "point_error_p90_over_gt_tooth_len": _safe_stat(dist_present_gtlen_t, "p90"),
            "point_within_5px_rate": float(np.mean(dist_present_t <= 5.0)) if dist_present_t.size > 0 else float("nan"),
            "false_point_rate_gt_absent": false_point_absent_rate,
            "valid_point_rate_when_pred_presence_pos": valid_point_when_pred_pos,
            "pred_presence_pos_count": int((pred_t > 0).sum()),
            "gt_present_count": int(present_mask_t.sum()),
            "gt_absent_count": int(absent_mask_t.sum()),
        }
        per_tooth_rows.append(row)

    per_quadrant: Dict[str, List[Dict]] = {}
    for row in per_tooth_rows:
        per_quadrant.setdefault(row["quadrant"], []).append(row)

    per_quadrant_rows: List[Dict] = []
    for quad, rows in per_quadrant.items():
        per_quadrant_rows.append(
            {
                "quadrant": quad,
                "presence_f1_mean": float(np.nanmean([r["presence_f1"] for r in rows])),
                "presence_auc_mean": float(np.nanmean([r["presence_auc"] for r in rows])),
                "point_error_mean_px": float(np.nanmean([r["point_error_mean_px"] for r in rows])),
                "point_within_5px_rate_mean": float(np.nanmean([r["point_within_5px_rate"] for r in rows])),
                "false_point_rate_gt_absent_mean": float(np.nanmean([r["false_point_rate_gt_absent"] for r in rows])),
                "num_teeth": len(rows),
            }
        )

    summary = {
        "mode": "multiroi_model_eval",
        "split": args.split,
        "num_samples": int(len(eval_samples)),
        "multiroi": {
            "center_ckpt": str(center_ckpt),
            "lateral_ckpt": str(lateral_ckpt),
            "device": str(models.device),
            "lib_infer_threshold": float(args.multiroi_infer_threshold),
        },
        "heatmap": {
            "not_applicable": True,
            "reason": "Modo multiroi_model usa predicoes por dente (p1/p2/score) e nao stack canonica 64xHxW.",
        },
        "localization": global_loc,
        "localization_operational_pred_presence": global_loc_operational,
        "localization_operational_pred_presence_alias": global_loc_operational_alias,
        "presence": {
            "auc_macro": presence["auc_macro"],
            "f1_macro": presence["f1_macro"],
            "precision_macro": presence["precision_macro"],
            "recall_macro": presence["recall_macro"],
            "accuracy_macro": presence["accuracy_macro"],
            "source": "multiroi_score",
            "threshold": presence["threshold"],
            "thresholds_per_tooth": presence["thresholds_per_tooth"],
        },
        "combined": {
            "false_point_rate_gt_absent_global": float(np.mean((y_true_eval < 0.5) & (y_pred_presence > 0))),
            "valid_point_rate_when_pred_presence_pos_global": float(
                np.mean(
                    (y_true_eval[y_pred_presence > 0] > 0.5)
                    & np.isfinite(tooth_dist_eval[y_pred_presence > 0])
                    & (tooth_dist_eval[y_pred_presence > 0] <= 5.0)
                )
            )
            if np.any(y_pred_presence > 0)
            else float("nan"),
        },
        "artifacts": {
            "metrics_per_tooth_csv": str(eval_dir / "metrics_per_tooth.csv"),
            "metrics_per_quadrant_csv": str(eval_dir / "metrics_per_quadrant.csv"),
            "pred_vs_gt_samples_dir": str(visual_out),
            "presence_thresholds_json": str(calibrated_thresholds_out_path) if calibrated_thresholds_out_path else None,
            "presence_thresholds_calibration_report_csv": str(calibration_report_path) if calibration_report_path else None,
            "failed_samples_json": str(eval_dir / "failed_samples.json") if failed_samples else None,
        },
    }

    with (eval_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (eval_dir / "metrics_per_tooth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_tooth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_tooth_rows)

    with (eval_dir / "metrics_per_quadrant.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_quadrant_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_quadrant_rows)

    if failed_samples:
        (eval_dir / "failed_samples.json").write_text(
            json.dumps(failed_samples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    try:
        register_record(
            experiment_root=output_base_dir,
            kind="runs",
            record={
                "id": f"hydra_eval_multiroi__{run_name}__{args.split}",
                "kind": "hydra_eval_multiroi",
                "experiment": output_base_dir.name,
                "run_name": run_name,
                "split": args.split,
                "summary": {
                    "presence_f1_macro": summary["presence"]["f1_macro"],
                    "presence_auc_macro": summary["presence"]["auc_macro"],
                    "point_error_median_px": summary["localization"]["point_error_median_px"],
                    "point_within_5px_rate": summary["localization"]["point_within_5px_rate"],
                    "false_point_rate_gt_absent_global": summary["combined"]["false_point_rate_gt_absent_global"],
                },
                "artifacts": {
                    "metrics_summary_json": rel_to_experiment(eval_dir / "metrics_summary.json", output_base_dir),
                    "metrics_per_tooth_csv": rel_to_experiment(eval_dir / "metrics_per_tooth.csv", output_base_dir),
                    "metrics_per_quadrant_csv": rel_to_experiment(eval_dir / "metrics_per_quadrant.csv", output_base_dir),
                    "pred_vs_gt_samples_dir": rel_to_experiment(visual_out, output_base_dir),
                },
            },
        )
    except Exception as e:
        print(f"[WARN] dashboard registry skipped in eval.py (multiroi): {e}")

    print(f"[EVAL] split={args.split} source=multiroi_model num_samples={len(eval_samples)}")
    print(f"[EVAL] point_error_median_px={summary['localization']['point_error_median_px']:.6f}")
    print(
        "[EVAL] point_error_median_px_when_pred_presence_pos={:.6f}".format(
            summary["localization_operational_pred_presence"]["point_error_median_px"]
        )
    )
    print(f"[EVAL] presence_source={summary['presence']['source']}")
    print(f"[EVAL] presence_f1_macro={summary['presence']['f1_macro']:.6f}")
    if failed_samples:
        print(f"[EVAL] failed_samples={len(failed_samples)} json={eval_dir / 'failed_samples.json'}")
    print(f"[EVAL] summary={eval_dir / 'metrics_summary.json'}")


def _run_rm_api_presence_eval(
    args: argparse.Namespace,
    repo_root: Path,
    output_base_dir: Path,
    split: Dict[str, List[str]],
    by_stem: Dict[str, object],
) -> None:
    run_name = args.run_name or _latest_run_name(output_base_dir) or "RM_API_LONGAXIS"
    run_dir = output_base_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / "eval_rm_api_longaxis"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stems_eval = _resolve_stems_for_split(split, args.split)
    if args.max_samples is not None:
        stems_eval = stems_eval[: max(0, int(args.max_samples))]
    eval_samples = [by_stem[s] for s in stems_eval if s in by_stem]
    if not eval_samples:
        raise RuntimeError("Nenhuma amostra para avaliacao RM API no split selecionado.")

    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")
    print(f"[DATA] source=rm_api_longaxis split={args.split} num_samples={len(eval_samples)}")

    headers = _rm_api_auth_headers(
        base=args.rm_api_base,
        username=args.rm_api_username,
        password=args.rm_api_password,
        timeout=int(args.rm_api_timeout),
    )

    failed_samples: List[Dict[str, str]] = []

    def _collect_for_samples(sample_list: List[object], tag: str) -> Tuple[np.ndarray, np.ndarray]:
        y_true_rows: List[np.ndarray] = []
        y_score_rows: List[np.ndarray] = []
        t0 = time.time()
        total = len(sample_list)
        for i, sample in enumerate(sample_list, start=1):
            y_true_rows.append(_gt_presence_from_json(sample.json_path))
            try:
                y_score_rows.append(
                    _rm_api_longaxis_scores_for_image(
                        image_path=sample.image_path,
                        base=args.rm_api_base,
                        headers=headers,
                        timeout=int(args.rm_api_timeout),
                        use_cache=bool(args.rm_api_use_cache),
                    )
                )
            except Exception as e:
                failed_samples.append(
                    {
                        "tag": tag,
                        "stem": sample.stem,
                        "image_path": str(sample.image_path),
                        "error": str(e),
                    }
                )
                if not bool(args.rm_api_skip_errors):
                    raise
                # fallback neutro: score zero para todos os dentes
                y_score_rows.append(np.zeros((32,), dtype=np.float64))
                print(f"[WARN] rm_api sample_failed tag={tag} stem={sample.stem} error={e}")
            if i == 1 or i % 25 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-8)
                eta = (total - i) / max(rate, 1e-8)
                print(f"[EVAL_PROGRESS] source=rm_api tag={tag} sample={i}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s")
        return np.stack(y_true_rows, axis=0), np.stack(y_score_rows, axis=0)

    y_true_eval, y_score_eval = _collect_for_samples(eval_samples, tag=args.split)

    thresholds_per_tooth: Dict[str, float] | None = None
    calibration_report_rows: List[Dict] = []
    calibrated_thresholds_out_path: Path | None = None
    calibration_report_path: Path | None = None

    if args.presence_thresholds_json is not None:
        thr_path = _resolve_path(repo_root, str(args.presence_thresholds_json))
        thr_data = load_json(thr_path)
        thr_map = thr_data.get("thresholds_per_tooth", thr_data) if isinstance(thr_data, dict) else thr_data
        missing = [t for t in CANONICAL_TEETH_32 if t not in thr_map]
        if missing:
            raise ValueError(f"presence-thresholds-json sem chaves: {missing}")
        thresholds_per_tooth = {t: float(thr_map[t]) for t in CANONICAL_TEETH_32}

    if args.calibrate_presence_thresholds:
        calibration_split = args.calibration_split or args.split
        stems_cal = _resolve_stems_for_split(split, calibration_split)
        if args.max_samples is not None:
            stems_cal = stems_cal[: max(0, int(args.max_samples))]
        cal_samples = [by_stem[s] for s in stems_cal if s in by_stem]
        if not cal_samples:
            raise RuntimeError(f"Nenhuma amostra para calibracao no split={calibration_split}.")

        y_true_cal, y_score_cal = _collect_for_samples(cal_samples, tag=f"cal:{calibration_split}")
        thresholds_per_tooth, calibration_report_rows = _calibrate_presence_thresholds_per_tooth(
            y_true=y_true_cal,
            y_score=y_score_cal,
            thr_min=float(args.calibration_threshold_min),
            thr_max=float(args.calibration_threshold_max),
            thr_step=float(args.calibration_threshold_step),
            default_thr=float(args.presence_threshold),
        )

        calib_out = args.calibrated_thresholds_out
        if calib_out is None:
            calib_out = eval_dir / f"presence_thresholds_rm_api_longaxis_calibrated_{calibration_split}.json"
        else:
            calib_out = _resolve_path(repo_root, str(calib_out))
        calib_out.parent.mkdir(parents=True, exist_ok=True)
        calib_payload = {
            "presence_source": "rm_api_longaxis_score",
            "split_used_for_calibration": calibration_split,
            "thresholds_per_tooth": thresholds_per_tooth,
            "calibration_grid": {
                "min": float(args.calibration_threshold_min),
                "max": float(args.calibration_threshold_max),
                "step": float(args.calibration_threshold_step),
            },
        }
        calib_out.write_text(json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        calibrated_thresholds_out_path = calib_out

        calib_report_path = eval_dir / f"presence_thresholds_rm_api_longaxis_calibration_report_{calibration_split}.csv"
        with calib_report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(calibration_report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibration_report_rows)
        calibration_report_path = calib_report_path

    presence_fixed = _presence_metrics(
        y_true=y_true_eval,
        y_score=y_score_eval,
        threshold=float(args.presence_threshold),
        thresholds_per_tooth=None,
    )
    y_pred_fixed = presence_fixed.pop("y_pred")
    per_tooth_fixed = _presence_rows_only(y_true_eval, y_pred_fixed, presence_fixed)

    presence_calibrated = None
    per_tooth_calibrated = None
    y_pred_calibrated = None
    if thresholds_per_tooth is not None:
        presence_calibrated = _presence_metrics(
            y_true=y_true_eval,
            y_score=y_score_eval,
            threshold=float(args.presence_threshold),
            thresholds_per_tooth=thresholds_per_tooth,
        )
        y_pred_calibrated = presence_calibrated.pop("y_pred")
        per_tooth_calibrated = _presence_rows_only(y_true_eval, y_pred_calibrated, presence_calibrated)

    summary = {
        "mode": "rm_api_longaxis_presence_eval",
        "split": args.split,
        "num_samples": int(len(eval_samples)),
        "api": {
            "base": args.rm_api_base,
            "endpoint": "/v1/panoramics/longaxis",
            "use_cache": bool(args.rm_api_use_cache),
        },
        "presence_fixed": {
            "auc_macro": presence_fixed["auc_macro"],
            "f1_macro": presence_fixed["f1_macro"],
            "precision_macro": presence_fixed["precision_macro"],
            "recall_macro": presence_fixed["recall_macro"],
            "accuracy_macro": presence_fixed["accuracy_macro"],
            "threshold": float(args.presence_threshold),
        },
        "presence_calibrated": None
        if presence_calibrated is None
        else {
            "auc_macro": presence_calibrated["auc_macro"],
            "f1_macro": presence_calibrated["f1_macro"],
            "precision_macro": presence_calibrated["precision_macro"],
            "recall_macro": presence_calibrated["recall_macro"],
            "accuracy_macro": presence_calibrated["accuracy_macro"],
            "thresholds_per_tooth": thresholds_per_tooth,
        },
        "artifacts": {
            "metrics_per_tooth_fixed_csv": str(eval_dir / "metrics_per_tooth_fixed.csv"),
            "metrics_per_tooth_calibrated_csv": str(eval_dir / "metrics_per_tooth_calibrated.csv")
            if per_tooth_calibrated is not None
            else None,
            "presence_thresholds_json": str(calibrated_thresholds_out_path) if calibrated_thresholds_out_path else None,
            "presence_thresholds_calibration_report_csv": str(calibration_report_path) if calibration_report_path else None,
            "failed_samples_json": str(eval_dir / "failed_samples.json") if failed_samples else None,
        },
    }

    (eval_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed_samples:
        (eval_dir / "failed_samples.json").write_text(json.dumps(failed_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (eval_dir / "metrics_per_tooth_fixed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_tooth_fixed[0].keys()))
        writer.writeheader()
        writer.writerows(per_tooth_fixed)

    if per_tooth_calibrated is not None:
        with (eval_dir / "metrics_per_tooth_calibrated.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_tooth_calibrated[0].keys()))
            writer.writeheader()
            writer.writerows(per_tooth_calibrated)

    print(f"[EVAL] split={args.split} source=rm_api_longaxis num_samples={len(eval_samples)}")
    print(f"[EVAL] fixed_presence_f1_macro={presence_fixed['f1_macro']:.6f}")
    if presence_calibrated is not None:
        print(f"[EVAL] calibrated_presence_f1_macro={presence_calibrated['f1_macro']:.6f}")
    if failed_samples:
        print(f"[EVAL] failed_samples={len(failed_samples)} json={eval_dir / 'failed_samples.json'}")
    print(f"[EVAL] summary={eval_dir / 'metrics_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydra eval")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Nome da run; se omitido usa latest_run.txt")
    parser.add_argument(
        "--inference-source",
        type=str,
        default="model",
        choices=["model", "rm_api", "multiroi_model"],
        help="Fonte das predicoes: checkpoint local, API RM (longaxis) ou inferencia MultiROI.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--calibration-split", type=str, default=None, choices=["train", "val", "test", "all"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Limita numero de amostras por split nesta avaliacao.")
    parser.add_argument(
        "--presence-source",
        type=str,
        default="logits",
        choices=["logits", "heatmap", "heatmap_composite"],
        help="Fonte da presenca: logits da head ou score derivado dos heatmaps.",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.5,
        help="Threshold global de presenca (tambem usado como default na calibracao).",
    )
    parser.add_argument(
        "--presence-thresholds-json",
        type=Path,
        default=None,
        help="JSON opcional com threshold por dente (chaves 11..48).",
    )
    parser.add_argument(
        "--calibrate-presence-thresholds",
        action="store_true",
        help="Calibra threshold por dente no split atual (recomendado: val) e salva JSON.",
    )
    parser.add_argument("--calibration-threshold-min", type=float, default=0.01)
    parser.add_argument("--calibration-threshold-max", type=float, default=0.99)
    parser.add_argument("--calibration-threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--calibrated-thresholds-out",
        type=Path,
        default=None,
        help="Saida do JSON calibrado. Se omitido, salva em eval_dir.",
    )
    parser.add_argument("--hm-comp-local-window", type=int, default=7, help="Janela local (impar) para energia de vizinhos.")
    parser.add_argument("--hm-comp-w-peak", type=float, default=1.0)
    parser.add_argument("--hm-comp-w-mass", type=float, default=1.0)
    parser.add_argument("--hm-comp-w-sharp", type=float, default=1.0)
    parser.add_argument("--hm-comp-w-balance", type=float, default=0.6)
    parser.add_argument("--hm-comp-w-dist", type=float, default=0.8)
    parser.add_argument("--hm-comp-sharpness-scale", type=float, default=12.0)
    parser.add_argument("--hm-comp-dist-min", type=float, default=4.0)
    parser.add_argument("--hm-comp-dist-max", type=float, default=90.0)
    parser.add_argument("--hm-comp-dist-sigma", type=float, default=12.0)
    parser.add_argument("--rm-api-base", type=str, default=os.getenv("RM_BASE_URL", "https://api.radiomemory.com.br/ia-idoc"))
    parser.add_argument("--rm-api-username", type=str, default=os.getenv("RM_USERNAME", "test"))
    parser.add_argument("--rm-api-password", type=str, default=os.getenv("RM_PASSWORD", "A)mks8aNKjanm9"))
    parser.add_argument("--rm-api-timeout", type=int, default=int(os.getenv("RM_TIMEOUT", "120")))
    parser.add_argument("--rm-api-use-cache", action="store_true")
    parser.add_argument(
        "--rm-api-skip-errors",
        action="store_true",
        help="Nao interrompe a avaliacao se alguma imagem falhar na API; salva lista em failed_samples.json.",
    )
    parser.add_argument("--multiroi-center-ckpt", type=Path, default=DEFAULT_CENTER_CKPT)
    parser.add_argument("--multiroi-lateral-ckpt", type=Path, default=DEFAULT_LATERAL_CKPT)
    parser.add_argument(
        "--multiroi-center-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1"),
        help="Output dir base da center para resolver latest best.ckpt quando --multiroi-use-latest-from-output-dirs.",
    )
    parser.add_argument(
        "--multiroi-lateral-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/lateral_shared20_nopres_absenthm1"),
        help="Output dir base da lateral para resolver latest best.ckpt quando --multiroi-use-latest-from-output-dirs.",
    )
    parser.add_argument(
        "--multiroi-use-latest-from-output-dirs",
        action="store_true",
        help="Se ligado, ignora --multiroi-center-ckpt/--multiroi-lateral-ckpt e usa latest_run.txt + best.ckpt.",
    )
    parser.add_argument(
        "--multiroi-infer-threshold",
        type=float,
        default=-1e6,
        help="Threshold interno da lib MultiROI; default muito baixo para sempre coletar score/pontos por dente.",
    )
    parser.add_argument(
        "--multiroi-skip-errors",
        action="store_true",
        help="Nao interrompe a avaliacao MultiROI se alguma imagem falhar; salva lista em failed_samples.json.",
    )
    parser.add_argument(
        "--max-visual-samples-multiroi",
        type=int,
        default=8,
        help="Numero maximo de paineis visuais em eval_multiroi_model/pred_vs_gt_samples.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    if args.smoke:
        smoke = cfg.get("smoke_test", {})
        if "masks_dir" in cfg.get("paths", {}):
            cfg["paths"]["masks_dir"] = smoke.get("masks_dir", cfg["paths"]["masks_dir"])
        cfg["paths"]["output_dir"] = smoke.get("output_dir", cfg["paths"]["output_dir"])

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    masks_dir_cfg = cfg["paths"].get("masks_dir")
    masks_dir = _resolve_path(repo_root, masks_dir_cfg) if masks_dir_cfg else None
    split_path = _resolve_path(repo_root, cfg["paths"]["splits_path"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])
    preset_path = _resolve_path(repo_root, cfg["paths"]["preset_path"])

    source_mode = str(cfg.get("data", {}).get("source_mode", "on_the_fly"))
    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        masks_dir=masks_dir,
        source_mode=source_mode,
    )
    if not samples:
        if source_mode == "on_the_fly":
            raise FileNotFoundError(f"Nenhum par JPG+JSON encontrado em {imgs_dir} e {json_dir}")
        raise FileNotFoundError(f"Nenhum triplo JPG+JSON+NPY encontrado em {imgs_dir}, {json_dir} e {masks_dir}")

    split = make_or_load_split(
        samples=samples,
        split_path=split_path,
        seed=int(cfg["split"].get("seed", cfg.get("seed", 123))),
        val_ratio=float(cfg["split"].get("val_ratio", 0.2)),
        test_ratio=float(cfg["split"].get("test_ratio", 0.0)),
        force_regen=False,
    )

    stems = _resolve_stems_for_split(split, args.split)
    if args.max_samples is not None:
        stems = stems[: max(0, int(args.max_samples))]

    by_stem = {s.stem: s for s in samples}
    eval_samples = [by_stem[s] for s in stems if s in by_stem]
    if not eval_samples:
        raise RuntimeError(
            "Split selecionado sem amostras apos reconciliar com dados disponiveis. "
            "Use --force-regenerate-split no treino para reconstruir o split."
        )

    if args.inference_source == "rm_api":
        _run_rm_api_presence_eval(
            args=args,
            repo_root=repo_root,
            output_base_dir=output_base_dir,
            split=split,
            by_stem=by_stem,
        )
        return
    if args.inference_source == "multiroi_model":
        _run_multiroi_model_eval(
            args=args,
            repo_root=repo_root,
            output_base_dir=output_base_dir,
            split=split,
            by_stem=by_stem,
        )
        return

    preset = load_json(preset_path)
    ds = HydraTeethDataset(
        samples=eval_samples,
        preset=preset,
        augment=False,
        source_mode=source_mode,
        seed=int(cfg.get("seed", 123)),
    )

    batch_size = int(cfg.get("evaluation", {}).get("batch_size", 2))
    num_workers = int(cfg.get("evaluation", {}).get("num_workers", 0))
    persistent_workers = bool(cfg.get("evaluation", {}).get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg.get("evaluation", {}).get("prefetch_factor", 2))

    def _make_loader_kwargs(nw: int) -> Dict:
        kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": nw,
            "pin_memory": torch.cuda.is_available(),
        }
        if nw > 0:
            kwargs["persistent_workers"] = persistent_workers
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    loader = DataLoader(ds, **_make_loader_kwargs(num_workers))
    try:
        _ = next(iter(loader))
    except RuntimeError as e:
        if "torch_shm_manager" not in str(e):
            raise
        print("[WARN] DataLoader multiprocessing indisponivel neste ambiente; fallback para num_workers=0.")
        num_workers = 0
        loader = DataLoader(ds, **_make_loader_kwargs(num_workers))

    device_name = str(cfg["training"].get("device", "auto"))
    if device_name == "auto":
        device = _auto_device()
    else:
        device = torch.device(device_name)
    print(f"[DEVICE] using {device}")

    model = HydraUNetMultiTask(
        in_channels=1,
        heatmap_out_channels=64,
        presence_out_channels=32,
        backbone=cfg["model"].get("backbone", "resnet34"),
        presence_dropout=float(cfg["model"].get("presence_dropout", 0.1)),
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is not None:
        ckpt_path = _resolve_path(repo_root, str(ckpt_path))
        run_dir = _resolve_path(repo_root, str(output_base_dir / "runs" / args.run_name)) if args.run_name else ckpt_path.parent
        run_name = args.run_name or run_dir.name
    else:
        run_name = args.run_name or _latest_run_name(output_base_dir)
        if not run_name:
            raise FileNotFoundError(
                f"Nenhuma run encontrada em {output_base_dir / 'runs'}; "
                "passe --run-name ou --checkpoint."
            )
        run_dir = output_base_dir / "runs" / run_name
        ckpt_path = run_dir / "best.ckpt"

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")
    print(f"[DATA] source_mode={source_mode} split={args.split} num_samples={len(eval_samples)}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mse_rows: List[np.ndarray] = []
    dice_rows: List[np.ndarray] = []
    dist_rows: List[np.ndarray] = []
    tooth_dist_rows: List[np.ndarray] = []
    y_true_presence_rows: List[np.ndarray] = []
    y_score_presence_logits_rows: List[np.ndarray] = []
    y_score_presence_heatmap_rows: List[np.ndarray] = []
    y_score_presence_heatmap_comp_rows: List[np.ndarray] = []

    visual_out = eval_dir / "pred_vs_gt_samples"
    visual_out.mkdir(parents=True, exist_ok=True)
    max_visuals = int(cfg.get("evaluation", {}).get("num_visual_samples", 8))
    vis_saved = 0

    with torch.no_grad():
        num_batches = len(loader)
        t0 = time.time()
        for bi, batch in enumerate(loader, start=1):
            stems_batch = list(batch["stem"])
            batch = _to_device(batch, device)

            x = batch["x"]
            y = batch["y_heatmap"]
            y_presence = batch["y_presence"]

            pred = model(x)
            y_pred = torch.sigmoid(pred["heatmap_logits"])
            p_score = torch.sigmoid(pred["presence_logits"])
            hm_presence_score_simple = _presence_scores_from_heatmap(y_pred)
            hm_presence_score_composite = _presence_scores_from_heatmap_composite(
                y_pred_probs=y_pred,
                local_window=int(args.hm_comp_local_window),
                w_peak=float(args.hm_comp_w_peak),
                w_mass=float(args.hm_comp_w_mass),
                w_sharp=float(args.hm_comp_w_sharp),
                w_balance=float(args.hm_comp_w_balance),
                w_dist=float(args.hm_comp_w_dist),
                sharpness_scale=float(args.hm_comp_sharpness_scale),
                dist_min=float(args.hm_comp_dist_min),
                dist_max=float(args.hm_comp_dist_max),
                dist_sigma=float(args.hm_comp_dist_sigma),
            )

            mse = ((y_pred - y) ** 2).mean(dim=(2, 3)).cpu().numpy()
            mse_rows.append(mse)

            inter = (y_pred * y).sum(dim=(2, 3))
            denom = y_pred.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
            dice = ((2.0 * inter + 1e-6) / (denom + 1e-6)).cpu().numpy()
            dice_rows.append(dice)

            b, _, h, w = y.shape
            pred_idx = y_pred.view(b, 64, -1).argmax(dim=-1)
            gt_idx = y.view(b, 64, -1).argmax(dim=-1)

            pred_y = (pred_idx // w).float()
            pred_x = (pred_idx % w).float()
            gt_y = (gt_idx // w).float()
            gt_x = (gt_idx % w).float()

            dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2).cpu().numpy()
            dist_rows.append(dist)

            tooth_dist = np.stack([(dist[:, 2 * i] + dist[:, 2 * i + 1]) * 0.5 for i in range(32)], axis=1)
            tooth_dist_rows.append(tooth_dist)

            y_true_presence_rows.append(y_presence.cpu().numpy())
            y_score_presence_logits_rows.append(p_score.cpu().numpy())
            y_score_presence_heatmap_rows.append(hm_presence_score_simple)
            y_score_presence_heatmap_comp_rows.append(hm_presence_score_composite)

            if vis_saved < max_visuals:
                xb = x.cpu().numpy()
                yb = y.cpu().numpy()
                yp = y_pred.cpu().numpy()
                for i in range(min(len(stems_batch), max_visuals - vis_saved)):
                    panel = _overlay_pred_gt_panel(xb[i, 0], yb[i], yp[i])
                    cv2.imwrite(str(visual_out / f"{vis_saved:02d}_{stems_batch[i]}.png"), panel)
                    vis_saved += 1

            if bi == 1 or bi % 50 == 0 or bi == num_batches:
                elapsed = time.time() - t0
                rate = bi / max(elapsed, 1e-8)
                eta = (num_batches - bi) / max(rate, 1e-8)
                print(
                    "[EVAL_PROGRESS] batch={}/{} elapsed={:.1f}s eta={:.1f}s".format(
                        bi, num_batches, elapsed, eta
                    )
                )

    mse_all = np.concatenate(mse_rows, axis=0)
    dice_all = np.concatenate(dice_rows, axis=0)
    dist_all = np.concatenate(dist_rows, axis=0)
    tooth_dist_all = np.concatenate(tooth_dist_rows, axis=0)
    y_true_presence = np.concatenate(y_true_presence_rows, axis=0)
    y_score_presence_logits = np.concatenate(y_score_presence_logits_rows, axis=0)
    y_score_presence_heatmap = np.concatenate(y_score_presence_heatmap_rows, axis=0)
    y_score_presence_heatmap_comp = np.concatenate(y_score_presence_heatmap_comp_rows, axis=0)

    if args.presence_source == "logits":
        y_score_presence = y_score_presence_logits
    elif args.presence_source == "heatmap":
        y_score_presence = y_score_presence_heatmap
    else:
        y_score_presence = y_score_presence_heatmap_comp

    thresholds_per_tooth: Dict[str, float] | None = None
    calibrated_thresholds_out_path: Path | None = None
    calibration_report_path: Path | None = None
    calibration_report_rows: List[Dict] = []
    if args.presence_thresholds_json is not None:
        thr_path = _resolve_path(repo_root, str(args.presence_thresholds_json))
        thr_data = load_json(thr_path)
        if isinstance(thr_data, dict) and "thresholds_per_tooth" in thr_data:
            thr_map = thr_data["thresholds_per_tooth"]
        else:
            thr_map = thr_data
        missing = [t for t in CANONICAL_TEETH_32 if t not in thr_map]
        if missing:
            raise ValueError(f"presence-thresholds-json sem chaves: {missing}")
        thresholds_per_tooth = {t: float(thr_map[t]) for t in CANONICAL_TEETH_32}

    if args.calibrate_presence_thresholds:
        thresholds_per_tooth, calibration_report_rows = _calibrate_presence_thresholds_per_tooth(
            y_true=y_true_presence,
            y_score=y_score_presence,
            thr_min=float(args.calibration_threshold_min),
            thr_max=float(args.calibration_threshold_max),
            thr_step=float(args.calibration_threshold_step),
            default_thr=float(args.presence_threshold),
        )

        calib_out = args.calibrated_thresholds_out
        if calib_out is None:
            calib_out = eval_dir / f"presence_thresholds_{args.presence_source}_calibrated_{args.split}.json"
        else:
            calib_out = _resolve_path(repo_root, str(calib_out))
        calib_out.parent.mkdir(parents=True, exist_ok=True)
        calib_payload = {
            "presence_source": args.presence_source,
            "split_used_for_calibration": args.split,
            "thresholds_per_tooth": thresholds_per_tooth,
            "calibration_grid": {
                "min": float(args.calibration_threshold_min),
                "max": float(args.calibration_threshold_max),
                "step": float(args.calibration_threshold_step),
            },
        }
        if args.presence_source == "heatmap_composite":
            calib_payload["heatmap_composite_params"] = {
                "local_window": int(args.hm_comp_local_window),
                "w_peak": float(args.hm_comp_w_peak),
                "w_mass": float(args.hm_comp_w_mass),
                "w_sharp": float(args.hm_comp_w_sharp),
                "w_balance": float(args.hm_comp_w_balance),
                "w_dist": float(args.hm_comp_w_dist),
                "sharpness_scale": float(args.hm_comp_sharpness_scale),
                "dist_min": float(args.hm_comp_dist_min),
                "dist_max": float(args.hm_comp_dist_max),
                "dist_sigma": float(args.hm_comp_dist_sigma),
            }
        calib_out.write_text(json.dumps(calib_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        calibrated_thresholds_out_path = calib_out

        calib_report_path = eval_dir / f"presence_thresholds_{args.presence_source}_calibration_report_{args.split}.csv"
        with calib_report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(calibration_report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calibration_report_rows)
        calibration_report_path = calib_report_path
        print(f"[EVAL] calibrated_thresholds_json={calib_out}")
        print(f"[EVAL] calibrated_thresholds_report={calib_report_path}")

    presence = _presence_metrics(
        y_true=y_true_presence,
        y_score=y_score_presence,
        threshold=float(args.presence_threshold),
        thresholds_per_tooth=thresholds_per_tooth,
    )
    y_pred_presence = presence.pop("y_pred")

    pred_channel_mask = np.repeat(y_pred_presence, 2, axis=1) > 0
    present_channel_mask = np.repeat(y_true_presence, 2, axis=1) > 0.5
    dist_present = dist_all[present_channel_mask]
    dist_pred_present = dist_all[pred_channel_mask]
    mse_present = mse_all[present_channel_mask]
    dice_present = dice_all[present_channel_mask]

    present_counts_per_channel = present_channel_mask.sum(axis=0)
    mse_mean_per_channel_present = np.array(
        [
            float(np.mean(mse_all[present_channel_mask[:, c], c])) if present_counts_per_channel[c] > 0 else float("nan")
            for c in range(mse_all.shape[1])
        ],
        dtype=np.float64,
    )
    dice_mean_per_channel_present = np.array(
        [
            float(np.mean(dice_all[present_channel_mask[:, c], c])) if present_counts_per_channel[c] > 0 else float("nan")
            for c in range(dice_all.shape[1])
        ],
        dtype=np.float64,
    )

    thresholds = [3.0, 5.0, 10.0]
    global_loc = {
        "point_error_mean_px": _safe_stat(dist_present, "mean"),
        "point_error_median_px": _safe_stat(dist_present, "median"),
        "point_error_p90_px": _safe_stat(dist_present, "p90"),
    }
    for thr in thresholds:
        global_loc[f"point_within_{int(thr)}px_rate"] = float(np.mean(dist_present <= thr)) if dist_present.size > 0 else float("nan")

    global_loc_operational = {
        "point_error_mean_px": _safe_stat(dist_pred_present, "mean"),
        "point_error_median_px": _safe_stat(dist_pred_present, "median"),
        "point_error_p90_px": _safe_stat(dist_pred_present, "p90"),
    }
    for thr in thresholds:
        global_loc_operational[f"point_within_{int(thr)}px_rate"] = (
            float(np.mean(dist_pred_present <= thr)) if dist_pred_present.size > 0 else float("nan")
        )

    global_loc_operational_alias = {
        "point_error_mean_px_when_pred_presence_pos": global_loc_operational["point_error_mean_px"],
        "point_error_median_px_when_pred_presence_pos": global_loc_operational["point_error_median_px"],
        "point_error_p90_px_when_pred_presence_pos": global_loc_operational["point_error_p90_px"],
        "point_within_3px_rate_when_pred_presence_pos": global_loc_operational["point_within_3px_rate"],
        "point_within_5px_rate_when_pred_presence_pos": global_loc_operational["point_within_5px_rate"],
        "point_within_10px_rate_when_pred_presence_pos": global_loc_operational["point_within_10px_rate"],
    }

    mse_mean_per_channel = mse_all.mean(axis=0)
    mse_std_per_channel = mse_all.std(axis=0)
    dice_mean_per_channel = dice_all.mean(axis=0)
    dice_std_per_channel = dice_all.std(axis=0)

    # Normalizadores por amostra/dente no espaço global da imagem:
    # - diagonal da imagem
    # - comprimento GT do dente
    n_eval = len(eval_samples)
    diag_per_sample = np.full((n_eval,), np.nan, dtype=np.float64)
    gt_tooth_len = np.full((n_eval, 32), np.nan, dtype=np.float64)
    for si, sample in enumerate(eval_samples):
        gt_presence_s, gt_p1_s, gt_p2_s = _gt_points_and_presence_from_json(sample.json_path)
        diag_s, gt_len_s = _image_diag_and_gt_tooth_length(
            image_path=sample.image_path,
            gt_presence=gt_presence_s,
            gt_p1=gt_p1_s,
            gt_p2=gt_p2_s,
        )
        diag_per_sample[si] = diag_s
        gt_tooth_len[si, :] = gt_len_s

    tooth_dist_norm_diag_all = tooth_dist_all / np.where(diag_per_sample.reshape(-1, 1) > 1e-8, diag_per_sample.reshape(-1, 1), np.nan)
    tooth_dist_norm_gtlen_all = tooth_dist_all / np.where(gt_tooth_len > 1e-8, gt_tooth_len, np.nan)

    per_tooth_rows: List[Dict] = []
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        gt_t = y_true_presence[:, i]
        pred_t = y_pred_presence[:, i]
        dist_t = tooth_dist_all[:, i]
        dist_diag_t = tooth_dist_norm_diag_all[:, i]
        dist_gtlen_t = tooth_dist_norm_gtlen_all[:, i]

        present_mask = gt_t > 0.5
        absent_mask = ~present_mask

        dist_present_t = dist_t[present_mask]
        dist_present_diag_t = dist_diag_t[present_mask]
        dist_present_gtlen_t = dist_gtlen_t[present_mask]
        dist_present_t = dist_present_t[np.isfinite(dist_present_t)]
        dist_present_diag_t = dist_present_diag_t[np.isfinite(dist_present_diag_t)]
        dist_present_gtlen_t = dist_present_gtlen_t[np.isfinite(dist_present_gtlen_t)]

        false_point_absent_rate = float(np.mean(pred_t[absent_mask] > 0)) if absent_mask.any() else float("nan")

        pred_pos = pred_t > 0
        valid_point_when_pred_pos = float(np.mean((gt_t[pred_pos] > 0.5) & (dist_t[pred_pos] <= 5.0))) if pred_pos.any() else float("nan")

        row = {
            "tooth": tooth,
            "quadrant": quadrant_for_tooth(tooth),
            "presence_auc": presence["per_tooth"][tooth]["auc"],
            "presence_f1": presence["per_tooth"][tooth]["f1"],
            "presence_precision": presence["per_tooth"][tooth]["precision"],
            "presence_recall": presence["per_tooth"][tooth]["recall"],
            "presence_accuracy": presence["per_tooth"][tooth]["accuracy"],
            "point_error_mean_px": _safe_stat(dist_present_t, "mean"),
            "point_error_median_px": _safe_stat(dist_present_t, "median"),
            "point_error_p90_px": _safe_stat(dist_present_t, "p90"),
            "point_error_mean_over_image_diag": _safe_stat(dist_present_diag_t, "mean"),
            "point_error_median_over_image_diag": _safe_stat(dist_present_diag_t, "median"),
            "point_error_p90_over_image_diag": _safe_stat(dist_present_diag_t, "p90"),
            "point_error_mean_over_gt_tooth_len": _safe_stat(dist_present_gtlen_t, "mean"),
            "point_error_median_over_gt_tooth_len": _safe_stat(dist_present_gtlen_t, "median"),
            "point_error_p90_over_gt_tooth_len": _safe_stat(dist_present_gtlen_t, "p90"),
            "point_within_5px_rate": float(np.mean(dist_present_t <= 5.0)) if dist_present_t.size > 0 else float("nan"),
            "false_point_rate_gt_absent": false_point_absent_rate,
            "valid_point_rate_when_pred_presence_pos": valid_point_when_pred_pos,
            "pred_presence_pos_count": int((pred_t > 0).sum()),
            "gt_present_count": int(present_mask.sum()),
            "gt_absent_count": int(absent_mask.sum()),
        }
        per_tooth_rows.append(row)

    per_quadrant: Dict[str, List[Dict]] = {}
    for row in per_tooth_rows:
        per_quadrant.setdefault(row["quadrant"], []).append(row)

    per_quadrant_rows: List[Dict] = []
    for quad, rows in per_quadrant.items():
        per_quadrant_rows.append(
            {
                "quadrant": quad,
                "presence_f1_mean": float(np.nanmean([r["presence_f1"] for r in rows])),
                "presence_auc_mean": float(np.nanmean([r["presence_auc"] for r in rows])),
                "point_error_mean_px": float(np.nanmean([r["point_error_mean_px"] for r in rows])),
                "point_within_5px_rate_mean": float(np.nanmean([r["point_within_5px_rate"] for r in rows])),
                "false_point_rate_gt_absent_mean": float(np.nanmean([r["false_point_rate_gt_absent"] for r in rows])),
                "num_teeth": len(rows),
            }
        )

    summary = {
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "num_samples": int(len(eval_samples)),
        "heatmap": {
            "mse_global_mean": float(np.mean(mse_all)),
            "mse_global_std": float(np.std(mse_all)),
            "dice_global_mean": float(np.mean(dice_all)),
            "dice_global_std": float(np.std(dice_all)),
            "mse_mean_per_channel": [float(v) for v in mse_mean_per_channel],
            "mse_std_per_channel": [float(v) for v in mse_std_per_channel],
            "dice_mean_per_channel": [float(v) for v in dice_mean_per_channel],
            "dice_std_per_channel": [float(v) for v in dice_std_per_channel],
            "dice_macro_64": float(np.mean(dice_mean_per_channel)),
        },
        "heatmap_present_only": {
            "mse_present_global_mean": float(np.mean(mse_present)) if mse_present.size > 0 else float("nan"),
            "mse_present_global_std": float(np.std(mse_present)) if mse_present.size > 0 else float("nan"),
            "dice_present_global_mean": float(np.mean(dice_present)) if dice_present.size > 0 else float("nan"),
            "dice_present_global_std": float(np.std(dice_present)) if dice_present.size > 0 else float("nan"),
            "mse_mean_per_channel_present": [float(v) for v in mse_mean_per_channel_present],
            "dice_mean_per_channel_present": [float(v) for v in dice_mean_per_channel_present],
            "dice_macro_present_channels": float(np.nanmean(dice_mean_per_channel_present)),
        },
        "localization": global_loc,
        "localization_operational_pred_presence": global_loc_operational,
        "localization_operational_pred_presence_alias": global_loc_operational_alias,
        "presence": {
            "auc_macro": presence["auc_macro"],
            "f1_macro": presence["f1_macro"],
            "precision_macro": presence["precision_macro"],
            "recall_macro": presence["recall_macro"],
            "accuracy_macro": presence["accuracy_macro"],
            "source": args.presence_source,
            "threshold": presence["threshold"],
            "thresholds_per_tooth": presence["thresholds_per_tooth"],
            "heatmap_composite_params": {
                "local_window": int(args.hm_comp_local_window),
                "w_peak": float(args.hm_comp_w_peak),
                "w_mass": float(args.hm_comp_w_mass),
                "w_sharp": float(args.hm_comp_w_sharp),
                "w_balance": float(args.hm_comp_w_balance),
                "w_dist": float(args.hm_comp_w_dist),
                "sharpness_scale": float(args.hm_comp_sharpness_scale),
                "dist_min": float(args.hm_comp_dist_min),
                "dist_max": float(args.hm_comp_dist_max),
                "dist_sigma": float(args.hm_comp_dist_sigma),
            }
            if args.presence_source == "heatmap_composite"
            else None,
        },
        "combined": {
            "false_point_rate_gt_absent_global": float(np.mean((y_true_presence < 0.5) & (y_pred_presence > 0))),
            "valid_point_rate_when_pred_presence_pos_global": float(
                np.mean(
                    (y_true_presence[y_pred_presence > 0] > 0.5)
                    & (tooth_dist_all[y_pred_presence > 0] <= 5.0)
                )
            )
            if np.any(y_pred_presence > 0)
            else float("nan"),
        },
        "artifacts": {
            "metrics_per_tooth_csv": str(eval_dir / "metrics_per_tooth.csv"),
            "metrics_per_quadrant_csv": str(eval_dir / "metrics_per_quadrant.csv"),
            "pred_vs_gt_samples_dir": str(visual_out),
            "presence_thresholds_json": str(calibrated_thresholds_out_path) if calibrated_thresholds_out_path else None,
            "presence_thresholds_calibration_report_csv": str(calibration_report_path) if calibration_report_path else None,
        },
    }

    with (eval_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (eval_dir / "metrics_per_tooth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_tooth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_tooth_rows)

    with (eval_dir / "metrics_per_quadrant.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_quadrant_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_quadrant_rows)

    # Registro no dashboard (não pode interromper o fluxo principal em caso de falha).
    try:
        register_record(
            experiment_root=output_base_dir,
            kind="runs",
            record={
                "id": f"hydra_eval__{run_name}__{args.split}",
                "kind": "hydra_eval",
                "experiment": output_base_dir.name,
                "run_name": run_name,
                "split": args.split,
                "summary": {
                    "presence_f1_macro": summary["presence"]["f1_macro"],
                    "presence_auc_macro": summary["presence"]["auc_macro"],
                    "point_error_median_px": summary["localization"]["point_error_median_px"],
                    "point_within_5px_rate": summary["localization"]["point_within_5px_rate"],
                    "false_point_rate_gt_absent_global": summary["combined"]["false_point_rate_gt_absent_global"],
                },
                "artifacts": {
                    "metrics_summary_json": rel_to_experiment(eval_dir / "metrics_summary.json", output_base_dir),
                    "metrics_per_tooth_csv": rel_to_experiment(eval_dir / "metrics_per_tooth.csv", output_base_dir),
                    "metrics_per_quadrant_csv": rel_to_experiment(eval_dir / "metrics_per_quadrant.csv", output_base_dir),
                    "pred_vs_gt_samples_dir": rel_to_experiment(visual_out, output_base_dir),
                    "train_visuals_html": rel_to_experiment(run_dir / "train_visuals" / "index.html", output_base_dir),
                },
            },
        )
    except Exception as e:
        print(f"[WARN] dashboard registry skipped in eval.py: {e}")

    print(f"[EVAL] split={args.split} num_samples={len(eval_samples)}")
    print(f"[EVAL] heatmap_dice_macro_64={summary['heatmap']['dice_macro_64']:.6f}")
    print(
        "[EVAL] heatmap_dice_macro_present_channels={:.6f}".format(
            summary["heatmap_present_only"]["dice_macro_present_channels"]
        )
    )
    print(f"[EVAL] point_error_median_px={summary['localization']['point_error_median_px']:.6f}")
    print(
        "[EVAL] point_error_median_px_when_pred_presence_pos={:.6f}".format(
            summary["localization_operational_pred_presence"]["point_error_median_px"]
        )
    )
    print(f"[EVAL] presence_source={summary['presence']['source']}")
    print(f"[EVAL] presence_f1_macro={summary['presence']['f1_macro']:.6f}")
    print(f"[EVAL] summary={eval_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
