#!/usr/bin/env python3
"""Audita anomalias morfologicas de anotacao e gera ranking + histogramas."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from hydra_data import discover_samples, load_json, make_or_load_split
from hydra_multitask_model import CANONICAL_TEETH_32
from dashboard_registry import rel_to_experiment, register_record

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


QUADRANT_TEETH: Dict[str, List[str]] = {
    "Q1": ["11", "12", "13", "14", "15", "16", "17", "18"],
    "Q2": ["21", "22", "23", "24", "25", "26", "27", "28"],
    "Q3": ["31", "32", "33", "34", "35", "36", "37", "38"],
    "Q4": ["41", "42", "43", "44", "45", "46", "47", "48"],
}


@dataclass(frozen=True)
class ToothGeometry:
    cx: float
    cy: float
    axis_len: float
    axis_tilt: float


@dataclass(frozen=True)
class SampleGeometry:
    stem: str
    teeth: Dict[str, ToothGeometry]
    jaw_gap: float | None
    upper_median_y: float | None
    lower_median_y: float | None


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _is_upper(tooth: str) -> bool:
    return tooth.startswith("1") or tooth.startswith("2")


def _is_lower(tooth: str) -> bool:
    return tooth.startswith("3") or tooth.startswith("4")


def _adjacent_pairs() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for teeth in QUADRANT_TEETH.values():
        for i in range(len(teeth) - 1):
            out.append((teeth[i], teeth[i + 1]))
    return out


def _load_tooth_geometry(json_path: Path) -> Dict[str, ToothGeometry]:
    data = load_json(json_path)
    out: Dict[str, ToothGeometry] = {}

    for ann in data:
        tooth = str(ann.get("label", ""))
        if tooth not in CANONICAL_TEETH_32 or tooth in out:
            continue
        pts = ann.get("pts", [])
        valid: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid.append((float(x), float(y)))
        if len(valid) < 2:
            continue

        x1, y1 = valid[0]
        x2, y2 = valid[1]
        dx = x2 - x1
        dy = y2 - y1
        axis_len = float(math.hypot(dx, dy))
        axis_tilt = float(abs(dx) / (abs(dy) + 1e-6))

        out[tooth] = ToothGeometry(
            cx=float((x1 + x2) * 0.5),
            cy=float((y1 + y2) * 0.5),
            axis_len=axis_len,
            axis_tilt=axis_tilt,
        )

    return out


def _extract_sample_geometry(stem: str, json_path: Path) -> SampleGeometry:
    teeth = _load_tooth_geometry(json_path)
    upper_ys = [g.cy for t, g in teeth.items() if _is_upper(t)]
    lower_ys = [g.cy for t, g in teeth.items() if _is_lower(t)]

    upper_med = float(np.median(np.array(upper_ys, dtype=np.float64))) if upper_ys else None
    lower_med = float(np.median(np.array(lower_ys, dtype=np.float64))) if lower_ys else None
    jaw_gap = (lower_med - upper_med) if (upper_med is not None and lower_med is not None) else None

    return SampleGeometry(
        stem=stem,
        teeth=teeth,
        jaw_gap=float(jaw_gap) if jaw_gap is not None else None,
        upper_median_y=upper_med,
        lower_median_y=lower_med,
    )


def _parse_semicolon_list(value: object) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    return [x for x in (item.strip() for item in s.split(";")) if x]


def _load_tooth_lines(json_path: Path) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    data = load_json(json_path)
    out: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    for ann in data:
        tooth = str(ann.get("label", ""))
        if tooth not in CANONICAL_TEETH_32 or tooth in out:
            continue
        pts = ann.get("pts", [])
        valid: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid.append((float(x), float(y)))
        if len(valid) >= 2:
            out[tooth] = (valid[0], valid[1])

    return out


def _midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[int, int]:
    return (int(round((p1[0] + p2[0]) * 0.5)), int(round((p1[1] + p2[1]) * 0.5)))


def _draw_morphology_overlay(
    img_gray: np.ndarray,
    tooth_lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    upper_low_teeth: set[str],
    lower_high_teeth: set[str],
    axis_len_teeth: set[str],
    axis_tilt_teeth: set[str],
    order_pairs: List[str],
    spacing_pairs: List[str],
    upper_median_y: float | None,
    lower_median_y: float | None,
) -> np.ndarray:
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    pair_teeth: set[str] = set()
    for pair in order_pairs + spacing_pairs:
        parts = pair.split("-")
        if len(parts) == 2:
            pair_teeth.update(parts)

    for tooth, (p1, p2) in tooth_lines.items():
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))

        color = (0, 255, 0)  # verde = normal
        if tooth in pair_teeth:
            color = (0, 200, 255)  # amarelo/laranja = par suspeito
        if tooth in axis_len_teeth or tooth in axis_tilt_teeth:
            color = (255, 180, 0)  # ciano = eixo anomalo
        if tooth in upper_low_teeth or tooth in lower_high_teeth:
            color = (0, 0, 255)  # vermelho = altura anomala

        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, cv2.LINE_AA)
        mx, my = _midpoint(p1, p2)
        cv2.putText(canvas, tooth, (mx + 2, my - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)

    for pair in order_pairs:
        parts = pair.split("-")
        if len(parts) != 2:
            continue
        a, b = parts
        la = tooth_lines.get(a)
        lb = tooth_lines.get(b)
        if la is None or lb is None:
            continue
        ma = _midpoint(la[0], la[1])
        mb = _midpoint(lb[0], lb[1])
        cv2.line(canvas, ma, mb, (0, 170, 255), 1, cv2.LINE_AA)

    for pair in spacing_pairs:
        parts = pair.split("-")
        if len(parts) != 2:
            continue
        a, b = parts
        la = tooth_lines.get(a)
        lb = tooth_lines.get(b)
        if la is None or lb is None:
            continue
        ma = _midpoint(la[0], la[1])
        mb = _midpoint(lb[0], lb[1])
        cv2.line(canvas, ma, mb, (255, 120, 0), 1, cv2.LINE_AA)

    h, w = canvas.shape[:2]
    if upper_median_y is not None:
        y = int(round(upper_median_y))
        cv2.line(canvas, (0, y), (w - 1, y), (255, 180, 180), 1, cv2.LINE_AA)
    if lower_median_y is not None:
        y = int(round(lower_median_y))
        cv2.line(canvas, (0, y), (w - 1, y), (180, 180, 255), 1, cv2.LINE_AA)

    return canvas


def _robust_stat(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0.0, "median": float("nan"), "scale": float("nan"), "mad": float("nan"), "std": float("nan")}

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = float(1.4826 * mad)
    std = float(np.std(arr))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = std
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    return {"count": float(arr.size), "median": med, "scale": scale, "mad": mad, "std": std}


def _z(x: float, stat: Dict[str, float]) -> float:
    scale = float(stat.get("scale", 1.0))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    med = float(stat.get("median", 0.0))
    return float((x - med) / scale)


def _stems_for_split(split: Dict[str, List[str]], split_name: str) -> List[str]:
    if split_name == "train":
        return split["train"]
    if split_name == "val":
        return split["val"]
    if split_name == "test":
        if "test" not in split:
            raise RuntimeError("Split 'test' indisponivel no arquivo de split atual.")
        return split["test"]
    return split["train"] + split["val"] + split.get("test", [])


def _build_baseline(
    geoms: List[SampleGeometry],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict], Dict[str, float]]:
    tooth_values: Dict[str, Dict[str, List[float]]] = {
        t: {"cy": [], "axis_len": [], "axis_tilt": []} for t in CANONICAL_TEETH_32
    }
    pair_values: Dict[str, Dict[str, List[float]]] = {
        f"{a}-{b}": {"dx": [], "abs_dx": []} for a, b in _adjacent_pairs()
    }
    jaw_gap_values: List[float] = []

    for sg in geoms:
        for tooth, g in sg.teeth.items():
            tooth_values[tooth]["cy"].append(g.cy)
            tooth_values[tooth]["axis_len"].append(g.axis_len)
            tooth_values[tooth]["axis_tilt"].append(g.axis_tilt)

        for a, b in _adjacent_pairs():
            ga = sg.teeth.get(a)
            gb = sg.teeth.get(b)
            if ga is None or gb is None:
                continue
            dx = float(gb.cx - ga.cx)
            pair_values[f"{a}-{b}"]["dx"].append(dx)
            pair_values[f"{a}-{b}"]["abs_dx"].append(abs(dx))

        if sg.jaw_gap is not None:
            jaw_gap_values.append(float(sg.jaw_gap))

    tooth_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tooth, d in tooth_values.items():
        tooth_stats[tooth] = {
            "cy": _robust_stat(d["cy"]),
            "axis_len": _robust_stat(d["axis_len"]),
            "axis_tilt": _robust_stat(d["axis_tilt"]),
        }

    pair_stats: Dict[str, Dict] = {}
    for key, d in pair_values.items():
        dx_stat = _robust_stat(d["dx"])
        abs_dx_stat = _robust_stat(d["abs_dx"])
        med_dx = float(dx_stat["median"]) if np.isfinite(dx_stat["median"]) else 0.0
        if med_dx > 0:
            expected_sign = 1
        elif med_dx < 0:
            expected_sign = -1
        else:
            expected_sign = 0
        pair_stats[key] = {
            "dx": dx_stat,
            "abs_dx": abs_dx_stat,
            "expected_sign": expected_sign,
        }

    jaw_gap_stat = _robust_stat(jaw_gap_values)
    return tooth_stats, pair_stats, jaw_gap_stat


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria morfologica de anotacoes (heuristica)")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--baseline-split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-baseline-count", type=int, default=30)
    parser.add_argument("--vertical-z-threshold", type=float, default=2.8)
    parser.add_argument("--spacing-z-threshold", type=float, default=3.0)
    parser.add_argument("--axis-len-z-threshold", type=float, default=3.0)
    parser.add_argument("--axis-tilt-z-threshold", type=float, default=3.0)
    parser.add_argument("--jaw-gap-z-threshold", type=float, default=3.0)
    parser.add_argument("--w-order", type=float, default=3.0)
    parser.add_argument("--w-spacing", type=float, default=2.0)
    parser.add_argument("--w-upper-low", type=float, default=2.5)
    parser.add_argument("--w-lower-high", type=float, default=2.5)
    parser.add_argument("--w-axis-len", type=float, default=1.2)
    parser.add_argument("--w-axis-tilt", type=float, default=1.2)
    parser.add_argument("--w-jaw-gap", type=float, default=2.0)
    parser.add_argument(
        "--overlay-top-k",
        type=int,
        default=None,
        help="Quantidade de overlays/itens no mosaico HTML (default: igual a --top-k; 0 desativa).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    split_path = _resolve_path(repo_root, cfg["paths"]["splits_path"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])

    samples = discover_samples(imgs_dir=imgs_dir, json_dir=json_dir, masks_dir=None, source_mode="on_the_fly")
    if not samples:
        raise FileNotFoundError(f"Nenhum par JPG+JSON encontrado em {imgs_dir} e {json_dir}")

    split = make_or_load_split(
        samples=samples,
        split_path=split_path,
        seed=int(cfg["split"].get("seed", cfg.get("seed", 123))),
        val_ratio=float(cfg["split"].get("val_ratio", 0.2)),
        test_ratio=float(cfg["split"].get("test_ratio", 0.0)),
        force_regen=False,
    )

    eval_stems = _stems_for_split(split, args.split)
    baseline_stems = _stems_for_split(split, args.baseline_split)

    by_stem = {s.stem: s for s in samples}
    eval_stems = [s for s in eval_stems if s in by_stem]
    baseline_stems = [s for s in baseline_stems if s in by_stem]
    if not eval_stems:
        raise RuntimeError("Split selecionado sem amostras apos reconciliar com dados disponiveis.")
    if not baseline_stems:
        baseline_stems = list(eval_stems)

    out_dir = args.out_dir or (output_base_dir / "annotation_audit" / f"morphology_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    needed_stems = sorted(set(eval_stems).union(baseline_stems))
    all_geom: Dict[str, SampleGeometry] = {}
    t0 = time.time()
    n_needed = len(needed_stems)
    print(
        f"[MORPH_AUDIT] split={args.split} baseline_split={args.baseline_split} "
        f"eval_samples={len(eval_stems)} baseline_samples={len(baseline_stems)}"
    )

    for i, stem in enumerate(needed_stems, start=1):
        sample = by_stem[stem]
        all_geom[stem] = _extract_sample_geometry(stem=stem, json_path=sample.json_path)
        if i == 1 or i % 400 == 0 or i == n_needed:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-8)
            eta = (n_needed - i) / max(rate, 1e-8)
            print(f"[MORPH_AUDIT_PROGRESS] load {i}/{n_needed} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    eval_geoms = [all_geom[s] for s in eval_stems if s in all_geom]
    baseline_geoms = [all_geom[s] for s in baseline_stems if s in all_geom]
    if not baseline_geoms:
        baseline_geoms = eval_geoms

    tooth_stats, pair_stats, jaw_gap_stat = _build_baseline(baseline_geoms)
    min_count = int(args.min_baseline_count)

    sample_rows: List[Dict] = []
    tooth_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    t1 = time.time()
    n_eval = len(eval_geoms)
    for i, sg in enumerate(eval_geoms, start=1):
        order_pairs: List[str] = []
        spacing_pairs: List[str] = []
        upper_low_teeth: List[str] = []
        lower_high_teeth: List[str] = []
        axis_len_teeth: List[str] = []
        axis_tilt_teeth: List[str] = []
        severity_sum = 0.0

        order_inversion_count = 0
        spacing_outlier_count = 0
        upper_low_count = 0
        lower_high_count = 0
        axis_len_outlier_count = 0
        axis_tilt_outlier_count = 0

        for tooth in CANONICAL_TEETH_32:
            g = sg.teeth.get(tooth)
            if g is None:
                continue
            tstat = tooth_stats[tooth]

            z_cy = float("nan")
            z_axis_len = float("nan")
            z_axis_tilt = float("nan")
            flag_upper_low = 0
            flag_lower_high = 0
            flag_axis_len = 0
            flag_axis_tilt = 0

            cy_count = int(tstat["cy"]["count"])
            if cy_count >= min_count:
                z_cy = _z(g.cy, tstat["cy"])
                if _is_upper(tooth) and z_cy > args.vertical_z_threshold:
                    flag_upper_low = 1
                    upper_low_count += 1
                    upper_low_teeth.append(tooth)
                    severity_sum += float(z_cy - args.vertical_z_threshold)
                elif _is_lower(tooth) and z_cy < -args.vertical_z_threshold:
                    flag_lower_high = 1
                    lower_high_count += 1
                    lower_high_teeth.append(tooth)
                    severity_sum += float((-z_cy) - args.vertical_z_threshold)

            len_count = int(tstat["axis_len"]["count"])
            if len_count >= min_count:
                z_axis_len = _z(g.axis_len, tstat["axis_len"])
                if abs(z_axis_len) > args.axis_len_z_threshold:
                    flag_axis_len = 1
                    axis_len_outlier_count += 1
                    axis_len_teeth.append(tooth)
                    severity_sum += float(abs(z_axis_len) - args.axis_len_z_threshold)

            tilt_count = int(tstat["axis_tilt"]["count"])
            if tilt_count >= min_count:
                z_axis_tilt = _z(g.axis_tilt, tstat["axis_tilt"])
                if z_axis_tilt > args.axis_tilt_z_threshold:
                    flag_axis_tilt = 1
                    axis_tilt_outlier_count += 1
                    axis_tilt_teeth.append(tooth)
                    severity_sum += float(z_axis_tilt - args.axis_tilt_z_threshold)

            tooth_rows.append(
                {
                    "stem": sg.stem,
                    "tooth": tooth,
                    "cx": float(g.cx),
                    "cy": float(g.cy),
                    "axis_len": float(g.axis_len),
                    "axis_tilt": float(g.axis_tilt),
                    "z_cy": float(z_cy),
                    "z_axis_len": float(z_axis_len),
                    "z_axis_tilt": float(z_axis_tilt),
                    "flag_upper_low": int(flag_upper_low),
                    "flag_lower_high": int(flag_lower_high),
                    "flag_axis_len_outlier": int(flag_axis_len),
                    "flag_axis_tilt_outlier": int(flag_axis_tilt),
                }
            )

        for a, b in _adjacent_pairs():
            ga = sg.teeth.get(a)
            gb = sg.teeth.get(b)
            if ga is None or gb is None:
                continue

            key = f"{a}-{b}"
            pstat = pair_stats[key]
            dx = float(gb.cx - ga.cx)
            abs_dx = float(abs(dx))
            expected_sign = int(pstat["expected_sign"])
            z_abs_dx = float("nan")
            flag_order = 0
            flag_spacing = 0

            abs_count = int(pstat["abs_dx"]["count"])
            if abs_count >= min_count:
                if expected_sign != 0:
                    dx_sign = 1 if dx > 0 else (-1 if dx < 0 else 0)
                    if dx_sign != expected_sign:
                        flag_order = 1
                        order_inversion_count += 1
                        order_pairs.append(key)
                        severity_sum += 1.0

                z_abs_dx = _z(abs_dx, pstat["abs_dx"])
                if abs(z_abs_dx) > args.spacing_z_threshold:
                    flag_spacing = 1
                    spacing_outlier_count += 1
                    spacing_pairs.append(key)
                    severity_sum += float(abs(z_abs_dx) - args.spacing_z_threshold)

            pair_rows.append(
                {
                    "stem": sg.stem,
                    "pair": key,
                    "dx": dx,
                    "abs_dx": abs_dx,
                    "expected_sign": expected_sign,
                    "z_abs_dx": z_abs_dx,
                    "flag_order_inversion": int(flag_order),
                    "flag_spacing_outlier": int(flag_spacing),
                }
            )

        jaw_gap_small_flag = 0
        jaw_gap_large_flag = 0
        z_jaw_gap = float("nan")
        if sg.jaw_gap is not None and int(jaw_gap_stat["count"]) >= min_count:
            z_jaw_gap = _z(float(sg.jaw_gap), jaw_gap_stat)
            if z_jaw_gap < -args.jaw_gap_z_threshold:
                jaw_gap_small_flag = 1
                severity_sum += float((-z_jaw_gap) - args.jaw_gap_z_threshold)
            elif z_jaw_gap > args.jaw_gap_z_threshold:
                jaw_gap_large_flag = 1
                severity_sum += float(z_jaw_gap - args.jaw_gap_z_threshold)

        total_flags = (
            order_inversion_count
            + spacing_outlier_count
            + upper_low_count
            + lower_high_count
            + axis_len_outlier_count
            + axis_tilt_outlier_count
            + jaw_gap_small_flag
            + jaw_gap_large_flag
        )

        suspect_score = (
            args.w_order * order_inversion_count
            + args.w_spacing * spacing_outlier_count
            + args.w_upper_low * upper_low_count
            + args.w_lower_high * lower_high_count
            + args.w_axis_len * axis_len_outlier_count
            + args.w_axis_tilt * axis_tilt_outlier_count
            + args.w_jaw_gap * (jaw_gap_small_flag + jaw_gap_large_flag)
            + 0.25 * severity_sum
        )

        sample_rows.append(
            {
                "stem": sg.stem,
                "teeth_with_2pts_count": int(len(sg.teeth)),
                "order_inversion_count": int(order_inversion_count),
                "spacing_outlier_count": int(spacing_outlier_count),
                "upper_low_count": int(upper_low_count),
                "lower_high_count": int(lower_high_count),
                "axis_len_outlier_count": int(axis_len_outlier_count),
                "axis_tilt_outlier_count": int(axis_tilt_outlier_count),
                "jaw_gap_small_flag": int(jaw_gap_small_flag),
                "jaw_gap_large_flag": int(jaw_gap_large_flag),
                "total_anomaly_flags": int(total_flags),
                "severity_sum": float(severity_sum),
                "suspect_score": float(suspect_score),
                "jaw_gap": float(sg.jaw_gap) if sg.jaw_gap is not None else float("nan"),
                "z_jaw_gap": float(z_jaw_gap),
                "order_pairs": ";".join(sorted(set(order_pairs))),
                "spacing_pairs": ";".join(sorted(set(spacing_pairs))),
                "upper_low_teeth": ";".join(sorted(set(upper_low_teeth))),
                "lower_high_teeth": ";".join(sorted(set(lower_high_teeth))),
                "axis_len_outlier_teeth": ";".join(sorted(set(axis_len_teeth))),
                "axis_tilt_outlier_teeth": ";".join(sorted(set(axis_tilt_teeth))),
            }
        )

        if i == 1 or i % 300 == 0 or i == n_eval:
            elapsed = time.time() - t1
            rate = i / max(elapsed, 1e-8)
            eta = (n_eval - i) / max(rate, 1e-8)
            print(f"[MORPH_AUDIT_PROGRESS] score {i}/{n_eval} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    rows_sorted = sorted(
        sample_rows,
        key=lambda r: (
            -r["suspect_score"],
            -r["total_anomaly_flags"],
            -r["order_inversion_count"],
            -r["upper_low_count"],
            -r["lower_high_count"],
            r["stem"],
        ),
    )
    top_k = max(1, int(args.top_k))
    top_rows = rows_sorted[:top_k]
    overlay_top_k = top_k if args.overlay_top_k is None else max(0, int(args.overlay_top_k))

    sample_csv = out_dir / "morphology_suspects_per_sample.csv"
    top_csv = out_dir / f"top_morphology_suspects_top{top_k}.csv"
    tooth_csv = out_dir / "morphology_features_per_tooth.csv"
    pair_csv = out_dir / "morphology_pair_checks.csv"
    indicator_hist_csv = out_dir / "morphology_indicator_histogram.csv"
    flag_hist_csv = out_dir / "morphology_flagcount_histogram.csv"
    summary_json = out_dir / "morphology_audit_summary.json"

    sample_fieldnames = list(rows_sorted[0].keys()) if rows_sorted else [
        "stem",
        "teeth_with_2pts_count",
        "order_inversion_count",
        "spacing_outlier_count",
        "upper_low_count",
        "lower_high_count",
        "axis_len_outlier_count",
        "axis_tilt_outlier_count",
        "jaw_gap_small_flag",
        "jaw_gap_large_flag",
        "total_anomaly_flags",
        "severity_sum",
        "suspect_score",
        "jaw_gap",
        "z_jaw_gap",
        "order_pairs",
        "spacing_pairs",
        "upper_low_teeth",
        "lower_high_teeth",
        "axis_len_outlier_teeth",
        "axis_tilt_outlier_teeth",
    ]
    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    with top_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    tooth_fieldnames = list(tooth_rows[0].keys()) if tooth_rows else [
        "stem",
        "tooth",
        "cx",
        "cy",
        "axis_len",
        "axis_tilt",
        "z_cy",
        "z_axis_len",
        "z_axis_tilt",
        "flag_upper_low",
        "flag_lower_high",
        "flag_axis_len_outlier",
        "flag_axis_tilt_outlier",
    ]
    with tooth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tooth_fieldnames)
        writer.writeheader()
        writer.writerows(tooth_rows)

    pair_fieldnames = list(pair_rows[0].keys()) if pair_rows else [
        "stem",
        "pair",
        "dx",
        "abs_dx",
        "expected_sign",
        "z_abs_dx",
        "flag_order_inversion",
        "flag_spacing_outlier",
    ]
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    indicator_specs = [
        ("order_inversion", "order_inversion_count"),
        ("spacing_outlier", "spacing_outlier_count"),
        ("upper_low", "upper_low_count"),
        ("lower_high", "lower_high_count"),
        ("axis_len_outlier", "axis_len_outlier_count"),
        ("axis_tilt_outlier", "axis_tilt_outlier_count"),
        ("jaw_gap_small", "jaw_gap_small_flag"),
        ("jaw_gap_large", "jaw_gap_large_flag"),
    ]
    indicator_rows: List[Dict] = []
    for name, field in indicator_specs:
        vals = np.array([int(r[field]) for r in rows_sorted], dtype=np.int32)
        indicator_rows.append(
            {
                "indicator": name,
                "num_samples_with_flag": int(np.sum(vals > 0)),
                "total_occurrences": int(np.sum(vals)),
                "fraction_samples_with_flag": float(np.mean(vals > 0)) if vals.size > 0 else float("nan"),
            }
        )
    with indicator_hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["indicator", "num_samples_with_flag", "total_occurrences", "fraction_samples_with_flag"]
        )
        writer.writeheader()
        writer.writerows(indicator_rows)

    flag_counts = np.array([int(r["total_anomaly_flags"]) for r in rows_sorted], dtype=np.int32)
    max_flags = int(flag_counts.max()) if flag_counts.size > 0 else 0
    flag_hist_rows: List[Dict] = []
    for k in range(max_flags + 1):
        count_k = int(np.sum(flag_counts == k))
        flag_hist_rows.append(
            {
                "total_anomaly_flags": k,
                "num_radiographs": count_k,
                "fraction": float(count_k / max(1, len(rows_sorted))),
            }
        )
    with flag_hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["total_anomaly_flags", "num_radiographs", "fraction"])
        writer.writeheader()
        writer.writerows(flag_hist_rows)

    indicator_hist_png = out_dir / "morphology_indicator_histogram.png"
    flag_hist_png = out_dir / "morphology_flagcount_histogram.png"
    indicator_hist_png_written = False
    flag_hist_png_written = False
    if plt is not None and indicator_rows:
        fig = plt.figure(figsize=(11, 4.5))
        ax = fig.add_subplot(111)
        xs = [r["indicator"] for r in indicator_rows]
        ys = [r["num_samples_with_flag"] for r in indicator_rows]
        ax.bar(xs, ys)
        ax.set_title("Morphology Indicators - Samples With Flag")
        ax.set_ylabel("Num radiografias")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(indicator_hist_png, dpi=140)
        plt.close(fig)
        indicator_hist_png_written = True

    if plt is not None and flag_hist_rows:
        fig = plt.figure(figsize=(11, 4.5))
        ax = fig.add_subplot(111)
        xs = [r["total_anomaly_flags"] for r in flag_hist_rows]
        ys = [r["num_radiographs"] for r in flag_hist_rows]
        ax.bar(xs, ys, width=0.85)
        ax.set_title("Morphology Anomaly Flags Per Radiograph")
        ax.set_xlabel("Total de flags por radiografia")
        ax.set_ylabel("Num radiografias")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(flag_hist_png, dpi=140)
        plt.close(fig)
        flag_hist_png_written = True

    overlay_html_path = None
    overlay_img_dir = None
    if overlay_top_k > 0:
        overlay_rows = top_rows[:overlay_top_k]
        overlay_img_dir = out_dir / "images"
        overlay_img_dir.mkdir(parents=True, exist_ok=True)
        cards: List[str] = []

        t2 = time.time()
        m = len(overlay_rows)
        print(f"[MORPH_AUDIT] generating_overlays top_k={m}")
        for i, row in enumerate(overlay_rows, start=1):
            stem = str(row["stem"])
            sample = by_stem.get(stem)
            if sample is None:
                continue

            img_gray = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue

            tooth_lines = _load_tooth_lines(sample.json_path)
            sg = all_geom.get(stem)
            overlay = _draw_morphology_overlay(
                img_gray=img_gray,
                tooth_lines=tooth_lines,
                upper_low_teeth=set(_parse_semicolon_list(row.get("upper_low_teeth"))),
                lower_high_teeth=set(_parse_semicolon_list(row.get("lower_high_teeth"))),
                axis_len_teeth=set(_parse_semicolon_list(row.get("axis_len_outlier_teeth"))),
                axis_tilt_teeth=set(_parse_semicolon_list(row.get("axis_tilt_outlier_teeth"))),
                order_pairs=_parse_semicolon_list(row.get("order_pairs")),
                spacing_pairs=_parse_semicolon_list(row.get("spacing_pairs")),
                upper_median_y=sg.upper_median_y if sg is not None else None,
                lower_median_y=sg.lower_median_y if sg is not None else None,
            )

            out_name = f"{i:04d}_{stem}.png"
            cv2.imwrite(str(overlay_img_dir / out_name), overlay)

            subtitle = (
                f"score={float(row['suspect_score']):.3f} | flags={int(row['total_anomaly_flags'])} | "
                f"ord={int(row['order_inversion_count'])} sp={int(row['spacing_outlier_count'])} "
                f"upLow={int(row['upper_low_count'])} lowHigh={int(row['lower_high_count'])}"
            )
            has_height = int(int(row["upper_low_count"]) > 0 or int(row["lower_high_count"]) > 0)
            has_axis = int(int(row["axis_len_outlier_count"]) > 0 or int(row["axis_tilt_outlier_count"]) > 0)
            has_pairs = int(int(row["order_inversion_count"]) > 0 or int(row["spacing_outlier_count"]) > 0)
            is_normal = int((has_height + has_axis + has_pairs) == 0)
            category_tags = []
            if is_normal:
                category_tags.append("normal")
            if has_height:
                category_tags.append("altura")
            if has_axis:
                category_tags.append("eixo")
            if has_pairs:
                category_tags.append("pares")
            categories_label = ",".join(category_tags) if category_tags else "normal"
            cards.append(
                (
                    f"<div class='card' data-normal='{is_normal}' data-height='{has_height}' data-axis='{has_axis}' data-pairs='{has_pairs}'>"
                    f"<img src='images/{html.escape(out_name)}' loading='lazy'/>"
                    "<div class='meta'>"
                    f"<div class='stem'>{html.escape(stem)}</div>"
                    f"<div class='sub'>{html.escape(subtitle)}</div>"
                    f"<div class='sub'>cats={html.escape(categories_label)}</div>"
                    "</div></div>"
                )
            )

            if i == 1 or i % 20 == 0 or i == m:
                elapsed = time.time() - t2
                rate = i / max(elapsed, 1e-8)
                eta = (m - i) / max(rate, 1e-8)
                print(f"[MORPH_AUDIT_PROGRESS] overlay {i}/{m} elapsed={elapsed:.1f}s eta={eta:.1f}s")

        overlay_html_path = out_dir / "index.html"
        html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Morphology Suspects Overlay</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1115; color: #e8edf2; }}
    .header {{ margin-bottom: 14px; }}
    .legend {{ font-size: 13px; opacity: 0.9; }}
    .filters {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .filters label {{ font-size: 13px; user-select: none; }}
    .filters input {{ margin-right: 4px; }}
    .muted {{ font-size: 12px; opacity: 0.8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
    .card {{ background: #171b22; border: 1px solid #2a3340; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; }}
    .meta {{ padding: 8px 10px; }}
    .stem {{ font-size: 13px; font-weight: 600; word-break: break-all; }}
    .sub {{ font-size: 12px; opacity: 0.9; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="header">
    <h2>Morphology Suspects Overlay</h2>
    <div class="legend">Verde = normal | Vermelho = altura anomala | Ciano = eixo anomalo | Amarelo/Laranja = pares suspeitos de ordem/espacamento</div>
    <div class="legend">Linhas claras horizontais = medianas de arcada (superior/inferior)</div>
    <div class="legend">split={html.escape(args.split)} | baseline_split={html.escape(args.baseline_split)} | top_k={len(cards)}</div>
    <div class="filters">
      <label><input type="checkbox" class="filt" value="normal" checked/>Normal (verde)</label>
      <label><input type="checkbox" class="filt" value="height" checked/>Altura anomala (vermelho)</label>
      <label><input type="checkbox" class="filt" value="axis" checked/>Eixo anomalo (ciano)</label>
      <label><input type="checkbox" class="filt" value="pairs" checked/>Pares suspeitos (amarelo/laranja)</label>
      <span id="visibleCount" class="muted"></span>
    </div>
  </div>
  <div class="grid">{''.join(cards)}</div>
  <script>
    const checks = Array.from(document.querySelectorAll('.filt'));
    const cards = Array.from(document.querySelectorAll('.card'));
    const visibleCount = document.getElementById('visibleCount');

    function selectedCategories() {{
      return checks.filter(c => c.checked).map(c => c.value);
    }}

    function applyFilters() {{
      const selected = selectedCategories();
      let shown = 0;
      for (const card of cards) {{
        const matches =
          selected.length === 0 ||
          selected.some(cat => card.dataset[cat] === '1');
        card.style.display = matches ? '' : 'none';
        if (matches) shown += 1;
      }}
      visibleCount.textContent = `visiveis: ${'{'}shown{'}'}/${'{'}cards.length{'}'}`;
    }}

    for (const c of checks) {{
      c.addEventListener('change', applyFilters);
    }}
    applyFilters();
  </script>
</body>
</html>
"""
        overlay_html_path.write_text(html_doc, encoding="utf-8")

    summary = {
        "split": args.split,
        "baseline_split": args.baseline_split,
        "num_eval_samples": len(rows_sorted),
        "num_baseline_samples": len(baseline_geoms),
        "thresholds": {
            "min_baseline_count": min_count,
            "vertical_z_threshold": float(args.vertical_z_threshold),
            "spacing_z_threshold": float(args.spacing_z_threshold),
            "axis_len_z_threshold": float(args.axis_len_z_threshold),
            "axis_tilt_z_threshold": float(args.axis_tilt_z_threshold),
            "jaw_gap_z_threshold": float(args.jaw_gap_z_threshold),
        },
        "weights": {
            "w_order": float(args.w_order),
            "w_spacing": float(args.w_spacing),
            "w_upper_low": float(args.w_upper_low),
            "w_lower_high": float(args.w_lower_high),
            "w_axis_len": float(args.w_axis_len),
            "w_axis_tilt": float(args.w_axis_tilt),
            "w_jaw_gap": float(args.w_jaw_gap),
            "severity_weight": 0.25,
        },
        "score_formula": (
            "w_order*order + w_spacing*spacing + w_upper_low*upper_low + "
            "w_lower_high*lower_high + w_axis_len*axis_len + w_axis_tilt*axis_tilt + "
            "w_jaw_gap*(jaw_gap_small+jaw_gap_large) + 0.25*severity_sum"
        ),
        "artifacts": {
            "sample_csv": str(sample_csv),
            "top_csv": str(top_csv),
            "tooth_csv": str(tooth_csv),
            "pair_csv": str(pair_csv),
            "indicator_histogram_csv": str(indicator_hist_csv),
            "flagcount_histogram_csv": str(flag_hist_csv),
            "indicator_histogram_png": str(indicator_hist_png) if indicator_hist_png_written else None,
            "flagcount_histogram_png": str(flag_hist_png) if flag_hist_png_written else None,
            "overlay_html": str(overlay_html_path) if overlay_html_path is not None else None,
            "overlay_images_dir": str(overlay_img_dir) if overlay_img_dir is not None else None,
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        register_record(
            experiment_root=output_base_dir,
            kind="audits",
            record={
                "id": f"morphology_audit__{args.split}__{args.baseline_split}",
                "kind": "morphology_audit",
                "experiment": output_base_dir.name,
                "run_name": None,
                "split": args.split,
                "summary": {
                    "num_eval_samples": summary["num_eval_samples"],
                    "num_baseline_samples": summary["num_baseline_samples"],
                    "vertical_z_threshold": summary["thresholds"]["vertical_z_threshold"],
                    "spacing_z_threshold": summary["thresholds"]["spacing_z_threshold"],
                },
                "artifacts": {
                    "summary_json": rel_to_experiment(summary_json, output_base_dir),
                    "sample_csv": rel_to_experiment(sample_csv, output_base_dir),
                    "top_csv": rel_to_experiment(top_csv, output_base_dir),
                    "tooth_csv": rel_to_experiment(tooth_csv, output_base_dir),
                    "pair_csv": rel_to_experiment(pair_csv, output_base_dir),
                    "indicator_histogram_csv": rel_to_experiment(indicator_hist_csv, output_base_dir),
                    "flagcount_histogram_csv": rel_to_experiment(flag_hist_csv, output_base_dir),
                    "overlay_html": rel_to_experiment(overlay_html_path, output_base_dir) if overlay_html_path else None,
                    "overlay_images_dir": rel_to_experiment(overlay_img_dir, output_base_dir) if overlay_img_dir else None,
                },
            },
        )
    except Exception as e:
        print(f"[WARN] dashboard registry skipped in audit_morphology_suspects.py: {e}")

    print(f"[MORPH_AUDIT] sample_csv={sample_csv}")
    print(f"[MORPH_AUDIT] top_csv={top_csv}")
    print(f"[MORPH_AUDIT] indicator_hist={indicator_hist_csv}")
    if overlay_html_path is not None:
        print(f"[MORPH_AUDIT] html={overlay_html_path}")
    print(f"[MORPH_AUDIT] summary={summary_json}")


if __name__ == "__main__":
    main()
