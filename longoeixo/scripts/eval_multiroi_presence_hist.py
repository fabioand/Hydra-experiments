#!/usr/bin/env python3
"""Avalia erros de presença/ausência no modo MultiROI e gera histogramas por imagem.

Gera:
- hist_presence_errors_all_teeth.png/json/csv
- hist_presence_errors_molars_premolars.png/json/csv
- hist_presence_errors_incisors_canines.png/json/csv
- per_image_presence_errors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from hydra_data import discover_samples, load_json, make_or_load_split
from hydra_multitask_model import CANONICAL_TEETH_32
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


def _default_run_name() -> str:
    return datetime.now().strftime("multiroi_presence_hist_%Y-%m-%d_%H-%M-%S")


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
    presence = np.zeros((32,), dtype=np.int32)
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}
    data = load_json(json_path)
    for ann in data:
        label = str(ann.get("label", ""))
        idx = idx_map.get(label)
        if idx is None:
            continue
        pts = ann.get("pts", [])
        valid_pts = 0
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid_pts += 1
        if valid_pts > 0:
            presence[idx] = 1
    return presence


def _multiroi_scores_for_image(image_path: Path, models, infer_threshold: float) -> np.ndarray:
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
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}
    for pred in result.predictions:
        idx = idx_map.get(pred.tooth)
        if idx is None:
            continue
        sc = float(pred.score)
        if sc >= scores[idx]:
            scores[idx] = sc
    return scores


def _is_molar_or_premolar(tooth: str) -> bool:
    if len(tooth) != 2 or not tooth.isdigit():
        return False
    d = int(tooth[1])
    return d in {4, 5, 6, 7, 8}


def _is_incisor_or_canine(tooth: str) -> bool:
    if len(tooth) != 2 or not tooth.isdigit():
        return False
    d = int(tooth[1])
    return d in {1, 2, 3}


def _save_histogram_png(
    out_png: Path,
    counter: Counter,
    max_errors: int,
    title: str,
    xlabel: str,
) -> None:
    xs = list(range(max_errors + 1))
    ys = [int(counter.get(x, 0)) for x in xs]
    y_max = max(ys) if ys else 0

    w, h = 1500, 820
    left, right, top, bottom = 90, 30, 90, 140
    plot_w = max(1, w - left - right)
    plot_h = max(1, h - top - bottom)

    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (230, 230, 230), -1)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (70, 70, 70), 1)

    n = max(1, len(xs))
    slot_w = plot_w / float(n)
    bar_w = max(1, int(slot_w * 0.78))

    for i, x in enumerate(xs):
        yv = ys[i]
        bh = 0 if y_max <= 0 else int(round((yv / float(y_max)) * (plot_h - 2)))
        x0 = int(round(left + i * slot_w + (slot_w - bar_w) * 0.5))
        x1 = x0 + bar_w
        y1 = top + plot_h - 1
        y0 = y1 - bh
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (91, 140, 229), -1)

    # Eixos
    cv2.line(canvas, (left, top + plot_h), (left + plot_w, top + plot_h), (30, 30, 30), 2)
    cv2.line(canvas, (left, top), (left, top + plot_h), (30, 30, 30), 2)

    # Ticks Y
    y_ticks = 5
    for j in range(y_ticks + 1):
        frac = j / float(y_ticks)
        yy = int(round(top + plot_h - frac * plot_h))
        val = int(round(frac * y_max))
        cv2.line(canvas, (left - 6, yy), (left, yy), (40, 40, 40), 1)
        cv2.putText(canvas, str(val), (8, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

    # Ticks X (amostrados)
    if max_errors <= 20:
        x_tick_values = list(range(max_errors + 1))
    else:
        step = max(1, int(round((max_errors + 1) / 15.0)))
        x_tick_values = list(range(0, max_errors + 1, step))
        if x_tick_values[-1] != max_errors:
            x_tick_values.append(max_errors)
    for xv in x_tick_values:
        xx = int(round(left + (xv + 0.5) * slot_w))
        cv2.line(canvas, (xx, top + plot_h), (xx, top + plot_h + 6), (40, 40, 40), 1)
        cv2.putText(
            canvas,
            str(xv),
            (xx - 8, top + plot_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(canvas, title[:120], (left, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, xlabel, (left, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Numero de imagens", (left, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)


def _save_histogram_json_csv(
    out_json: Path,
    out_csv: Path,
    counter: Counter,
    max_errors: int,
) -> None:
    rows = []
    for k in range(max_errors + 1):
        rows.append({"num_erros": int(k), "num_imagens": int(counter.get(k, 0))})

    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["num_erros", "num_imagens"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Histograma de erros de presença no modo MultiROI")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--presence-threshold", type=float, default=0.1)
    parser.add_argument(
        "--multiroi-infer-threshold",
        type=float,
        default=-1e6,
        help="Threshold interno da lib MultiROI; mantenha baixo para coletar score de todos os dentes.",
    )
    parser.add_argument("--multiroi-center-ckpt", type=Path, default=DEFAULT_CENTER_CKPT)
    parser.add_argument("--multiroi-lateral-ckpt", type=Path, default=DEFAULT_LATERAL_CKPT)
    parser.add_argument(
        "--multiroi-center-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1"),
    )
    parser.add_argument(
        "--multiroi-lateral-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/lateral_shared20_nopres_absenthm1"),
    )
    parser.add_argument(
        "--multiroi-use-latest-from-output-dirs",
        action="store_true",
        help="Se ligado, ignora ckpts explicitos e usa latest best.ckpt dos output dirs.",
    )
    parser.add_argument(
        "--multiroi-skip-errors",
        action="store_true",
        help="Nao interrompe em amostras com falha na inferencia; salva lista de falhas.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    masks_dir_cfg = cfg["paths"].get("masks_dir")
    masks_dir = _resolve_path(repo_root, masks_dir_cfg) if masks_dir_cfg else None
    split_path = _resolve_path(repo_root, cfg["paths"]["splits_path"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])

    source_mode = str(cfg.get("data", {}).get("source_mode", "on_the_fly"))
    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        masks_dir=masks_dir,
        source_mode=source_mode,
    )
    if not samples:
        raise RuntimeError("Nenhuma amostra encontrada para avaliacao.")

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
        raise RuntimeError("Split selecionado sem amostras apos reconciliar com dados disponiveis.")

    if bool(args.multiroi_use_latest_from_output_dirs):
        center_ckpt = latest_best_ckpt(resolve_multiroi_path(repo_root, args.multiroi_center_output_dir))
        lateral_ckpt = latest_best_ckpt(resolve_multiroi_path(repo_root, args.multiroi_lateral_output_dir))
    else:
        center_ckpt = resolve_multiroi_path(repo_root, args.multiroi_center_ckpt)
        lateral_ckpt = resolve_multiroi_path(repo_root, args.multiroi_lateral_ckpt)

    run_name = args.run_name or _default_run_name()
    run_dir = output_base_dir / "runs" / run_name
    eval_dir = run_dir / "eval_multiroi_presence_hist"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    models = load_multiroi_models(center_ckpt=center_ckpt, lateral_ckpt=lateral_ckpt)
    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")
    print(f"[DATA] split={args.split} num_samples={len(eval_samples)}")
    print(f"[MULTIROI] center_ckpt={center_ckpt}")
    print(f"[MULTIROI] lateral_ckpt={lateral_ckpt}")
    print(f"[MULTIROI] device={models.device}")

    idx_pm = [i for i, t in enumerate(CANONICAL_TEETH_32) if _is_molar_or_premolar(t)]
    idx_ic = [i for i, t in enumerate(CANONICAL_TEETH_32) if _is_incisor_or_canine(t)]
    failed_samples: List[Dict[str, str]] = []
    per_image_rows: List[Dict[str, object]] = []
    hist_all = Counter()
    hist_pm = Counter()
    hist_ic = Counter()

    t0 = time.time()
    total = len(eval_samples)
    for i, sample in enumerate(eval_samples, start=1):
        gt_presence = _gt_presence_from_json(sample.json_path)
        try:
            scores = _multiroi_scores_for_image(
                image_path=sample.image_path,
                models=models,
                infer_threshold=float(args.multiroi_infer_threshold),
            )
        except Exception as e:
            failed_samples.append(
                {
                    "stem": sample.stem,
                    "image_path": str(sample.image_path),
                    "error": str(e),
                }
            )
            if not bool(args.multiroi_skip_errors):
                raise
            print(f"[WARN] sample_failed stem={sample.stem} error={e}")
            continue

        pred_presence = (scores >= float(args.presence_threshold)).astype(np.int32)
        err_all = int((pred_presence != gt_presence).sum())
        err_pm = int((pred_presence[idx_pm] != gt_presence[idx_pm]).sum())
        err_ic = int((pred_presence[idx_ic] != gt_presence[idx_ic]).sum())
        hist_all[err_all] += 1
        hist_pm[err_pm] += 1
        hist_ic[err_ic] += 1
        per_image_rows.append(
            {
                "stem": sample.stem,
                "num_presence_errors_all_teeth": err_all,
                "num_presence_errors_molars_premolars": err_pm,
                "num_presence_errors_incisors_canines": err_ic,
            }
        )

        if i == 1 or i % 25 == 0 or i == total:
            elapsed = time.time() - t0
            it_s = elapsed / max(i, 1)
            eta = (total - i) * it_s
            print(f"[PROGRESS] {i}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    max_all = max(hist_all.keys()) if hist_all else 0
    max_pm = max(hist_pm.keys()) if hist_pm else 0
    max_ic = max(hist_ic.keys()) if hist_ic else 0

    _save_histogram_png(
        out_png=eval_dir / "hist_presence_errors_all_teeth.png",
        counter=hist_all,
        max_errors=max_all,
        title="Histograma: numero de dentes com erro de presenca por imagem (todos os dentes)",
        xlabel="Numero de dentes com erro de presenca (0..32)",
    )
    _save_histogram_png(
        out_png=eval_dir / "hist_presence_errors_molars_premolars.png",
        counter=hist_pm,
        max_errors=max_pm,
        title="Histograma: erros de presenca por imagem (apenas molares e pre-molares)",
        xlabel="Numero de dentes com erro de presenca (0..20)",
    )
    _save_histogram_png(
        out_png=eval_dir / "hist_presence_errors_incisors_canines.png",
        counter=hist_ic,
        max_errors=max_ic,
        title="Histograma: erros de presenca por imagem (apenas incisivos e caninos)",
        xlabel="Numero de dentes com erro de presenca (0..12)",
    )

    _save_histogram_json_csv(
        out_json=eval_dir / "hist_presence_errors_all_teeth.json",
        out_csv=eval_dir / "hist_presence_errors_all_teeth.csv",
        counter=hist_all,
        max_errors=max_all,
    )
    _save_histogram_json_csv(
        out_json=eval_dir / "hist_presence_errors_molars_premolars.json",
        out_csv=eval_dir / "hist_presence_errors_molars_premolars.csv",
        counter=hist_pm,
        max_errors=max_pm,
    )
    _save_histogram_json_csv(
        out_json=eval_dir / "hist_presence_errors_incisors_canines.json",
        out_csv=eval_dir / "hist_presence_errors_incisors_canines.csv",
        counter=hist_ic,
        max_errors=max_ic,
    )

    per_csv = eval_dir / "per_image_presence_errors.csv"
    with per_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "num_presence_errors_all_teeth",
                "num_presence_errors_molars_premolars",
                "num_presence_errors_incisors_canines",
            ],
        )
        w.writeheader()
        for row in per_image_rows:
            w.writerow(row)

    summary = {
        "mode": "multiroi_presence_hist",
        "split": args.split,
        "num_samples_requested": len(eval_samples),
        "num_samples_ok": len(per_image_rows),
        "num_samples_failed": len(failed_samples),
        "presence_threshold": float(args.presence_threshold),
        "multiroi_infer_threshold": float(args.multiroi_infer_threshold),
        "center_ckpt": str(center_ckpt),
        "lateral_ckpt": str(lateral_ckpt),
        "output_dir": str(eval_dir),
        "hist_all_max_errors": int(max_all),
        "hist_pm_max_errors": int(max_pm),
        "hist_ic_max_errors": int(max_ic),
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed_samples:
        (eval_dir / "failed_samples.json").write_text(
            json.dumps(failed_samples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[DONE] eval_dir={eval_dir}")
    print(f"[DONE] summary={eval_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
