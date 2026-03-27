#!/usr/bin/env python3
"""Conta duplas FN/FP (trocas) a partir da lista por imagem do histograma MultiROI.

Entrada:
- per_image_presence_errors.csv (saida do eval_multiroi_presence_hist.py)

Processo:
- Para cada stem da lista, roda inferencia MultiROI oficial e compara com GT.
- Extrai FN/FP por dente (presenca) e conta trocas por pares definidos.

Saidas:
- swap_pairs_summary.json
- swap_pairs_by_pair.csv
- swap_pairs_by_exam.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np

from hydra_data import discover_samples, load_json
from hydra_multitask_model import CANONICAL_TEETH_32
from longoeixo.scripts.multiroi_composed_inference import (
    DEFAULT_CENTER_CKPT,
    DEFAULT_LATERAL_CKPT,
    infer_multiroi_from_image,
    latest_best_ckpt,
    load_multiroi_models,
    resolve_path,
)


PREMOLAR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("14", "15"),
    ("24", "25"),
    ("34", "35"),
    ("44", "45"),
)

MOLAR_2ND_3RD_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("17", "18"),
    ("27", "28"),
    ("37", "38"),
    ("47", "48"),
)


def _read_stems(per_image_csv: Path) -> List[str]:
    stems: List[str] = []
    with per_image_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "stem" not in (r.fieldnames or []):
            raise ValueError("CSV sem coluna 'stem'.")
        for row in r:
            stem = str(row.get("stem", "")).strip()
            if stem:
                stems.append(stem)
    return stems


def _gt_presence_from_json(json_path: Path) -> np.ndarray:
    arr = np.zeros((32,), dtype=np.int32)
    idx_map = {t: i for i, t in enumerate(CANONICAL_TEETH_32)}
    data = load_json(json_path)
    for ann in data:
        label = str(ann.get("label", ""))
        idx = idx_map.get(label)
        if idx is None:
            continue
        pts = ann.get("pts", [])
        valid = 0
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid += 1
        if valid > 0:
            arr[idx] = 1
    return arr


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
    idx_map = {t: i for i, t in enumerate(CANONICAL_TEETH_32)}
    for pred in result.predictions:
        idx = idx_map.get(pred.tooth)
        if idx is None:
            continue
        sc = float(pred.score)
        if sc >= scores[idx]:
            scores[idx] = sc
    return scores


def _count_pair_swaps(fn_teeth: Set[str], fp_teeth: Set[str], a: str, b: str) -> Tuple[int, int, int]:
    a_fn_b_fp = int((a in fn_teeth) and (b in fp_teeth))
    b_fn_a_fp = int((b in fn_teeth) and (a in fp_teeth))
    return int(a_fn_b_fp or b_fn_a_fp), a_fn_b_fp, b_fn_a_fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Conta trocas FN/FP por pares usando lista do histograma.")
    parser.add_argument("--per-image-errors-csv", type=Path, required=True)
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--json-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument("--presence-threshold", type=float, default=0.1)
    parser.add_argument("--multiroi-infer-threshold", type=float, default=0.1)
    parser.add_argument("--center-ckpt", type=Path, default=DEFAULT_CENTER_CKPT)
    parser.add_argument("--lateral-ckpt", type=Path, default=DEFAULT_LATERAL_CKPT)
    parser.add_argument(
        "--center-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1"),
    )
    parser.add_argument(
        "--lateral-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/lateral_shared20_nopres_absenthm1"),
    )
    parser.add_argument("--use-latest-from-output-dirs", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    per_image_csv = resolve_path(repo_root, args.per_image_errors_csv)
    out_dir = resolve_path(repo_root, args.output_dir) if args.output_dir else per_image_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_latest_from_output_dirs:
        center_ckpt = latest_best_ckpt(resolve_path(repo_root, args.center_output_dir))
        lateral_ckpt = latest_best_ckpt(resolve_path(repo_root, args.lateral_output_dir))
    else:
        center_ckpt = resolve_path(repo_root, args.center_ckpt)
        lateral_ckpt = resolve_path(repo_root, args.lateral_ckpt)

    stems = _read_stems(per_image_csv)
    stem_set = set(stems)
    if not stems:
        raise RuntimeError("Lista de stems vazia no per_image_presence_errors.csv")

    imgs_dir = resolve_path(repo_root, args.imgs_dir)
    json_dir = resolve_path(repo_root, args.json_dir)
    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        masks_dir=None,
        source_mode="on_the_fly",
    )
    by_stem = {s.stem: s for s in samples}
    selected = [by_stem[s] for s in stems if s in by_stem]
    missing = [s for s in stems if s not in by_stem]

    models = load_multiroi_models(center_ckpt=center_ckpt, lateral_ckpt=lateral_ckpt)
    print(f"[INFO] selected={len(selected)} missing={len(missing)} device={models.device}")

    pair_rows: Dict[str, Dict[str, object]] = {}
    for a, b in PREMOLAR_PAIRS:
        key = f"{a}<->{b}"
        pair_rows[key] = {
            "group": "premolar_adjacent",
            "pair": key,
            "num_exams_with_swap": 0,
            "num_swaps_a_fn_b_fp": 0,
            "num_swaps_b_fn_a_fp": 0,
        }
    for a, b in MOLAR_2ND_3RD_PAIRS:
        key = f"{a}<->{b}"
        pair_rows[key] = {
            "group": "molar_2nd_3rd",
            "pair": key,
            "num_exams_with_swap": 0,
            "num_swaps_a_fn_b_fp": 0,
            "num_swaps_b_fn_a_fp": 0,
        }

    idx_map = {t: i for i, t in enumerate(CANONICAL_TEETH_32)}
    by_exam_rows: List[Dict[str, object]] = []
    premolar_hits = 0
    molar_hits = 0

    for i, sample in enumerate(selected, start=1):
        gt = _gt_presence_from_json(sample.json_path)
        scores = _multiroi_scores_for_image(sample.image_path, models, infer_threshold=float(args.multiroi_infer_threshold))
        pred = (scores >= float(args.presence_threshold)).astype(np.int32)

        fn_idx = np.where((gt == 1) & (pred == 0))[0]
        fp_idx = np.where((gt == 0) & (pred == 1))[0]
        fn_teeth = {CANONICAL_TEETH_32[j] for j in fn_idx}
        fp_teeth = {CANONICAL_TEETH_32[j] for j in fp_idx}

        exam_pairs: List[str] = []
        exam_groups: Set[str] = set()
        for a, b in PREMOLAR_PAIRS:
            hit, a2b, b2a = _count_pair_swaps(fn_teeth, fp_teeth, a, b)
            if not hit:
                continue
            key = f"{a}<->{b}"
            r = pair_rows[key]
            r["num_exams_with_swap"] = int(r["num_exams_with_swap"]) + 1
            r["num_swaps_a_fn_b_fp"] = int(r["num_swaps_a_fn_b_fp"]) + int(a2b)
            r["num_swaps_b_fn_a_fp"] = int(r["num_swaps_b_fn_a_fp"]) + int(b2a)
            exam_pairs.append(key)
            exam_groups.add("premolar_adjacent")
            premolar_hits += 1

        for a, b in MOLAR_2ND_3RD_PAIRS:
            hit, a2b, b2a = _count_pair_swaps(fn_teeth, fp_teeth, a, b)
            if not hit:
                continue
            key = f"{a}<->{b}"
            r = pair_rows[key]
            r["num_exams_with_swap"] = int(r["num_exams_with_swap"]) + 1
            r["num_swaps_a_fn_b_fp"] = int(r["num_swaps_a_fn_b_fp"]) + int(a2b)
            r["num_swaps_b_fn_a_fp"] = int(r["num_swaps_b_fn_a_fp"]) + int(b2a)
            exam_pairs.append(key)
            exam_groups.add("molar_2nd_3rd")
            molar_hits += 1

        by_exam_rows.append(
            {
                "stem": sample.stem,
                "has_any_swap_pair": int(bool(exam_pairs)),
                "num_swap_pairs": len(exam_pairs),
                "swap_pairs": ";".join(sorted(exam_pairs)),
                "swap_groups": ";".join(sorted(exam_groups)),
                "fn_teeth": ";".join(sorted(fn_teeth)),
                "fp_teeth": ";".join(sorted(fp_teeth)),
            }
        )

        if i == 1 or i % 25 == 0 or i == len(selected):
            print(f"[PROGRESS] {i}/{len(selected)}")

    by_pair_sorted = sorted(
        pair_rows.values(),
        key=lambda r: (str(r["group"]), -int(r["num_exams_with_swap"]), str(r["pair"])),
    )

    by_exam_csv = out_dir / "swap_pairs_by_exam.csv"
    with by_exam_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "has_any_swap_pair",
                "num_swap_pairs",
                "swap_pairs",
                "swap_groups",
                "fn_teeth",
                "fp_teeth",
            ],
        )
        w.writeheader()
        w.writerows(by_exam_rows)

    by_pair_csv = out_dir / "swap_pairs_by_pair.csv"
    with by_pair_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "pair",
                "num_exams_with_swap",
                "num_swaps_a_fn_b_fp",
                "num_swaps_b_fn_a_fp",
            ],
        )
        w.writeheader()
        w.writerows(by_pair_sorted)

    num_exams_with_any = int(sum(int(r["has_any_swap_pair"]) for r in by_exam_rows))
    summary = {
        "per_image_errors_csv": str(per_image_csv),
        "num_stems_in_list": len(stems),
        "num_samples_found": len(selected),
        "num_missing_stems": len(missing),
        "num_exams_with_any_swap_pair": num_exams_with_any,
        "presence_threshold": float(args.presence_threshold),
        "multiroi_infer_threshold": float(args.multiroi_infer_threshold),
        "center_ckpt": str(center_ckpt),
        "lateral_ckpt": str(lateral_ckpt),
        "premolar_adjacent_swap_pairs_total": int(premolar_hits),
        "molar_2nd_3rd_swap_pairs_total": int(molar_hits),
    }
    summary_json = out_dir / "swap_pairs_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if missing:
        missing_json = out_dir / "swap_pairs_missing_stems.json"
        missing_json.write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] summary={summary_json}")
    print(f"[DONE] by_pair={by_pair_csv}")
    print(f"[DONE] by_exam={by_exam_csv}")
    print(
        "[SUMMARY] "
        f"premolar_pairs={premolar_hits} molar_2nd_3rd_pairs={molar_hits} "
        f"exams_with_any_swap={num_exams_with_any}"
    )


if __name__ == "__main__":
    main()
