#!/usr/bin/env python3
"""Mosaico qualitativo Multi-ROI usando biblioteca composta.

Este script apenas:
- seleciona amostras
- chama a biblioteca de inferencia composta
- renderiza paineis
- salva HTML + JSONs de predicoes
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from hydra_data import discover_samples, load_json
from hydra_multitask_model import CANONICAL_TEETH_32
from longoeixo.scripts.multiroi_composed_inference import (
    DEFAULT_CENTER_CKPT,
    DEFAULT_LATERAL_CKPT,
    ToothPrediction,
    infer_multiroi_from_image,
    latest_best_ckpt,
    load_multiroi_models,
    resolve_path,
)


DEFAULT_OUTPUT_DIR = Path(
    "longoeixo/experiments/hydra_roi_fixed_shared_lateral/"
    "qualitative_multiroi_100_lib"
)


def _draw_axes_on_bgr(image_bgr: np.ndarray, preds: List[ToothPrediction]) -> np.ndarray:
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


def _draw_overlay(image_gray: np.ndarray, preds: List[ToothPrediction]) -> np.ndarray:
    out = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    return _draw_axes_on_bgr(out, preds)


def _draw_heatmap_fusion(image_gray: np.ndarray, hm_global_max: np.ndarray, alpha: float = 0.50) -> np.ndarray:
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


def _make_horizontal_panels(panels: List[np.ndarray], gap: int = 10) -> np.ndarray:
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


def _write_html(output_dir: Path, records: List[Dict]) -> Path:
    html_path = output_dir / "index.html"
    items = []
    for rec in records:
        rel = rec["overlay_file"]
        stem = rec["stem"]
        n = rec["num_predicted_teeth"]
        err_type = rec.get("presence_error_type", "")
        err_tooth = rec.get("presence_error_tooth", "")
        err_txt = rec.get("presence_error_desc", "")
        err_count = rec.get("presence_error_count", -1)
        items.append(
            f"""
            <div class="card">
              <img src="{rel}" alt="{stem}" loading="lazy" />
              <div class="meta"><b>{stem}</b><br/>pred teeth: {n}<br/>num_errors: {err_count}<br/>erro presenca: {err_txt}<br/>tipo: {err_type} | dente: {err_tooth}</div>
            </div>
            """
        )
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Hydra Multi-ROI Qualitativo (Lib)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0f1115; color:#e7e9ee; margin:16px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:12px; }}
    .card {{ background:#1a1f2b; border:1px solid #2a3142; border-radius:8px; overflow:hidden; }}
    img {{ width:100%; height:auto; display:block; background:#000; }}
    .meta {{ padding:8px 10px; font-size:12px; color:#c8cfdd; }}
  </style>
</head>
<body>
  <h2>Hydra Multi-ROI (via biblioteca composta)</h2>
  <p>[1] radiografia+eixos | [2] heatmap global (logits crus)+eixos | threshold={records[0]["threshold"] if records else "n/a"}</p>
  <div class="grid">
    {''.join(items)}
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _gt_presence_from_json(json_path: Path) -> np.ndarray:
    arr = np.zeros((32,), dtype=np.int32)
    idx_map = {t: i for i, t in enumerate(CANONICAL_TEETH_32)}
    data = load_json(json_path)
    for ann in data:
        t = str(ann.get("label", ""))
        idx = idx_map.get(t)
        if idx is None:
            continue
        pts = ann.get("pts", [])
        valid = 0
        for pt in pts:
            if pt.get("x") is not None and pt.get("y") is not None:
                valid += 1
        if valid > 0:
            arr[idx] = 1
    return arr


def _pred_presence_from_predictions(predictions, threshold: float) -> np.ndarray:
    idx_map = {t: i for i, t in enumerate(CANONICAL_TEETH_32)}
    scores = np.full((32,), -1e6, dtype=np.float64)
    for p in predictions:
        idx = idx_map.get(p.tooth)
        if idx is None:
            continue
        score = float(p.score)
        if score >= scores[idx]:
            scores[idx] = score
    return (scores >= float(threshold)).astype(np.int32)


def _presence_error_desc(sample_json: Path, predictions, threshold: float) -> Dict[str, str]:
    gt = _gt_presence_from_json(sample_json)
    pred = _pred_presence_from_predictions(predictions, threshold=threshold)
    fn_idx = np.where((gt == 1) & (pred == 0))[0]
    fp_idx = np.where((gt == 0) & (pred == 1))[0]

    if len(fn_idx) == 1 and len(fp_idx) == 0:
        t = CANONICAL_TEETH_32[int(fn_idx[0])]
        return {"presence_error_type": "FN", "presence_error_tooth": t, "presence_error_desc": f"FN:{t}"}
    if len(fn_idx) == 0 and len(fp_idx) == 1:
        t = CANONICAL_TEETH_32[int(fp_idx[0])]
        return {"presence_error_type": "FP", "presence_error_tooth": t, "presence_error_desc": f"FP:{t}"}

    if len(fn_idx) == 0 and len(fp_idx) == 0:
        return {"presence_error_type": "NONE", "presence_error_tooth": "-", "presence_error_desc": "NONE"}

    fn_teeth = [CANONICAL_TEETH_32[int(i)] for i in fn_idx]
    fp_teeth = [CANONICAL_TEETH_32[int(i)] for i in fp_idx]
    return {
        "presence_error_type": "MIXED",
        "presence_error_tooth": f"FN:{';'.join(fn_teeth)} FP:{';'.join(fp_teeth)}",
        "presence_error_desc": f"MIXED fn={len(fn_teeth)} fp={len(fp_teeth)}",
    }


def _stems_from_error_csv(
    per_image_csv: Path,
    error_count_eq: int | None,
    error_count_min: int | None,
    error_count_max: int | None,
    sort_by_error_count_desc: bool,
) -> List[tuple[str, int]]:
    rows: List[tuple[str, int]] = []
    with per_image_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "stem" not in (r.fieldnames or []):
            raise ValueError(f"CSV sem coluna 'stem': {per_image_csv}")
        for row in r:
            stem = str(row.get("stem", "")).strip()
            if not stem:
                continue
            nerr = int(row.get("num_presence_errors_all_teeth", "0"))
            if error_count_eq is not None and nerr != int(error_count_eq):
                continue
            if error_count_min is not None and nerr < int(error_count_min):
                continue
            if error_count_max is not None and nerr > int(error_count_max):
                continue
            rows.append((stem, nerr))
    if sort_by_error_count_desc:
        rows.sort(key=lambda x: (-int(x[1]), x[0]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia qualitativa Multi-ROI via biblioteca composta")
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
        "--per-image-errors-csv",
        type=Path,
        default=None,
        help="CSV de erros por imagem (ex.: per_image_presence_errors.csv). Se informado, seleciona stems por esse arquivo.",
    )
    parser.add_argument(
        "--error-count-eq",
        type=int,
        default=None,
        help="Filtra no CSV por num_presence_errors_all_teeth == valor (ex.: 1).",
    )
    parser.add_argument(
        "--error-count-min",
        type=int,
        default=None,
        help="Filtra no CSV por num_presence_errors_all_teeth >= valor.",
    )
    parser.add_argument(
        "--error-count-max",
        type=int,
        default=None,
        help="Filtra no CSV por num_presence_errors_all_teeth <= valor.",
    )
    parser.add_argument(
        "--sort-by-error-count-desc",
        action="store_true",
        help="Quando usar --per-image-errors-csv, ordena por maior numero de erros primeiro.",
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
        center_ckpt = latest_best_ckpt(resolve_path(repo_root, args.center_output_dir))
        lateral_ckpt = latest_best_ckpt(resolve_path(repo_root, args.lateral_output_dir))
    else:
        center_ckpt = resolve_path(repo_root, args.center_ckpt)
        lateral_ckpt = resolve_path(repo_root, args.lateral_ckpt)
    imgs_dir = resolve_path(repo_root, args.imgs_dir)
    json_dir = resolve_path(repo_root, args.json_dir)
    output_dir = resolve_path(repo_root, args.output_dir)

    overlays_dir = output_dir / "overlays"
    preds_dir = output_dir / "predictions_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    models = load_multiroi_models(center_ckpt=center_ckpt, lateral_ckpt=lateral_ckpt)
    print(f"[INFO] device={models.device}")
    print(f"[INFO] center_ckpt={center_ckpt}")
    print(f"[INFO] lateral_ckpt={lateral_ckpt}")

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

    if args.per_image_errors_csv is not None:
        per_image_csv = resolve_path(repo_root, args.per_image_errors_csv)
        rows = _stems_from_error_csv(
            per_image_csv=per_image_csv,
            error_count_eq=args.error_count_eq,
            error_count_min=args.error_count_min,
            error_count_max=args.error_count_max,
            sort_by_error_count_desc=bool(args.sort_by_error_count_desc),
        )
        stems = [s for s, _ in rows]
        err_by_stem = {s: n for s, n in rows}
        chosen = [by_stem[s] for s in stems if s in by_stem]
        if not bool(args.sort_by_error_count_desc):
            chosen = sorted(chosen, key=lambda s: s.stem)
        print(
            f"[INFO] per_image_errors_csv={per_image_csv} | "
            f"selected_images={len(chosen)} / listed={len(stems)} / total={len(pool)}"
        )
    elif args.split_path is not None:
        split_path = resolve_path(repo_root, args.split_path)
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
        image_gray = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            print(f"[WARN] falha ao ler imagem, pulando: {sample.image_path}")
            continue

        result = infer_multiroi_from_image(
            image_gray=image_gray,
            models=models,
            threshold=float(args.threshold),
        )

        panel_1 = _draw_overlay(image_gray, result.predictions)
        panel_2 = _draw_axes_on_bgr(
            _draw_heatmap_fusion(image_gray, result.heatmaps.global_max, alpha=0.50),
            result.predictions,
        )
        overlay = _make_horizontal_panels([panel_1, panel_2], gap=10)

        overlay_name = f"{sample.stem}_overlay.png"
        overlay_path = overlays_dir / overlay_name
        cv2.imwrite(str(overlay_path), overlay)

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
                for p in result.predictions
            ],
        }
        (preds_dir / f"{sample.stem}.json").write_text(
            json.dumps(pred_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        records.append(
            {
                "stem": sample.stem,
                "overlay_file": f"overlays/{overlay_name}",
                "num_predicted_teeth": len(result.predictions),
                "threshold": float(args.threshold),
                "presence_error_count": int(err_by_stem[sample.stem]) if args.per_image_errors_csv is not None else -1,
                **_presence_error_desc(sample_json=sample.json_path, predictions=result.predictions, threshold=float(args.threshold)),
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
