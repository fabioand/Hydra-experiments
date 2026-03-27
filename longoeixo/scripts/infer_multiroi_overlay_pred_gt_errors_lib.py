#!/usr/bin/env python3
"""Mosaico MultiROI com overlay Pred (vermelho) vs GT (verde), filtrado por erro.

Filtro vem do CSV `per_image_presence_errors.csv` gerado por
`eval_multiroi_presence_hist.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from hydra_data import discover_samples, load_json
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
    "qualitative_multiroi_pred_gt_errors_filtered"
)


def _read_errors_csv(path: Path, min_errors_exclusive: int, error_column: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = str(row.get("stem", "")).strip()
            if not stem:
                continue
            try:
                nerr = int(row.get(error_column, "0"))
            except Exception:
                continue
            if nerr > int(min_errors_exclusive):
                out[stem] = nerr
    return out


def _gt_axes_from_json(json_path: Path) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    data = load_json(json_path)
    out: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for ann in data:
        label = str(ann.get("label", "")).strip()
        if not label:
            continue
        pts = ann.get("pts", [])
        valid: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid.append((float(x), float(y)))
        if not valid:
            continue
        p1 = valid[0]
        p2 = valid[1] if len(valid) >= 2 else valid[0]
        out[label] = (p1, p2)
    return out


def _pred_best_by_tooth(pred: List[ToothPrediction]) -> Dict[str, ToothPrediction]:
    best: Dict[str, ToothPrediction] = {}
    for p in pred:
        cur = best.get(p.tooth)
        if cur is None or float(p.score) >= float(cur.score):
            best[p.tooth] = p
    return best


def _draw_axis(
    out,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Tuple[int, int, int],
    label: str | None = None,
    thickness: int = 2,
) -> None:
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.circle(out, (x1, y1), 2, color, -1, cv2.LINE_AA)
    cv2.circle(out, (x2, y2), 2, color, -1, cv2.LINE_AA)
    if label:
        xm = int(round((x1 + x2) * 0.5))
        ym = int(round((y1 + y2) * 0.5))
        cv2.putText(out, label, (xm + 2, ym - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def _draw_pred_gt_overlay(
    image_gray,
    pred: List[ToothPrediction],
    gt_axes: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
):
    out = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    pred_map = _pred_best_by_tooth(pred)
    teeth = sorted(set(gt_axes.keys()) | set(pred_map.keys()))

    # Esquema visual classico:
    # - GT sempre verde
    # - Pred sempre vermelho
    green = (0, 230, 0)
    red = (0, 0, 255)

    tp = 0
    fp = 0
    fn = 0
    for t in teeth:
        has_gt = t in gt_axes
        has_pred = t in pred_map
        if has_gt and has_pred:
            pp = pred_map[t]
            g1, g2 = gt_axes[t]
            _draw_axis(out, g1, g2, green, label=f"GT:{t}", thickness=2)
            _draw_axis(out, pp.p1, pp.p2, red, label=f"PR:{t}", thickness=2)
            tp += 1
        elif (not has_gt) and has_pred:
            pp = pred_map[t]
            _draw_axis(out, pp.p1, pp.p2, red, label=f"PR:{t}", thickness=2)
            fp += 1
        elif has_gt and (not has_pred):
            g1, g2 = gt_axes[t]
            _draw_axis(out, g1, g2, green, label=f"GT:{t}", thickness=2)
            fn += 1

    cv2.putText(out, "GT: verde | Pred: vermelho", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(out, f"TP={tp} FP={fp} FN={fn}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    return out, tp, fp, fn


def _write_html(output_dir: Path, records: List[Dict]) -> Path:
    html_path = output_dir / "index.html"
    cards = []
    for rec in records:
        cards.append(
            f"""
            <div class="card">
              <img src="{rec['overlay_file']}" alt="{rec['stem']}" loading="lazy" />
              <div class="meta">
                <b>{rec['stem']}</b><br/>
                erros_presenca_filtro: {rec['num_presence_errors_filter']}<br/>
                pred_teeth: {rec['num_predicted_teeth']} | gt_teeth: {rec['num_gt_teeth']}<br/>
                TP: {rec['tp']} | FP: {rec['fp']} | FN: {rec['fn']}
              </div>
            </div>
            """
        )

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>MultiROI Pred vs GT (Filtrado por Erros)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0f1115; color:#e7e9ee; margin:16px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:12px; }}
    .card {{ background:#1a1f2b; border:1px solid #2a3142; border-radius:8px; overflow:hidden; }}
    img {{ width:100%; height:auto; display:block; background:#000; }}
    .meta {{ padding:8px 10px; font-size:12px; color:#c8cfdd; }}
  </style>
</head>
<body>
  <h2>MultiROI Pred (vermelho) vs GT (verde)</h2>
  <p>Filtro: {records[0]['error_column'] if records else "error_column"} &gt; {records[0]['min_errors_exclusive'] if records else "n/a"} | threshold={records[0]['threshold'] if records else "n/a"}</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaico Pred vs GT MultiROI filtrado por erros de presença")
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
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--json-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--errors-csv",
        type=Path,
        required=True,
        help="CSV per_image_presence_errors.csv do eval_multiroi_presence_hist.py",
    )
    parser.add_argument(
        "--error-column",
        type=str,
        default="num_presence_errors_all_teeth",
        choices=[
            "num_presence_errors_all_teeth",
            "num_presence_errors_molars_premolars",
        ],
        help="Coluna do CSV usada para filtrar casos.",
    )
    parser.add_argument(
        "--min-errors-exclusive",
        type=int,
        default=4,
        help="Seleciona imagens com valor da coluna > este valor.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0=sem limite")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    errors_csv = resolve_path(repo_root, args.errors_csv)
    output_dir = resolve_path(repo_root, args.output_dir)
    overlays_dir = output_dir / "overlays"
    preds_dir = output_dir / "predictions_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    err_map = _read_errors_csv(
        errors_csv,
        min_errors_exclusive=int(args.min_errors_exclusive),
        error_column=str(args.error_column),
    )
    if not err_map:
        raise RuntimeError(
            f"Nenhuma imagem com {args.error_column} > {args.min_errors_exclusive} no CSV: {errors_csv}"
        )

    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        masks_dir=None,
        source_mode="on_the_fly",
    )
    by_stem = {s.stem: s for s in samples}

    chosen = [(stem, err) for stem, err in err_map.items() if stem in by_stem]
    chosen.sort(key=lambda x: (-x[1], x[0]))
    if args.max_images and int(args.max_images) > 0:
        chosen = chosen[: int(args.max_images)]
    if not chosen:
        raise RuntimeError("Nenhuma amostra selecionada apos reconciliar CSV com dataset.")

    models = load_multiroi_models(center_ckpt=center_ckpt, lateral_ckpt=lateral_ckpt)
    print(f"[INFO] device={models.device}")
    print(f"[INFO] center_ckpt={center_ckpt}")
    print(f"[INFO] lateral_ckpt={lateral_ckpt}")
    print(f"[INFO] selected_images={len(chosen)}")

    records: List[Dict] = []
    for i, (stem, nerr) in enumerate(chosen, start=1):
        sample = by_stem[stem]
        image_gray = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(f"Falha ao ler imagem: {sample.image_path}")

        result = infer_multiroi_from_image(
            image_gray=image_gray,
            models=models,
            threshold=float(args.threshold),
        )
        gt_axes = _gt_axes_from_json(sample.json_path)
        overlay, tp, fp, fn = _draw_pred_gt_overlay(image_gray, result.predictions, gt_axes)

        overlay_name = f"{stem}_pred_gt.png"
        cv2.imwrite(str(overlays_dir / overlay_name), overlay)

        pred_payload = {
            "stem": stem,
            "threshold": float(args.threshold),
            "error_column": str(args.error_column),
            "min_errors_exclusive": int(args.min_errors_exclusive),
            "num_presence_errors_filter": int(nerr),
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
        (preds_dir / f"{stem}.json").write_text(json.dumps(pred_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        records.append(
            {
                "stem": stem,
                "overlay_file": f"overlays/{overlay_name}",
                "num_presence_errors_filter": int(nerr),
                "num_predicted_teeth": int(len(result.predictions)),
                "num_gt_teeth": int(len(gt_axes)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "threshold": float(args.threshold),
                "error_column": str(args.error_column),
                "min_errors_exclusive": int(args.min_errors_exclusive),
            }
        )
        if i % 20 == 0 or i == len(chosen):
            print(f"[INFO] processed {i}/{len(chosen)}")

    html_path = _write_html(output_dir, records)
    summary = {
        "num_selected_images": len(records),
        "threshold": float(args.threshold),
        "error_column": str(args.error_column),
        "min_errors_exclusive": int(args.min_errors_exclusive),
        "errors_csv": str(errors_csv),
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
