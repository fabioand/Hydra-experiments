#!/usr/bin/env python3
"""Audita inversao de eixo (p1/p2) nas anotacoes e gera ranking + mosaico HTML."""

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

from hydra_data import discover_samples, load_json, make_or_load_split
from hydra_multitask_model import CANONICAL_TEETH_32
from dashboard_registry import rel_to_experiment, register_record


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _tooth_quadrant(tooth: str) -> str:
    if tooth.startswith("1"):
        return "Q1"
    if tooth.startswith("2"):
        return "Q2"
    if tooth.startswith("3"):
        return "Q3"
    return "Q4"


def _is_upper_tooth(tooth: str) -> bool:
    return tooth.startswith("1") or tooth.startswith("2")


def _is_lower_tooth(tooth: str) -> bool:
    return tooth.startswith("3") or tooth.startswith("4")


def _load_lines_and_inversions(json_path: Path) -> Tuple[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]], List[str]]:
    data = load_json(json_path)
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    inverted: List[str] = []

    for ann in data:
        tooth = str(ann.get("label", ""))
        if tooth not in CANONICAL_TEETH_32:
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
        if tooth in lines:
            continue

        p1 = valid[0]
        p2 = valid[1]
        lines[tooth] = (p1, p2)

        # Regra solicitada pelo usuario (origem da imagem no topo):
        # - superiores: esperado p1.y > p2.y, inversao se p1.y < p2.y
        # - inferiores: esperado p1.y < p2.y, inversao se p1.y > p2.y
        if _is_upper_tooth(tooth) and (p1[1] < p2[1]):
            inverted.append(tooth)
        elif _is_lower_tooth(tooth) and (p1[1] > p2[1]):
            inverted.append(tooth)

    return lines, sorted(inverted)


def _draw_overlay(
    img_gray: np.ndarray,
    lines: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    inverted_set: set[str],
) -> np.ndarray:
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for tooth, (p1, p2) in lines.items():
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))

        color = (0, 0, 255) if tooth in inverted_set else (0, 255, 0)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, cv2.LINE_AA)

        lx = int(round((x1 + x2) * 0.5 + 2))
        ly = int(round((y1 + y2) * 0.5 - 2))
        cv2.putText(canvas, tooth, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria de inversao de eixo p1/p2")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    split_path = _resolve_path(repo_root, cfg["paths"]["splits_path"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])

    samples = discover_samples(imgs_dir=imgs_dir, json_dir=json_dir, masks_dir=None, source_mode="on_the_fly")
    split = make_or_load_split(
        samples=samples,
        split_path=split_path,
        seed=int(cfg["split"].get("seed", cfg.get("seed", 123))),
        val_ratio=float(cfg["split"].get("val_ratio", 0.2)),
        test_ratio=float(cfg["split"].get("test_ratio", 0.0)),
        force_regen=False,
    )

    if args.split == "train":
        stems = split["train"]
    elif args.split == "val":
        stems = split["val"]
    elif args.split == "test":
        if "test" not in split:
            raise RuntimeError("Split 'test' indisponivel no arquivo de split atual.")
        stems = split["test"]
    else:
        stems = split["train"] + split["val"] + split.get("test", [])

    by_stem = {s.stem: s for s in samples}
    eval_samples = [by_stem[s] for s in stems if s in by_stem]

    out_dir = args.out_dir or (output_base_dir / "annotation_audit" / f"axis_inversion_{args.split}")
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    t0 = time.time()
    n = len(eval_samples)
    print(f"[AXIS_AUDIT] split={args.split} samples={n}")

    for i, sample in enumerate(eval_samples, start=1):
        lines, inverted = _load_lines_and_inversions(sample.json_path)
        n_lines = len(lines)
        inv_count = len(inverted)
        inv_rate = float(inv_count / max(1, n_lines))
        score = float(inv_count + inv_rate)  # prioridade por contagem + taxa

        inv_q1 = sum(1 for t in inverted if _tooth_quadrant(t) == "Q1")
        inv_q2 = sum(1 for t in inverted if _tooth_quadrant(t) == "Q2")
        inv_q3 = sum(1 for t in inverted if _tooth_quadrant(t) == "Q3")
        inv_q4 = sum(1 for t in inverted if _tooth_quadrant(t) == "Q4")

        rows.append(
            {
                "stem": sample.stem,
                "inverted_count": inv_count,
                "inverted_rate": inv_rate,
                "total_teeth_with_2pts": n_lines,
                "inv_q1": inv_q1,
                "inv_q2": inv_q2,
                "inv_q3": inv_q3,
                "inv_q4": inv_q4,
                "inverted_teeth": ";".join(inverted),
                "suspect_score": score,
            }
        )

        if i == 1 or i % 200 == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-8)
            eta = (n - i) / max(rate, 1e-8)
            print(f"[AXIS_AUDIT_PROGRESS] scan {i}/{n} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -r["inverted_count"],
            -r["inverted_rate"],
            -r["suspect_score"],
            r["stem"],
        ),
    )
    top_k = max(1, int(args.top_k))
    top_rows = rows_sorted[:top_k]

    full_csv = out_dir / "axis_inversion_per_sample.csv"
    top_csv = out_dir / f"axis_inversion_top_errors_top{top_k}.csv"

    fieldnames = list(rows_sorted[0].keys()) if rows_sorted else [
        "stem",
        "inverted_count",
        "inverted_rate",
        "total_teeth_with_2pts",
        "inv_q1",
        "inv_q2",
        "inv_q3",
        "inv_q4",
        "inverted_teeth",
        "suspect_score",
    ]

    with full_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_sorted)

    with top_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(top_rows)

    # Overlay do top-K
    cards: List[str] = []
    t1 = time.time()
    m = len(top_rows)
    for i, row in enumerate(top_rows, start=1):
        stem = row["stem"]
        sample = by_stem.get(stem)
        if sample is None:
            continue

        img_gray = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        lines, inverted = _load_lines_and_inversions(sample.json_path)
        overlay = _draw_overlay(img_gray, lines=lines, inverted_set=set(inverted))

        out_name = f"{i:04d}_{stem}.png"
        cv2.imwrite(str(img_dir / out_name), overlay)

        subtitle = (
            f"inverted={row['inverted_count']} | rate={float(row['inverted_rate']):.3f} | "
            f"teeth={row['inverted_teeth']}"
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

        if i == 1 or i % 20 == 0 or i == m:
            elapsed = time.time() - t1
            rate = i / max(elapsed, 1e-8)
            eta = (m - i) / max(rate, 1e-8)
            print(f"[AXIS_AUDIT_PROGRESS] overlay {i}/{m} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    html_path = out_dir / "index.html"
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Axis Inversion Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1115; color: #e8edf2; }}
    .header {{ margin-bottom: 14px; }}
    .legend {{ font-size: 13px; opacity: 0.9; }}
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
    <h2>Axis Inversion Audit (p1/p2)</h2>
    <div class="legend">Verde = eixo consistente | Vermelho = suspeita de eixo invertido</div>
    <div class="legend">Regra usada: superiores invertidos quando p1.y &lt; p2.y; inferiores invertidos quando p1.y &gt; p2.y.</div>
    <div class="legend">split={html.escape(args.split)} | top_k={len(cards)}</div>
  </div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    summary = {
        "split": args.split,
        "num_samples": len(eval_samples),
        "top_k": top_k,
        "rule": {
            "upper_inverted_if": "p1.y < p2.y",
            "lower_inverted_if": "p1.y > p2.y",
        },
        "artifacts": {
            "full_csv": str(full_csv),
            "top_csv": str(top_csv),
            "html": str(html_path),
            "images_dir": str(img_dir),
        },
    }
    summary_path = out_dir / "axis_inversion_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        register_record(
            experiment_root=output_base_dir,
            kind="audits",
            record={
                "id": f"axis_inversion_audit__{args.split}__{top_k}",
                "kind": "axis_inversion_audit",
                "experiment": output_base_dir.name,
                "run_name": None,
                "split": args.split,
                "summary": {
                    "num_samples": summary["num_samples"],
                    "top_k": summary["top_k"],
                },
                "artifacts": {
                    "summary_json": rel_to_experiment(summary_path, output_base_dir),
                    "full_csv": rel_to_experiment(full_csv, output_base_dir),
                    "top_csv": rel_to_experiment(top_csv, output_base_dir),
                    "html": rel_to_experiment(html_path, output_base_dir),
                    "images_dir": rel_to_experiment(img_dir, output_base_dir),
                },
            },
        )
    except Exception as e:
        print(f"[WARN] dashboard registry skipped in audit_axis_inversion.py: {e}")

    print(f"[AXIS_AUDIT] top_csv={top_csv}")
    print(f"[AXIS_AUDIT] html={html_path}")
    print(f"[AXIS_AUDIT] summary={summary_path}")


if __name__ == "__main__":
    main()
