#!/usr/bin/env python3
"""Gera graficos de metricas por dente a partir de metrics_per_tooth.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


CANONICAL_32 = [
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
]


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _load_rows(csv_path: Path) -> List[Dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    if not rows:
        raise RuntimeError(f"CSV sem linhas: {csv_path}")
    return rows


def _sort_rows(rows: List[Dict]) -> List[Dict]:
    rank = {t: i for i, t in enumerate(CANONICAL_32)}
    return sorted(rows, key=lambda r: (rank.get(r.get("tooth", ""), 999), r.get("tooth", "")))


def _plot_presence(rows: List[Dict], out_path: Path, title: str, dpi: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    teeth = [r["tooth"] for r in rows]
    precision = np.array([_to_float(r.get("presence_precision", "")) for r in rows], dtype=np.float64)
    recall = np.array([_to_float(r.get("presence_recall", "")) for r in rows], dtype=np.float64)
    f1 = np.array([_to_float(r.get("presence_f1", "")) for r in rows], dtype=np.float64)

    x = np.arange(len(teeth))
    w = 0.27

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)
    ax.bar(x - w, precision, width=w, label="precision")
    ax.bar(x, recall, width=w, label="recall")
    ax.bar(x + w, f1, width=w, label="f1")

    ax.set_title(f"{title} - Presence Metrics by Tooth")
    ax.set_xlabel("Tooth")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(teeth, rotation=90)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_geometry(rows: List[Dict], out_path: Path, title: str, dpi: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    teeth = [r["tooth"] for r in rows]
    mean_px = np.array([_to_float(r.get("point_error_mean_px", "")) for r in rows], dtype=np.float64)
    median_px = np.array([_to_float(r.get("point_error_median_px", "")) for r in rows], dtype=np.float64)
    p90_px = np.array([_to_float(r.get("point_error_p90_px", "")) for r in rows], dtype=np.float64)
    within5 = np.array([_to_float(r.get("point_within_5px_rate", "")) for r in rows], dtype=np.float64)

    x = np.arange(len(teeth))

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(211)
    ax1.plot(x, median_px, marker="o", linewidth=1.5, label="median px")
    ax1.plot(x, mean_px, marker="o", linewidth=1.2, label="mean px")
    ax1.plot(x, p90_px, marker="o", linewidth=1.2, label="p90 px")
    ax1.set_title(f"{title} - Geometric Error by Tooth")
    ax1.set_ylabel("Error (px)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(teeth, rotation=90)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(212)
    ax2.bar(x, within5)
    ax2.set_ylabel("Within 5px rate")
    ax2.set_xlabel("Tooth")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xticks(x)
    ax2.set_xticklabels(teeth, rotation=90)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _write_worst_summary(rows: List[Dict], out_csv: Path, top_k: int) -> None:
    scored = []
    for r in rows:
        scored.append(
            {
                "tooth": r["tooth"],
                "presence_precision": _to_float(r.get("presence_precision", "")),
                "presence_recall": _to_float(r.get("presence_recall", "")),
                "presence_f1": _to_float(r.get("presence_f1", "")),
                "point_error_median_px": _to_float(r.get("point_error_median_px", "")),
                "point_error_mean_px": _to_float(r.get("point_error_mean_px", "")),
                "point_error_p90_px": _to_float(r.get("point_error_p90_px", "")),
                "point_within_5px_rate": _to_float(r.get("point_within_5px_rate", "")),
            }
        )

    worst_recall = sorted(scored, key=lambda r: (r["presence_recall"], r["tooth"]))[:top_k]
    worst_precision = sorted(scored, key=lambda r: (r["presence_precision"], r["tooth"]))[:top_k]
    worst_geom = sorted(scored, key=lambda r: (-r["point_error_median_px"], r["tooth"]))[:top_k]

    out_rows: List[Dict] = []
    for rank, row in enumerate(worst_recall, start=1):
        out_rows.append({"group": "worst_recall", "rank": rank, **row})
    for rank, row in enumerate(worst_precision, start=1):
        out_rows.append({"group": "worst_precision", "rank": rank, **row})
    for rank, row in enumerate(worst_geom, start=1):
        out_rows.append({"group": "worst_geometric_median_px", "rank": rank, **row})

    fields = [
        "group",
        "rank",
        "tooth",
        "presence_precision",
        "presence_recall",
        "presence_f1",
        "point_error_median_px",
        "point_error_mean_px",
        "point_error_p90_px",
        "point_within_5px_rate",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plota metricas por dente (presence + erro geometrico).")
    parser.add_argument("--csv", type=Path, required=True, help="Arquivo metrics_per_tooth.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Diretorio de saida dos graficos")
    parser.add_argument("--title", type=str, default="Hydra")
    parser.add_argument("--top-k", type=int, default=8, help="Top piores dentes no CSV resumo")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as _plt  # type: ignore  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "[ERRO] matplotlib indisponivel no ambiente atual. "
            "Instale com: .venv/bin/pip install matplotlib. "
            f"Detalhe: {exc}"
        )

    rows = _sort_rows(_load_rows(args.csv))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    presence_png = out_dir / "presence_precision_recall_f1_per_tooth.png"
    geom_png = out_dir / "geometric_error_per_tooth.png"
    worst_csv = out_dir / "worst_teeth_summary.csv"

    _plot_presence(rows=rows, out_path=presence_png, title=args.title, dpi=int(args.dpi))
    _plot_geometry(rows=rows, out_path=geom_png, title=args.title, dpi=int(args.dpi))
    _write_worst_summary(rows=rows, out_csv=worst_csv, top_k=max(1, int(args.top_k)))

    print(f"[PLOT] presence={presence_png}")
    print(f"[PLOT] geometry={geom_png}")
    print(f"[PLOT] worst_summary={worst_csv}")


if __name__ == "__main__":
    main()
