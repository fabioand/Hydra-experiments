#!/usr/bin/env python3
"""Calcula estatisticas horizontais do dataset longoeixo.

Métricas:
1) Imagens: largura (px) e aspect ratio (w/h) -> media, minimo, maximo.
2) Bounding box dos pontos: largura horizontal (max_x - min_x) -> media, minimo, maximo.
3) Razao: largura_bbox_pontos / largura_imagem -> media, minimo, maximo.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image


@dataclass
class Stats:
    mean: float
    min: float
    max: float
    count: int


def summarize(values: Iterable[float]) -> Stats:
    vals = list(values)
    if not vals:
        return Stats(mean=float("nan"), min=float("nan"), max=float("nan"), count=0)
    return Stats(
        mean=sum(vals) / len(vals),
        min=min(vals),
        max=max(vals),
        count=len(vals),
    )


def load_annotations(json_path: Path) -> list:
    try:
        text = json_path.read_text(encoding="utf-8").strip()
    except Exception:
        return []

    if not text:
        return []

    try:
        data = json.loads(text)
    except Exception:
        return []

    return data if isinstance(data, list) else []


def collect_points(data: list) -> List[tuple[float, float]]:
    pts: List[tuple[float, float]] = []
    for ann in data:
        if not isinstance(ann, dict):
            continue
        ann_pts = ann.get("pts", [])
        if not isinstance(ann_pts, list):
            continue
        for pt in ann_pts:
            if not isinstance(pt, dict):
                continue
            x = pt.get("x")
            y = pt.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                pts.append((float(x), float(y)))
    return pts


def main() -> int:
    parser = argparse.ArgumentParser(description="Estatisticas de largura/aspect ratio e bbox dos pontos.")
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--json-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument("--top-k", type=int, default=300, help="Quantidade de imagens no mosaico de menor aspect ratio.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("longoeixo/annotation_audit/aspect_ratio_bottom"),
        help="Pasta de saida para CSV ranking + HTML mosaico.",
    )
    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    json_dir = args.json_dir
    out_dir = args.out_dir

    if not imgs_dir.exists() or not imgs_dir.is_dir():
        raise SystemExit(f"[ERRO] Pasta de imagens invalida: {imgs_dir}")
    if not json_dir.exists() or not json_dir.is_dir():
        raise SystemExit(f"[ERRO] Pasta de anotacoes invalida: {json_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_widths: List[float] = []
    aspect_ratios: List[float] = []
    bbox_widths: List[float] = []
    bbox_to_img_ratios: List[float] = []
    image_rows: List[dict] = []

    total_json = 0
    missing_image = 0
    unreadable_image = 0
    empty_or_invalid_json = 0
    json_without_points = 0

    for json_path in sorted(json_dir.glob("*.json")):
        total_json += 1
        stem = json_path.stem
        image_path = imgs_dir / f"{stem}.jpg"
        if not image_path.exists():
            missing_image += 1
            continue

        try:
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception:
            unreadable_image += 1
            continue

        if h <= 0 or w <= 0:
            unreadable_image += 1
            continue

        image_widths.append(float(w))
        ar = float(w) / float(h)
        aspect_ratios.append(ar)
        image_rows.append(
            {
                "stem": stem,
                "width_px": int(w),
                "height_px": int(h),
                "aspect_ratio_w_over_h": float(ar),
                "image_path": str(image_path.resolve()),
            }
        )

        data = load_annotations(json_path)
        if not data:
            empty_or_invalid_json += 1
            continue

        pts = collect_points(data)
        if not pts:
            json_without_points += 1
            continue

        xs = [p[0] for p in pts]
        bbox_w = max(xs) - min(xs)
        bbox_widths.append(bbox_w)
        bbox_to_img_ratios.append(bbox_w / float(w))

    img_stats = summarize(image_widths)
    ar_stats = summarize(aspect_ratios)
    bbox_stats = summarize(bbox_widths)
    ratio_stats = summarize(bbox_to_img_ratios)

    print("== Dataset Horizontal Stats ==")
    print(f"json_total: {total_json}")
    print(f"imgs_consideradas: {img_stats.count}")
    print(f"json_sem_imagem: {missing_image}")
    print(f"imgs_ilegiveis: {unreadable_image}")
    print(f"json_vazio_ou_invalido: {empty_or_invalid_json}")
    print(f"json_sem_pts: {json_without_points}")
    print()

    print("1) Imagens")
    print(f"   largura_px -> media={img_stats.mean:.6f} min={img_stats.min:.6f} max={img_stats.max:.6f}")
    print(f"   aspect_ratio(w/h) -> media={ar_stats.mean:.6f} min={ar_stats.min:.6f} max={ar_stats.max:.6f}")
    print()

    print("2) BBox dos pontos (largura horizontal)")
    print(
        "   largura_bbox_pts_px -> "
        f"media={bbox_stats.mean:.6f} min={bbox_stats.min:.6f} max={bbox_stats.max:.6f}"
    )
    print(f"   amostras_com_bbox: {bbox_stats.count}")
    print()

    print("3) Razao bbox pontos / largura imagem")
    print(
        "   ratio_bbox_img -> "
        f"media={ratio_stats.mean:.6f} min={ratio_stats.min:.6f} max={ratio_stats.max:.6f}"
    )
    print(f"   amostras_com_ratio: {ratio_stats.count}")

    # Ranking e mosaico: menores aspect ratios (mais "altas/estreitas" proporcionalmente).
    top_k = max(0, int(args.top_k))
    ranked = sorted(image_rows, key=lambda r: (r["aspect_ratio_w_over_h"], r["stem"]))
    top_rows = ranked[:top_k]

    ranking_csv = out_dir / f"bottom_aspect_ratio_top{top_k}.csv"
    with ranking_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "stem", "width_px", "height_px", "aspect_ratio_w_over_h", "image_path"]
        )
        writer.writeheader()
        for i, row in enumerate(top_rows, start=1):
            writer.writerow(
                {
                    "rank": i,
                    "stem": row["stem"],
                    "width_px": row["width_px"],
                    "height_px": row["height_px"],
                    "aspect_ratio_w_over_h": f"{float(row['aspect_ratio_w_over_h']):.8f}",
                    "image_path": row["image_path"],
                }
            )

    html_path = out_dir / "index.html"
    images_out_dir = out_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    cards: List[str] = []
    for i, row in enumerate(top_rows, start=1):
        stem = str(row["stem"])
        ar = float(row["aspect_ratio_w_over_h"])
        w = int(row["width_px"])
        h = int(row["height_px"])
        src_path = Path(str(row["image_path"])).resolve()
        out_name = f"{i:04d}_{stem}.jpg"
        dst_path = images_out_dir / out_name
        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            # Se copia falhar, pula card para nao quebrar o HTML.
            continue
        subtitle = f"rank={i} | ar(w/h)={ar:.6f} | {w}x{h}"
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

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Bottom Aspect Ratio Mosaic</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1115; color: #e8edf2; }}
    .header {{ margin-bottom: 14px; }}
    .legend {{ font-size: 13px; opacity: 0.9; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
    .card {{ background: #171b22; border: 1px solid #2a3340; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; background: #000; }}
    .meta {{ padding: 8px 10px; }}
    .stem {{ font-size: 13px; font-weight: 600; word-break: break-all; }}
    .sub {{ font-size: 12px; opacity: 0.9; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="header">
    <h2>Bottom Aspect Ratio (w/h)</h2>
    <div class="legend">Mostrando as {len(cards)} imagens com menor aspect ratio.</div>
  </div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    print()
    print("[ASPECT_RATIO_AUDIT]")
    print(f"ranking_csv: {ranking_csv}")
    print(f"html_mosaic: {html_path}")
    print(f"images_dir: {images_out_dir}")
    print(f"top_k: {len(top_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
