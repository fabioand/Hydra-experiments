#!/usr/bin/env python3
"""Desenha 3 janelas fixas por dimensao da imagem:
- LEFT: metade esquerda
- RIGHT: metade direita
- CENTER: mesma largura de uma metade, centralizada
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def list_images(input_dir: Path, limit: int) -> List[Path]:
    files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png")))
    if limit > 0:
        return files[:limit]
    return files


def fixed_rects(w: int, h: int) -> Dict[str, List[int]]:
    half = w // 2
    left = [0, 0, half, h]
    right = [w - half, 0, w, h]
    center_x1 = (w - half) // 2
    center = [center_x1, 0, center_x1 + half, h]
    return {"LEFT": left, "CENTER": center, "RIGHT": right}


def draw_rects(image_path: Path, out_path: Path) -> Dict[str, object]:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = img.size

    rects = fixed_rects(w, h)
    colors = {"LEFT": (0, 255, 255), "CENTER": (255, 180, 0), "RIGHT": (0, 255, 0)}

    for name in ["LEFT", "CENTER", "RIGHT"]:
        x1, y1, x2, y2 = rects[name]
        draw.rectangle([(x1, y1), (x2, y2)], outline=colors[name], width=4)
        tb = draw.textbbox((x1 + 6, y1 + 6), name, font=font)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((x1 + 6, y1 + 6), name, fill=(255, 255, 255), font=font)

    banner = f"Fixed windows: LEFT half | CENTER half-width centered | RIGHT half"
    bb = draw.textbbox((10, 10), banner, font=font)
    draw.rectangle(bb, fill=(0, 0, 0))
    draw.text((10, 10), banner, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return {
        "input_image": str(image_path.resolve()),
        "out": str(out_path.resolve()),
        "image_size": [w, h],
        "rectangles": rects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render 3 fixed windows in panoramic images.")
    parser.add_argument("--input-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("radiomemory_api_tools/outputs/fixed_three_windows"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("radiomemory_api_tools/outputs/fixed_three_windows/report.json"),
    )
    args = parser.parse_args()

    imgs = list_images(args.input_dir, args.limit)
    if not imgs:
        raise RuntimeError(f"No images found in {args.input_dir}")

    rows = []
    for p in imgs:
        out = args.out_dir / f"{p.stem}_fixed_windows.png"
        rows.append(draw_rects(p, out))

    report = {"count": len(rows), "results": rows}
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report_json.resolve()), "count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
