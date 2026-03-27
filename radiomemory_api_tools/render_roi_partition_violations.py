#!/usr/bin/env python3
"""Renderiza overlays dos casos com violacao da auditoria ROI partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


TOOTH_SETS_BY_RECT: Dict[str, List[str]] = {
    "R_LEFT": ["24", "25", "26", "27", "28", "34", "35", "36", "37", "38"],
    "R_RIGHT": ["14", "15", "16", "17", "18", "44", "45", "46", "47", "48"],
    "R_CENTER": ["11", "12", "13", "21", "22", "23", "31", "32", "33", "41", "42", "43"],
}
TOOTH_TO_RECT = {t: r for r, teeth in TOOTH_SETS_BY_RECT.items() for t in teeth}

RECT_COLORS = {
    "R_LEFT": (0, 255, 0),
    "R_CENTER": (255, 180, 0),
    "R_RIGHT": (0, 180, 255),
}


def load_gt_points(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Tuple[float, float]]] = {}
    if not isinstance(data, list):
        return out
    for ann in data:
        if not isinstance(ann, dict):
            continue
        label = str(ann.get("label", ""))
        pts = ann.get("pts", [])
        valid: List[Tuple[float, float]] = []
        if isinstance(pts, list):
            for p in pts:
                if not isinstance(p, dict):
                    continue
                x = p.get("x")
                y = p.get("y")
                if x is None or y is None:
                    continue
                valid.append((float(x), float(y)))
        if label and valid:
            out[label] = valid
    return out


def point_in_rect(pt: Tuple[float, float], rect: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_one(
    item: dict,
    out_path: Path,
) -> None:
    image_path = Path(item["image"])
    gt_path = Path(item["gt_json"])
    rects = item["rectangles"]
    gt = load_gt_points(gt_path)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Draw rectangles
    for rname in ["R_LEFT", "R_CENTER", "R_RIGHT"]:
        rect = rects[rname]
        color = RECT_COLORS[rname]
        draw.rectangle([(rect[0], rect[1]), (rect[2], rect[3])], outline=color, width=4)
        tb = draw.textbbox((rect[0] + 4, max(0, rect[1] - 14)), rname, font=font)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((rect[0] + 4, max(0, rect[1] - 14)), rname, fill=(255, 255, 255), font=font)

    # Named anchor points from API
    np = item.get("named_points", {})
    for name, color in [
        ("condilo_esquerdo", (0, 255, 0)),
        ("ena", (255, 255, 0)),
        ("condilo_direito", (0, 180, 255)),
        ("mentoniano", (255, 80, 0)),
    ]:
        if name not in np:
            continue
        x, y = np[name]
        draw.ellipse([(x - 6, y - 6), (x + 6, y + 6)], outline=color, width=3)
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))

    # Draw all audit teeth points. Outside points in red.
    for tooth, rect_name in TOOTH_TO_RECT.items():
        pts = gt.get(tooth, [])
        if not pts:
            continue
        rect = rects[rect_name]
        for idx, pt in enumerate(pts):
            inside = point_in_rect(pt, rect)
            x, y = pt
            if inside:
                color = (240, 240, 240)
                draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], outline=color, width=2)
            else:
                color = (255, 0, 0)
                draw.ellipse([(x - 6, y - 6), (x + 6, y + 6)], outline=color, width=3)
                # X mark
                draw.line([(x - 5, y - 5), (x + 5, y + 5)], fill=color, width=2)
                draw.line([(x - 5, y + 5), (x + 5, y - 5)], fill=color, width=2)
                label = f"{tooth}.p{idx+1}"
                tb = draw.textbbox((x + 6, y - 8), label, font=font)
                draw.rectangle(tb, fill=(0, 0, 0))
                draw.text((x + 6, y - 8), label, fill=(255, 255, 255), font=font)

    # Header
    nviol = len(item.get("violations", []))
    header = f"{item['stem']} | violations={nviol}"
    hb = draw.textbbox((10, 10), header, font=font)
    draw.rectangle(hb, fill=(0, 0, 0))
    draw.text((10, 10), header, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render overlays for ROI partition audit violations.")
    parser.add_argument(
        "--audit-details",
        type=Path,
        default=Path("radiomemory_api_tools/outputs/roi_partition_audit_full999/audit_details.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("radiomemory_api_tools/outputs/roi_partition_audit_full999/violation_overlays"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means all violation cases")
    args = parser.parse_args()

    rows = json.loads(args.audit_details.read_text(encoding="utf-8"))
    viol = [r for r in rows if r.get("ok") and r.get("has_violation")]
    viol = sorted(viol, key=lambda r: len(r.get("violations", [])), reverse=True)
    if args.limit > 0:
        viol = viol[: args.limit]

    rendered = []
    for item in viol:
        out_path = args.out_dir / f"{item['stem']}_violations.png"
        draw_one(item, out_path)
        rendered.append(
            {
                "stem": item["stem"],
                "violations": len(item.get("violations", [])),
                "out": str(out_path.resolve()),
            }
        )

    manifest = {"count": len(rendered), "items": rendered}
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "count": len(rendered)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
