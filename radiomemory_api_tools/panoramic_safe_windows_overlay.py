#!/usr/bin/env python3
import argparse
import base64
import json
import sys
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTH_DIR = REPO_ROOT / "radiomemory_auth"
if str(AUTH_DIR) not in sys.path:
    sys.path.append(str(AUTH_DIR))

from login import LINKAPI, LoginAPI  # noqa: E402


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in value if not unicodedata.combining(ch)).lower()


def encode_image_b64(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("ascii")


def auth_headers() -> dict[str, str]:
    auth = LoginAPI()
    token_type = auth.get("token_type")
    access_token = auth.get("access_token")
    if not token_type or not access_token:
        raise RuntimeError(f"Auth failed: {auth}")
    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def build_body(image_path: Path) -> dict[str, Any]:
    return {
        "base64_image": encode_image_b64(image_path),
        "output_width": 0,
        "output_height": 0,
        "threshold": 0.0,
        "resource": "describe",
        "lang": "pt-br",
        "use_cache": False,
    }


def call_anatomic_points(base: str, image_path: Path, timeout: int) -> tuple[dict[str, Any], int, str]:
    headers = auth_headers()
    body = build_body(image_path)
    url = f"{base.rstrip('/')}/v1/panoramics/anatomic_points"
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Non-JSON response ({resp.status_code}): {resp.text[:400]}") from exc
    if resp.status_code >= 400:
        raise RuntimeError(f"API error {resp.status_code}: {json.dumps(payload, ensure_ascii=False)[:800]}")
    return payload, resp.status_code, url


def extract_named_points(payload: dict[str, Any], img_w: int, img_h: int) -> dict[str, tuple[float, float]]:
    entities = payload.get("entities") if isinstance(payload, dict) else None
    if not isinstance(entities, list):
        raise RuntimeError("Payload has no entities list.")

    src_w = float(payload.get("output_width") or img_w)
    src_h = float(payload.get("output_height") or img_h)
    sx = (img_w / src_w) if src_w > 0 else 1.0
    sy = (img_h / src_h) if src_h > 0 else 1.0

    found: dict[str, tuple[float, float]] = {}
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        point = ent.get("point")
        if not (isinstance(point, (list, tuple)) and len(point) >= 2):
            continue
        try:
            x = float(point[0]) * sx
            y = float(point[1]) * sy
        except Exception:
            continue
        name = normalize_text(str(ent.get("class_name") or ""))
        if "condilo - esquerdo" in name:
            found["condilo_esquerdo"] = (x, y)
        elif "condilo - direito" in name:
            found["condilo_direito"] = (x, y)
        elif "e.n.a." in name or "ena" in name:
            found["ena"] = (x, y)
        elif "mentoniano" in name:
            found["mentoniano"] = (x, y)

    missing = [k for k in ["condilo_esquerdo", "condilo_direito", "ena", "mentoniano"] if k not in found]
    if missing:
        raise RuntimeError(f"Missing required anatomic points: {missing}")
    return found


def draw_overlay(
    image_path: Path,
    named_points: dict[str, tuple[float, float]],
    out_path: Path,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, h = image.size

    x_cond_esq = named_points["condilo_esquerdo"][0]
    y_cond_esq = named_points["condilo_esquerdo"][1]
    x_cond_dir = named_points["condilo_direito"][0]
    y_cond_dir = named_points["condilo_direito"][1]
    x_ena = named_points["ena"][0]
    y_mentoniano = named_points["mentoniano"][1]

    # Rectangle 1: condilo esquerdo <-> ENA, vertical from condilo esquerdo to image bottom.
    r1_x1 = int(round(min(x_cond_esq, x_ena)))
    r1_x2 = int(round(max(x_cond_esq, x_ena)))
    r1_y1 = int(round(y_cond_esq))
    r1_y2 = int(round(y_mentoniano))

    # Rectangle 2: ENA <-> condilo direito, vertical from condilo direito to image bottom.
    r2_x1 = int(round(min(x_ena, x_cond_dir)))
    r2_x2 = int(round(max(x_ena, x_cond_dir)))
    r2_y1 = int(round(y_cond_dir))
    r2_y2 = int(round(y_mentoniano))

    # Rectangle 3 (center): lateral bounds are midpoints between each condyle and ENA.
    x_mid_left = 0.5 * (x_cond_esq + x_ena)
    x_mid_right = 0.5 * (x_cond_dir + x_ena)
    r3_x1 = int(round(min(x_mid_left, x_mid_right)))
    r3_x2 = int(round(max(x_mid_left, x_mid_right)))
    # Top at condyle line (using the highest condyle to avoid cutting superior area).
    r3_y1 = int(round(min(y_cond_esq, y_cond_dir)))
    r3_y2 = int(round(y_mentoniano))

    r1_x1 = max(0, min(w - 1, r1_x1))
    r1_x2 = max(0, min(w, r1_x2))
    r1_y1 = max(0, min(h - 1, r1_y1))
    r1_y2 = max(1, min(h, r1_y2))
    r2_x1 = max(0, min(w - 1, r2_x1))
    r2_x2 = max(0, min(w, r2_x2))
    r2_y1 = max(0, min(h - 1, r2_y1))
    r2_y2 = max(1, min(h, r2_y2))
    r3_x1 = max(0, min(w - 1, r3_x1))
    r3_x2 = max(0, min(w, r3_x2))
    r3_y1 = max(0, min(h - 1, r3_y1))
    r3_y2 = max(1, min(h, r3_y2))

    if r1_y2 <= r1_y1:
        r1_y2 = min(h, r1_y1 + 1)
    if r2_y2 <= r2_y1:
        r2_y2 = min(h, r2_y1 + 1)
    if r3_y2 <= r3_y1:
        r3_y2 = min(h, r3_y1 + 1)

    draw.rectangle([(r1_x1, r1_y1), (r1_x2, r1_y2)], outline=(0, 255, 0), width=4)
    draw.rectangle([(r2_x1, r2_y1), (r2_x2, r2_y2)], outline=(0, 180, 255), width=4)
    draw.rectangle([(r3_x1, r3_y1), (r3_x2, r3_y2)], outline=(255, 180, 0), width=4)

    for key, color in [
        ("condilo_esquerdo", (0, 255, 0)),
        ("ena", (255, 255, 0)),
        ("condilo_direito", (0, 180, 255)),
        ("mentoniano", (255, 120, 0)),
    ]:
        x, y = named_points[key]
        draw.ellipse([(x - 7, y - 7), (x + 7, y + 7)], outline=color, width=3)
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 0, 0))
        label = key
        tb = draw.textbbox((x + 8, y - 10), label, font=font)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((x + 8, y - 10), label, fill=(255, 255, 255), font=font)

    draw.rectangle([(10, 10), (860, 74)], fill=(0, 0, 0))
    draw.text((16, 18), "R1: Condilo Esq<->ENA, Y: cond_esq..Mentoniano", fill=(255, 255, 255), font=font)
    draw.text((16, 38), "R2: ENA<->Condilo Dir, Y: cond_dir..Mentoniano | R3: mids(cond-ENA), Y: condilo..Mentoniano", fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    return {
        "out": str(out_path),
        "image_size": [w, h],
        "rect_left": [r1_x1, r1_y1, r1_x2, r1_y2],
        "rect_right": [r2_x1, r2_y1, r2_x2, r2_y2],
        "rect_center": [r3_x1, r3_y1, r3_x2, r3_y2],
        "points": {
            "condilo_esquerdo": [named_points["condilo_esquerdo"][0], named_points["condilo_esquerdo"][1]],
            "ena": [named_points["ena"][0], named_points["ena"][1]],
            "condilo_direito": [named_points["condilo_direito"][0], named_points["condilo_direito"][1]],
            "mentoniano": [named_points["mentoniano"][0], named_points["mentoniano"][1]],
        },
    }


def list_images(input_dir: Path, limit: int) -> list[Path]:
    files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png")))
    if limit > 0:
        return files[:limit]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw robust full-height bilateral windows using Condyles + ENA.")
    parser.add_argument("--input-dir", default=str(REPO_ROOT / "longoeixo" / "imgs"))
    parser.add_argument("--limit", type=int, default=4, help="Number of images to process.")
    parser.add_argument("--base", default=LINKAPI)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "radiomemory_api_tools" / "outputs" / "safe_windows"))
    parser.add_argument("--report-json", default=str(REPO_ROOT / "radiomemory_api_tools" / "outputs" / "safe_windows_report.json"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()

    images = list_images(input_dir, args.limit)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    results: list[dict[str, Any]] = []
    for image_path in images:
        entry: dict[str, Any] = {"input_image": str(image_path)}
        try:
            payload, status, url = call_anatomic_points(args.base, image_path, timeout=args.timeout)
            img = Image.open(image_path)
            named_points = extract_named_points(payload, img.size[0], img.size[1])
            out_path = out_dir / f"{image_path.stem}_safe_windows.png"
            overlay = draw_overlay(image_path, named_points, out_path)
            entry.update(overlay)
            entry["api_url"] = url
            entry["http_status"] = status
            entry["ok"] = True
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = str(exc)
        results.append(entry)

    report = {"count": len(results), "results": results}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
