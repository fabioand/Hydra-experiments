#!/usr/bin/env python3
import argparse
import base64
import json
import sys
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


def pick_image(image_arg: Optional[str]) -> Path:
    if image_arg:
        path = Path(image_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    imgs_dir = REPO_ROOT / "longoeixo" / "imgs"
    candidates = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.jpeg")) + list(imgs_dir.glob("*.png")))
    if not candidates:
        raise FileNotFoundError(f"No dataset image found in: {imgs_dir}")
    return candidates[0]


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


def extract_entities(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        return [e for e in payload["entities"] if isinstance(e, dict)]
    return []


def parse_response_payload(response: requests.Response, expected_model: Optional[str] = None) -> dict[str, Any]:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    # Fallback for newline-delimited JSON responses (describe-like streams).
    if expected_model:
        for line in (response.text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if str(obj.get("model_name", "")).lower() == expected_model.lower():
                return obj

    raise RuntimeError(f"Non-JSON/unknown response ({response.status_code}): {response.text[:500]}")


def call_panoramic_model(
    base: str,
    image_path: Path,
    endpoint: str,
    timeout: int,
    expected_model: Optional[str] = None,
) -> tuple[dict[str, Any], int, str]:
    headers = auth_headers()
    body = build_body(image_path)
    url = f"{base.rstrip('/')}/v1/panoramics/{endpoint.lstrip('/')}"
    response = requests.post(url, headers=headers, json=body, timeout=timeout)
    payload = parse_response_payload(response, expected_model=expected_model)
    if response.status_code >= 400:
        raise RuntimeError(f"API error {response.status_code}: {json.dumps(payload, ensure_ascii=False)[:800]}")
    return payload, response.status_code, url


def pick_mandibular_entity(entities: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not entities:
        return None
    ranked = []
    for ent in entities:
        name = str(ent.get("class_name") or "").lower()
        contour = ent.get("contour")
        contour_len = len(contour) if isinstance(contour, list) else 0
        if contour_len < 2:
            continue
        score = 0
        if "contmand" in name:
            score += 100
        if "mand" in name:
            score += 30
        if "infer" in name or "lower" in name:
            score += 20
        score += min(contour_len, 200) / 10.0
        ranked.append((score, ent))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def _safe_points_from_contour(contour: Any, sx: float, sy: float) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    if not isinstance(contour, list):
        return pts
    for p in contour:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        try:
            x = float(p[0]) * sx
            y = float(p[1]) * sy
        except Exception:
            continue
        pts.append((x, y))
    return pts


def _quantile_index(n: int, q: float) -> int:
    idx = int(round((n - 1) * q))
    return max(0, min(n - 1, idx))


def compute_molar_crop_from_contour(
    contour_points: list[tuple[float, float]],
    image_size: tuple[int, int],
    side: str,
) -> dict[str, Any]:
    img_w, img_h = image_size
    if len(contour_points) < 20:
        raise RuntimeError("Mandibular contour has too few points for stable crop heuristic.")

    pts_sorted = sorted(contour_points, key=lambda p: p[0])
    n = len(pts_sorted)
    # Avoid extreme posterior ramus and move toward posterior dentition region.
    if side == "right":
        i0 = _quantile_index(n, 0.30)
        i1 = _quantile_index(n, 0.45)
    else:
        i0 = _quantile_index(n, 0.55)
        i1 = _quantile_index(n, 0.70)
    if i1 <= i0:
        i0, i1 = 0, max(1, n // 5)
    band = pts_sorted[i0 : i1 + 1]
    xs = sorted(p[0] for p in band)
    ys = sorted(p[1] for p in band)

    cx = xs[len(xs) // 2]
    # Use upper contour quantile in the band (closer to alveolar crest region).
    cy_top = ys[_quantile_index(len(ys), 0.25)]

    crop_w = 0.18 * img_w
    crop_h = 0.24 * img_h
    # Move slightly down from top mandibular contour to center the lower molar area.
    cy = cy_top + 0.10 * img_h

    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x2 = int(round(cx + crop_w / 2))
    y2 = int(round(cy + crop_h / 2))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 = img_w
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 = img_h
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return {
        "side": side,
        "quantile_band": [i0, i1],
        "center_curve": [cx, cy_top],
        "center_crop": [cx, cy],
        "box": [x1, y1, x2, y2],
        "box_size": [x2 - x1, y2 - y1],
    }


def draw_overlay(
    image_path: Path,
    payload_points: dict[str, Any],
    payload_panorogram: dict[str, Any],
    out_path: Path,
    side: str = "right",
    radius: int = 10,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    img_w, img_h = image.size
    src_w = float(payload_points.get("output_width") or img_w)
    src_h = float(payload_points.get("output_height") or img_h)

    # Scale points if API coordinates are based on a different output size.
    sx = (img_w / src_w) if src_w > 0 else 1.0
    sy = (img_h / src_h) if src_h > 0 else 1.0

    entities = extract_entities(payload_points)
    drawn = 0
    skipped = 0

    for idx, ent in enumerate(entities, start=1):
        pt = ent.get("point")
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            skipped += 1
            continue
        try:
            x = float(pt[0]) * sx
            y = float(pt[1]) * sy
        except Exception:
            skipped += 1
            continue

        label_base = str(ent.get("class_name") or f"point_{idx}")
        score = ent.get("score")
        if isinstance(score, (int, float)):
            label = f"{label_base} ({score:.2f})"
        else:
            label = label_base

        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], outline=(255, 64, 64), width=3)
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=(255, 255, 0))

        tx = x + radius + 4
        ty = y - radius - 2
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
        drawn += 1

    pano_src_w = float(payload_panorogram.get("output_width") or img_w)
    pano_src_h = float(payload_panorogram.get("output_height") or img_h)
    psx = (img_w / pano_src_w) if pano_src_w > 0 else 1.0
    psy = (img_h / pano_src_h) if pano_src_h > 0 else 1.0
    pano_entities = extract_entities(payload_panorogram)
    mand_ent = pick_mandibular_entity(pano_entities)
    if mand_ent is None:
        raise RuntimeError("No suitable mandibular contour found in panorogram response.")

    mand_points = _safe_points_from_contour(mand_ent.get("contour"), psx, psy)
    crop = compute_molar_crop_from_contour(mand_points, (img_w, img_h), side=side)
    x1, y1, x2, y2 = crop["box"]

    # Draw mandibular contour (reference) and crop rectangle.
    if len(mand_points) >= 2:
        draw.line(mand_points, fill=(255, 0, 255), width=2)
    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=4)
    label = f"Molar crop ({side})"
    tb = draw.textbbox((x1 + 4, max(0, y1 - 16)), label, font=font)
    draw.rectangle(tb, fill=(0, 80, 0))
    draw.text((x1 + 4, max(0, y1 - 16)), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    return {
        "image_size": [img_w, img_h],
        "api_output_size": [payload_points.get("output_width"), payload_points.get("output_height")],
        "scale": [sx, sy],
        "entities_total": len(entities),
        "points_drawn": drawn,
        "points_skipped": skipped,
        "panorogram_entities_total": len(pano_entities),
        "mandibular_class_name": mand_ent.get("class_name"),
        "molar_crop": crop,
        "out": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Call RM panoramic anatomic-points endpoint and draw overlay labels.")
    parser.add_argument("--image", default=None, help="Path to panoramic image (defaults to first image in longoeixo/imgs).")
    parser.add_argument("--base", default=LINKAPI, help="RM API base URL.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--side", choices=["right", "left"], default="right", help="Which side molar window to crop.")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "radiomemory_api_tools" / "outputs" / "panoramic_anatomic_points_overlay.png"),
        help="Output image path.",
    )
    parser.add_argument("--save-json", default=None, help="Optional path to save raw JSON response.")
    args = parser.parse_args()

    image_path = pick_image(args.image)
    payload_points, status_points, url_points = call_panoramic_model(
        args.base, image_path, endpoint="anatomic_points", timeout=args.timeout, expected_model="anatomic_points"
    )
    payload_pano, status_pano, url_pano = call_panoramic_model(
        args.base, image_path, endpoint="panorogram", timeout=args.timeout, expected_model="panorogram"
    )
    report = draw_overlay(
        image_path,
        payload_points,
        payload_pano,
        Path(args.out).expanduser().resolve(),
        side=args.side,
    )

    if args.save_json:
        save_json = Path(args.save_json).expanduser().resolve()
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(
            json.dumps(
                {
                    "anatomic_points": payload_points,
                    "panorogram": payload_pano,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        report["raw_json"] = str(save_json)

    report.update(
        {
            "input_image": str(image_path),
            "anatomic_points_url": url_points,
            "anatomic_points_http_status": status_points,
            "anatomic_points_model_name": payload_points.get("model_name"),
            "panorogram_url": url_pano,
            "panorogram_http_status": status_pano,
            "panorogram_model_name": payload_pano.get("model_name"),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
