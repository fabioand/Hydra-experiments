#!/usr/bin/env python3
"""Script simples para chamar panorogram da RM e desenhar curvas na radiografia."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTH_DIR = REPO_ROOT / "radiomemory_auth"
if str(AUTH_DIR) not in sys.path:
    sys.path.append(str(AUTH_DIR))

from login import LINKAPI, LoginAPI  # noqa: E402


def pick_image(image_arg: str | None) -> Path:
    if image_arg:
        p = Path(image_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Imagem nao encontrada: {p}")
        return p

    imgs_dir = REPO_ROOT / "longoeixo" / "imgs"
    candidates = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.jpeg")) + list(imgs_dir.glob("*.png")))
    if not candidates:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {imgs_dir}")
    return candidates[0]


def encode_image_b64(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("ascii")


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


def auth_headers() -> dict[str, str]:
    auth = LoginAPI()
    token_type = auth.get("token_type")
    access_token = auth.get("access_token")
    if not token_type or not access_token:
        raise RuntimeError(f"Falha de autenticacao: {auth}")
    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def parse_payload(response: requests.Response, expected_model: str) -> dict[str, Any]:
    # Caso normal: JSON dict.
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    # Fallback: stream com uma linha JSON por modelo.
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

    raise RuntimeError(f"Resposta nao reconhecida (status={response.status_code}): {response.text[:500]}")


def call_panorogram(base: str, image_path: Path, timeout: int) -> tuple[dict[str, Any], str, int]:
    url = f"{base.rstrip('/')}/v1/panoramics/panorogram"
    resp = requests.post(url, headers=auth_headers(), json=build_body(image_path), timeout=timeout)
    payload = parse_payload(resp, expected_model="panorogram")
    if resp.status_code >= 400:
        raise RuntimeError(f"Erro API {resp.status_code}: {json.dumps(payload, ensure_ascii=False)[:800]}")
    return payload, url, resp.status_code


def extract_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return []
    return [e for e in entities if isinstance(e, dict)]


def draw_panorogram_contours(image_path: Path, payload: dict[str, Any], out_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Falha ao abrir imagem: {image_path}")
    img_h, img_w = image.shape[:2]

    src_w = float(payload.get("output_width") or img_w)
    src_h = float(payload.get("output_height") or img_h)
    sx = (img_w / src_w) if src_w > 0 else 1.0
    sy = (img_h / src_h) if src_h > 0 else 1.0

    entities = extract_entities(payload)
    drawn = 0

    for ent in entities:
        contour = ent.get("contour")
        if not isinstance(contour, list) or len(contour) < 2:
            continue

        pts: list[list[int]] = []
        for p in contour:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            try:
                x = int(round(float(p[0]) * sx))
                y = int(round(float(p[1]) * sy))
            except Exception:
                continue
            pts.append([x, y])

        if len(pts) < 2:
            continue

        # OpenCV expects contour as Nx1x2 int32.
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [arr], isClosed=False, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        drawn += 1

    cv2.rectangle(image, (8, 8), (560, 44), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"Panorogram contours: {drawn}",
        (14, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)

    return {
        "image_size": [img_w, img_h],
        "api_output_size": [payload.get("output_width"), payload.get("output_height")],
        "scale": [sx, sy],
        "entities_total": len(entities),
        "contours_drawn": drawn,
        "out_image": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chama RM panorogram e desenha curvas na radiografia.")
    parser.add_argument("--image", default=None, help="Imagem panoramica (default: primeira em longoeixo/imgs).")
    parser.add_argument("--base", default=LINKAPI, help="Base da API RM.")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout HTTP em segundos.")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "radiomemory_api_tools" / "outputs" / "panorogram_simple_overlay.png"),
        help="PNG de saida.",
    )
    parser.add_argument(
        "--save-json",
        default=str(REPO_ROOT / "radiomemory_api_tools" / "outputs" / "panorogram_simple_response.json"),
        help="JSON para salvar payload bruto.",
    )
    args = parser.parse_args()

    image_path = pick_image(args.image)
    payload, url, status = call_panorogram(args.base, image_path, timeout=args.timeout)

    out_path = Path(args.out).expanduser().resolve()
    report = draw_panorogram_contours(image_path, payload, out_path)

    save_json_path = Path(args.save_json).expanduser().resolve()
    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report.update(
        {
            "input_image": str(image_path),
            "api_url": url,
            "http_status": status,
            "model_name": payload.get("model_name"),
            "raw_json": str(save_json_path),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
