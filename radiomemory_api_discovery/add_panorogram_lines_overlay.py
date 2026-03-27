#!/usr/bin/env python3
import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
import sys

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUTH_DIR = ROOT / "radiomemory_auth"
sys.path.append(str(AUTH_DIR))
from login import LINKAPI, LoginAPI  # noqa: E402


def encode_jpg_b64(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode()


def build_body(image_path: Path) -> dict:
    return {
        "base64_image": encode_jpg_b64(image_path),
        "output_width": 0,
        "output_height": 0,
        "threshold": 0.0,
        "resource": "describe",
        "lang": "pt-br",
        "use_cache": False,
    }


def auth_headers() -> dict:
    auth = LoginAPI()
    token_type = auth.get("token_type")
    access_token = auth.get("access_token")
    if not token_type or not access_token:
        raise RuntimeError(f"Falha autenticacao: {auth}")
    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def extract_entities_by_model(resp_text: str, resp_json, model_name: str):
    target = model_name.lower().strip()

    if isinstance(resp_json, dict):
        m = str(resp_json.get("model_name") or "").lower()
        if m == target and isinstance(resp_json.get("entities"), list):
            return resp_json["entities"]

    entities = []
    for line in (resp_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        m = str(obj.get("model_name") or "").lower()
        if m == target and isinstance(obj.get("entities"), list):
            entities.extend(obj["entities"])

    if entities:
        return entities

    if isinstance(resp_json, dict) and isinstance(resp_json.get("entities"), list):
        # fallback para respostas que nao trazem model_name no topo
        return resp_json["entities"]

    return []


def fetch_model_entities(image_path: Path, base: str, model_name: str, endpoints: list[str], timeout: int = 120):
    headers = auth_headers()
    body = build_body(image_path)

    last_error = None
    for ep in endpoints:
        url = f"{base.rstrip('/')}{ep}"
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)

        parsed = None
        try:
            parsed = resp.json()
        except Exception:
            parsed = None

        entities = extract_entities_by_model(resp.text, parsed, model_name=model_name)
        if entities:
            return {"url": url, "status": resp.status_code, "entities": entities}

        last_error = {"url": url, "status": resp.status_code, "preview": (resp.text or "")[:400]}

    raise RuntimeError(f"Nao encontrei model={model_name} em nenhum endpoint: {last_error}")


def draw_contours(overlay_bgr: np.ndarray, entities: list, color=(255, 0, 255), thickness=2):
    out = overlay_bgr.copy()
    drawn = 0

    for ent in entities:
        contour = ent.get("contour")
        if not isinstance(contour, list) or len(contour) < 2:
            continue

        pts = []
        for p in contour:
            if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                continue
            try:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
            except Exception:
                continue
            pts.append([x, y])

        if len(pts) < 2:
            continue

        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        drawn += 1

    return out, drawn


def main():
    parser = argparse.ArgumentParser(description="Adiciona linhas de panorograma no overlay existente")
    parser.add_argument("--image", type=Path, required=True, help="Imagem panoramica original (.jpg)")
    parser.add_argument("--overlay", type=Path, required=True, help="Overlay existente para desenhar por cima")
    parser.add_argument("--out", type=Path, required=True, help="PNG de saida")
    parser.add_argument("--base", default=LINKAPI, help="Base da API RM")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {args.image}")
    if not args.overlay.exists():
        raise FileNotFoundError(f"Overlay nao encontrado: {args.overlay}")

    overlay = cv2.imread(str(args.overlay), cv2.IMREAD_COLOR)
    if overlay is None:
        raise RuntimeError(f"Falha ao abrir overlay: {args.overlay}")

    pano_info = fetch_model_entities(
        args.image,
        args.base,
        model_name="panorogram",
        endpoints=["/v1/panoramics/panorogram", "/v1/panoramics/anatomic_points"],
    )
    out_img, pano_drawn = draw_contours(overlay, pano_info["entities"], color=(255, 0, 255), thickness=2)

    seg_info = fetch_model_entities(
        args.image,
        args.base,
        model_name="teeth_segmentation",
        endpoints=["/v1/panoramics/teeth_segmentation", "/v1/panoramics/describe"],
    )
    out_img, seg_drawn = draw_contours(out_img, seg_info["entities"], color=(255, 255, 0), thickness=1)

    cv2.rectangle(out_img, (8, 8), (560, 72), (0, 0, 0), -1)
    cv2.putText(out_img, f"Panorogram: {pano_drawn} (magenta)", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out_img, f"Teeth segmentation: {seg_drawn} (cyan)", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out_img)

    print(json.dumps({
        "panorogram": {
            "status": pano_info["status"],
            "url": pano_info["url"],
            "entities": len(pano_info["entities"]),
            "drawn_contours": pano_drawn,
        },
        "teeth_segmentation": {
            "status": seg_info["status"],
            "url": seg_info["url"],
            "entities": len(seg_info["entities"]),
            "drawn_contours": seg_drawn,
        },
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
