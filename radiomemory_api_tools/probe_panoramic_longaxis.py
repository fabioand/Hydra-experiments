#!/usr/bin/env python3
import argparse
import base64
import json
import sys
from pathlib import Path
from io import BytesIO

import requests
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radiomemory_auth.login import LINKAPI, LoginAPI  # noqa: E402


DEFAULT_ENDPOINT = "https://api.radiomemory.com.br/ia-dev/api/v1/panoramics/longaxis"


def pick_default_image() -> Path:
    imgs_dir = REPO_ROOT / "longoeixo" / "imgs"
    candidates = sorted(
        [
            *imgs_dir.glob("*.jpg"),
            *imgs_dir.glob("*.jpeg"),
            *imgs_dir.glob("*.png"),
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {imgs_dir}")
    return candidates[0]


def image_to_base64_like_aihub(path: Path) -> str:
    # AIHub.from_array: Image.fromarray(img) -> save JPEG -> base64
    # Here we load from disk, normalize to numpy array, and re-encode as JPEG
    # to match the same transport format used by AIHub.
    with Image.open(path) as pil_img:
        arr = np.array(pil_img)
    img = Image.fromarray(arr)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def summarize_response(resp: requests.Response) -> dict:
    summary = {
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type"),
        "is_json": False,
        "response_type": type(resp.text).__name__,
    }

    try:
        payload = resp.json()
        summary["is_json"] = True
        summary["response_type"] = type(payload).__name__

        if isinstance(payload, dict):
            summary["top_level_keys"] = sorted(payload.keys())
            summary["top_level_value_types"] = {
                k: type(v).__name__ for k, v in payload.items()
            }
        elif isinstance(payload, list):
            summary["list_length"] = len(payload)
            if payload:
                summary["first_item_type"] = type(payload[0]).__name__

        summary["body_preview"] = json.dumps(payload, ensure_ascii=False)[:1200]
    except Exception:
        summary["body_preview"] = (resp.text or "")[:1200]

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pega uma panorâmica do dataset longoeixo e chama o endpoint longaxis da RM."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Caminho da panorâmica. Se omitido, usa a primeira de longoeixo/imgs.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Endpoint da API para longaxis panorâmico.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout HTTP em segundos.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Se informado, salva o resumo da resposta nesse JSON.",
    )
    args = parser.parse_args()

    image_path = args.image if args.image else pick_default_image()
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    auth = LoginAPI()
    token_type = auth.get("token_type")
    access_token = auth.get("access_token")
    if not token_type or not access_token:
        raise RuntimeError(f"Falha na autenticação. Resposta: {auth}")

    headers = {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "text/plain",
    }
    body = {"base64_image": image_to_base64_like_aihub(image_path)}

    resp = requests.post(args.endpoint, headers=headers, json=body, timeout=args.timeout)

    summary = {
        "auth_base": LINKAPI,
        "endpoint": args.endpoint,
        "image_path": str(image_path),
        "image_size_bytes": image_path.stat().st_size,
        "response": summarize_response(resp),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
