import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from login import LINKAPI, LoginAPI


def image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def auth_headers() -> dict:
    auth = LoginAPI()
    token_type = auth.get("token_type")
    access_token = auth.get("access_token")

    if not token_type or not access_token:
        raise RuntimeError(f"Failed to authenticate: {auth}")

    return {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }


def post_json(url: str, body: dict, use_auth: bool = True, timeout: int = 90) -> requests.Response:
    headers = auth_headers() if use_auth else {
        "Content-type": "application/json",
        "Accept": "application/json",
    }
    return requests.post(url, json=body, headers=headers, timeout=timeout)


def build_base64_body(image_path: str, threshold: float = 0.0, resource: str = "describe") -> dict:
    return {
        "base64_image": image_to_base64(image_path),
        "output_width": 0,
        "output_height": 0,
        "threshold": threshold,
        "resource": resource,
        "lang": "pt-br",
        "use_cache": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RM IA API client helper")
    sub = parser.add_subparsers(dest="command", required=True)

    v1 = sub.add_parser("v1", help="Call /v1 endpoints (auth required)")
    v1.add_argument("endpoint", help="Example: panoramics/longaxis")
    v1.add_argument("image", help="Path to image file")
    v1.add_argument("--base", default=LINKAPI, help="Base URL (default from login.py LINKAPI)")
    v1.add_argument("--threshold", type=float, default=0.0)
    v1.add_argument("--resource", default="describe")

    tomo = sub.add_parser("tomo", help="Call /internal/tomos endpoints")
    tomo.add_argument("endpoint", help="Example: SagitalClass, BigFOVSeg, KeypointsAxial")
    tomo.add_argument("image", help="Path to image file")
    tomo.add_argument("--base", default=LINKAPI, help="Base URL (default from login.py LINKAPI)")
    tomo.add_argument("--threshold", type=float, default=0.0)
    tomo.add_argument("--resource", default="describe")
    tomo.add_argument("--auth", action="store_true", help="Use token auth (not usually required for /internal/tomos)")

    tomo_legacy = sub.add_parser("tomo-legacy", help="Call legacy tomos autoencoder points endpoints")
    tomo_legacy.add_argument("endpoint", choices=["bbox-autoencoder-boca", "bbox-autoencoder-maxila", "bbox-autoencoder-mandibula"])
    tomo_legacy.add_argument("points", help="JSON list, e.g. '[[1,2],[3,4]]'")
    tomo_legacy.add_argument("--base", default=LINKAPI, help="Base URL (default from login.py LINKAPI)")
    tomo_legacy.add_argument("--auth", action="store_true", help="Use token auth")

    args = parser.parse_args()

    if args.command == "v1":
        url = f"{args.base.rstrip('/')}/v1/{args.endpoint.lstrip('/')}"
        body = build_base64_body(args.image, threshold=args.threshold, resource=args.resource)
        response = post_json(url, body, use_auth=True)

    elif args.command == "tomo":
        url = f"{args.base.rstrip('/')}/internal/tomos/{args.endpoint.lstrip('/')}"
        body = build_base64_body(args.image, threshold=args.threshold, resource=args.resource)
        response = post_json(url, body, use_auth=args.auth)

    else:
        url = f"{args.base.rstrip('/')}/internal/tomos/{args.endpoint}"
        points = json.loads(args.points)
        body = {"points": points}
        response = post_json(url, body, use_auth=args.auth)

    print(f"URL: {url}")
    print(f"HTTP {response.status_code}")

    text = response.text.strip()
    try:
        parsed = response.json()
        print(json.dumps(parsed, ensure_ascii=False, indent=2)[:8000])
    except Exception:
        print(text[:8000])


if __name__ == "__main__":
    main()
