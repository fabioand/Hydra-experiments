import argparse
import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List

import requests
from PIL import Image

from login import LINKAPI, LoginAPI


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_to_base64(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
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


def collect_images(input_dir: Path, recursive: bool, limit: int) -> List[Path]:
    walker = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = [p for p in walker if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    images.sort()

    if limit > 0:
        return images[:limit]
    return images


def build_body(image_path: Path, threshold: float, resource: str) -> dict:
    return {
        "base64_image": image_to_base64(image_path),
        "output_width": 0,
        "output_height": 0,
        "threshold": threshold,
        "resource": resource,
        "lang": "pt-br",
        "use_cache": False,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for RM IA endpoints")
    parser.add_argument("mode", choices=["v1", "tomo"], help="Endpoint group")
    parser.add_argument("endpoint", help="Example: panoramics/longaxis or SagitalClass")
    parser.add_argument("--input-dir", default=".", help="Input folder with images (default: current dir)")
    parser.add_argument("--output-dir", default="./rm_ia_batch_out", help="Output folder for JSON responses")
    parser.add_argument("--limit", type=int, default=10, help="Max images to process (0 = all)")
    parser.add_argument("--base", default=LINKAPI, help="Base URL (default from login.py LINKAPI)")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--resource", default="describe")
    parser.add_argument("--recursive", action="store_true", help="Scan input dir recursively")
    parser.add_argument("--auth", action="store_true", help="Force auth for tomo mode")
    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Invalid input dir: {input_dir}")

    images = collect_images(input_dir=input_dir, recursive=args.recursive, limit=args.limit)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    if args.mode == "v1":
        url = f"{args.base.rstrip('/')}/v1/{args.endpoint.lstrip('/')}"
        headers = auth_headers()
    else:
        url = f"{args.base.rstrip('/')}/internal/tomos/{args.endpoint.lstrip('/')}"
        headers = auth_headers() if args.auth else {
            "Content-type": "application/json",
            "Accept": "application/json",
        }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{run_id}_{args.mode}_{args.endpoint.replace('/', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "base": args.base,
        "url": url,
        "mode": args.mode,
        "endpoint": args.endpoint,
        "input_dir": str(input_dir),
        "output_dir": str(run_dir),
        "requested_limit": args.limit,
        "processed": 0,
        "ok": 0,
        "errors": 0,
        "items": [],
    }

    print(f"URL: {url}")
    print(f"Images found: {len(images)}")
    print(f"Output dir: {run_dir}")

    for idx, image_path in enumerate(images, start=1):
        body = build_body(image_path, threshold=args.threshold, resource=args.resource)
        response_file = run_dir / f"{idx:04d}_{image_path.stem}.json"

        item = {
            "index": idx,
            "image": str(image_path),
            "response_file": str(response_file),
        }

        try:
            response = requests.post(url, json=body, headers=headers, timeout=args.timeout)
            item["status_code"] = response.status_code
            item["ok"] = response.ok

            try:
                payload = response.json()
            except Exception:
                payload = {"raw": response.text}

            write_json(
                response_file,
                {
                    "request": {
                        "url": url,
                        "image": str(image_path),
                        "mode": args.mode,
                        "endpoint": args.endpoint,
                    },
                    "response": {
                        "status_code": response.status_code,
                        "ok": response.ok,
                        "body": payload,
                    },
                },
            )

            if response.ok:
                summary["ok"] += 1
            else:
                summary["errors"] += 1

        except Exception as exc:
            item["ok"] = False
            item["status_code"] = None
            item["error"] = str(exc)
            summary["errors"] += 1
            write_json(
                response_file,
                {
                    "request": {
                        "url": url,
                        "image": str(image_path),
                        "mode": args.mode,
                        "endpoint": args.endpoint,
                    },
                    "error": str(exc),
                },
            )

        summary["processed"] += 1
        summary["items"].append(item)
        print(f"[{idx}/{len(images)}] {image_path.name} -> {item.get('status_code')} ({'ok' if item['ok'] else 'error'})")

    summary_file = run_dir / "summary.json"
    write_json(summary_file, summary)

    print("Done.")
    print(f"Processed: {summary['processed']} | OK: {summary['ok']} | Errors: {summary['errors']}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
