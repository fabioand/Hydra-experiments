#!/usr/bin/env python3
import argparse
import base64
import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path
import sys
from typing import Optional

import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radiomemory_auth.login import LINKAPI, LoginAPI  # noqa: E402


KEYWORD_PATTERNS = [
    re.compile(r"cond", re.IGNORECASE),
    re.compile(r"gon", re.IGNORECASE),
    re.compile(r"ena|ans", re.IGNORECASE),
    re.compile(r"ment", re.IGNORECASE),
    re.compile(r"nasi|sella|porion|orbitale", re.IGNORECASE),
]


def pick_sample_image(image_arg: Optional[str]) -> Path:
    if image_arg:
        path = Path(image_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    imgs_dir = REPO_ROOT / "longoeixo" / "imgs"
    candidates = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.jpeg")) + list(imgs_dir.glob("*.png")))
    if not candidates:
        raise FileNotFoundError(f"No images in {imgs_dir}")
    return candidates[0]


def encode_image_b64(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("ascii")


def auth_headers() -> dict:
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


def extract_entities_from_json(payload):
    if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        return payload["entities"]
    return []


def extract_entities_from_describe_stream(text: str):
    entities = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("model_name") == "anatomic_points" and isinstance(obj.get("entities"), list):
            entities.extend(obj["entities"])
    return entities


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value


def count_keyword_hits(class_names):
    hits = []
    for name in class_names:
        normalized = normalize_text(name)
        for pat in KEYWORD_PATTERNS:
            if pat.search(normalized):
                hits.append(name)
                break
    return sorted(set(hits))


def probe_endpoint(base: str, path: str, body: dict, headers: dict, timeout: int):
    url = f"{base.rstrip('/')}{path}"
    response = requests.post(url, headers=headers, json=body, timeout=timeout)

    result = {
        "path": path,
        "url": url,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "entity_count": 0,
        "class_names": [],
        "keyword_hits": [],
        "looks_like_target": False,
        "body_preview": response.text[:1200],
    }

    entities = []
    parsed = None
    try:
        parsed = response.json()
        entities = extract_entities_from_json(parsed)
    except Exception:
        entities = extract_entities_from_describe_stream(response.text)

    if entities:
        class_names = []
        for e in entities:
            c = e.get("class_name")
            if c is not None:
                class_names.append(str(c))

        uniq = sorted(set(class_names))
        keyword_hits = count_keyword_hits(uniq)

        result["entity_count"] = len(entities)
        result["class_names"] = uniq
        result["keyword_hits"] = keyword_hits

        # Target profile requested: few points (~8-14) and anatomical names.
        # In practice, some studies can return 6 points only.
        result["looks_like_target"] = (6 <= len(uniq) <= 20) and (len(keyword_hits) > 0)

        if parsed is not None:
            result["body_preview"] = json.dumps(parsed, ensure_ascii=False)[:1200]

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Find panoramic anatomical-points endpoint in RM API")
    parser.add_argument("--base", default=LINKAPI, help="Base URL")
    parser.add_argument("--image", default=None, help="Path to panoramic image")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--save", default=str(REPO_ROOT / "radiomemory_api_discovery" / "panoramic_anatomic_endpoint_report.json"))
    args = parser.parse_args()

    image_path = pick_sample_image(args.image)
    image_b64 = encode_image_b64(image_path)

    headers = auth_headers()
    body = {
        "base64_image": image_b64,
        "output_width": 0,
        "output_height": 0,
        "threshold": 0.0,
        "resource": "describe",
        "lang": "pt-br",
        "use_cache": False,
    }

    candidates = [
        "/v1/panoramics/anatomic_points",
        "/v1/panoramics/describe",
        "/radiobot/panoramics/describe",
    ]

    probes = [probe_endpoint(args.base, p, body, headers, args.timeout) for p in candidates]

    found = [p for p in probes if p["looks_like_target"]]

    report = {
        "base": args.base,
        "image": str(image_path),
        "probes": probes,
        "found_candidates": found,
        "best_guess": found[0] if found else None,
    }

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "image": str(image_path),
        "best_guess_path": report["best_guess"]["path"] if report["best_guess"] else None,
        "best_guess_url": report["best_guess"]["url"] if report["best_guess"] else None,
        "best_guess_status": report["best_guess"]["status_code"] if report["best_guess"] else None,
        "best_guess_entity_count": report["best_guess"]["entity_count"] if report["best_guess"] else None,
        "best_guess_keyword_hits": report["best_guess"]["keyword_hits"] if report["best_guess"] else None,
        "report": str(save_path),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
