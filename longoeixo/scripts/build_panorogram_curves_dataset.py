#!/usr/bin/env python3
"""Gera dataset de curvas alveolares normalizadas (128 pontos) a partir do endpoint panorogram RM.

Para cada imagem em `longoeixo/imgs`, salva um JSON com mesmo stem em pasta de saída.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from radiomemory_auth.login import LINKAPI, LoginAPI  # noqa: E402
from radiomemory_api_tools.normalize_alveolar_curves import normalize_alveolar_curves  # noqa: E402


THREAD_LOCAL = threading.local()


def _session() -> requests.Session:
    sess = getattr(THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        THREAD_LOCAL.session = sess
    return sess


def pick_images(imgs_dir: Path, max_images: int = 0) -> list[Path]:
    images = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.jpeg")) + list(imgs_dir.glob("*.png")))
    if max_images > 0:
        images = images[:max_images]
    return images


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


def parse_payload(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    for line in (response.text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if str(obj.get("model_name", "")).lower() == "panorogram":
            return obj

    raise RuntimeError(f"Resposta nao reconhecida status={response.status_code}: {response.text[:400]}")


def call_panorogram(image_path: Path, base: str, headers: dict[str, str], timeout: int) -> tuple[dict[str, Any], int]:
    url = f"{base.rstrip('/')}/v1/panoramics/panorogram"
    resp = _session().post(url, headers=headers, json=build_body(image_path), timeout=timeout)
    payload = parse_payload(resp)
    if resp.status_code >= 400:
        raise RuntimeError(f"Erro API {resp.status_code}: {json.dumps(payload, ensure_ascii=False)[:600]}")
    return payload, resp.status_code


def process_one(
    image_path: Path,
    out_dir: Path,
    base: str,
    headers: dict[str, str],
    timeout: int,
    n_points: int,
    retries: int,
) -> dict[str, Any]:
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            payload, status = call_panorogram(image_path, base=base, headers=headers, timeout=timeout)
            curves = normalize_alveolar_curves(payload, n_points_sup=n_points, n_points_inf=n_points)
            out_path = out_dir / f"{image_path.stem}.json"
            out = {
                "image_stem": image_path.stem,
                "image_name": image_path.name,
                "model_name": payload.get("model_name"),
                "http_status": status,
                "n_points_sup": n_points,
                "n_points_inf": n_points,
                "RebAlvSup": curves["RebAlvSup"].tolist(),
                "RebAlvInf": curves["RebAlvInf"].tolist(),
                "shape_sup": list(curves["RebAlvSup"].shape),
                "shape_inf": list(curves["RebAlvInf"].shape),
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"ok": True, "stem": image_path.stem, "out": str(out_path), "attempt": attempt}
        except Exception as exc:
            last_err = str(exc)
    return {"ok": False, "stem": image_path.stem, "error": last_err}


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera JSONs de curvas alveolares normalizadas via panorogram RM.")
    parser.add_argument("--imgs-dir", default=str(REPO_ROOT / "longoeixo" / "imgs"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "longoeixo" / "data_panorogram_curves128"))
    parser.add_argument("--base", default=LINKAPI)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=0, help="0 = todas")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    imgs_dir = Path(args.imgs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = pick_images(imgs_dir, max_images=int(args.max_images))
    if not images:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {imgs_dir}")

    if not args.overwrite:
        images = [p for p in images if not (out_dir / f"{p.stem}.json").exists()]

    total = len(images)
    if total == 0:
        print(json.dumps({"status": "nothing_to_do", "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
        return 0

    headers = auth_headers()
    ok_count = 0
    fail_count = 0
    failed: list[dict[str, Any]] = []

    print(f"[START] images={total} workers={args.workers} out_dir={out_dir}")
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [
            ex.submit(
                process_one,
                image_path=p,
                out_dir=out_dir,
                base=args.base,
                headers=headers,
                timeout=int(args.timeout),
                n_points=int(args.n_points),
                retries=int(args.retries),
            )
            for p in images
        ]
        for i, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            if res["ok"]:
                ok_count += 1
            else:
                fail_count += 1
                failed.append(res)

            if i % 10 == 0 or i == total:
                print(f"[PROGRESS] done={i}/{total} ok={ok_count} fail={fail_count}")

    report = {
        "imgs_dir": str(imgs_dir),
        "out_dir": str(out_dir),
        "total_requested": total,
        "ok": ok_count,
        "fail": fail_count,
        "n_points": int(args.n_points),
        "workers": int(args.workers),
    }
    report_path = out_dir / "_build_report.json"
    report_path.write_text(json.dumps({"summary": report, "failed": failed}, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

