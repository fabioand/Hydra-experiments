#!/usr/bin/env python3
"""Audita particao de dentes por ROI (3 retangulos) no dataset longoeixo.

Fluxo:
1) Para cada amostra (jpg+json), obtem pontos anatomicos da API RM
   (`/v1/panoramics/anatomic_points`) com cache em disco.
2) Constroi os 3 retangulos definidos no plano.
3) Verifica, no JSON de longoeixo, se cada dente cai dentro do retangulo
   do seu grupo (R_LEFT, R_RIGHT, R_CENTER).
4) Gera relatorio JSON/CSV com violacoes por amostra e por dente.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radiomemory_auth.login import LINKAPI, LoginAPI


TOOTH_SETS_BY_RECT: Dict[str, List[str]] = {
    "R_LEFT": ["24", "25", "26", "27", "28", "34", "35", "36", "37", "38"],
    "R_RIGHT": ["14", "15", "16", "17", "18", "44", "45", "46", "47", "48"],
    "R_CENTER": ["11", "12", "13", "21", "22", "23", "31", "32", "33", "41", "42", "43"],
}

TOOTH_TO_RECT: Dict[str, str] = {
    tooth: rect for rect, teeth in TOOTH_SETS_BY_RECT.items() for tooth in teeth
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in value if not unicodedata.combining(ch)).lower()


def encode_image_b64(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("ascii")


def build_body(image_path: Path) -> dict:
    return {
        "base64_image": encode_image_b64(image_path),
        "output_width": 0,
        "output_height": 0,
        "threshold": 0.0,
        "resource": "describe",
        "lang": "pt-br",
        "use_cache": False,
    }


class RMClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers: Optional[Dict[str, str]] = None
        self._lock = threading.Lock()

    def _login(self) -> Dict[str, str]:
        auth = LoginAPI()
        token_type = auth.get("token_type")
        access_token = auth.get("access_token")
        if not token_type or not access_token:
            raise RuntimeError(f"Authentication failed: {auth}")
        return {
            "Authorization": f"{token_type} {access_token}",
            "Content-type": "application/json",
            "Accept": "application/json",
        }

    def _get_headers(self) -> Dict[str, str]:
        with self._lock:
            if self._headers is None:
                self._headers = self._login()
            return dict(self._headers)

    def _refresh_headers(self) -> None:
        with self._lock:
            self._headers = self._login()

    def get_anatomic_points(self, image_path: Path, retries: int = 3) -> dict:
        url = f"{self.base_url}/v1/panoramics/anatomic_points"
        body = build_body(image_path)

        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            headers = self._get_headers()
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.5 * attempt)
                continue

            if resp.status_code == 401:
                self._refresh_headers()
                time.sleep(0.2 * attempt)
                continue

            try:
                payload = resp.json()
            except Exception:
                raise RuntimeError(f"Non-JSON response {resp.status_code}: {resp.text[:500]}")

            if resp.status_code >= 400:
                raise RuntimeError(f"API error {resp.status_code}: {json.dumps(payload, ensure_ascii=False)[:800]}")
            return payload

        if last_error:
            raise RuntimeError(f"API request failed after retries: {last_error}")
        raise RuntimeError("API request failed after retries")


def extract_named_points(payload: dict, image_size: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
    img_w, img_h = image_size
    entities = payload.get("entities")
    if not isinstance(entities, list):
        raise RuntimeError("Payload has no entities list")

    src_w = float(payload.get("output_width") or img_w)
    src_h = float(payload.get("output_height") or img_h)
    sx = (img_w / src_w) if src_w > 0 else 1.0
    sy = (img_h / src_h) if src_h > 0 else 1.0

    out: Dict[str, Tuple[float, float]] = {}
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        pt = ent.get("point")
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            continue
        try:
            x = float(pt[0]) * sx
            y = float(pt[1]) * sy
        except Exception:
            continue
        name = normalize_text(str(ent.get("class_name") or ""))
        if "condilo - esquerdo" in name:
            out["condilo_esquerdo"] = (x, y)
        elif "condilo - direito" in name:
            out["condilo_direito"] = (x, y)
        elif "e.n.a." in name or "ena" in name:
            out["ena"] = (x, y)
        elif "mentoniano" in name:
            out["mentoniano"] = (x, y)

    required = ["condilo_esquerdo", "condilo_direito", "ena", "mentoniano"]
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"Missing required points: {missing}")
    return out


def build_rectangles(points: Dict[str, Tuple[float, float]], image_size: Tuple[int, int]) -> Dict[str, List[int]]:
    w, h = image_size
    x_ce, y_ce = points["condilo_esquerdo"]
    x_cd, y_cd = points["condilo_direito"]
    x_ena, _ = points["ena"]
    _, y_men = points["mentoniano"]

    r_left = [int(round(min(x_ce, x_ena))), int(round(y_ce)), int(round(max(x_ce, x_ena))), int(round(y_men))]
    r_right = [int(round(min(x_cd, x_ena))), int(round(y_cd)), int(round(max(x_cd, x_ena))), int(round(y_men))]

    x_mid_left = 0.5 * (x_ce + x_ena)
    x_mid_right = 0.5 * (x_cd + x_ena)
    r_center = [
        int(round(min(x_mid_left, x_mid_right))),
        int(round(min(y_ce, y_cd))),
        int(round(max(x_mid_left, x_mid_right))),
        int(round(y_men)),
    ]

    def clamp_rect(rect: List[int]) -> List[int]:
        x1, y1, x2, y2 = rect
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return [x1, y1, x2, y2]

    return {
        "R_LEFT": clamp_rect(r_left),
        "R_RIGHT": clamp_rect(r_right),
        "R_CENTER": clamp_rect(r_center),
    }


def load_teeth_points(json_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
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
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def audit_sample(
    stem: str,
    img_path: Path,
    gt_json_path: Path,
    client: RMClient,
    cache_dir: Path,
    use_cache_only: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "stem": stem,
        "image": str(img_path),
        "gt_json": str(gt_json_path),
        "ok": False,
    }

    if not img_path.exists() or not gt_json_path.exists():
        result["error"] = "missing_image_or_json"
        return result

    img = Image.open(img_path)
    image_size = img.size

    cache_path = cache_dir / f"{stem}.json"
    payload: Optional[dict] = None
    cache_hit = False

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_hit = True
        except Exception:
            payload = None

    if payload is None:
        if use_cache_only:
            result["error"] = "cache_miss"
            return result
        payload = client.get_anatomic_points(img_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        named_points = extract_named_points(payload, image_size=image_size)
        rects = build_rectangles(named_points, image_size=image_size)
        gt = load_teeth_points(gt_json_path)
    except Exception as exc:
        result["error"] = f"prep_failed: {exc}"
        result["cache_hit"] = cache_hit
        return result

    violations: List[Dict[str, Any]] = []
    present_teeth = 0
    audited_points = 0

    for tooth, rect_name in TOOTH_TO_RECT.items():
        pts = gt.get(tooth, [])
        if not pts:
            continue
        present_teeth += 1
        rect = rects[rect_name]
        outside = []
        for i, pt in enumerate(pts):
            audited_points += 1
            if not point_in_rect(pt, rect):
                outside.append({"index": i, "x": pt[0], "y": pt[1]})
        if outside:
            violations.append(
                {
                    "tooth": tooth,
                    "rect": rect_name,
                    "rect_xyxy": rect,
                    "outside_points": outside,
                    "num_points": len(pts),
                }
            )

    result.update(
        {
            "ok": True,
            "cache_hit": cache_hit,
            "image_size": [image_size[0], image_size[1]],
            "named_points": {
                "condilo_esquerdo": list(named_points["condilo_esquerdo"]),
                "condilo_direito": list(named_points["condilo_direito"]),
                "ena": list(named_points["ena"]),
                "mentoniano": list(named_points["mentoniano"]),
            },
            "rectangles": rects,
            "present_teeth_in_partition": present_teeth,
            "audited_points": audited_points,
            "has_violation": len(violations) > 0,
            "violations": violations,
            "cache_path": str(cache_path),
        }
    )
    return result


def iter_stems(imgs_dir: Path, gt_dir: Path, limit: int) -> List[str]:
    imgs = {p.stem for p in imgs_dir.glob("*.jpg")}
    gts = {p.stem for p in gt_dir.glob("*.json")}
    stems = sorted(imgs & gts)
    if limit > 0:
        stems = stems[:limit]
    return stems


def write_summary_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "ok",
                "cache_hit",
                "has_violation",
                "present_teeth_in_partition",
                "audited_points",
                "num_violations",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "stem": r.get("stem"),
                    "ok": r.get("ok", False),
                    "cache_hit": r.get("cache_hit", False),
                    "has_violation": r.get("has_violation", False),
                    "present_teeth_in_partition": r.get("present_teeth_in_partition", 0),
                    "audited_points": r.get("audited_points", 0),
                    "num_violations": len(r.get("violations", []) or []),
                    "error": r.get("error", ""),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit longoeixo tooth partition vs ROI rectangles with API anatomic points.")
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--gt-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument("--base", default=LINKAPI, help="RM API base URL")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=999, help="0 means all")
    parser.add_argument("--use-cache-only", action="store_true", help="Do not call API; fail on cache miss.")
    parser.add_argument("--out-dir", type=Path, default=Path("radiomemory_api_tools/outputs/roi_partition_audit"))
    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    gt_dir = args.gt_dir
    out_dir = args.out_dir
    cache_dir = out_dir / "anatomic_points_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    limit = 0 if args.limit < 0 else args.limit
    stems = iter_stems(imgs_dir=imgs_dir, gt_dir=gt_dir, limit=limit)
    if not stems:
        raise RuntimeError("No paired samples found.")

    client = RMClient(base_url=args.base, timeout=args.timeout)
    total = len(stems)

    results: List[Dict[str, Any]] = []
    processed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = []
        for stem in stems:
            img_path = imgs_dir / f"{stem}.jpg"
            gt_path = gt_dir / f"{stem}.json"
            futures.append(
                ex.submit(
                    audit_sample,
                    stem,
                    img_path,
                    gt_path,
                    client,
                    cache_dir,
                    args.use_cache_only,
                )
            )
        for fut in as_completed(futures):
            processed += 1
            item = fut.result()
            results.append(item)
            if processed % 25 == 0 or processed == total:
                print(f"[{processed}/{total}] processed")

    results.sort(key=lambda r: r.get("stem", ""))

    ok_results = [r for r in results if r.get("ok")]
    err_results = [r for r in results if not r.get("ok")]
    viol_results = [r for r in ok_results if r.get("has_violation")]

    violations_by_tooth: Dict[str, int] = {}
    violations_by_rect: Dict[str, int] = {"R_LEFT": 0, "R_RIGHT": 0, "R_CENTER": 0}
    total_violations = 0
    for r in viol_results:
        for v in r.get("violations", []):
            tooth = str(v.get("tooth"))
            rect = str(v.get("rect"))
            violations_by_tooth[tooth] = violations_by_tooth.get(tooth, 0) + 1
            if rect in violations_by_rect:
                violations_by_rect[rect] += 1
            total_violations += 1

    summary = {
        "base_url": args.base,
        "total_samples": total,
        "ok_samples": len(ok_results),
        "error_samples": len(err_results),
        "samples_with_violation": len(viol_results),
        "total_tooth_violations": total_violations,
        "violations_by_rect": violations_by_rect,
        "violations_by_tooth": dict(sorted(violations_by_tooth.items(), key=lambda kv: kv[0])),
        "tooth_sets_by_rect": TOOTH_SETS_BY_RECT,
        "workers": args.workers,
        "limit": args.limit,
        "use_cache_only": args.use_cache_only,
        "cache_dir": str(cache_dir),
    }

    summary_path = out_dir / "audit_summary.json"
    details_path = out_dir / "audit_details.json"
    csv_path = out_dir / "audit_summary.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(csv_path, results)

    print(json.dumps({"summary": str(summary_path), "details": str(details_path), "csv": str(csv_path)}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
