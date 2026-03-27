import argparse
import base64
import json
import math
import statistics as st
from io import BytesIO
from pathlib import Path
import sys

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
AUTH_DIR = ROOT / "radiomemory_auth"
sys.path.append(str(AUTH_DIR))
from login import LoginAPI  # noqa: E402


def load_gt(json_path: Path):
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    out = {}
    for item in raw:
        label = str(item["label"])
        p1 = (float(item["pts"][0]["x"]), float(item["pts"][0]["y"]))
        p2 = (float(item["pts"][1]["x"]), float(item["pts"][1]["y"]))
        out[label] = (p1, p2)
    return out


def encode_jpg_b64(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    buff = BytesIO()
    image.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode()


def call_rm_longaxis(image_path: Path, base_url: str):
    auth = LoginAPI()
    token_type = auth.get("token_type")
    access_token = auth.get("access_token")

    if not token_type or not access_token:
        raise RuntimeError(f"Auth failed: {auth}")

    headers = {
        "Authorization": f"{token_type} {access_token}",
        "Content-type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "base64_image": encode_jpg_b64(image_path),
        "output_width": 0,
        "output_height": 0,
        "threshold": 0.0,
        "resource": "describe",
        "lang": "pt-br",
        "use_cache": False,
    }

    url = f"{base_url.rstrip('/')}/v1/panoramics/longaxis"
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    obj = resp.json()
    return resp.status_code, url, obj


def main():
    parser = argparse.ArgumentParser(description="Compare RM longaxis output against repo dataset JSON")
    parser.add_argument("sample", help="Sample basename (without .jpg/.json)")
    parser.add_argument("--base", default="https://api.radiomemory.com.br/ia-idoc")
    parser.add_argument("--imgs-dir", default=str(ROOT / "longoeixo" / "imgs"))
    parser.add_argument("--json-dir", default=str(ROOT / "longoeixo" / "data_longoeixo"))
    parser.add_argument("--out", default="/tmp/rm_longaxis_vs_dataset_compare.json")
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.0,
        help="Only predictions with score >= threshold are considered present teeth",
    )
    args = parser.parse_args()

    img_path = Path(args.imgs_dir) / f"{args.sample}.jpg"
    json_path = Path(args.json_dir) / f"{args.sample}.json"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    gt = load_gt(json_path)

    status, url, pred_obj = call_rm_longaxis(img_path, args.base)

    orig_w, orig_h = Image.open(img_path).size
    out_w = float(pred_obj.get("output_width", 0.0) or 0.0)
    out_h = float(pred_obj.get("output_height", 0.0) or 0.0)
    sx = (orig_w / out_w) if out_w else 1.0
    sy = (orig_h / out_h) if out_h else 1.0

    pred_all = {}
    for e in pred_obj.get("entities", []):
        label = str(e.get("class_name"))
        line = e.get("line")
        if not (isinstance(line, list) and len(line) == 2):
            continue
        p1 = (float(line[0][0]) * sx, float(line[0][1]) * sy)
        p2 = (float(line[1][0]) * sx, float(line[1][1]) * sy)
        pred_all[label] = {
            "p1": p1,
            "p2": p2,
            "score": float(e.get("score", 0.0) or 0.0),
        }

    pred = {
        label: value
        for label, value in pred_all.items()
        if value["score"] >= args.presence_threshold
    }

    gt_labels = set(gt.keys())
    pred_labels = set(pred.keys())
    matched = sorted(gt_labels & pred_labels, key=lambda x: int(x))
    missing_in_pred = sorted(gt_labels - pred_labels, key=lambda x: int(x))
    extra_in_pred = sorted(pred_labels - gt_labels, key=lambda x: int(x))

    rows = []
    endpoint_dists = []
    midpoint_dists = []
    swapped_count = 0

    for label in matched:
        g1, g2 = gt[label]
        p1, p2 = pred[label]["p1"], pred[label]["p2"]

        d_same = (math.dist(p1, g1), math.dist(p2, g2))
        d_swap = (math.dist(p1, g2), math.dist(p2, g1))

        if sum(d_swap) < sum(d_same):
            d1, d2 = d_swap
            swapped = True
            swapped_count += 1
        else:
            d1, d2 = d_same
            swapped = False

        endpoint_mean = (d1 + d2) / 2.0

        pm = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        gm = ((g1[0] + g2[0]) / 2.0, (g1[1] + g2[1]) / 2.0)
        midpoint = math.dist(pm, gm)

        rows.append(
            {
                "label": label,
                "score": pred[label]["score"],
                "endpoint_p1_px": d1,
                "endpoint_p2_px": d2,
                "endpoint_mean_px": endpoint_mean,
                "midpoint_px": midpoint,
                "swapped": swapped,
            }
        )

        endpoint_dists.extend([d1, d2])
        midpoint_dists.append(midpoint)

    rows_sorted = sorted(rows, key=lambda r: r["endpoint_mean_px"])

    summary = {
        "sample": args.sample,
        "api_url": url,
        "http_status": status,
        "image_size": [orig_w, orig_h],
        "model_output_size": [out_w, out_h],
        "scale_applied": [sx, sy],
        "gt_teeth_count": len(gt_labels),
        "pred_teeth_count": len(pred_labels),
        "pred_teeth_count_all": len(pred_all),
        "presence_threshold": args.presence_threshold,
        "matched_teeth_count": len(matched),
        "missing_in_prediction": missing_in_pred,
        "extra_in_prediction": extra_in_pred,
        "swapped_line_count": swapped_count,
        "endpoint_mae_px": (sum(endpoint_dists) / len(endpoint_dists)) if endpoint_dists else None,
        "endpoint_median_px": st.median(endpoint_dists) if endpoint_dists else None,
        "endpoint_p95_px": sorted(endpoint_dists)[int(0.95 * (len(endpoint_dists) - 1))] if endpoint_dists else None,
        "midpoint_mae_px": (sum(midpoint_dists) / len(midpoint_dists)) if midpoint_dists else None,
        "best_5": rows_sorted[:5],
        "worst_8": sorted(rows, key=lambda r: r["endpoint_mean_px"], reverse=True)[:8],
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved comparison: {out_path}")
    print(json.dumps({
        "sample": summary["sample"],
        "http_status": summary["http_status"],
        "matched": summary["matched_teeth_count"],
        "missing": summary["missing_in_prediction"],
        "extra": summary["extra_in_prediction"],
        "endpoint_mae_px": summary["endpoint_mae_px"],
        "endpoint_p95_px": summary["endpoint_p95_px"],
        "midpoint_mae_px": summary["midpoint_mae_px"],
        "swapped_line_count": summary["swapped_line_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
