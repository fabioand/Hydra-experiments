#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from panorama_foundation.models import PanoramicResNetAutoencoder


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _discover_images(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in files})


def _load_model(ckpt_path: Path, device: torch.device) -> PanoramicResNetAutoencoder:
    model = PanoramicResNetAutoencoder(backbone="resnet34").to(device)
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw.get("model_state_dict", raw)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _to_u8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)


def _preprocess(gray_u8: np.ndarray, size: int) -> torch.Tensor:
    x = cv2.resize(gray_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


def _build_html(run_name: str, rows: List[dict]) -> str:
    cards = []
    for r in rows:
        stem = html.escape(r["stem"])
        orig_rel = html.escape(r["orig_rel"])
        enh_rel = html.escape(r["enh_rel"])
        cards.append(
            f"""
            <div class="card">
              <div class="title">{stem}</div>
              <div class="pair">
                <figure>
                  <img src="{orig_rel}" loading="lazy" />
                  <figcaption>Original</figcaption>
                </figure>
                <figure>
                  <img src="{enh_rel}" loading="lazy" />
                  <figcaption>Enhanced</figcaption>
                </figure>
              </div>
            </div>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AE Batch Enhance - {html.escape(run_name)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #0f1217; color: #e7edf6; }}
    .wrap {{ padding: 14px; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .meta {{ color: #9bb0c7; margin-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 12px; }}
    .card {{ background: #151b22; border: 1px solid #283445; border-radius: 8px; padding: 8px; }}
    .title {{ font-size: 12px; color: #cbd7e8; margin-bottom: 6px; word-break: break-all; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; display: block; background: #000; border-radius: 6px; }}
    figcaption {{ font-size: 11px; color: #9bb0c7; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AE Batch Enhance</h1>
    <div class="meta">Run: {html.escape(run_name)} | Pares: {len(rows)}</div>
    <div class="grid">
      {"".join(cards)}
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica filtro AE em lote e gera mosaico HTML.")
    parser.add_argument("--images-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--ckpt", type=Path, default=Path("ae_radiograph_filter/models/ae_identity_bestE21.ckpt"))
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--fs", type=float, default=0.3)
    parser.add_argument("--fa", type=float, default=0.3)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    images_dir = args.images_dir if args.images_dir.is_absolute() else (root / args.images_dir)
    ckpt = args.ckpt if args.ckpt.is_absolute() else (root / args.ckpt)
    run_name = args.run_name or datetime.now().strftime("AE_ENHANCE_%Y-%m-%d_%H-%M-%S")

    out_dir = root / "ae_radiograph_filter" / "outputs" / run_name
    out_orig = out_dir / "original"
    out_enh = out_dir / "enhanced"
    out_orig.mkdir(parents=True, exist_ok=True)
    out_enh.mkdir(parents=True, exist_ok=True)

    image_paths = _discover_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    device = _auto_device() if args.device == "auto" else torch.device(args.device)
    model = _load_model(ckpt, device=device)

    rows: List[dict] = []
    infer_times_s: List[float] = []
    t0_all = time.perf_counter()
    for i, path in enumerate(image_paths, start=1):
        orig_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if orig_u8 is None:
            continue
        h0, w0 = orig_u8.shape[:2]
        x = _preprocess(orig_u8, args.input_size).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            recon = model(x)["recon"].detach().cpu().numpy()[0, 0]
        infer_times_s.append(time.perf_counter() - t0)

        recon_orig = cv2.resize(recon, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        orig01 = orig_u8.astype(np.float32) / 255.0
        enhanced01 = np.clip(orig01 - args.fs * recon_orig + args.fa * (orig01 - recon_orig), 0.0, 1.0)
        enhanced_u8 = _to_u8(enhanced01)

        stem = path.stem
        orig_name = f"{i:04d}_{stem}_orig.jpg"
        enh_name = f"{i:04d}_{stem}_enh.jpg"
        cv2.imwrite(str(out_orig / orig_name), orig_u8)
        cv2.imwrite(str(out_enh / enh_name), enhanced_u8)

        rows.append(
            {
                "stem": stem,
                "orig_rel": f"original/{orig_name}",
                "enh_rel": f"enhanced/{enh_name}",
            }
        )
    total_elapsed_s = time.perf_counter() - t0_all

    html_text = _build_html(run_name, rows)
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")

    n_inf = max(1, len(infer_times_s))
    infer_mean_ms = float(np.mean(infer_times_s) * 1000.0) if infer_times_s else 0.0
    infer_p90_ms = float(np.percentile(np.array(infer_times_s), 90) * 1000.0) if infer_times_s else 0.0
    infer_total_s = float(np.sum(infer_times_s)) if infer_times_s else 0.0
    summary = {
        "run_name": run_name,
        "images_dir": str(images_dir),
        "ckpt": str(ckpt),
        "device": str(device),
        "num_images": len(rows),
        "fs": float(args.fs),
        "fa": float(args.fa),
        "inference_mean_ms_per_image": infer_mean_ms,
        "inference_p90_ms_per_image": infer_p90_ms,
        "inference_total_s": infer_total_s,
        "wall_total_s": float(total_elapsed_s),
        "throughput_images_per_s_wall": float(len(rows) / max(total_elapsed_s, 1e-9)),
        "throughput_images_per_s_inference_only": float(len(rows) / max(infer_total_s, 1e-9)),
        "output_dir": str(out_dir),
        "html": str(out_dir / "index.html"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
