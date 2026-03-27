#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from panorama_foundation.models import PanoramicResNetAutoencoder


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(model: PanoramicResNetAutoencoder, ckpt_path: Path) -> None:
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw.get("model_state_dict", raw)
    model.load_state_dict(state, strict=False)


def _discover_images(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted({p.resolve() for p in files})


def _gray01(path: Path, image_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel abrir imagem: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], image_size: int):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.image_paths[idx]
        x = _gray01(path, self.image_size)
        x_t = torch.from_numpy(x).unsqueeze(0)
        return {"x": x_t, "path": str(path), "stem": path.stem}


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out: Dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _panel(inp01: np.ndarray, rec01: np.ndarray, out_path: Path) -> None:
    err = np.clip(np.abs(rec01 - inp01), 0.0, 1.0)
    inp_u8 = (inp01 * 255.0).astype(np.uint8)
    rec_u8 = (rec01 * 255.0).astype(np.uint8)
    err_u8 = (err * 255.0).astype(np.uint8)

    inp_bgr = cv2.cvtColor(inp_u8, cv2.COLOR_GRAY2BGR)
    rec_bgr = cv2.cvtColor(rec_u8, cv2.COLOR_GRAY2BGR)
    err_heat = cv2.applyColorMap(err_u8, cv2.COLORMAP_JET)

    h, w = inp_bgr.shape[:2]
    bar_h = 28
    panel = np.zeros((h + bar_h, w * 3, 3), dtype=np.uint8)
    panel[bar_h:, 0:w] = inp_bgr
    panel[bar_h:, w : 2 * w] = rec_bgr
    panel[bar_h:, 2 * w : 3 * w] = err_heat
    cv2.putText(panel, "Input", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "Reconstruction", (w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "Abs Error", (2 * w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "stem", "path", "mae", "mse", "flagged_percentile"],
        )
        writer.writeheader()
        for i, r in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": i,
                    "stem": r["stem"],
                    "path": r["path"],
                    "mae": f"{r['mae']:.8f}",
                    "mse": f"{r['mse']:.8f}",
                    "flagged_percentile": int(r["flagged"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="AE-based radiograph filter")
    parser.add_argument("--ckpt", type=Path, default=Path("ae_radiograph_filter/models/ae_identity_bestE21.ckpt"))
    parser.add_argument("--images-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=300, help="Numero maximo de radiografias para rodar")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentil de corte para flag")
    parser.add_argument("--top-k-panels", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    ckpt = args.ckpt if args.ckpt.is_absolute() else (root / args.ckpt)
    images_dir = args.images_dir if args.images_dir.is_absolute() else (root / args.images_dir)

    run_name = args.run_name or datetime.now().strftime("AE_FILTER_%Y-%m-%d_%H-%M-%S")
    out_dir = root / "ae_radiograph_filter" / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _discover_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    ds = SimpleImageDataset(image_paths=image_paths, image_size=args.image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = _auto_device() if args.device == "auto" else torch.device(args.device)
    model = PanoramicResNetAutoencoder(backbone="resnet34").to(device)
    _load_checkpoint(model, ckpt)
    model.eval()

    rows: List[Dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            x = batch["x"]
            pred = model(x)
            recon = pred["recon"]

            mae = torch.mean(torch.abs(recon - x), dim=(1, 2, 3)).detach().cpu().numpy()
            mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            r_np = recon.detach().cpu().numpy()

            for i in range(x.shape[0]):
                rows.append(
                    {
                        "stem": batch["stem"][i],
                        "path": batch["path"][i],
                        "mae": float(mae[i]),
                        "mse": float(mse[i]),
                        "x": x_np[i, 0],
                        "recon": r_np[i, 0],
                    }
                )

    rows.sort(key=lambda r: r["mae"], reverse=True)
    mae_vals = np.array([r["mae"] for r in rows], dtype=np.float64)
    thr = float(np.percentile(mae_vals, args.percentile))
    for r in rows:
        r["flagged"] = 1 if r["mae"] >= thr else 0

    flagged = [r for r in rows if r["flagged"] == 1]
    _write_csv(out_dir / "scores_all.csv", rows)
    _write_csv(out_dir / "scores_flagged.csv", flagged)

    top_k = min(args.top_k_panels, len(rows))
    for i in range(top_k):
        r = rows[i]
        _panel(r["x"], r["recon"], out_dir / "panels" / "top_errors" / f"{i+1:03d}_{r['stem']}.png")

    low = list(reversed(rows[-top_k:])) if top_k > 0 else []
    for i, r in enumerate(low, start=1):
        _panel(r["x"], r["recon"], out_dir / "panels" / "low_errors" / f"{i:03d}_{r['stem']}.png")

    summary = {
        "run_name": run_name,
        "ckpt": str(ckpt),
        "images_dir": str(images_dir),
        "num_images_scored": len(rows),
        "device": str(device),
        "percentile": float(args.percentile),
        "mae_threshold": thr,
        "num_flagged": len(flagged),
        "flagged_ratio": len(flagged) / max(1, len(rows)),
        "mae_mean": float(mae_vals.mean()),
        "mae_std": float(mae_vals.std()),
        "mae_p95": float(np.percentile(mae_vals, 95)),
        "mae_p99": float(np.percentile(mae_vals, 99)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

