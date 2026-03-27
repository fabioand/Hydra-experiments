"""Callbacks visuais para treino do denoising autoencoder de coordenadas."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

from hydra_multitask_model import CANONICAL_TEETH_32


def _to_numpy_cpu(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _coords_to_xyxy(coords_128: np.ndarray, size: int = 512) -> List[tuple[int, int, int, int]]:
    coords = np.clip(coords_128.reshape(32, 4), 0.0, 1.0)
    out: List[tuple[int, int, int, int]] = []
    for i in range(32):
        x1 = int(round(coords[i, 0] * (size - 1)))
        y1 = int(round(coords[i, 1] * (size - 1)))
        x2 = int(round(coords[i, 2] * (size - 1)))
        y2 = int(round(coords[i, 3] * (size - 1)))
        out.append((x1, y1, x2, y2))
    return out


def _draw_grid(canvas: np.ndarray, step: int = 64) -> None:
    h, w = canvas.shape[:2]
    color = (36, 43, 56)
    for x in range(0, w, step):
        cv2.line(canvas, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(canvas, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)


def _draw_coords_panel(
    coords_128: np.ndarray,
    knocked_teeth_mask_32: np.ndarray,
    mode: str,
    size: int = 512,
) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (15, 19, 25)
    _draw_grid(canvas)

    lines = _coords_to_xyxy(coords_128, size=size)

    for i, (x1, y1, x2, y2) in enumerate(lines):
        knocked = bool(knocked_teeth_mask_32[i] > 0.5)
        if mode == "input" and knocked:
            continue

        if mode == "input":
            color = (235, 190, 65)
            thickness = 1
        elif mode == "pred":
            color = (64, 164, 255) if knocked == 0 else (0, 219, 198)
            thickness = 2 if knocked else 1
        elif mode == "gt":
            color = (90, 220, 90) if knocked == 0 else (50, 255, 255)
            thickness = 2 if knocked else 1
        else:
            raise ValueError(f"mode invalido: {mode}")

        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, cv2.LINE_AA)

        if knocked:
            tx = int(round((x1 + x2) * 0.5 + 2))
            ty = int(round((y1 + y2) * 0.5 - 2))
            cv2.putText(
                canvas,
                CANONICAL_TEETH_32[i],
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    return canvas


def _save_panel_h(images: Sequence[np.ndarray], titles: Sequence[str], out_path: Path) -> None:
    imgs = list(images)
    tts = list(titles)
    if len(imgs) == len(tts):
        pass
    else:
        raise ValueError("images e titles devem ter mesmo tamanho")

    if imgs:
        pass
    else:
        raise ValueError("images vazio")

    h, w = imgs[0].shape[:2]
    bar_h = 30
    panel = np.zeros((h + bar_h, w * len(imgs), 3), dtype=np.uint8)

    for i, (img, title) in enumerate(zip(imgs, tts)):
        x0 = i * w
        panel[bar_h:, x0 : x0 + w] = img
        cv2.putText(
            panel,
            title,
            (x0 + 8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _read_manifest_rows(out_dir: Path) -> List[Dict]:
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        pass
    else:
        return []

    rows: List[Dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _render_index_html(out_dir: Path) -> None:
    rows = _read_manifest_rows(out_dir)
    rows = sorted(rows, key=lambda r: (int(r.get("epoch", 0)), int(r.get("sample_idx", 0))))

    cards: List[str] = []
    for r in rows:
        stem = html.escape(str(r.get("stem", "")))
        path = html.escape(str(r.get("path", "")))
        knocked = html.escape(str(r.get("knocked_teeth", "")))
        cards.append(
            (
                "<div class='card'>"
                f"<img src='{path}' loading='lazy'/>"
                "<div class='meta'>"
                f"<div>epoch={r.get('epoch')} | sample={r.get('sample_idx')} | stem={stem}</div>"
                f"<div>knocked={knocked}</div>"
                f"<div><a href='{path}' target='_blank'>{path}</a></div>"
                "</div></div>"
            )
        )

    body = f"""<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>DAE Coordinate Visuals</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #0f1115; color: #e8edf2; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; }}
    .card {{ background: #171b22; border: 1px solid #2a3340; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; display: block; }}
    .meta {{ padding: 8px 10px; font-size: 12px; line-height: 1.5; }}
    a {{ color: #7eb7ff; }}
  </style>
</head>
<body>
  <h2>DAE Coordinate Visuals</h2>
  <div>Total artifacts: {len(rows)}</div>
  <div class='grid'>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(body, encoding="utf-8")


def _append_manifest(out_dir: Path, records: List[Dict]) -> None:
    if records:
        pass
    else:
        return

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _render_index_html(out_dir)


def save_imputation_panels(
    out_dir: Path,
    epoch: int,
    stems: Sequence[str],
    x_noisy_coords: torch.Tensor,
    y_true_coords: torch.Tensor,
    y_pred_coords: torch.Tensor,
    knocked_teeth_mask: torch.Tensor,
    max_samples: int = 8,
) -> List[Dict]:
    noisy = _to_numpy_cpu(x_noisy_coords)
    y_true = _to_numpy_cpu(y_true_coords)
    y_pred = _to_numpy_cpu(y_pred_coords)
    ko = _to_numpy_cpu(knocked_teeth_mask)

    n = min(max_samples, noisy.shape[0])
    (out_dir / f"epoch_{epoch:04d}" / "imputation").mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    for i in range(n):
        rel_path = Path(f"epoch_{epoch:04d}/imputation/sample_{i:02d}_{stems[i]}.png")
        p_in = _draw_coords_panel(noisy[i], ko[i], mode="input")
        p_pred = _draw_coords_panel(y_pred[i], ko[i], mode="pred")
        p_gt = _draw_coords_panel(y_true[i], ko[i], mode="gt")

        knocked_list = [CANONICAL_TEETH_32[j] for j in np.where(ko[i] > 0.5)[0]]
        knocked_txt = ",".join(knocked_list) if knocked_list else "none"

        _save_panel_h(
            [p_in, p_pred, p_gt],
            [f"Input noisy ({stems[i]})", f"Pred ({len(knocked_list)} knocked)", "GT complete"],
            out_dir / rel_path,
        )

        records.append(
            {
                "timestamp_utc": ts,
                "epoch": int(epoch),
                "sample_idx": int(i),
                "stem": str(stems[i]),
                "group": "imputation",
                "path": rel_path.as_posix(),
                "knocked_teeth": knocked_txt,
            }
        )

    return records


def capture_epoch_visuals(
    out_dir: Path,
    epoch: int,
    stems: Sequence[str],
    x_noisy_coords: torch.Tensor,
    y_true_coords: torch.Tensor,
    y_pred_coords: torch.Tensor,
    knocked_teeth_mask: torch.Tensor,
    interval: int = 1,
    max_samples: int = 8,
) -> None:
    if interval < 1:
        raise ValueError("interval deve ser >= 1")
    if epoch % interval == 0:
        pass
    else:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = save_imputation_panels(
        out_dir=out_dir,
        epoch=epoch,
        stems=stems,
        x_noisy_coords=x_noisy_coords,
        y_true_coords=y_true_coords,
        y_pred_coords=y_pred_coords,
        knocked_teeth_mask=knocked_teeth_mask,
        max_samples=max_samples,
    )
    _append_manifest(out_dir, rec)


__all__ = [
    "capture_epoch_visuals",
    "save_imputation_panels",
]
