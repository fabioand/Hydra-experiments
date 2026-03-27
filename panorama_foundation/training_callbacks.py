"""Callbacks visuais para treino do Panorama Autoencoder.

Padrao alinhado ao Hydra:
- artefatos por epoca em subpastas
- manifest.jsonl
- index.html para inspeção visual
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from hydra_multitask_model import build_attention_maps

VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Panorama AE Training Visuals</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f1217; color: #e9eef5; }
    .wrap { padding: 12px 14px; }
    h1 { margin: 0 0 10px; font-size: 20px; }
    .controls { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 8px; margin-bottom: 12px; }
    label { display: flex; flex-direction: column; font-size: 12px; gap: 4px; }
    select, input { background: #151b22; color: #e9eef5; border: 1px solid #2a3442; border-radius: 6px; padding: 6px 8px; }
    .meta { margin: 6px 0 10px; font-size: 12px; color: #a9b6c7; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; }
    .card { background: #151b22; border: 1px solid #263140; border-radius: 8px; overflow: hidden; }
    .card img { width: 100%; display: block; background: #000; }
    .cap { padding: 8px; font-size: 12px; line-height: 1.4; color: #c9d4e3; }
    .kv { color: #8ea3bd; }
    .empty { padding: 20px; color: #98a9be; }
    .footer { margin-top: 10px; font-size: 12px; color: #98a9be; }
    a { color: #7eb7ff; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Panorama AE Training Visuals</h1>
    <div class="controls">
      <label>Epoch
        <select id="epoch"></select>
      </label>
      <label>Sample
        <select id="sample"></select>
      </label>
      <label>Group
        <select id="group"></select>
      </label>
      <label>Mode
        <select id="mode"></select>
      </label>
      <label>Layer
        <select id="layer"></select>
      </label>
      <label>Search Path
        <input id="search" placeholder="contains..." />
      </label>
    </div>
    <div class="meta" id="meta"></div>
    <div class="grid" id="grid"></div>
    <div class="footer">
      Source: <code>manifest.jsonl</code> in this same folder.
    </div>
  </div>
  <script>
    const els = {
      epoch: document.getElementById('epoch'),
      sample: document.getElementById('sample'),
      group: document.getElementById('group'),
      mode: document.getElementById('mode'),
      layer: document.getElementById('layer'),
      search: document.getElementById('search'),
      meta: document.getElementById('meta'),
      grid: document.getElementById('grid'),
    };

    let rows = [];
    function uniq(arr) { return [...new Set(arr)].sort((a,b)=>String(a).localeCompare(String(b), undefined, {numeric:true})); }
    function fillSelect(el, values, allLabel='All') {
      const current = el.value;
      el.innerHTML = '';
      const optAll = document.createElement('option');
      optAll.value = '';
      optAll.textContent = allLabel;
      el.appendChild(optAll);
      for (const v of values) {
        const o = document.createElement('option');
        o.value = String(v);
        o.textContent = String(v);
        el.appendChild(o);
      }
      if ([...el.options].some(o => o.value === current)) el.value = current;
    }

    function applyFilters() {
      const e = els.epoch.value;
      const s = els.sample.value;
      const g = els.group.value;
      const m = els.mode.value;
      const l = els.layer.value;
      const q = els.search.value.trim().toLowerCase();

      const filtered = rows.filter(r =>
        (!e || String(r.epoch) === e) &&
        (!s || String(r.sample_idx) === s) &&
        (!g || String(r.group||'') === g) &&
        (!m || String(r.mode||'') === m) &&
        (!l || String(r.layer||'') === l) &&
        (!q || String(r.path||'').toLowerCase().includes(q))
      );
      els.meta.textContent = `${filtered.length} / ${rows.length} artifacts`;
      renderGrid(filtered);
    }

    function renderGrid(list) {
      els.grid.innerHTML = '';
      if (!list.length) {
        const d = document.createElement('div');
        d.className = 'empty';
        d.textContent = 'No artifacts match the current filters.';
        els.grid.appendChild(d);
        return;
      }
      for (const r of list) {
        const card = document.createElement('div');
        card.className = 'card';
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.src = r.path;
        img.alt = r.path;
        const cap = document.createElement('div');
        cap.className = 'cap';
        cap.innerHTML = `
          <div><span class="kv">epoch:</span> ${r.epoch} | <span class="kv">sample:</span> ${r.sample_idx}</div>
          <div><span class="kv">group:</span> ${r.group || '-'} | <span class="kv">mode:</span> ${r.mode || '-'} | <span class="kv">layer:</span> ${r.layer || '-'}</div>
          <div><span class="kv">path:</span> <a href="${r.path}" target="_blank">${r.path}</a></div>
        `;
        card.appendChild(img);
        card.appendChild(cap);
        els.grid.appendChild(card);
      }
    }

    async function boot() {
      try {
        const resp = await fetch('manifest.jsonl', { cache: 'no-store' });
        const txt = await resp.text();
        rows = txt.split('\\n').map(s => s.trim()).filter(Boolean).map(s => JSON.parse(s));
      } catch (err) {
        els.meta.textContent = 'Failed to load manifest.jsonl. Serve this folder with HTTP and ensure manifest exists.';
        return;
      }
      fillSelect(els.epoch, uniq(rows.map(r => r.epoch)), 'All epochs');
      fillSelect(els.sample, uniq(rows.map(r => r.sample_idx)), 'All samples');
      fillSelect(els.group, uniq(rows.map(r => r.group).filter(Boolean)), 'All groups');
      fillSelect(els.mode, uniq(rows.map(r => r.mode).filter(Boolean)), 'All modes');
      fillSelect(els.layer, uniq(rows.map(r => r.layer).filter(Boolean)), 'All layers');
      ['epoch','sample','group','mode','layer'].forEach(k => els[k].addEventListener('change', applyFilters));
      els.search.addEventListener('input', applyFilters);
      applyFilters();
    }
    boot();
  </script>
</body>
</html>
"""


def _to_numpy_cpu(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _gray_to_bgr_u8(img01: np.ndarray) -> np.ndarray:
    base = np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def _overlay_scalar_red(img01: np.ndarray, scalar01: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    base_bgr = _gray_to_bgr_u8(img01)
    red = np.zeros_like(base_bgr, dtype=np.uint8)
    red[:, :, 2] = np.clip(scalar01 * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.addWeighted(base_bgr, 1.0, red, alpha, 0.0)


def _overlay_attention(img01: np.ndarray, att01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base_bgr = _gray_to_bgr_u8(img01)
    att_u8 = np.clip(att01 * 255.0, 0.0, 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(att_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heat, alpha, 0.0)


def _save_panel_h(images: Iterable[np.ndarray], titles: Iterable[str], out_path: Path) -> None:
    imgs = list(images)
    tts = list(titles)
    if len(imgs) != len(tts):
        raise ValueError("images e titles devem ter o mesmo tamanho")
    if not imgs:
        raise ValueError("images vazio")
    h, w, _ = imgs[0].shape
    bar_h = 28
    panel = np.zeros((h + bar_h, w * len(imgs), 3), dtype=np.uint8)
    for i, (img, title) in enumerate(zip(imgs, tts)):
        x0 = i * w
        panel[bar_h:, x0 : x0 + w] = img
        cv2.putText(panel, title, (x0 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _append_manifest(out_dir: Path, records: List[Dict]) -> None:
    if not records:
        return
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _ensure_viewer(out_dir: Path) -> None:
    idx = out_dir / "index.html"
    if not idx.exists():
        idx.write_text(VIEWER_HTML, encoding="utf-8")


def save_augmentation_panels(
    out_dir: Path,
    epoch: int,
    x_before: torch.Tensor,
    x_after: torch.Tensor,
    corruption_mask: torch.Tensor | None = None,
    max_samples: int = 8,
) -> List[Dict]:
    xb = _to_numpy_cpu(x_before)
    xa = _to_numpy_cpu(x_after)
    cm = _to_numpy_cpu(corruption_mask) if corruption_mask is not None else None
    n = min(max_samples, xb.shape[0])
    ts = datetime.now(timezone.utc).isoformat()
    records: List[Dict] = []
    (out_dir / f"epoch_{epoch:04d}" / "augmentation").mkdir(parents=True, exist_ok=True)

    for i in range(n):
        diff = np.abs(xa[i, 0] - xb[i, 0])
        before_img = _gray_to_bgr_u8(xb[i, 0])
        after_img = _gray_to_bgr_u8(xa[i, 0])
        diff_overlay = _overlay_scalar_red(xa[i, 0], np.clip(diff, 0.0, 1.0), alpha=0.6)
        mask_overlay = _overlay_scalar_red(xa[i, 0], np.clip(cm[i, 0], 0.0, 1.0), alpha=0.6) if cm is not None else None
        rel_path = Path(f"epoch_{epoch:04d}/augmentation/sample_{i:02d}.png")
        if mask_overlay is not None:
            _save_panel_h(
                [before_img, after_img, diff_overlay, mask_overlay],
                ["Before Aug", "After Aug", "Abs Diff", "Corruption Mask"],
                out_dir / rel_path,
            )
        else:
            _save_panel_h(
                [before_img, after_img, diff_overlay],
                ["Before Aug", "After Aug", "Abs Diff (mask-like)"],
                out_dir / rel_path,
            )
        records.append(
            {
                "timestamp_utc": ts,
                "epoch": int(epoch),
                "sample_idx": int(i),
                "group": "augmentation",
                "view": "before_after_diff",
                "mode": None,
                "layer": None,
                "path": rel_path.as_posix(),
            }
        )
    return records


def save_reconstruction_panels(
    out_dir: Path,
    epoch: int,
    x_after: torch.Tensor,
    y_target: torch.Tensor,
    recon: torch.Tensor,
    max_samples: int = 8,
) -> List[Dict]:
    xa = _to_numpy_cpu(x_after)
    yt = _to_numpy_cpu(y_target)
    rc = _to_numpy_cpu(recon)
    n = min(max_samples, xa.shape[0])
    ts = datetime.now(timezone.utc).isoformat()
    records: List[Dict] = []
    (out_dir / f"epoch_{epoch:04d}" / "reconstruction").mkdir(parents=True, exist_ok=True)

    for i in range(n):
        err = np.abs(rc[i, 0] - yt[i, 0])
        inp = _gray_to_bgr_u8(xa[i, 0])
        gt = _gray_to_bgr_u8(yt[i, 0])
        rc_img = _gray_to_bgr_u8(rc[i, 0])
        err_overlay = _overlay_scalar_red(yt[i, 0], np.clip(err, 0.0, 1.0), alpha=0.6)
        rel_path = Path(f"epoch_{epoch:04d}/reconstruction/sample_{i:02d}.png")

        _save_panel_h(
            [inp, gt, rc_img, err_overlay],
            ["Input (After Aug)", "Target (Clean)", "Reconstruction", "Abs Error Overlay"],
            out_dir / rel_path,
        )
        records.append(
            {
                "timestamp_utc": ts,
                "epoch": int(epoch),
                "sample_idx": int(i),
                "group": "reconstruction",
                "view": "input_target_recon_error",
                "mode": None,
                "layer": None,
                "path": rel_path.as_posix(),
            }
        )
    return records


def save_attention_panels(
    out_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    x_after: torch.Tensor,
    y_target: torch.Tensor,
    max_samples: int = 8,
    att_alpha: float = 0.5,
    agg_modes: Tuple[str, ...] = ("mean", "max"),
) -> List[Dict]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(x_after, return_intermediates=True)
    if was_training:
        model.train()

    intermediates = pred.get("intermediates")
    recon = pred.get("recon")
    if intermediates is None:
        raise RuntimeError("Modelo nao retornou intermediates no modo return_intermediates=True.")
    if recon is None:
        raise RuntimeError("Modelo nao retornou recon no forward.")

    xa = _to_numpy_cpu(x_after)
    yt = _to_numpy_cpu(y_target)
    rc = _to_numpy_cpu(recon)
    h, w = xa.shape[-2], xa.shape[-1]
    n = min(max_samples, xa.shape[0])
    ts = datetime.now(timezone.utc).isoformat()
    records: List[Dict] = []
    (out_dir / f"epoch_{epoch:04d}" / "attention").mkdir(parents=True, exist_ok=True)

    for mode in agg_modes:
        att_maps = build_attention_maps(intermediates, mode=mode, normalize=True, out_size=(h, w))
        for i in range(n):
            err = np.clip(np.abs(rc[i, 0] - yt[i, 0]), 0.0, 1.0)
            base_overlay = _overlay_scalar_red(yt[i, 0], err, alpha=0.55)
            for layer_name, att in att_maps.items():
                a = _to_numpy_cpu(att)[i, 0]
                att_overlay = _overlay_attention(xa[i, 0], a, alpha=att_alpha)
                rel_path = Path(f"epoch_{epoch:04d}/attention/sample_{i:02d}_{mode}_{layer_name}.png")
                _save_panel_h(
                    [base_overlay, att_overlay],
                    ["Recon Error Overlay", f"Attention {mode} - {layer_name}"],
                    out_dir / rel_path,
                )
                records.append(
                    {
                        "timestamp_utc": ts,
                        "epoch": int(epoch),
                        "sample_idx": int(i),
                        "group": "attention",
                        "view": "error_vs_attention",
                        "mode": str(mode),
                        "layer": str(layer_name),
                        "path": rel_path.as_posix(),
                    }
                )
    return records


def capture_epoch_visuals(
    out_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    x_before: torch.Tensor,
    x_after: torch.Tensor,
    y_target: torch.Tensor,
    corruption_mask: torch.Tensor | None = None,
    interval: int = 1,
    max_samples: int = 8,
) -> None:
    if interval < 1:
        raise ValueError("interval deve ser >= 1")
    if epoch % interval != 0:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_viewer(out_dir)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(x_after)
    if was_training:
        model.train()

    recon = pred["recon"]
    rec_aug = save_augmentation_panels(
        out_dir=out_dir,
        epoch=epoch,
        x_before=x_before,
        x_after=x_after,
        corruption_mask=corruption_mask,
        max_samples=max_samples,
    )
    rec_recon = save_reconstruction_panels(
        out_dir=out_dir,
        epoch=epoch,
        x_after=x_after,
        y_target=y_target,
        recon=recon,
        max_samples=max_samples,
    )
    rec_att = save_attention_panels(
        out_dir=out_dir,
        epoch=epoch,
        model=model,
        x_after=x_after,
        y_target=y_target,
        max_samples=max_samples,
    )
    _append_manifest(out_dir, rec_aug + rec_recon + rec_att)


__all__ = [
    "capture_epoch_visuals",
    "save_augmentation_panels",
    "save_reconstruction_panels",
    "save_attention_panels",
]
