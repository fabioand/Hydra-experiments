#!/usr/bin/env python3
"""Normaliza curvas de rebordo alveolar para tamanho fixo de pontos.

Entrada esperada:
- JSON do endpoint panorogram da RM (com entities e contour).

Saída:
- duas curvas reamostradas com número fixo de pontos:
  - RebAlvSup
  - RebAlvInf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_points_array(contour: Any) -> np.ndarray:
    pts = []
    if not isinstance(contour, list):
        return np.zeros((0, 2), dtype=np.float32)
    for p in contour:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception:
            continue
        pts.append((x, y))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _ensure_left_to_right(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] < 2:
        return points_xy
    # Mantém orientação previsível (x crescente em média).
    return points_xy if points_xy[-1, 0] >= points_xy[0, 0] else points_xy[::-1].copy()


def resample_curve_arc_length(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    """Reamostra curva 2D por comprimento de arco para n_points fixos."""
    if n_points < 2:
        raise ValueError("n_points deve ser >= 2")
    if points_xy.shape[0] == 0:
        return np.zeros((n_points, 2), dtype=np.float32)
    if points_xy.shape[0] == 1:
        return np.repeat(points_xy.astype(np.float32), n_points, axis=0)

    pts = _ensure_left_to_right(points_xy.astype(np.float32))
    diffs = np.diff(pts, axis=0)
    seg_len = np.sqrt(np.sum(diffs * diffs, axis=1))
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])

    if total <= 1e-8:
        return np.repeat(pts[:1], n_points, axis=0)

    targets = np.linspace(0.0, total, n_points, dtype=np.float32)
    x = np.interp(targets, cum, pts[:, 0]).astype(np.float32)
    y = np.interp(targets, cum, pts[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=1)


def _extract_curve_by_name(payload: dict[str, Any], class_name: str) -> np.ndarray:
    entities = payload.get("entities")
    if not isinstance(entities, list):
        return np.zeros((0, 2), dtype=np.float32)
    target = class_name.lower()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        if str(ent.get("class_name", "")).lower() != target:
            continue
        return _to_points_array(ent.get("contour"))
    return np.zeros((0, 2), dtype=np.float32)


def normalize_alveolar_curves(
    payload: dict[str, Any],
    n_points_sup: int = 128,
    n_points_inf: int = 128,
) -> dict[str, np.ndarray]:
    """Extrai e normaliza RebAlvSup/RebAlvInf para tamanho fixo."""
    sup = _extract_curve_by_name(payload, "RebAlvSup")
    inf = _extract_curve_by_name(payload, "RebAlvInf")

    sup_n = resample_curve_arc_length(sup, n_points=n_points_sup)
    inf_n = resample_curve_arc_length(inf, n_points=n_points_inf)
    return {"RebAlvSup": sup_n, "RebAlvInf": inf_n}


def main() -> int:
    parser = argparse.ArgumentParser(description="Normaliza curvas de rebordo alveolar para número fixo de pontos.")
    parser.add_argument("--input-json", required=True, help="JSON do panorogram (payload bruto RM).")
    parser.add_argument("--out-json", required=True, help="JSON de saída com curvas normalizadas.")
    parser.add_argument("--n-sup", type=int, default=128, help="Número de pontos da curva superior.")
    parser.add_argument("--n-inf", type=int, default=128, help="Número de pontos da curva inferior.")
    args = parser.parse_args()

    in_path = Path(args.input_json).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    payload = json.loads(in_path.read_text(encoding="utf-8"))

    curves = normalize_alveolar_curves(payload, n_points_sup=args.n_sup, n_points_inf=args.n_inf)

    out = {
        "source_json": str(in_path),
        "n_sup": int(args.n_sup),
        "n_inf": int(args.n_inf),
        "RebAlvSup": curves["RebAlvSup"].tolist(),
        "RebAlvInf": curves["RebAlvInf"].tolist(),
        "shape_sup": list(curves["RebAlvSup"].shape),
        "shape_inf": list(curves["RebAlvInf"].shape),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_json": str(in_path),
                "out_json": str(out_path),
                "shape_sup": out["shape_sup"],
                "shape_inf": out["shape_inf"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

