#!/usr/bin/env python3
"""Conta cobertura de denticao em anotacoes de long-eixo.

Métricas principais:
- full_32: amostras com os 32 dentes canônicos (2 pontos validos por dente)
- upto_second_molars: amostras com todos os dentes ate segundos molares
  (permite faltar 18/28/38/48)
- histograma de quantidade de dentes presentes (0..32)

Por padrao, varre apenas pares JPG+JSON (interseccao por stem), que e o
modo recomendado para treino.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CANONICAL_32: List[str] = [
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
]

THIRD_MOLARS = {"18", "28", "38", "48"}
REQUIRED_UPTO_SECOND = [t for t in CANONICAL_32 if t not in THIRD_MOLARS]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _points_by_label(json_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    data = _load_json(json_path)
    out: Dict[str, List[Tuple[float, float]]] = {}

    for ann in data:
        label = str(ann.get("label", ""))
        pts = ann.get("pts", [])

        valid: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid.append((float(x), float(y)))

        if label not in out and valid:
            out[label] = valid

    return out


def _is_tooth_present(points_by_label: Dict[str, List[Tuple[float, float]]], tooth: str) -> bool:
    pts = points_by_label.get(tooth)
    return pts is not None and len(pts) >= 2


def _iter_sample_stems(imgs_dir: Path, json_dir: Path, use_json_only: bool) -> Iterable[str]:
    json_stems = {p.stem for p in json_dir.glob("*.json")}
    if use_json_only:
        return sorted(json_stems)

    img_stems = {p.stem for p in imgs_dir.glob("*.jpg")}
    return sorted(img_stems & json_stems)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conta cobertura de denticao em data_longoeixo")
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--json-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument(
        "--use-json-only",
        action="store_true",
        help="Usa todos os JSONs (ignora necessidade de JPG correspondente)",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Arquivo JSON opcional de saida")
    parser.add_argument("--out-hist-csv", type=Path, default=None, help="CSV opcional do histograma")
    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    json_dir = args.json_dir

    if not json_dir.exists():
        raise FileNotFoundError(f"json_dir nao encontrado: {json_dir}")
    if not args.use_json_only and not imgs_dir.exists():
        raise FileNotFoundError(f"imgs_dir nao encontrado: {imgs_dir}")

    stems = list(_iter_sample_stems(imgs_dir=imgs_dir, json_dir=json_dir, use_json_only=args.use_json_only))
    if not stems:
        raise RuntimeError("Nenhuma amostra encontrada para o criterio informado")

    full_32 = 0
    upto_second = 0
    upto_second_missing_1plus_third = 0

    hist_0_32 = {k: 0 for k in range(33)}

    for stem in stems:
        json_path = json_dir / f"{stem}.json"
        pb = _points_by_label(json_path)

        present_count = 0
        for tooth in CANONICAL_32:
            if _is_tooth_present(pb, tooth):
                present_count += 1
        hist_0_32[present_count] += 1

        is_full_32 = present_count == 32
        if is_full_32:
            full_32 += 1

        is_upto_second = all(_is_tooth_present(pb, t) for t in REQUIRED_UPTO_SECOND)
        if is_upto_second:
            upto_second += 1
            missing_third = [t for t in THIRD_MOLARS if not _is_tooth_present(pb, t)]
            if len(missing_third) >= 1:
                upto_second_missing_1plus_third += 1

    summary = {
        "mode": "json_only" if args.use_json_only else "paired_jpg_json",
        "imgs_dir": str(imgs_dir),
        "json_dir": str(json_dir),
        "num_samples": len(stems),
        "full_32": full_32,
        "upto_second_molars": upto_second,
        "upto_second_missing_1plus_third": upto_second_missing_1plus_third,
        "upto_second_with_all_third_present": upto_second - upto_second_missing_1plus_third,
        "hist_num_present_teeth": {str(k): hist_0_32[k] for k in range(32, -1, -1)},
    }

    print(f"mode={summary['mode']}")
    print(f"num_samples={summary['num_samples']}")
    print(f"full_32={summary['full_32']}")
    print(f"upto_second_molars={summary['upto_second_molars']}")
    print(f"upto_second_missing_1plus_third={summary['upto_second_missing_1plus_third']}")
    print(f"upto_second_with_all_third_present={summary['upto_second_with_all_third_present']}")
    print("hist_num_present_teeth (32..0):")
    print(" ".join(f"{k}:{hist_0_32[k]}" for k in range(32, -1, -1)))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_hist_csv is not None:
        args.out_hist_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_hist_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["num_present_teeth", "num_samples"])
            writer.writeheader()
            for k in range(32, -1, -1):
                writer.writerow({"num_present_teeth": k, "num_samples": hist_0_32[k]})


if __name__ == "__main__":
    main()
