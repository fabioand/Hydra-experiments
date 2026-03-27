#!/usr/bin/env python3
"""Gera split reprodutivel train/val/test a partir de JPG+JSON."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val/test split")
    parser.add_argument("--imgs-dir", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise SystemExit(f"Ratios devem somar 1.0 (recebido {total_ratio})")
    if min(args.train_ratio, args.val_ratio, args.test_ratio) <= 0:
        raise SystemExit("Ratios devem ser > 0")

    img_stems = {p.stem for p in args.imgs_dir.glob("*.jpg")}
    json_stems = {p.stem for p in args.json_dir.glob("*.json")}
    stems = sorted(img_stems & json_stems)
    if not stems:
        raise SystemExit("Nenhum par JPG+JSON encontrado")

    rng = random.Random(args.seed)
    rng.shuffle(stems)

    n_total = len(stems)
    n_train = int(round(n_total * args.train_ratio))
    n_val = int(round(n_total * args.val_ratio))
    n_test = n_total - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise SystemExit(
            f"Split invalido para n_total={n_total}: train={n_train}, val={n_val}, test={n_test}"
        )

    train = stems[:n_train]
    val = stems[n_train : n_train + n_val]
    test = stems[n_train + n_val :]

    payload = {
        "seed": args.seed,
        "num_samples": n_total,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train": train,
        "val": val,
        "test": test,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[SPLIT] out={args.out}")
    print(f"[SPLIT] total={n_total} train={len(train)} val={len(val)} test={len(test)} seed={args.seed}")


if __name__ == "__main__":
    main()
