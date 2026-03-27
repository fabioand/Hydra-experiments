#!/usr/bin/env python3
"""Conta trocas FN<->FP em pares de dentes (presenca/ausencia) por exame.

Entrada esperada:
- CSV com colunas `stem`, `fn_teeth`, `fp_teeth`
  (formato gerado por eval_presence_top_errors.py).

Saidas:
- swap_pairs_summary.json
- swap_pairs_by_pair.csv
- swap_pairs_by_exam.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


PREMOLAR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("14", "15"),
    ("24", "25"),
    ("34", "35"),
    ("44", "45"),
)

SECOND_THIRD_MOLAR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("17", "18"),
    ("27", "28"),
    ("37", "38"),
    ("47", "48"),
)


@dataclass(frozen=True)
class PairRule:
    group: str
    a: str
    b: str

    @property
    def pair_key(self) -> str:
        return f"{self.a}<->{self.b}"


def _tokenize_teeth(raw: str) -> Set[str]:
    out: Set[str] = set()
    if not raw:
        return out
    for chunk in raw.replace(",", ";").replace("|", ";").split(";"):
        t = chunk.strip()
        if len(t) == 2 and t.isdigit():
            out.add(t)
    return out


def _build_rules() -> List[PairRule]:
    rules: List[PairRule] = []
    for a, b in PREMOLAR_PAIRS:
        rules.append(PairRule(group="premolar_adjacent", a=a, b=b))
    for a, b in SECOND_THIRD_MOLAR_PAIRS:
        rules.append(PairRule(group="molar_2nd_3rd", a=a, b=b))
    return rules


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []
    expected = {"stem", "fn_teeth", "fp_teeth"}
    missing = [k for k in expected if k not in rows[0]]
    if missing:
        raise ValueError(
            f"CSV sem colunas obrigatorias {missing}. "
            "Esperado: stem, fn_teeth, fp_teeth."
        )
    return rows


def _count_swaps(
    rows: Sequence[Dict[str, str]],
    rules: Sequence[PairRule],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], Dict[str, int]]:
    by_exam: List[Dict[str, object]] = []
    by_pair: Dict[str, Dict[str, object]] = {}
    group_totals: Dict[str, int] = {}

    for rule in rules:
        by_pair[rule.pair_key] = {
            "group": rule.group,
            "pair": rule.pair_key,
            "num_exams_with_swap": 0,
            "num_swaps_a_fn_b_fp": 0,
            "num_swaps_b_fn_a_fp": 0,
        }
        group_totals[rule.group] = 0

    for row in rows:
        stem = str(row.get("stem", "")).strip()
        fn = _tokenize_teeth(str(row.get("fn_teeth", "")))
        fp = _tokenize_teeth(str(row.get("fp_teeth", "")))
        exam_hits: List[str] = []
        exam_groups: Set[str] = set()

        for rule in rules:
            a_fn_b_fp = int(rule.a in fn and rule.b in fp)
            b_fn_a_fp = int(rule.b in fn and rule.a in fp)
            hit = int(a_fn_b_fp or b_fn_a_fp)
            if not hit:
                continue

            exam_hits.append(rule.pair_key)
            exam_groups.add(rule.group)
            pair_row = by_pair[rule.pair_key]
            pair_row["num_exams_with_swap"] = int(pair_row["num_exams_with_swap"]) + 1
            pair_row["num_swaps_a_fn_b_fp"] = int(pair_row["num_swaps_a_fn_b_fp"]) + a_fn_b_fp
            pair_row["num_swaps_b_fn_a_fp"] = int(pair_row["num_swaps_b_fn_a_fp"]) + b_fn_a_fp
            group_totals[rule.group] += 1

        by_exam.append(
            {
                "stem": stem,
                "has_any_swap_pair": int(bool(exam_hits)),
                "num_swap_pairs": len(exam_hits),
                "swap_pairs": ";".join(sorted(exam_hits)),
                "swap_groups": ";".join(sorted(exam_groups)),
            }
        )

    return by_exam, by_pair, group_totals


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisa trocas FN<->FP em pares de dentes.")
    parser.add_argument(
        "--errors-csv",
        type=Path,
        required=True,
        help="CSV com colunas stem/fn_teeth/fp_teeth (ex.: presence_errors_per_sample.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretorio de saida. Default: pasta do CSV de entrada.",
    )
    args = parser.parse_args()

    errors_csv = args.errors_csv.resolve()
    out_dir = args.output_dir.resolve() if args.output_dir is not None else errors_csv.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(errors_csv)
    rules = _build_rules()
    by_exam, by_pair, group_totals = _count_swaps(rows=rows, rules=rules)

    num_exams = len(rows)
    num_exams_with_swap = int(sum(int(r["has_any_swap_pair"]) for r in by_exam))
    summary = {
        "errors_csv": str(errors_csv),
        "num_exams": num_exams,
        "num_exams_with_any_swap_pair": num_exams_with_swap,
        "groups": {
            "premolar_adjacent": {
                "pairs": [f"{a}<->{b}" for a, b in PREMOLAR_PAIRS],
                "num_swap_pairs_found": int(group_totals["premolar_adjacent"]),
            },
            "molar_2nd_3rd": {
                "pairs": [f"{a}<->{b}" for a, b in SECOND_THIRD_MOLAR_PAIRS],
                "num_swap_pairs_found": int(group_totals["molar_2nd_3rd"]),
            },
        },
    }

    summary_json = out_dir / "swap_pairs_summary.json"
    per_pair_csv = out_dir / "swap_pairs_by_pair.csv"
    per_exam_csv = out_dir / "swap_pairs_by_exam.csv"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pair_rows = sorted(
        by_pair.values(),
        key=lambda r: (
            str(r["group"]),
            -int(r["num_exams_with_swap"]),
            str(r["pair"]),
        ),
    )
    _write_csv(
        per_pair_csv,
        rows=pair_rows,
        fieldnames=[
            "group",
            "pair",
            "num_exams_with_swap",
            "num_swaps_a_fn_b_fp",
            "num_swaps_b_fn_a_fp",
        ],
    )
    _write_csv(
        per_exam_csv,
        rows=by_exam,
        fieldnames=[
            "stem",
            "has_any_swap_pair",
            "num_swap_pairs",
            "swap_pairs",
            "swap_groups",
        ],
    )

    print(f"[DONE] summary={summary_json}")
    print(f"[DONE] by_pair={per_pair_csv}")
    print(f"[DONE] by_exam={per_exam_csv}")
    print(
        "[SUMMARY] "
        f"exams={num_exams} exams_with_swap={num_exams_with_swap} "
        f"premolar_swaps={group_totals['premolar_adjacent']} "
        f"molar_2nd_3rd_swaps={group_totals['molar_2nd_3rd']}"
    )


if __name__ == "__main__":
    main()
