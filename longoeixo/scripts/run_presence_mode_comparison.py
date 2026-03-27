#!/usr/bin/env python3
"""Executa comparativo 2x3 de presença (head vs heatmap fixo vs heatmap calibrado).

Matriz padrão:
- Runs: A e B (duas redes/checkpoints)
- Modos de presença:
  1) logits
  2) heatmap_fixed (threshold fixo)
  3) heatmap_calibrated (calibra no val e aplica no test)

Gera um CSV consolidado com métricas de presença e geometria operacional.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _run(cmd: List[str], cwd: Path) -> None:
    print("[CMP] $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_summary(eval_dir: Path) -> Dict:
    p = eval_dir / "metrics_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"metrics_summary.json nao encontrado: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_row(summary: Dict, base_run_name: str, eval_run_name: str, mode: str) -> Dict:
    presence = summary["presence"]
    loc_op = summary["localization_operational_pred_presence"]
    combined = summary["combined"]
    return {
        "base_run_name": base_run_name,
        "eval_run_name": eval_run_name,
        "mode": mode,
        "presence_source": presence.get("source"),
        "presence_f1_macro": presence.get("f1_macro"),
        "presence_precision_macro": presence.get("precision_macro"),
        "presence_recall_macro": presence.get("recall_macro"),
        "presence_auc_macro": presence.get("auc_macro"),
        "presence_accuracy_macro": presence.get("accuracy_macro"),
        "point_error_median_px_operational": loc_op.get("point_error_median_px"),
        "point_error_mean_px_operational": loc_op.get("point_error_mean_px"),
        "point_error_p90_px_operational": loc_op.get("point_error_p90_px"),
        "point_within_5px_rate_operational": loc_op.get("point_within_5px_rate"),
        "valid_point_rate_when_pred_presence_pos_global": combined.get(
            "valid_point_rate_when_pred_presence_pos_global"
        ),
        "false_point_rate_gt_absent_global": combined.get("false_point_rate_gt_absent_global"),
        "num_samples": summary.get("num_samples"),
        "split": summary.get("split"),
    }


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "base_run_name",
        "eval_run_name",
        "mode",
        "presence_source",
        "presence_f1_macro",
        "presence_precision_macro",
        "presence_recall_macro",
        "presence_auc_macro",
        "presence_accuracy_macro",
        "point_error_median_px_operational",
        "point_error_mean_px_operational",
        "point_error_p90_px_operational",
        "point_within_5px_rate_operational",
        "valid_point_rate_when_pred_presence_pos_global",
        "false_point_rate_gt_absent_global",
        "num_samples",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _run_for_model(
    repo_root: Path,
    python_bin: Path,
    eval_script: Path,
    experiment_root: Path,
    cfg_path: Path,
    base_run_name: str,
    fixed_threshold: float,
    cal_min: float,
    cal_max: float,
    cal_step: float,
    include_heatmap_composite: bool,
) -> List[Dict]:
    rows: List[Dict] = []
    ckpt = experiment_root / "runs" / base_run_name / "best.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {ckpt}")

    # 1) logits
    eval_run_logits = f"{base_run_name}__cmp_logits"
    _run(
        [
            str(python_bin),
            str(eval_script),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--run-name",
            eval_run_logits,
            "--split",
            "test",
            "--presence-source",
            "logits",
        ],
        cwd=repo_root,
    )
    sum_logits = _load_summary(experiment_root / "runs" / eval_run_logits / "eval")
    rows.append(_extract_row(sum_logits, base_run_name=base_run_name, eval_run_name=eval_run_logits, mode="logits"))

    # 2) heatmap + threshold fixo
    eval_run_hm_fixed = f"{base_run_name}__cmp_heatmap_fixed"
    _run(
        [
            str(python_bin),
            str(eval_script),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--run-name",
            eval_run_hm_fixed,
            "--split",
            "test",
            "--presence-source",
            "heatmap",
            "--presence-threshold",
            str(fixed_threshold),
        ],
        cwd=repo_root,
    )
    sum_hm_fixed = _load_summary(experiment_root / "runs" / eval_run_hm_fixed / "eval")
    rows.append(
        _extract_row(
            sum_hm_fixed,
            base_run_name=base_run_name,
            eval_run_name=eval_run_hm_fixed,
            mode=f"heatmap_fixed_{fixed_threshold}",
        )
    )

    # 3) heatmap + threshold calibrado no val
    eval_run_hm_cal = f"{base_run_name}__cmp_heatmap_cal"
    _run(
        [
            str(python_bin),
            str(eval_script),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--run-name",
            eval_run_hm_cal,
            "--split",
            "val",
            "--presence-source",
            "heatmap",
            "--presence-threshold",
            str(fixed_threshold),
            "--calibrate-presence-thresholds",
            "--calibration-threshold-min",
            str(cal_min),
            "--calibration-threshold-max",
            str(cal_max),
            "--calibration-threshold-step",
            str(cal_step),
        ],
        cwd=repo_root,
    )
    calibrated_json = (
        experiment_root / "runs" / eval_run_hm_cal / "eval" / "presence_thresholds_heatmap_calibrated_val.json"
    )
    if not calibrated_json.exists():
        raise FileNotFoundError(f"JSON calibrado nao encontrado: {calibrated_json}")

    _run(
        [
            str(python_bin),
            str(eval_script),
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--run-name",
            eval_run_hm_cal,
            "--split",
            "test",
            "--presence-source",
            "heatmap",
            "--presence-thresholds-json",
            str(calibrated_json),
        ],
        cwd=repo_root,
    )
    sum_hm_cal = _load_summary(experiment_root / "runs" / eval_run_hm_cal / "eval")
    rows.append(
        _extract_row(
            sum_hm_cal,
            base_run_name=base_run_name,
            eval_run_name=eval_run_hm_cal,
            mode="heatmap_calibrated_val",
        )
    )

    if include_heatmap_composite:
        # 4) heatmap_composite + threshold fixo
        eval_run_hm_comp_fixed = f"{base_run_name}__cmp_heatmap_comp_fixed"
        _run(
            [
                str(python_bin),
                str(eval_script),
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(ckpt),
                "--run-name",
                eval_run_hm_comp_fixed,
                "--split",
                "test",
                "--presence-source",
                "heatmap_composite",
                "--presence-threshold",
                str(fixed_threshold),
            ],
            cwd=repo_root,
        )
        sum_hm_comp_fixed = _load_summary(experiment_root / "runs" / eval_run_hm_comp_fixed / "eval")
        rows.append(
            _extract_row(
                sum_hm_comp_fixed,
                base_run_name=base_run_name,
                eval_run_name=eval_run_hm_comp_fixed,
                mode=f"heatmap_composite_fixed_{fixed_threshold}",
            )
        )

        # 5) heatmap_composite + threshold calibrado no val
        eval_run_hm_comp_cal = f"{base_run_name}__cmp_heatmap_comp_cal"
        _run(
            [
                str(python_bin),
                str(eval_script),
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(ckpt),
                "--run-name",
                eval_run_hm_comp_cal,
                "--split",
                "val",
                "--presence-source",
                "heatmap_composite",
                "--presence-threshold",
                str(fixed_threshold),
                "--calibrate-presence-thresholds",
                "--calibration-threshold-min",
                str(cal_min),
                "--calibration-threshold-max",
                str(cal_max),
                "--calibration-threshold-step",
                str(cal_step),
            ],
            cwd=repo_root,
        )
        calibrated_comp_json = (
            experiment_root / "runs" / eval_run_hm_comp_cal / "eval" / "presence_thresholds_heatmap_composite_calibrated_val.json"
        )
        if not calibrated_comp_json.exists():
            raise FileNotFoundError(f"JSON calibrado composto nao encontrado: {calibrated_comp_json}")

        _run(
            [
                str(python_bin),
                str(eval_script),
                "--config",
                str(cfg_path),
                "--checkpoint",
                str(ckpt),
                "--run-name",
                eval_run_hm_comp_cal,
                "--split",
                "test",
                "--presence-source",
                "heatmap_composite",
                "--presence-thresholds-json",
                str(calibrated_comp_json),
            ],
            cwd=repo_root,
        )
        sum_hm_comp_cal = _load_summary(experiment_root / "runs" / eval_run_hm_comp_cal / "eval")
        rows.append(
            _extract_row(
                sum_hm_comp_cal,
                base_run_name=base_run_name,
                eval_run_name=eval_run_hm_comp_cal,
                mode="heatmap_composite_calibrated_val",
            )
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara modos de presença em duas runs.")
    parser.add_argument("--run-a-name", type=str, required=True)
    parser.add_argument("--run-a-config", type=Path, required=True)
    parser.add_argument("--run-b-name", type=str, required=True)
    parser.add_argument("--run-b-config", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("longoeixo/experiments/hydra_unet_multitask"),
    )
    parser.add_argument("--python-bin", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--eval-script", type=Path, default=Path("eval.py"))
    parser.add_argument("--fixed-threshold", type=float, default=0.1)
    parser.add_argument("--calibration-threshold-min", type=float, default=0.01)
    parser.add_argument("--calibration-threshold-max", type=float, default=0.99)
    parser.add_argument("--calibration-threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--include-heatmap-composite",
        action="store_true",
        help="Inclui modos extras com score composto de heatmap (fixo + calibrado).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("longoeixo/experiments/hydra_unet_multitask/comparisons/presence_mode_comparison.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("longoeixo/experiments/hydra_unet_multitask/comparisons/presence_mode_comparison.json"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python_bin = _resolve(repo_root, str(args.python_bin))
    eval_script = _resolve(repo_root, str(args.eval_script))
    exp_root = _resolve(repo_root, str(args.experiment_root))
    run_a_cfg = _resolve(repo_root, str(args.run_a_config))
    run_b_cfg = _resolve(repo_root, str(args.run_b_config))
    out_csv = _resolve(repo_root, str(args.out_csv))
    out_json = _resolve(repo_root, str(args.out_json))

    for p in (python_bin, eval_script, exp_root, run_a_cfg, run_b_cfg):
        if not p.exists():
            raise FileNotFoundError(f"Path nao encontrado: {p}")

    all_rows: List[Dict] = []
    all_rows.extend(
        _run_for_model(
            repo_root=repo_root,
            python_bin=python_bin,
            eval_script=eval_script,
            experiment_root=exp_root,
            cfg_path=run_a_cfg,
            base_run_name=args.run_a_name,
            fixed_threshold=float(args.fixed_threshold),
            cal_min=float(args.calibration_threshold_min),
            cal_max=float(args.calibration_threshold_max),
            cal_step=float(args.calibration_threshold_step),
            include_heatmap_composite=bool(args.include_heatmap_composite),
        )
    )
    all_rows.extend(
        _run_for_model(
            repo_root=repo_root,
            python_bin=python_bin,
            eval_script=eval_script,
            experiment_root=exp_root,
            cfg_path=run_b_cfg,
            base_run_name=args.run_b_name,
            fixed_threshold=float(args.fixed_threshold),
            cal_min=float(args.calibration_threshold_min),
            cal_max=float(args.calibration_threshold_max),
            cal_step=float(args.calibration_threshold_step),
            include_heatmap_composite=bool(args.include_heatmap_composite),
        )
    )

    _write_csv(out_csv, all_rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CMP] rows={len(all_rows)}")
    print(f"[CMP] csv={out_csv}")
    print(f"[CMP] json={out_json}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[CMP][ERRO] comando falhou com exit={e.returncode}: {e.cmd}", file=sys.stderr)
        raise
