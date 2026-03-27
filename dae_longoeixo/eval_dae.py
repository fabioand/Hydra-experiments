#!/usr/bin/env python3
"""Avaliacao do denoising autoencoder de coordenadas de long-eixo."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from dae_longoeixo.dae_data import (
    DaeCoordinateDataset,
    discover_samples,
    load_json,
    make_or_load_split,
)
from dae_longoeixo.dae_model import CoordinateDenoisingAutoencoder, DaeImputationLoss, point_distance_px
from dae_longoeixo.dae_visuals import save_imputation_panels
from hydra_multitask_model import CANONICAL_TEETH_32


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_run_name(output_base_dir: Path) -> str | None:
    latest_file = output_base_dir / "latest_run.txt"
    if latest_file.exists():
        name = latest_file.read_text(encoding="utf-8").strip()
        if name:
            return name

    runs_dir = output_base_dir / "runs"
    if runs_dir.exists():
        pass
    else:
        return None

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if run_dirs:
        pass
    else:
        return None

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def _safe_mean(a: np.ndarray) -> float:
    return float(np.mean(a)) if a.size > 0 else float("nan")


def _safe_median(a: np.ndarray) -> float:
    return float(np.median(a)) if a.size > 0 else float("nan")


def _safe_p90(a: np.ndarray) -> float:
    return float(np.percentile(a, 90)) if a.size > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="DAE eval")
    parser.add_argument("--config", type=Path, default=Path("dae_longoeixo/dae_train_config.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Nome da run; se omitido usa latest_run.txt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--num-knockout-passes", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Usa output_dir de smoke_test")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))
    if args.smoke:
        smoke = cfg.get("smoke_test", {})
        cfg["paths"]["output_dir"] = smoke.get("output_dir", cfg["paths"]["output_dir"])

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    curves_json_dir_cfg = cfg["paths"].get("curves_json_dir")
    curves_json_dir = _resolve_path(repo_root, curves_json_dir_cfg) if curves_json_dir_cfg else None
    split_path = _resolve_path(repo_root, cfg["paths"]["splits_path"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])
    preset_path = _resolve_path(repo_root, cfg["paths"]["preset_path"])

    preset = load_json(preset_path)

    data_cfg = cfg.get("data", {})
    sample_filter = str(data_cfg.get("sample_filter", "full_32_only"))
    min_teeth_present = int(data_cfg.get("min_teeth_present", 1))
    discovery_progress_interval = int(data_cfg.get("discovery_progress_interval", 500))
    n_curve_points = int(preset.get("model", {}).get("n_curve_points", 128))
    require_curves = bool(data_cfg.get("require_curves_for_samples", False))
    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        sample_filter=sample_filter,
        min_teeth_present=min_teeth_present,
        progress_interval=discovery_progress_interval,
        curves_json_dir=curves_json_dir,
        n_curve_points=n_curve_points,
        require_curves=require_curves,
    )
    if samples:
        pass
    else:
        raise FileNotFoundError(
            "Nenhuma amostra elegivel encontrada para os criterios em "
            f"{imgs_dir} e {json_dir}. sample_filter={sample_filter} min_teeth_present={min_teeth_present}"
        )

    if args.max_samples > 0:
        rng = np.random.default_rng(int(cfg.get("seed", 123)))
        idxs = np.arange(len(samples), dtype=np.int64)
        rng.shuffle(idxs)
        keep = sorted(idxs[: args.max_samples].tolist())
        samples = [samples[i] for i in keep]

    split = make_or_load_split(
        samples=samples,
        split_path=split_path,
        seed=int(cfg["split"].get("seed", cfg.get("seed", 123))),
        val_ratio=float(cfg["split"].get("val_ratio", 0.2)),
        test_ratio=float(cfg["split"].get("test_ratio", 0.0)),
        force_regen=False,
    )

    if args.split == "train":
        stems = split["train"]
    elif args.split == "val":
        stems = split["val"]
    elif args.split == "test":
        if "test" in split:
            stems = split["test"]
        else:
            raise RuntimeError("Split 'test' indisponivel no arquivo de split atual")
    else:
        stems = split["train"] + split["val"] + split.get("test", [])

    by_stem = {s.stem: s for s in samples}
    eval_samples = [by_stem[s] for s in stems if s in by_stem]
    if eval_samples:
        pass
    else:
        raise RuntimeError(
            "Split selecionado sem amostras apos reconciliar com dados disponiveis. "
            "Use --force-regenerate-split no treino para reconstruir o split"
        )

    ds = DaeCoordinateDataset(
        samples=eval_samples,
        preset=preset,
        stage="eval",
        seed=int(cfg.get("seed", 123)),
    )

    batch_size = int(cfg.get("evaluation", {}).get("batch_size", 64))
    num_workers = int(cfg.get("evaluation", {}).get("num_workers", 0))
    persistent_workers = bool(cfg.get("evaluation", {}).get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg.get("evaluation", {}).get("prefetch_factor", 2))

    def _make_loader_kwargs(nw: int) -> Dict:
        kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": nw,
            "pin_memory": torch.cuda.is_available(),
        }
        if nw > 0:
            kwargs["persistent_workers"] = persistent_workers
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    loader = DataLoader(ds, **_make_loader_kwargs(num_workers))
    try:
        _ = next(iter(loader))
    except RuntimeError as e:
        if "torch_shm_manager" in str(e):
            print("[WARN] DataLoader multiprocessing indisponivel neste ambiente; fallback para num_workers=0")
            num_workers = 0
            loader = DataLoader(ds, **_make_loader_kwargs(num_workers))
        else:
            raise

    device_name = str(cfg["training"].get("device", "auto"))
    if device_name == "auto":
        device = _auto_device()
    else:
        device = torch.device(device_name)
    print(f"[DEVICE] using {device}")

    model_cfg = preset.get("model", {})
    model = CoordinateDenoisingAutoencoder(
        input_dim=ds.input_dim,
        output_dim=ds.output_dim,
        hidden_dims=tuple(int(v) for v in model_cfg.get("hidden_dims", [512, 256])),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_activation=str(model_cfg.get("output_activation", "sigmoid")),
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is not None:
        ckpt_path = _resolve_path(repo_root, str(ckpt_path))
        if args.run_name:
            run_dir = output_base_dir / "runs" / args.run_name
            run_name = args.run_name
        else:
            run_dir = ckpt_path.parent
            run_name = run_dir.name
    else:
        run_name = args.run_name or _latest_run_name(output_base_dir)
        if run_name:
            pass
        else:
            raise FileNotFoundError(
                f"Nenhuma run encontrada em {output_base_dir / 'runs'}; "
                "passe --run-name ou --checkpoint"
            )
        run_dir = output_base_dir / "runs" / run_name
        ckpt_path = run_dir / "best.ckpt"

    eval_dir = run_dir / "eval"
    visual_out = eval_dir / "pred_vs_gt_samples"
    eval_dir.mkdir(parents=True, exist_ok=True)
    visual_out.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")
    print(f"[DATA] split={args.split} num_samples={len(eval_samples)}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loss_cfg = preset.get("loss", {})
    criterion = DaeImputationLoss(
        w_knocked=float(cfg["training"].get("w_knocked", loss_cfg.get("w_knocked", 0.85))),
        w_observed=float(cfg["training"].get("w_observed", loss_cfg.get("w_observed", 0.15))),
        w_all=float(cfg["training"].get("w_all", loss_cfg.get("w_all", 0.0))),
        w_curves=float(cfg["training"].get("w_curves", loss_cfg.get("w_curves", 0.0))),
        w_arc_spacing=float(cfg["training"].get("w_arc_spacing", loss_cfg.get("w_arc_spacing", 0.0))),
        w_anchor_rel=float(cfg["training"].get("w_anchor_rel", loss_cfg.get("w_anchor_rel", 0.0))),
    )

    point_grid_size = int(cfg.get("training", {}).get("point_grid_size", 256))
    num_passes = int(args.num_knockout_passes)
    if num_passes <= 0:
        num_passes = int(cfg.get("evaluation", {}).get("num_knockout_passes", 3))

    max_visuals = int(cfg.get("evaluation", {}).get("num_visual_samples", 12))
    visuals_saved = 0

    rows_mse_all: List[np.ndarray] = []
    rows_mse_knocked: List[np.ndarray] = []
    rows_mse_observed: List[np.ndarray] = []
    rows_mae_all: List[np.ndarray] = []
    rows_mae_knocked: List[np.ndarray] = []
    rows_mae_observed: List[np.ndarray] = []
    rows_dist_all: List[np.ndarray] = []
    rows_dist_knocked: List[np.ndarray] = []

    per_tooth_mae_rows: List[np.ndarray] = []
    per_tooth_dist_rows: List[np.ndarray] = []
    per_tooth_knock_rows: List[np.ndarray] = []

    per_sample_rows: List[Dict] = []

    with torch.no_grad():
        for pass_id in range(num_passes):
            ds.set_eval_seed_offset(pass_id * 1_000_003)

            for batch in loader:
                stems_batch = list(batch["stem"])
                batch = _to_device(batch, device)

                x = batch["x_input"]
                y = batch["y_coords"]
                knocked = batch["knocked_teeth_mask"]
                gt_available = batch["gt_available_teeth_mask"]
                x_noisy = batch["x_noisy_coords"]
                y_curves = batch.get("y_alveolar_curves")
                y_curves_available = batch.get("y_curves_available_mask")

                model_out = model(x)
                pred = model_out["coords_pred"]
                pred_curves = model_out.get("curves_pred")
                out = criterion(
                    pred,
                    y,
                    knocked,
                    gt_available_teeth_mask=gt_available,
                    pred_curves=pred_curves,
                    target_curves=y_curves if pred_curves is not None else None,
                    curves_available_mask=y_curves_available if pred_curves is not None else None,
                )

                sq = (pred - y) ** 2
                abs_err = (pred - y).abs()

                available_xy = gt_available.repeat_interleave(4, dim=1)
                knocked_xy = knocked.repeat_interleave(4, dim=1) * available_xy
                observed_xy = (1.0 - knocked.repeat_interleave(4, dim=1)) * available_xy

                mse_all = sq.mean(dim=1)
                mae_all = abs_err.mean(dim=1)

                mse_knocked = (sq * knocked_xy).sum(dim=1) / (knocked_xy.sum(dim=1) + 1e-8)
                mse_observed = (sq * observed_xy).sum(dim=1) / (observed_xy.sum(dim=1) + 1e-8)
                mae_knocked = (abs_err * knocked_xy).sum(dim=1) / (knocked_xy.sum(dim=1) + 1e-8)
                mae_observed = (abs_err * observed_xy).sum(dim=1) / (observed_xy.sum(dim=1) + 1e-8)

                dist64 = point_distance_px(pred, y, grid_size=point_grid_size)
                available_points = gt_available.repeat_interleave(2, dim=1)
                knocked_points = knocked.repeat_interleave(2, dim=1) * available_points
                dist_knocked = (dist64 * knocked_points).sum(dim=1) / (knocked_points.sum(dim=1) + 1e-8)

                per_tooth_mae = abs_err.view(-1, 32, 4).mean(dim=2)
                per_tooth_dist = dist64.view(-1, 32, 2).mean(dim=2)

                rows_mse_all.append(mse_all.cpu().numpy())
                rows_mse_knocked.append(mse_knocked.cpu().numpy())
                rows_mse_observed.append(mse_observed.cpu().numpy())
                rows_mae_all.append(mae_all.cpu().numpy())
                rows_mae_knocked.append(mae_knocked.cpu().numpy())
                rows_mae_observed.append(mae_observed.cpu().numpy())
                rows_dist_all.append(dist64.cpu().numpy().reshape(-1))
                rows_dist_knocked.append((dist64 * knocked_points).cpu().numpy().reshape(-1))

                per_tooth_mae_rows.append(per_tooth_mae.cpu().numpy())
                per_tooth_dist_rows.append(per_tooth_dist.cpu().numpy())
                per_tooth_knock_rows.append(knocked.cpu().numpy() * gt_available.cpu().numpy())

                bsz = x.shape[0]
                for i in range(bsz):
                    per_sample_rows.append(
                        {
                            "stem": stems_batch[i],
                            "pass_id": int(pass_id),
                            "knocked_teeth_count": int(float(knocked[i].sum().item())),
                            "total_loss": float(out.total.detach().item()),
                            "mse_knocked": float(mse_knocked[i].detach().item()),
                            "mae_knocked": float(mae_knocked[i].detach().item()),
                            "point_error_knocked_px": float(dist_knocked[i].detach().item()),
                            "mae_all": float(mae_all[i].detach().item()),
                        }
                    )

                if visuals_saved < max_visuals and pass_id == 0:
                    remaining = max_visuals - visuals_saved
                    rec = save_imputation_panels(
                        out_dir=visual_out,
                        epoch=1,
                        stems=stems_batch,
                        x_noisy_coords=x_noisy,
                        y_true_coords=y,
                        y_pred_coords=pred,
                        knocked_teeth_mask=knocked,
                        max_samples=remaining,
                    )
                    visuals_saved += len(rec)

    mse_all = np.concatenate(rows_mse_all, axis=0)
    mse_knocked = np.concatenate(rows_mse_knocked, axis=0)
    mse_observed = np.concatenate(rows_mse_observed, axis=0)
    mae_all = np.concatenate(rows_mae_all, axis=0)
    mae_knocked = np.concatenate(rows_mae_knocked, axis=0)
    mae_observed = np.concatenate(rows_mae_observed, axis=0)

    dist_all_flat = np.concatenate(rows_dist_all, axis=0)
    dist_knocked_raw = np.concatenate(rows_dist_knocked, axis=0)
    dist_knocked_flat = dist_knocked_raw[dist_knocked_raw > 0]

    per_tooth_mae_all = np.concatenate(per_tooth_mae_rows, axis=0)
    per_tooth_dist_all = np.concatenate(per_tooth_dist_rows, axis=0)
    per_tooth_knock_all = np.concatenate(per_tooth_knock_rows, axis=0)

    per_tooth_rows: List[Dict] = []
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        mae_i_all = per_tooth_mae_all[:, i]
        dist_i_all = per_tooth_dist_all[:, i]
        knock_i = per_tooth_knock_all[:, i] > 0.5

        mae_i_kn = mae_i_all[knock_i]
        dist_i_kn = dist_i_all[knock_i]

        row = {
            "tooth": tooth,
            "mae_all": _safe_mean(mae_i_all),
            "mae_knocked": _safe_mean(mae_i_kn),
            "point_error_mean_px_all": _safe_mean(dist_i_all),
            "point_error_median_px_all": _safe_median(dist_i_all),
            "point_error_mean_px_knocked": _safe_mean(dist_i_kn),
            "point_error_median_px_knocked": _safe_median(dist_i_kn),
            "point_within_5px_rate_knocked": float(np.mean(dist_i_kn <= 5.0)) if dist_i_kn.size > 0 else float("nan"),
            "point_within_10px_rate_knocked": float(np.mean(dist_i_kn <= 10.0)) if dist_i_kn.size > 0 else float("nan"),
            "knocked_count": int(knock_i.sum()),
            "total_count": int(len(knock_i)),
        }
        per_tooth_rows.append(row)

    per_sample_rows = sorted(
        per_sample_rows,
        key=lambda r: (r["point_error_knocked_px"], r["mae_knocked"]),
        reverse=True,
    )

    summary = {
        "split": args.split,
        "num_samples": int(len(eval_samples)),
        "num_knockout_passes": int(num_passes),
        "checkpoint": str(ckpt_path),
        "metrics": {
            "mse_all_mean": _safe_mean(mse_all),
            "mse_knocked_mean": _safe_mean(mse_knocked),
            "mse_observed_mean": _safe_mean(mse_observed),
            "mae_all_mean": _safe_mean(mae_all),
            "mae_knocked_mean": _safe_mean(mae_knocked),
            "mae_observed_mean": _safe_mean(mae_observed),
            "point_error_mean_px_all": _safe_mean(dist_all_flat),
            "point_error_median_px_all": _safe_median(dist_all_flat),
            "point_error_p90_px_all": _safe_p90(dist_all_flat),
            "point_error_mean_px_knocked": _safe_mean(dist_knocked_flat),
            "point_error_median_px_knocked": _safe_median(dist_knocked_flat),
            "point_error_p90_px_knocked": _safe_p90(dist_knocked_flat),
            "point_within_3px_rate_knocked": float(np.mean(dist_knocked_flat <= 3.0)) if dist_knocked_flat.size > 0 else float("nan"),
            "point_within_5px_rate_knocked": float(np.mean(dist_knocked_flat <= 5.0)) if dist_knocked_flat.size > 0 else float("nan"),
            "point_within_10px_rate_knocked": float(np.mean(dist_knocked_flat <= 10.0)) if dist_knocked_flat.size > 0 else float("nan"),
        },
        "artifacts": {
            "metrics_summary": str(eval_dir / "metrics_summary.json"),
            "metrics_per_tooth_csv": str(eval_dir / "metrics_per_tooth.csv"),
            "per_sample_errors_csv": str(eval_dir / "per_sample_errors.csv"),
            "visual_samples_dir": str(visual_out),
        },
    }

    with (eval_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (eval_dir / "metrics_per_tooth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_tooth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_tooth_rows)

    with (eval_dir / "per_sample_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print(f"[EVAL] split={args.split} num_samples={len(eval_samples)} passes={num_passes}")
    print(f"[EVAL] mae_knocked_mean={summary['metrics']['mae_knocked_mean']:.6f}")
    print(f"[EVAL] point_error_median_px_knocked={summary['metrics']['point_error_median_px_knocked']:.4f}")
    print(f"[EVAL] summary={eval_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
