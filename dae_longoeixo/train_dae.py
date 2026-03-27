#!/usr/bin/env python3
"""Treino do denoising autoencoder de coordenadas de long-eixo."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
from dae_longoeixo.dae_visuals import capture_epoch_visuals

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class EpochMetrics:
    train_total_loss: float
    train_mse_knocked: float
    train_mse_observed: float
    train_mae_knocked: float
    train_mae_observed: float
    train_mse_curves: float
    train_arc_spacing_loss: float
    train_anchor_rel_loss: float
    train_point_dist_knocked_px: float
    val_total_loss: float
    val_mse_knocked: float
    val_mse_observed: float
    val_mae_knocked: float
    val_mae_observed: float
    val_mse_curves: float
    val_arc_spacing_loss: float
    val_anchor_rel_loss: float
    val_point_dist_knocked_px: float
    lr: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _default_run_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _select_samples_deterministically(samples: list, max_samples: int, seed: int) -> list:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    keep = sorted(idxs[:max_samples])
    return [samples[i] for i in keep]


def _run_one_epoch(
    model: CoordinateDenoisingAutoencoder,
    loader: DataLoader,
    criterion: DaeImputationLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp_enabled: bool,
    point_grid_size: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    mse_knocked = 0.0
    mse_observed = 0.0
    mae_knocked = 0.0
    mae_observed = 0.0
    mse_curves = 0.0
    arc_spacing_loss = 0.0
    anchor_rel_loss = 0.0
    point_knocked_px = 0.0
    count = 0

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        batch = _to_device(batch, device)
        x = batch["x_input"]
        y = batch["y_coords"]
        knocked = batch["knocked_teeth_mask"]
        gt_available = batch["gt_available_teeth_mask"]
        y_curves = batch.get("y_alveolar_curves")
        y_curves_available = batch.get("y_curves_available_mask")

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
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
                loss = out.total

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            dist64 = point_distance_px(pred, y, grid_size=point_grid_size)
            gt_available_points = gt_available.repeat_interleave(2, dim=1)
            knocked_points = knocked.repeat_interleave(2, dim=1)
            valid_knocked_points = knocked_points * gt_available_points
            denom = valid_knocked_points.sum()
            if float(denom.item()) > 0:
                dist_knocked = (dist64 * valid_knocked_points).sum() / (denom + 1e-8)
            else:
                dist_knocked = dist64.new_tensor(0.0)

        bsz = x.shape[0]
        total_loss += float(out.total.detach().item()) * bsz
        mse_knocked += float(out.mse_knocked.detach().item()) * bsz
        mse_observed += float(out.mse_observed.detach().item()) * bsz
        mae_knocked += float(out.mae_knocked.detach().item()) * bsz
        mae_observed += float(out.mae_observed.detach().item()) * bsz
        mse_curves += float(out.mse_curves.detach().item()) * bsz
        arc_spacing_loss += float(out.arc_spacing.detach().item()) * bsz
        anchor_rel_loss += float(out.anchor_relative.detach().item()) * bsz
        point_knocked_px += float(dist_knocked.detach().item()) * bsz
        count += bsz

    if count == 0:
        return {
            "total": 0.0,
            "mse_knocked": 0.0,
            "mse_observed": 0.0,
            "mae_knocked": 0.0,
            "mae_observed": 0.0,
            "mse_curves": 0.0,
            "arc_spacing": 0.0,
            "anchor_rel": 0.0,
            "point_knocked_px": 0.0,
        }

    return {
        "total": total_loss / count,
        "mse_knocked": mse_knocked / count,
        "mse_observed": mse_observed / count,
        "mae_knocked": mae_knocked / count,
        "mae_observed": mae_observed / count,
        "mse_curves": mse_curves / count,
        "arc_spacing": arc_spacing_loss / count,
        "anchor_rel": anchor_rel_loss / count,
        "point_knocked_px": point_knocked_px / count,
    }


def _write_csv_row(csv_path: Path, metrics: EpochMetrics, epoch: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", *asdict(metrics).keys()])
        if is_new:
            writer.writeheader()
        row = {"epoch": epoch}
        row.update(asdict(metrics))
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAE coordenadas de long-eixo")
    parser.add_argument("--config", type=Path, default=Path("dae_longoeixo/dae_train_config.json"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Aplica overrides de smoke test")
    parser.add_argument("--force-regenerate-split", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Limita numero de amostras completas")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    if args.smoke:
        smoke = cfg.get("smoke_test", {})
        cfg["paths"]["output_dir"] = smoke.get("output_dir", cfg["paths"]["output_dir"])
        cfg["training"]["epochs"] = int(smoke.get("epochs", cfg["training"]["epochs"]))
        cfg["training"]["batch_size"] = int(smoke.get("batch_size", cfg["training"]["batch_size"]))
        cfg["training"]["max_train_batches"] = int(smoke.get("max_train_batches", 0))
        cfg["training"]["max_val_batches"] = int(smoke.get("max_val_batches", 0))
        cfg["visuals"]["interval"] = int(smoke.get("visual_interval", cfg["visuals"]["interval"]))
        cfg["data"]["max_samples"] = int(smoke.get("max_samples", cfg.get("data", {}).get("max_samples", 0)))

    seed = int(cfg.get("seed", 123))
    set_seed(seed)

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

    cfg_max_samples = int(cfg.get("data", {}).get("max_samples", 0))
    cli_max_samples = int(args.max_samples)
    limit = cli_max_samples if cli_max_samples > 0 else cfg_max_samples
    samples = _select_samples_deterministically(samples, max_samples=limit, seed=seed)

    split = make_or_load_split(
        samples=samples,
        split_path=split_path,
        seed=int(cfg["split"].get("seed", seed)),
        val_ratio=float(cfg["split"].get("val_ratio", 0.2)),
        test_ratio=float(cfg["split"].get("test_ratio", 0.0)),
        force_regen=args.force_regenerate_split,
    )

    by_stem = {s.stem: s for s in samples}
    train_samples = [by_stem[s] for s in split["train"] if s in by_stem]
    val_samples = [by_stem[s] for s in split["val"] if s in by_stem]
    if train_samples and val_samples:
        pass
    else:
        raise RuntimeError(
            "Split invalido: train/val vazio apos reconciliar com amostras disponiveis. "
            "Use --force-regenerate-split."
        )

    mean_teeth = float(np.mean([s.num_present_teeth for s in samples]))
    print(
        "[DATA] sample_filter={} eligible_samples={} train={} val={} mean_present_teeth={:.2f}".format(
            sample_filter,
            len(samples),
            len(train_samples),
            len(val_samples),
            mean_teeth,
        )
    )

    train_ds = DaeCoordinateDataset(samples=train_samples, preset=preset, stage="train", seed=seed)
    val_ds = DaeCoordinateDataset(samples=val_samples, preset=preset, stage="val", seed=seed)

    batch_size = int(cfg["training"].get("batch_size", preset.get("training", {}).get("batch_size", 64)))
    num_workers = int(cfg["training"].get("num_workers", 0))
    persistent_workers = bool(cfg["training"].get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg["training"].get("prefetch_factor", 2))

    def _make_loader_kwargs(nw: int, shuffle: bool) -> Dict:
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": nw,
            "pin_memory": torch.cuda.is_available(),
        }
        if nw > 0:
            kwargs["persistent_workers"] = persistent_workers
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    train_loader = DataLoader(train_ds, **_make_loader_kwargs(num_workers, shuffle=True))
    val_loader = DataLoader(val_ds, **_make_loader_kwargs(num_workers, shuffle=False))

    print(
        "[DATALOADER] workers={} prefetch_factor={} persistent_workers={} parallel={}".format(
            num_workers,
            prefetch_factor if num_workers > 0 else 0,
            persistent_workers if num_workers > 0 else False,
            "ON" if num_workers > 0 else "OFF",
        )
    )

    run_name = args.run_name or _default_run_name()
    run_dir = output_base_dir / "runs" / run_name
    visuals_dir = run_dir / "train_visuals"
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    cfg["active_run_name"] = run_name
    cfg["run_dir"] = str(run_dir)
    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")

    device_name = str(cfg["training"].get("device", "auto"))
    if device_name == "auto":
        device = _auto_device()
    else:
        device = torch.device(device_name)
    print(f"[DEVICE] using {device}")

    model_cfg = preset.get("model", {})
    hidden_dims = tuple(int(v) for v in model_cfg.get("hidden_dims", [512, 256]))
    latent_dim = int(model_cfg.get("latent_dim", 128))
    dropout = float(model_cfg.get("dropout", 0.1))
    output_activation = str(model_cfg.get("output_activation", "sigmoid"))

    model = CoordinateDenoisingAutoencoder(
        input_dim=train_ds.input_dim,
        output_dim=train_ds.output_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        output_activation=output_activation,
    ).to(device)

    loss_cfg = preset.get("loss", {})
    criterion = DaeImputationLoss(
        w_knocked=float(cfg["training"].get("w_knocked", loss_cfg.get("w_knocked", 0.85))),
        w_observed=float(cfg["training"].get("w_observed", loss_cfg.get("w_observed", 0.15))),
        w_all=float(cfg["training"].get("w_all", loss_cfg.get("w_all", 0.0))),
        w_curves=float(cfg["training"].get("w_curves", loss_cfg.get("w_curves", 0.0))),
        w_arc_spacing=float(cfg["training"].get("w_arc_spacing", loss_cfg.get("w_arc_spacing", 0.0))),
        w_anchor_rel=float(cfg["training"].get("w_anchor_rel", loss_cfg.get("w_anchor_rel", 0.0))),
    )

    lr = float(cfg["training"].get("lr", preset.get("training", {}).get("lr", 3e-4)))
    weight_decay = float(cfg["training"].get("weight_decay", preset.get("training", {}).get("weight_decay", 1e-4)))
    epochs = int(cfg["training"].get("epochs", preset.get("training", {}).get("epochs", 120)))
    patience = int(
        cfg["training"].get(
            "early_stopping_patience",
            preset.get("training", {}).get("early_stopping_patience", 20),
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(cfg["training"].get("lr_min", 1e-6)),
    )

    amp_requested = bool(cfg["training"].get("amp", preset.get("training", {}).get("amp", True)))
    amp_enabled = bool(amp_requested and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    writer = None
    tb_dir = run_dir / "tensorboard"
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))
    else:
        print("[WARN] tensorboard nao instalado; seguiremos com CSV apenas")

    csv_path = run_dir / "metrics.csv"
    ckpt_best = run_dir / "best.ckpt"
    ckpt_last = run_dir / "last.ckpt"
    if csv_path.exists():
        csv_path.unlink()

    run_cfg_out = run_dir / "resolved_config.json"
    run_cfg_out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_base_dir / "latest_run.txt").write_text(run_name, encoding="utf-8")

    point_grid_size = int(cfg.get("training", {}).get("point_grid_size", 256))

    try:
        first_batch = next(iter(train_loader))
    except RuntimeError as e:
        if "torch_shm_manager" in str(e):
            print("[WARN] DataLoader multiprocessing indisponivel neste ambiente; fallback para num_workers=0")
            num_workers = 0
            train_loader = DataLoader(train_ds, **_make_loader_kwargs(num_workers, shuffle=True))
            val_loader = DataLoader(val_ds, **_make_loader_kwargs(num_workers, shuffle=False))
            print("[DATALOADER] workers=0 prefetch_factor=0 persistent_workers=False parallel=OFF")
            first_batch = next(iter(train_loader))
        else:
            raise

    print(
        "[SHAPES] X_input={} X_noisy={} Y={} KO={}".format(
            tuple(first_batch["x_input"].shape),
            tuple(first_batch["x_noisy_coords"].shape),
            tuple(first_batch["y_coords"].shape),
            tuple(first_batch["knocked_teeth_mask"].shape),
        )
    )
    print(
        "[SHAPES] Y_curves={} curves_available={}".format(
            tuple(first_batch["y_alveolar_curves"].shape),
            tuple(first_batch["y_curves_available_mask"].shape),
        )
    )

    best_val = float("inf")
    stale_epochs = 0

    max_train_batches = int(cfg["training"].get("max_train_batches", 0))
    max_val_batches = int(cfg["training"].get("max_val_batches", 0))

    for epoch in range(1, epochs + 1):
        train_stats = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            point_grid_size=point_grid_size,
            max_batches=max_train_batches,
        )
        val_stats = _run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            amp_enabled=amp_enabled,
            point_grid_size=point_grid_size,
            max_batches=max_val_batches,
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        metrics = EpochMetrics(
            train_total_loss=train_stats["total"],
            train_mse_knocked=train_stats["mse_knocked"],
            train_mse_observed=train_stats["mse_observed"],
            train_mae_knocked=train_stats["mae_knocked"],
            train_mae_observed=train_stats["mae_observed"],
            train_mse_curves=train_stats["mse_curves"],
            train_arc_spacing_loss=train_stats["arc_spacing"],
            train_anchor_rel_loss=train_stats["anchor_rel"],
            train_point_dist_knocked_px=train_stats["point_knocked_px"],
            val_total_loss=val_stats["total"],
            val_mse_knocked=val_stats["mse_knocked"],
            val_mse_observed=val_stats["mse_observed"],
            val_mae_knocked=val_stats["mae_knocked"],
            val_mae_observed=val_stats["mae_observed"],
            val_mse_curves=val_stats["mse_curves"],
            val_arc_spacing_loss=val_stats["arc_spacing"],
            val_anchor_rel_loss=val_stats["anchor_rel"],
            val_point_dist_knocked_px=val_stats["point_knocked_px"],
            lr=lr_now,
        )
        _write_csv_row(csv_path, metrics, epoch)

        if writer is not None:
            writer.add_scalar("loss/train_total", metrics.train_total_loss, epoch)
            writer.add_scalar("loss/val_total", metrics.val_total_loss, epoch)
            writer.add_scalar("mse/train_knocked", metrics.train_mse_knocked, epoch)
            writer.add_scalar("mse/val_knocked", metrics.val_mse_knocked, epoch)
            writer.add_scalar("mse/train_observed", metrics.train_mse_observed, epoch)
            writer.add_scalar("mse/val_observed", metrics.val_mse_observed, epoch)
            writer.add_scalar("mae/train_knocked", metrics.train_mae_knocked, epoch)
            writer.add_scalar("mae/val_knocked", metrics.val_mae_knocked, epoch)
            writer.add_scalar("curves/mse_train", metrics.train_mse_curves, epoch)
            writer.add_scalar("curves/mse_val", metrics.val_mse_curves, epoch)
            writer.add_scalar("arc/train_spacing", metrics.train_arc_spacing_loss, epoch)
            writer.add_scalar("arc/val_spacing", metrics.val_arc_spacing_loss, epoch)
            writer.add_scalar("anchor/train_rel", metrics.train_anchor_rel_loss, epoch)
            writer.add_scalar("anchor/val_rel", metrics.val_anchor_rel_loss, epoch)
            writer.add_scalar("point_dist/train_knocked_px", metrics.train_point_dist_knocked_px, epoch)
            writer.add_scalar("point_dist/val_knocked_px", metrics.val_point_dist_knocked_px, epoch)
            writer.add_scalar("lr", metrics.lr, epoch)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "config": cfg,
            "preset": preset,
            "input_dim": train_ds.input_dim,
            "output_dim": train_ds.output_dim,
        }
        torch.save(ckpt, ckpt_last)

        if metrics.val_total_loss < best_val:
            best_val = metrics.val_total_loss
            stale_epochs = 0
            ckpt["best_val"] = best_val
            torch.save(ckpt, ckpt_best)
        else:
            stale_epochs += 1

        print(
            "[EPOCH {:03d}] train_total={:.6f} val_total={:.6f} val_mse_knocked={:.6f} val_mse_curves={:.6f} val_arc={:.6f} val_anchor={:.6f} val_point_knocked_px={:.4f} lr={:.6e}".format(
                epoch,
                metrics.train_total_loss,
                metrics.val_total_loss,
                metrics.val_mse_knocked,
                metrics.val_mse_curves,
                metrics.val_arc_spacing_loss,
                metrics.val_anchor_rel_loss,
                metrics.val_point_dist_knocked_px,
                metrics.lr,
            )
        )

        debug_batch = next(iter(val_loader))
        debug_batch = _to_device(debug_batch, device)
        with torch.no_grad():
            debug_pred = model(debug_batch["x_input"])["coords_pred"]

        capture_epoch_visuals(
            out_dir=visuals_dir,
            epoch=epoch,
            stems=list(debug_batch["stem"]),
            x_noisy_coords=debug_batch["x_noisy_coords"],
            y_true_coords=debug_batch["y_coords"],
            y_pred_coords=debug_pred,
            knocked_teeth_mask=debug_batch["knocked_teeth_mask"],
            interval=int(cfg["visuals"].get("interval", 1)),
            max_samples=int(cfg["visuals"].get("max_samples", 8)),
        )

        if stale_epochs >= patience:
            print(f"[EARLY_STOP] sem melhora por {patience} epocas")
            break

    if writer is not None:
        writer.flush()
        writer.close()

    print(f"[DONE] best.ckpt={ckpt_best}")
    print(f"[DONE] last.ckpt={ckpt_last}")
    print(f"[DONE] metrics.csv={csv_path}")


if __name__ == "__main__":
    main()
