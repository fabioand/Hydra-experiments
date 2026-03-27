#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from panorama_foundation.dataset import (
    PanoramaAutoencoderDataset,
    discover_panoramic_samples,
    load_json,
    make_or_load_split,
)
from panorama_foundation.models import PanoramicResNetAutoencoder
from panorama_foundation.training_callbacks import capture_epoch_visuals

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class EpochMetrics:
    train_loss: float
    train_l1: float
    train_mse: float
    val_loss: float
    val_l1: float
    val_mse: float
    lr: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_run_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def _to_device(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    corruption_mask: torch.Tensor | None,
    w_l1: float,
    w_mse: float,
    focus_corrupted_regions: bool,
    w_corrupted: float,
    w_clean: float,
) -> Dict[str, torch.Tensor]:
    if focus_corrupted_regions and corruption_mask is not None:
        m = torch.clamp(corruption_mask, 0.0, 1.0)
        m_inv = 1.0 - m
        denom_corr = torch.clamp(m.sum(), min=1.0)
        denom_clean = torch.clamp(m_inv.sum(), min=1.0)

        l1_map = torch.abs(recon - target)
        mse_map = (recon - target) ** 2

        l1_corr = (l1_map * m).sum() / denom_corr
        l1_clean = (l1_map * m_inv).sum() / denom_clean
        mse_corr = (mse_map * m).sum() / denom_corr
        mse_clean = (mse_map * m_inv).sum() / denom_clean

        l1 = w_corrupted * l1_corr + w_clean * l1_clean
        mse = w_corrupted * mse_corr + w_clean * mse_clean
    else:
        l1 = F.l1_loss(recon, target)
        mse = F.mse_loss(recon, target)

    total = w_l1 * l1 + w_mse * mse
    return {"total": total, "l1": l1, "mse": mse}


def _run_one_epoch(
    model: PanoramicResNetAutoencoder,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp_enabled: bool,
    w_l1: float,
    w_mse: float,
    focus_corrupted_regions: bool,
    w_corrupted: float,
    w_clean: float,
    max_batches: int = 0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    agg_total = 0.0
    agg_l1 = 0.0
    agg_mse = 0.0
    count = 0

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        batch = _to_device(batch, device)
        x = batch["x"]
        y = batch["y"]
        m = batch.get("corruption_mask")

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(x)
                losses = _recon_loss(
                    out["recon"],
                    y,
                    corruption_mask=m,
                    w_l1=w_l1,
                    w_mse=w_mse,
                    focus_corrupted_regions=focus_corrupted_regions,
                    w_corrupted=w_corrupted,
                    w_clean=w_clean,
                )
                loss = losses["total"]

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bsz = x.shape[0]
        agg_total += float(losses["total"].detach().item()) * bsz
        agg_l1 += float(losses["l1"].detach().item()) * bsz
        agg_mse += float(losses["mse"].detach().item()) * bsz
        count += bsz

    if count == 0:
        return {"total": 0.0, "l1": 0.0, "mse": 0.0}
    return {"total": agg_total / count, "l1": agg_l1 / count, "mse": agg_mse / count}


def _build_loader_kwargs(batch_size: int, num_workers: int, shuffle: bool, prefetch_factor: int) -> Dict:
    out = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        out["persistent_workers"] = True
        out["prefetch_factor"] = prefetch_factor
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Panorama autoencoder training")
    parser.add_argument("--config", type=Path, default=Path("panorama_foundation/configs/ae_local_smoke.json"))
    parser.add_argument("--force-regenerate-split", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override de device (ex.: cuda, cuda:0, mps, cpu). Se omitido usa config/auto.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_json(args.config if args.config.is_absolute() else (repo_root / args.config))

    seed = int(cfg.get("seed", 123))
    set_seed(seed)

    paths = cfg["paths"]
    images_dir = Path(paths["images_dir"])
    if not images_dir.is_absolute():
        images_dir = repo_root / images_dir
    split_path = Path(paths["split_path"])
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    output_dir = Path(paths["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    train_cfg = cfg["training"]
    visuals_cfg = cfg.get("visuals", {})
    run_name = args.run_name or _default_run_name()
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = run_dir / "train_visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    resolved = dict(cfg)
    resolved["run_name"] = run_name
    resolved["run_dir"] = str(run_dir)
    _write_json(run_dir / "resolved_config.json", resolved)

    samples = discover_panoramic_samples(images_dir)
    if not samples:
        raise FileNotFoundError(f"Nenhuma panoramica encontrada em {images_dir}")

    split = make_or_load_split(
        samples=samples,
        split_path=split_path,
        seed=seed,
        val_ratio=float(cfg["split"].get("val_ratio", 0.2)),
        test_ratio=float(cfg["split"].get("test_ratio", 0.0)),
        force_regen=args.force_regenerate_split,
    )

    by_stem = {s.stem: s for s in samples}
    train_samples = [by_stem[s] for s in split["train"] if s in by_stem]
    val_samples = [by_stem[s] for s in split["val"] if s in by_stem]
    if not train_samples or not val_samples:
        raise RuntimeError("Split invalido: train/val vazio.")

    image_size = int(cfg["data"].get("image_size", 256))
    train_ds = PanoramaAutoencoderDataset(
        samples=train_samples,
        image_size_hw=(image_size, image_size),
        augment=bool(cfg["data"].get("augment_train", True)),
        pretext_cfg=cfg.get("pretext", {"mode": "identity"}),
    )
    val_ds = PanoramaAutoencoderDataset(
        samples=val_samples,
        image_size_hw=(image_size, image_size),
        augment=False,
        pretext_cfg=cfg.get("pretext", {"mode": "identity"}),
    )

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    train_loader = DataLoader(train_ds, **_build_loader_kwargs(batch_size, num_workers, True, prefetch_factor))
    val_loader = DataLoader(val_ds, **_build_loader_kwargs(batch_size, num_workers, False, prefetch_factor))

    device_name = args.device or str(train_cfg.get("device", "auto"))
    device = _auto_device() if device_name == "auto" else torch.device(device_name)
    model = PanoramicResNetAutoencoder(
        in_channels=1,
        backbone=str(cfg["model"].get("backbone", "resnet34")),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(train_cfg.get("lr_min", 1e-6)),
    )

    amp_requested = bool(train_cfg.get("amp", True))
    amp_enabled = bool(amp_requested and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    w_l1 = float(train_cfg.get("loss_w_l1", 0.8))
    w_mse = float(train_cfg.get("loss_w_mse", 0.2))
    loss_cfg = cfg.get("loss", {})
    focus_corrupted_regions = bool(loss_cfg.get("focus_corrupted_regions", False))
    w_corrupted = float(loss_cfg.get("w_corrupted", 0.8))
    w_clean = float(loss_cfg.get("w_clean", 0.2))
    max_train_batches = int(train_cfg.get("max_train_batches", 0))
    max_val_batches = int(train_cfg.get("max_val_batches", 0))
    patience = int(train_cfg.get("early_stopping_patience", 10))
    visual_interval = int(visuals_cfg.get("interval", 1))
    visual_max_samples = int(visuals_cfg.get("max_samples", 8))

    best_val = float("inf")
    no_improve = 0
    metrics_csv = run_dir / "metrics.csv"

    for epoch in range(1, epochs + 1):
        train_stats = _run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            w_l1=w_l1,
            w_mse=w_mse,
            focus_corrupted_regions=focus_corrupted_regions,
            w_corrupted=w_corrupted,
            w_clean=w_clean,
            max_batches=max_train_batches,
        )
        val_stats = _run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            amp_enabled=amp_enabled,
            w_l1=w_l1,
            w_mse=w_mse,
            focus_corrupted_regions=focus_corrupted_regions,
            w_corrupted=w_corrupted,
            w_clean=w_clean,
            max_batches=max_val_batches,
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        row = EpochMetrics(
            train_loss=train_stats["total"],
            train_l1=train_stats["l1"],
            train_mse=train_stats["mse"],
            val_loss=val_stats["total"],
            val_l1=val_stats["l1"],
            val_mse=val_stats["mse"],
            lr=lr_now,
        )
        _write_csv_row(metrics_csv, row, epoch)

        if writer is not None:
            writer.add_scalar("loss/train_total", row.train_loss, epoch)
            writer.add_scalar("loss/val_total", row.val_loss, epoch)
            writer.add_scalar("loss/train_l1", row.train_l1, epoch)
            writer.add_scalar("loss/val_l1", row.val_l1, epoch)
            writer.add_scalar("loss/train_mse", row.train_mse, epoch)
            writer.add_scalar("loss/val_mse", row.val_mse, epoch)
            writer.add_scalar("lr", row.lr, epoch)

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val,
            "config": resolved,
        }
        torch.save(ckpt_payload, run_dir / "last.ckpt")

        if row.val_loss < best_val:
            best_val = row.val_loss
            no_improve = 0
            torch.save(ckpt_payload, run_dir / "best.ckpt")
            torch.save(model.encoder.state_dict(), run_dir / "best_encoder.ckpt")
        else:
            no_improve += 1

        scheduler.step()
        print(
            f"[E{epoch:03d}] train={row.train_loss:.6f} val={row.val_loss:.6f} "
            f"l1={row.val_l1:.6f} mse={row.val_mse:.6f} lr={row.lr:.2e}"
        )

        debug_batch = next(iter(train_loader))
        debug_batch = _to_device(debug_batch, device)
        capture_epoch_visuals(
            out_dir=visuals_dir,
            epoch=epoch,
            model=model,
            x_before=debug_batch["x_before"],
            x_after=debug_batch["x"],
            y_target=debug_batch["y"],
            corruption_mask=debug_batch.get("corruption_mask"),
            interval=visual_interval,
            max_samples=visual_max_samples,
        )

        if no_improve >= patience:
            print(f"[EARLY_STOP] patience={patience} atingido em epoch={epoch}")
            break

    if writer is not None:
        writer.close()

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best_val_loss": best_val,
        "best_ckpt": str(run_dir / "best.ckpt"),
        "best_encoder_ckpt": str(run_dir / "best_encoder.ckpt"),
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
