#!/usr/bin/env python3
"""Treino local end-to-end do Hydra U-Net MultiTask."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from hydra_data import HydraTeethDataset, discover_samples, load_json, make_or_load_split
from hydra_multitask_model import HydraMultiTaskLoss, HydraUNetMultiTask
from hydra_training_callbacks import capture_epoch_visuals
from longoeixo.scripts.roi_lateral_shared_config import (
    CENTER_TEETH,
    CENTER_RIGHT_TO_LEFT,
    LATERAL_RIGHT_TEETH,
    LEFT_TO_RIGHT,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class EpochMetrics:
    train_total_loss: float
    train_heatmap_loss: float
    train_presence_loss: float
    val_total_loss: float
    val_heatmap_loss: float
    val_presence_loss: float
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


def _to_device(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
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


def _run_one_epoch(
    model: HydraUNetMultiTask,
    loader: DataLoader,
    criterion: HydraMultiTaskLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp_enabled: bool,
    max_batches: int = 0,
    grad_clip_norm: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    heatmap_loss = 0.0
    presence_loss = 0.0
    count = 0

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break

        batch = _to_device(batch, device)
        x = batch["x"]
        y_heatmap = batch["y_heatmap"]
        y_presence = batch["y_presence"]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(x)
                out = criterion(pred, y_heatmap, y_presence)
                loss = out.total

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

        bsz = x.shape[0]
        total_loss += float(out.total.detach().item()) * bsz
        heatmap_loss += float(out.heatmap_total.detach().item()) * bsz
        presence_loss += float(out.presence_bce.detach().item()) * bsz
        count += bsz

    if count == 0:
        return {"total": 0.0, "heatmap": 0.0, "presence": 0.0}

    return {
        "total": total_loss / count,
        "heatmap": heatmap_loss / count,
        "presence": presence_loss / count,
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
    parser = argparse.ArgumentParser(description="Hydra U-Net MultiTask training")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--smoke", action="store_true", help="Aplica overrides de smoke test")
    parser.add_argument("--force-regenerate-split", action="store_true")
    parser.add_argument("--run-name", type=str, default=None, help="Nome da run; se omitido usa timestamp")
    parser.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Checkpoint opcional para inicializar pesos do modelo (warm start/fine-tune).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    if args.smoke:
        smoke = cfg.get("smoke_test", {})
        if "masks_dir" in cfg.get("paths", {}):
            cfg["paths"]["masks_dir"] = smoke.get("masks_dir", cfg["paths"]["masks_dir"])
        cfg["paths"]["output_dir"] = smoke.get("output_dir", cfg["paths"]["output_dir"])
        cfg["training"]["epochs"] = int(smoke.get("epochs", cfg["training"]["epochs"]))
        cfg["training"]["batch_size"] = int(smoke.get("batch_size", cfg["training"]["batch_size"]))
        cfg["training"]["max_train_batches"] = int(smoke.get("max_train_batches", 0))
        cfg["training"]["max_val_batches"] = int(smoke.get("max_val_batches", 0))
        cfg["visuals"]["interval"] = int(smoke.get("visual_interval", cfg["visuals"]["interval"]))

    seed = int(cfg.get("seed", 123))
    set_seed(seed)

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    masks_dir_cfg = cfg["paths"].get("masks_dir")
    masks_dir = _resolve_path(repo_root, masks_dir_cfg) if masks_dir_cfg else None
    split_path = _resolve_path(repo_root, cfg["paths"]["splits_path"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])
    preset_path = _resolve_path(repo_root, cfg["paths"]["preset_path"])

    run_name = args.run_name or _default_run_name()
    run_dir = output_base_dir / "runs" / run_name
    visuals_dir = run_dir / "train_visuals"
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    cfg["active_run_name"] = run_name
    cfg["run_dir"] = str(run_dir)
    cfg["visuals"]["resolved_out_dir"] = str(visuals_dir)
    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")

    preset = load_json(preset_path)

    source_mode = str(cfg.get("data", {}).get("source_mode", "on_the_fly"))
    samples = discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        masks_dir=masks_dir,
        source_mode=source_mode,
    )
    if not samples:
        if source_mode == "on_the_fly":
            raise FileNotFoundError(f"Nenhum par JPG+JSON encontrado em {imgs_dir} e {json_dir}")
        raise FileNotFoundError(f"Nenhum triplo JPG+JSON+NPY encontrado em {imgs_dir}, {json_dir} e {masks_dir}")

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
    if not train_samples or not val_samples:
        raise RuntimeError(
            "Split invalido: train/val vazio apos reconciliar com amostras disponiveis. "
            "Use --force-regenerate-split."
        )
    print(f"[DATA] source_mode={source_mode} train={len(train_samples)} val={len(val_samples)}")

    data_cfg = cfg.get("data", {})
    roi_profile = str(data_cfg.get("roi_profile", "full64"))
    if roi_profile not in {"full64", "center24", "center24_shared_flip", "lateral_shared20"}:
        raise ValueError(f"data.roi_profile invalido: {roi_profile}")

    if roi_profile == "full64":
        train_ds = HydraTeethDataset(
            samples=train_samples,
            preset=preset,
            augment=True,
            source_mode=source_mode,
            seed=seed,
        )
        val_ds = HydraTeethDataset(
            samples=val_samples,
            preset=preset,
            augment=False,
            source_mode=source_mode,
            seed=seed,
        )
        heatmap_out_channels = train_ds.heatmap_channels
        presence_out_channels = train_ds.presence_channels

    elif roi_profile == "center24":
        train_ds = HydraTeethDataset(
            samples=train_samples,
            preset=preset,
            augment=True,
            source_mode=source_mode,
            seed=seed,
            teeth_subset=CENTER_TEETH,
            window_name="CENTER",
            flip_horizontal=False,
        )
        val_ds = HydraTeethDataset(
            samples=val_samples,
            preset=preset,
            augment=False,
            source_mode=source_mode,
            seed=seed,
            teeth_subset=CENTER_TEETH,
            window_name="CENTER",
            flip_horizontal=False,
        )
        heatmap_out_channels = train_ds.heatmap_channels
        presence_out_channels = train_ds.presence_channels

    elif roi_profile == "center24_shared_flip":
        # Centro compartilhado com espelhamento:
        # - ramo canonico: CENTER direto
        # - ramo simetrico: CENTER flipado + remapeamento de labels centrais
        train_center = HydraTeethDataset(
            samples=train_samples,
            preset=preset,
            augment=True,
            source_mode=source_mode,
            seed=seed,
            teeth_subset=CENTER_TEETH,
            window_name="CENTER",
            flip_horizontal=False,
        )
        train_center_flip = HydraTeethDataset(
            samples=train_samples,
            preset=preset,
            augment=True,
            source_mode=source_mode,
            seed=seed + 1,
            teeth_subset=CENTER_TEETH,
            window_name="CENTER",
            flip_horizontal=True,
            label_remap=CENTER_RIGHT_TO_LEFT,
        )
        val_center = HydraTeethDataset(
            samples=val_samples,
            preset=preset,
            augment=False,
            source_mode=source_mode,
            seed=seed,
            teeth_subset=CENTER_TEETH,
            window_name="CENTER",
            flip_horizontal=False,
        )
        val_center_flip = HydraTeethDataset(
            samples=val_samples,
            preset=preset,
            augment=False,
            source_mode=source_mode,
            seed=seed + 1,
            teeth_subset=CENTER_TEETH,
            window_name="CENTER",
            flip_horizontal=True,
            label_remap=CENTER_RIGHT_TO_LEFT,
        )
        train_ds = ConcatDataset([train_center, train_center_flip])
        val_ds = ConcatDataset([val_center, val_center_flip])
        heatmap_out_channels = train_center.heatmap_channels
        presence_out_channels = train_center.presence_channels

    else:  # lateral_shared20
        # Importante: em panoramicas RM, o lado anatomico direito aparece
        # majoritariamente no lado esquerdo da imagem.
        # Portanto, usamos LEFT como ramo canonico (direito anatomico)
        # e RIGHT flipado+remapeado como aumento simetrico.
        train_right = HydraTeethDataset(
            samples=train_samples,
            preset=preset,
            augment=True,
            source_mode=source_mode,
            seed=seed,
            teeth_subset=LATERAL_RIGHT_TEETH,
            window_name="LEFT",
            flip_horizontal=False,
        )
        train_left_flip = HydraTeethDataset(
            samples=train_samples,
            preset=preset,
            augment=True,
            source_mode=source_mode,
            seed=seed + 1,
            teeth_subset=LATERAL_RIGHT_TEETH,
            window_name="RIGHT",
            flip_horizontal=True,
            label_remap=LEFT_TO_RIGHT,
            label_remap_only_keys=True,
        )
        val_right = HydraTeethDataset(
            samples=val_samples,
            preset=preset,
            augment=False,
            source_mode=source_mode,
            seed=seed,
            teeth_subset=LATERAL_RIGHT_TEETH,
            window_name="LEFT",
            flip_horizontal=False,
        )
        val_left_flip = HydraTeethDataset(
            samples=val_samples,
            preset=preset,
            augment=False,
            source_mode=source_mode,
            seed=seed + 1,
            teeth_subset=LATERAL_RIGHT_TEETH,
            window_name="RIGHT",
            flip_horizontal=True,
            label_remap=LEFT_TO_RIGHT,
            label_remap_only_keys=True,
        )
        train_ds = ConcatDataset([train_right, train_left_flip])
        val_ds = ConcatDataset([val_right, val_left_flip])
        heatmap_out_channels = train_right.heatmap_channels
        presence_out_channels = train_right.presence_channels

    print(
        f"[ROI] profile={roi_profile} heatmap_out={heatmap_out_channels} presence_out={presence_out_channels}"
    )

    batch_size = int(cfg["training"].get("batch_size", preset["training"].get("batch_size", 8)))
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

    device_name = str(cfg["training"].get("device", "auto"))
    if device_name == "auto":
        device = _auto_device()
    else:
        device = torch.device(device_name)
    print(f"[DEVICE] using {device}")

    model = HydraUNetMultiTask(
        in_channels=1,
        heatmap_out_channels=heatmap_out_channels,
        presence_out_channels=presence_out_channels,
        enable_presence_head=bool(cfg["training"].get("use_presence_head", True)),
        backbone=cfg["model"].get("backbone", "resnet34"),
        presence_dropout=float(cfg["model"].get("presence_dropout", 0.1)),
    ).to(device)

    if args.init_ckpt is not None:
        init_ckpt = _resolve_path(repo_root, str(args.init_ckpt))
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-ckpt nao encontrado: {init_ckpt}")
        init_obj = torch.load(init_ckpt, map_location="cpu")
        state = init_obj.get("model_state_dict", None)
        if state is None:
            state = init_obj.get("state_dict", None)
        if state is None:
            raise KeyError(f"Checkpoint sem model_state_dict/state_dict: {init_ckpt}")
        model.load_state_dict(state, strict=True)
        print(
            "[INIT] loaded model weights from {} (epoch={} best_val={})".format(
                init_ckpt,
                init_obj.get("epoch"),
                init_obj.get("best_val"),
            )
        )

    criterion = HydraMultiTaskLoss(
        w_heatmap=float(cfg["training"].get("lambda_heatmap", 1.0)),
        w_presence=float(cfg["training"].get("lambda_presence", 0.3)),
        w_mse=float(cfg["training"].get("loss_mse", 0.8)),
        w_dice=float(cfg["training"].get("loss_dice", 0.2)),
        absent_heatmap_weight=float(cfg["training"].get("absent_heatmap_weight", 1.0)),
    )

    lr = float(cfg["training"].get("lr", preset["training"].get("lr", 3e-4)))
    weight_decay = float(cfg["training"].get("weight_decay", preset["training"].get("weight_decay", 1e-4)))
    epochs = int(cfg["training"].get("epochs", preset["training"].get("epochs", 120)))
    patience = int(
        cfg["training"].get(
            "early_stopping_patience",
            preset["training"].get("early_stopping_patience", 20),
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(cfg["training"].get("lr_min", 1e-6)),
    )

    amp_requested = bool(cfg["training"].get("amp", preset["training"].get("amp", True)))
    amp_enabled = bool(amp_requested and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    writer = None
    tb_dir = run_dir / "tensorboard"
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))
    else:
        print("[WARN] tensorboard nao instalado; seguiremos com CSV apenas.")

    csv_path = run_dir / "metrics.csv"
    ckpt_best = run_dir / "best.ckpt"
    ckpt_last = run_dir / "last.ckpt"
    if csv_path.exists():
        csv_path.unlink()

    run_cfg_out = run_dir / "resolved_config.json"
    run_cfg_out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_base_dir / "latest_run.txt").write_text(run_name, encoding="utf-8")

    try:
        first_batch = next(iter(train_loader))
    except RuntimeError as e:
        if "torch_shm_manager" not in str(e):
            raise
        print("[WARN] DataLoader multiprocessing indisponivel neste ambiente; fallback para num_workers=0.")
        num_workers = 0
        train_loader = DataLoader(train_ds, **_make_loader_kwargs(num_workers, shuffle=True))
        val_loader = DataLoader(val_ds, **_make_loader_kwargs(num_workers, shuffle=False))
        print("[DATALOADER] workers=0 prefetch_factor=0 persistent_workers=False parallel=OFF")
        first_batch = next(iter(train_loader))
    print(
        "[SHAPES] X={} Y_heatmap={} Y_presence={}".format(
            tuple(first_batch["x"].shape),
            tuple(first_batch["y_heatmap"].shape),
            tuple(first_batch["y_presence"].shape),
        )
    )

    best_val = float("inf")
    stale_epochs = 0

    max_train_batches = int(cfg["training"].get("max_train_batches", 0))
    max_val_batches = int(cfg["training"].get("max_val_batches", 0))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 0.0))

    for epoch in range(1, epochs + 1):
        train_stats = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=max_train_batches,
            grad_clip_norm=grad_clip_norm,
        )
        val_stats = _run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=max_val_batches,
            grad_clip_norm=0.0,
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        metrics = EpochMetrics(
            train_total_loss=train_stats["total"],
            train_heatmap_loss=train_stats["heatmap"],
            train_presence_loss=train_stats["presence"],
            val_total_loss=val_stats["total"],
            val_heatmap_loss=val_stats["heatmap"],
            val_presence_loss=val_stats["presence"],
            lr=lr_now,
        )
        _write_csv_row(csv_path, metrics, epoch)

        if writer is not None:
            writer.add_scalar("loss/train_total", metrics.train_total_loss, epoch)
            writer.add_scalar("loss/train_heatmap", metrics.train_heatmap_loss, epoch)
            writer.add_scalar("loss/train_presence", metrics.train_presence_loss, epoch)
            writer.add_scalar("loss/val_total", metrics.val_total_loss, epoch)
            writer.add_scalar("loss/val_heatmap", metrics.val_heatmap_loss, epoch)
            writer.add_scalar("loss/val_presence", metrics.val_presence_loss, epoch)
            writer.add_scalar("lr", metrics.lr, epoch)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "config": cfg,
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
            "[EPOCH {:03d}] train_total={:.6f} val_total={:.6f} val_heatmap={:.6f} val_presence={:.6f} lr={:.6e}".format(
                epoch,
                metrics.train_total_loss,
                metrics.val_total_loss,
                metrics.val_heatmap_loss,
                metrics.val_presence_loss,
                metrics.lr,
            )
        )

        debug_batch = next(iter(train_loader))
        debug_batch = _to_device(debug_batch, device)
        capture_epoch_visuals(
            out_dir=visuals_dir,
            epoch=epoch,
            model=model,
            x_before=debug_batch["x_before"],
            y_before=debug_batch["y_before"],
            x_after=debug_batch["x"],
            y_after=debug_batch["y_heatmap"],
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
