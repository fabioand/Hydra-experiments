#!/usr/bin/env python3
from __future__ import annotations

"""
Esqueleto de treino downstream com estrategia:
1) freeze do encoder
2) unfreeze para fine-tuning

Este arquivo propositalmente nao implementa dataset/loss especifica de tarefa,
mas deixa o contrato pronto para classificacao, regressao e segmentacao.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable

import torch

from panorama_foundation.dataset import load_json
from panorama_foundation.models import (
    PanoramicEncoderClassifier,
    PanoramicEncoderRegressor,
    PanoramicUNetSegmenter,
    load_encoder_from_checkpoint,
)


def _pick_model(task: str, cfg: Dict) -> torch.nn.Module:
    backbone = str(cfg["model"].get("backbone", "resnet34"))
    if task == "classification":
        return PanoramicEncoderClassifier(
            num_classes=int(cfg["task"]["num_classes"]),
            backbone=backbone,
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    if task == "regression":
        return PanoramicEncoderRegressor(
            out_dim=int(cfg["task"].get("out_dim", 1)),
            backbone=backbone,
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    if task == "segmentation":
        return PanoramicUNetSegmenter(
            out_channels=int(cfg["task"]["out_channels"]),
            backbone=backbone,
        )
    raise ValueError(f"Tarefa nao suportada: {task}")


def _make_optimizer_two_groups(
    model: torch.nn.Module,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )


def _count_trainable_params(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in params if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream transfer skeleton")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    task = str(cfg["task"]["type"])
    model = _pick_model(task=task, cfg=cfg)

    encoder_ckpt = Path(cfg["paths"]["encoder_checkpoint"])
    missing = load_encoder_from_checkpoint(model, encoder_ckpt)
    print(f"[LOAD] encoder carregado de {encoder_ckpt} (missing_keys={missing})")

    # Fase 1: freeze
    model.freeze_encoder()
    opt_stage1 = _make_optimizer_two_groups(
        model=model,
        lr_encoder=0.0,
        lr_head=float(cfg["training"]["lr_head_stage1"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    print(f"[STAGE1] trainable_params={_count_trainable_params(model.parameters())}")
    print(f"[STAGE1] lr_head={opt_stage1.param_groups[1]['lr']}")

    # Fase 2: unfreeze
    model.unfreeze_encoder()
    opt_stage2 = _make_optimizer_two_groups(
        model=model,
        lr_encoder=float(cfg["training"]["lr_encoder_stage2"]),
        lr_head=float(cfg["training"]["lr_head_stage2"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    print(f"[STAGE2] trainable_params={_count_trainable_params(model.parameters())}")
    print(
        "[STAGE2] lr_encoder={} lr_head={}".format(
            opt_stage2.param_groups[0]["lr"], opt_stage2.param_groups[1]["lr"]
        )
    )

    print(
        "[TODO] Implementar dataloaders, losses e loops por tarefa usando o protocolo freeze->unfreeze "
        "ja configurado neste esqueleto."
    )


if __name__ == "__main__":
    main()

