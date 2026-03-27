"""Hydra U-Net MultiTask

Modelo multitarefa para radiografias panoramicas:
- head 1: heatmaps de pontos (64 canais)
- head 2: presenca por dente (32 logits)

O design segue os documentos:
- docs/arquitetura/HYDRA_UNET_MULTITASK_SPEC.md
- docs/runbooks/HYDRA_TRAINING_RUNBOOK.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CANONICAL_TEETH_32: List[str] = [
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


def derive_presence_from_stack64(stack64: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Deriva vetor de presenca (32) a partir do target stack64.

    Args:
        stack64: tensor com shape (B,64,H,W) ou (64,H,W)
        eps: threshold minimo para considerar presenca

    Returns:
        Tensor binario float32 com shape (B,32) ou (32,)
    """
    if stack64.dim() == 3:
        stack64 = stack64.unsqueeze(0)
        squeeze_back = True
    elif stack64.dim() == 4:
        squeeze_back = False
    else:
        raise ValueError(f"Esperado shape (64,H,W) ou (B,64,H,W), recebido: {tuple(stack64.shape)}")

    if stack64.size(1) != 64:
        raise ValueError(f"Esperado 64 canais, recebido: {stack64.size(1)}")

    b, _, h, w = stack64.shape
    pairs = stack64.view(b, 32, 2, h, w)
    # Se qualquer pixel em qualquer um dos dois canais do dente for > eps, dente presente.
    presence = (pairs.amax(dim=(2, 3, 4)) > eps).float()

    if squeeze_back:
        return presence[0]
    return presence


def _normalize_map_01(att: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normaliza mapa por amostra para [0,1]."""
    b = att.shape[0]
    flat = att.view(b, -1)
    mn = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    mx = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
    return (att - mn) / (mx - mn + eps)


def aggregate_attention_map(
    feat: torch.Tensor, mode: str = "mean", normalize: bool = True
) -> torch.Tensor:
    """Agrega tensor de feature (B,C,H,W) em mapa (B,1,H,W) via mean ou max."""
    if feat.dim() != 4:
        raise ValueError(f"Esperado feat com shape (B,C,H,W), recebido {tuple(feat.shape)}")

    if mode == "mean":
        att = feat.mean(dim=1, keepdim=True)
    elif mode == "max":
        att = feat.max(dim=1, keepdim=True).values
    else:
        raise ValueError(f"Modo invalido: {mode}. Use 'mean' ou 'max'.")

    if normalize:
        att = _normalize_map_01(att)
    return att


def build_attention_maps(
    feature_dict: Dict[str, torch.Tensor],
    mode: str = "mean",
    normalize: bool = True,
    out_size: Tuple[int, int] | None = None,
) -> Dict[str, torch.Tensor]:
    """Gera mapas de atenção agregados para um conjunto de features.

    Args:
        feature_dict: dict nome->tensor (B,C,H,W)
        mode: 'mean' ou 'max'
        normalize: normalizar cada mapa para [0,1]
        out_size: (H,W) destino opcional para resize bilinear
    """
    out: Dict[str, torch.Tensor] = {}
    for name, feat in feature_dict.items():
        att = aggregate_attention_map(feat, mode=mode, normalize=normalize)
        if out_size is not None:
            att = F.interpolate(att, size=out_size, mode="bilinear", align_corners=False)
        out[name] = att
    return out


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """Encoder estilo ResNet simplificado para U-Net multitarefa.

    Variantes suportadas:
    - resnet18: layers [2,2,2,2]
    - resnet34: layers [3,4,6,3]
    """

    def __init__(self, in_channels: int = 1, variant: str = "resnet34"):
        super().__init__()

        if variant == "resnet18":
            layers = [2, 2, 2, 2]
        elif variant == "resnet34":
            layers = [3, 4, 6, 3]
        else:
            raise ValueError(f"Backbone nao suportado: {variant}. Use resnet18 ou resnet34.")

        self.variant = variant
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.out_channels = 512

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x0: 1/2
        x0 = self.relu(self.bn1(self.conv1(x)))

        # x1: 1/4
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        # x2: 1/8, x3: 1/16, x4: 1/32
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
        }


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, decoder_channels: Tuple[int, int, int, int, int] = (256, 128, 64, 32, 16)):
        super().__init__()

        c4, c3, c2, c1, c0 = decoder_channels

        # Bottleneck: 512
        self.up4 = UpBlock(in_ch=512, skip_ch=256, out_ch=c4)  # 1/32 -> 1/16
        self.up3 = UpBlock(in_ch=c4, skip_ch=128, out_ch=c3)  # 1/16 -> 1/8
        self.up2 = UpBlock(in_ch=c3, skip_ch=64, out_ch=c2)  # 1/8 -> 1/4
        self.up1 = UpBlock(in_ch=c2, skip_ch=64, out_ch=c1)  # 1/4 -> 1/2

        # Ultimo passo sem skip para ir a resolucao total (1/1)
        self.final_conv1 = ConvBNAct(c1, c0)
        self.final_conv2 = ConvBNAct(c0, c0)

        self.out_channels = c0

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0, x1, x2, x3, x4 = feats["x0"], feats["x1"], feats["x2"], feats["x3"], feats["x4"]

        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        d0 = F.interpolate(d1, scale_factor=2.0, mode="bilinear", align_corners=False)
        d0 = self.final_conv1(d0)
        d0 = self.final_conv2(d0)

        return d0


class PresenceHead(nn.Module):
    def __init__(self, in_channels: int, out_classes: int = 32, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class HydraUNetMultiTask(nn.Module):
    """Modelo Hydra multitarefa.

    Forward returns:
    - heatmap_logits: (B,64,H,W)
    - presence_logits: (B,32)
    """

    def __init__(
        self,
        in_channels: int = 1,
        heatmap_out_channels: int = 64,
        presence_out_channels: int = 32,
        enable_presence_head: bool = True,
        backbone: str = "resnet34",
        decoder_channels: Tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
        presence_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, variant=backbone)
        self.decoder = UNetDecoder(decoder_channels=decoder_channels)

        self.heatmap_head = nn.Conv2d(self.decoder.out_channels, heatmap_out_channels, kernel_size=1)
        self.enable_presence_head = bool(enable_presence_head)
        if self.enable_presence_head:
            self.presence_head = PresenceHead(
                in_channels=self.encoder.out_channels,
                out_classes=presence_out_channels,
                dropout=presence_dropout,
            )
        else:
            self.presence_head = None

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)

        dec = self.decoder(feats)
        heatmap_logits = self.heatmap_head(dec)
        presence_logits = self.presence_head(feats["x4"]) if self.presence_head is not None else None

        out = {
            "heatmap_logits": heatmap_logits,
            "presence_logits": presence_logits,
        }
        if return_intermediates:
            out["intermediates"] = {
                "enc_x1": feats["x1"],
                "enc_x2": feats["x2"],
                "enc_x3": feats["x3"],
                "bottleneck_x4": feats["x4"],
                "decoder_final": dec,
            }
        return out


def soft_dice_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice medio por batch e por canal."""
    if probs.shape != target.shape:
        raise ValueError(f"Shape mismatch: probs={tuple(probs.shape)} target={tuple(target.shape)}")

    dims = (2, 3)
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


@dataclass
class MultiTaskLossOutput:
    total: torch.Tensor
    heatmap_total: torch.Tensor
    heatmap_mse: torch.Tensor
    heatmap_dice: torch.Tensor
    presence_bce: torch.Tensor


class HydraMultiTaskLoss(nn.Module):
    """Loss multitarefa para HydraUNetMultiTask.

    L = w_heatmap * (w_mse*MSE + w_dice*Dice) + w_presence * BCEWithLogits
    """

    def __init__(
        self,
        w_heatmap: float = 1.0,
        w_presence: float = 0.3,
        w_mse: float = 0.8,
        w_dice: float = 0.2,
        absent_heatmap_weight: float = 1.0,
    ):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.w_presence = w_presence
        self.w_mse = w_mse
        self.w_dice = w_dice
        self.absent_heatmap_weight = absent_heatmap_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target_heatmap: torch.Tensor,
        target_presence: torch.Tensor,
    ) -> MultiTaskLossOutput:
        heatmap_logits = pred["heatmap_logits"]
        presence_logits = pred.get("presence_logits", None)

        heatmap_probs = torch.sigmoid(heatmap_logits)
        if self.absent_heatmap_weight != 1.0:
            if target_presence.ndim != 2:
                raise ValueError(f"target_presence must be (B,32), got shape={tuple(target_presence.shape)}")
            if target_presence.shape[1] * 2 != heatmap_probs.shape[1]:
                raise ValueError(
                    "target_presence and heatmap channels mismatch: "
                    f"presence={tuple(target_presence.shape)} heatmap={tuple(heatmap_probs.shape)}"
                )

            present_mask = target_presence.repeat_interleave(2, dim=1)
            channel_weight = present_mask + (1.0 - present_mask) * self.absent_heatmap_weight
            channel_weight_2d = channel_weight.unsqueeze(-1).unsqueeze(-1)

            mse_map = (heatmap_probs - target_heatmap) ** 2
            hm_mse = (mse_map * channel_weight_2d).sum() / (
                channel_weight_2d.sum() * heatmap_probs.shape[2] * heatmap_probs.shape[3] + 1e-8
            )

            intersection = (heatmap_probs * target_heatmap).sum(dim=(2, 3))
            denom = heatmap_probs.sum(dim=(2, 3)) + target_heatmap.sum(dim=(2, 3))
            dice_per_channel = 1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6)
            hm_dice = (dice_per_channel * channel_weight).sum() / (channel_weight.sum() + 1e-8)
        else:
            hm_mse = F.mse_loss(heatmap_probs, target_heatmap)
            hm_dice = soft_dice_loss_from_probs(heatmap_probs, target_heatmap)
        hm_total = self.w_mse * hm_mse + self.w_dice * hm_dice

        if self.w_presence > 0.0:
            if presence_logits is None:
                raise ValueError("presence head disabled but w_presence > 0")
            presence_bce = self.bce(presence_logits, target_presence)
        else:
            presence_bce = torch.zeros((), device=heatmap_logits.device, dtype=heatmap_logits.dtype)

        total = self.w_heatmap * hm_total + self.w_presence * presence_bce
        return MultiTaskLossOutput(
            total=total,
            heatmap_total=hm_total,
            heatmap_mse=hm_mse,
            heatmap_dice=hm_dice,
            presence_bce=presence_bce,
        )


__all__ = [
    "CANONICAL_TEETH_32",
    "derive_presence_from_stack64",
    "aggregate_attention_map",
    "build_attention_maps",
    "HydraUNetMultiTask",
    "HydraMultiTaskLoss",
    "MultiTaskLossOutput",
]
