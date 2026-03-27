from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra_multitask_model import PresenceHead, ResNetEncoder, UNetDecoder


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpNoSkipBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PlainDecoderNoSkips(nn.Module):
    """Decoder sem skip-connections para pretreino de autoencoder."""

    def __init__(self, decoder_channels: Tuple[int, int, int, int, int] = (256, 128, 64, 32, 16)):
        super().__init__()
        c4, c3, c2, c1, c0 = decoder_channels
        self.up4 = UpNoSkipBlock(512, c4)  # 1/32 -> 1/16
        self.up3 = UpNoSkipBlock(c4, c3)  # 1/16 -> 1/8
        self.up2 = UpNoSkipBlock(c3, c2)  # 1/8 -> 1/4
        self.up1 = UpNoSkipBlock(c2, c1)  # 1/4 -> 1/2
        self.up0 = UpNoSkipBlock(c1, c0)  # 1/2 -> 1/1
        self.out_channels = c0

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        x = self.up4(x4)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.up0(x)
        return x


class PanoramicResNetAutoencoder(nn.Module):
    """Autoencoder de panoramica com encoder ResNet + decoder SEM skips."""

    def __init__(
        self,
        in_channels: int = 1,
        backbone: str = "resnet34",
        decoder_channels: Tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, variant=backbone)
        self.decoder = PlainDecoderNoSkips(decoder_channels=decoder_channels)
        self.reconstruction_head = nn.Conv2d(self.decoder.out_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        feats = self.encoder(x)
        dec = self.decoder(feats["x4"])
        recon_logits = self.reconstruction_head(dec)
        recon = torch.sigmoid(recon_logits)
        out = {"recon_logits": recon_logits, "recon": recon}
        if return_intermediates:
            out["intermediates"] = {
                "enc_x1": feats["x1"],
                "enc_x2": feats["x2"],
                "enc_x3": feats["x3"],
                "bottleneck_x4": feats["x4"],
                "decoder_final": dec,
            }
        return out

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True


class PanoramicEncoderClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        backbone: str = "resnet34",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, variant=backbone)
        self.head = PresenceHead(
            in_channels=self.encoder.out_channels,
            out_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats["x4"])

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True


class PanoramicEncoderRegressor(nn.Module):
    def __init__(
        self,
        out_dim: int = 1,
        in_channels: int = 1,
        backbone: str = "resnet34",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, variant=backbone)
        self.head = PresenceHead(
            in_channels=self.encoder.out_channels,
            out_classes=out_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats["x4"])

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True


class PanoramicUNetSegmenter(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int = 1,
        backbone: str = "resnet34",
        decoder_channels: Tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, variant=backbone)
        self.decoder = UNetDecoder(decoder_channels=decoder_channels)
        self.segmentation_head = nn.Conv2d(self.decoder.out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        dec = self.decoder(feats)
        return self.segmentation_head(dec)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True


def load_encoder_from_checkpoint(model: nn.Module, ckpt_path: Path) -> int:
    """Carrega pesos no encoder de um checkpoint de AE ou encoder-only."""
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw.get("model_state_dict", raw)

    if any(k.startswith("encoder.") for k in state.keys()):
        encoder_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    else:
        encoder_state = state

    result = model.encoder.load_state_dict(encoder_state, strict=False)
    missing = len(result.missing_keys)
    return missing
