"""Modelo e loss para denoising autoencoder de coordenadas de long-eixo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn


class CoordinateDenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dims: Sequence[int] = (512, 256),
        latent_dim: int = 128,
        dropout: float = 0.1,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim/output_dim devem ser > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim deve ser > 0")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_activation = output_activation
        self.coords_dim = 128
        if self.output_dim < self.coords_dim:
            raise ValueError(f"output_dim deve ser >= {self.coords_dim}")

        enc_layers = []
        prev = self.input_dim
        for h in hidden_dims:
            h = int(h)
            enc_layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.to_latent = nn.Linear(prev, latent_dim)

        dec_layers = []
        prev = latent_dim
        for h in reversed(list(hidden_dims)):
            h = int(h)
            dec_layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            prev = h
        dec_layers.append(nn.Linear(prev, self.output_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.to_latent(self.encoder(x))
        out = self.decoder(z)

        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.output_activation == "none":
            pass
        else:
            raise ValueError(f"output_activation invalida: {self.output_activation}")

        return {
            "coords_pred": out[:, : self.coords_dim],
            "curves_pred": out[:, self.coords_dim :] if self.output_dim > self.coords_dim else None,
            "latent": z,
        }


@dataclass
class DaeLossOutput:
    total: torch.Tensor
    mse_all: torch.Tensor
    mse_knocked: torch.Tensor
    mse_observed: torch.Tensor
    mae_all: torch.Tensor
    mae_knocked: torch.Tensor
    mae_observed: torch.Tensor
    mse_curves: torch.Tensor
    mae_curves: torch.Tensor
    arc_spacing: torch.Tensor
    anchor_relative: torch.Tensor


class DaeImputationLoss(nn.Module):
    """Loss ponderada por tipo de ponto (nocauteado vs observado)."""

    def __init__(
        self,
        w_knocked: float = 0.85,
        w_observed: float = 0.15,
        w_all: float = 0.0,
        w_curves: float = 0.0,
        w_arc_spacing: float = 0.0,
        w_anchor_rel: float = 0.0,
    ):
        super().__init__()
        self.w_knocked = float(w_knocked)
        self.w_observed = float(w_observed)
        self.w_all = float(w_all)
        self.w_curves = float(w_curves)
        self.w_arc_spacing = float(w_arc_spacing)
        self.w_anchor_rel = float(w_anchor_rel)

    @staticmethod
    def _arc_spacing_regularization(
        pred_coords: torch.Tensor,
        gt_available_teeth_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Regulariza a regularidade das distancias entre centros vizinhos.

        Para cada quadrante (8 dentes):
        - calcula centros por dente
        - calcula distancias entre vizinhos consecutivos (7)
        - penaliza variacao entre distancias adjacentes (6): |d_{i+1} - d_i|

        O termo so e aplicado quando ha GT disponivel no triplo (i, i+1, i+2).
        """
        b = pred_coords.shape[0]
        coords = pred_coords.view(b, 32, 4)
        centers = torch.stack(
            [
                0.5 * (coords[:, :, 0] + coords[:, :, 2]),
                0.5 * (coords[:, :, 1] + coords[:, :, 3]),
            ],
            dim=-1,
        )  # (B,32,2)

        losses: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        eps = 1e-8

        for start in (0, 8, 16, 24):
            c = centers[:, start : start + 8, :]  # (B,8,2)
            m = gt_available_teeth_mask[:, start : start + 8]  # (B,8)

            d = torch.sqrt(((c[:, 1:, :] - c[:, :-1, :]) ** 2).sum(dim=-1) + eps)  # (B,7)
            reg = (d[:, 1:] - d[:, :-1]).abs()  # (B,6)

            triplet_mask = m[:, :-2] * m[:, 1:-1] * m[:, 2:]  # (B,6)
            losses.append(reg)
            masks.append(triplet_mask)

        reg_all = torch.cat(losses, dim=1)  # (B,24)
        mask_all = torch.cat(masks, dim=1)  # (B,24)

        denom = mask_all.sum()
        if float(denom.item()) <= 0.0:
            return pred_coords.new_tensor(0.0)
        return (reg_all * mask_all).sum() / (denom + eps)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        denom = mask.sum()
        if float(denom.item()) <= 0.0:
            return x.new_tensor(0.0)
        return (x * mask).sum() / (denom + eps)

    @staticmethod
    def _anchor_relative_regularization(
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        knocked_teeth_mask: torch.Tensor,
        gt_available_teeth_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Regulariza dente nocauteado em relacao a vizinhos observados (ancora no GT).

        Para cada dente i nocauteado, usa vizinhos imediatos (esquerda/direita no mesmo
        quadrante) se estiverem observados e com GT. Compara vetor relativo:
          (c_i_pred - c_j_gt)  vs  (c_i_gt - c_j_gt)
        """
        b = pred_coords.shape[0]
        pred = pred_coords.view(b, 32, 4)
        target = target_coords.view(b, 32, 4)

        c_pred = torch.stack(
            [0.5 * (pred[:, :, 0] + pred[:, :, 2]), 0.5 * (pred[:, :, 1] + pred[:, :, 3])],
            dim=-1,
        )  # (B,32,2)
        c_gt = torch.stack(
            [0.5 * (target[:, :, 0] + target[:, :, 2]), 0.5 * (target[:, :, 1] + target[:, :, 3])],
            dim=-1,
        )  # (B,32,2)

        terms: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for q_start in (0, 8, 16, 24):
            for local_i in range(8):
                i = q_start + local_i
                # Condicao base para dente alvo (nocauteado e com GT disponivel)
                mask_i = knocked_teeth_mask[:, i] * gt_available_teeth_mask[:, i]  # (B,)

                for local_j in (local_i - 1, local_i + 1):
                    if local_j < 0 or local_j > 7:
                        continue
                    j = q_start + local_j

                    # Vizinho observado e com GT
                    mask_j = (1.0 - knocked_teeth_mask[:, j]) * gt_available_teeth_mask[:, j]  # (B,)
                    m = mask_i * mask_j  # (B,)

                    v_pred = c_pred[:, i, :] - c_gt[:, j, :]
                    v_gt = c_gt[:, i, :] - c_gt[:, j, :]
                    reg = (v_pred - v_gt).abs().sum(dim=-1)  # (B,) L1 no vetor (dx,dy)

                    terms.append(reg)
                    masks.append(m)

        if not terms:
            return pred_coords.new_tensor(0.0)

        reg_all = torch.stack(terms, dim=1)  # (B,Npairs)
        mask_all = torch.stack(masks, dim=1)  # (B,Npairs)
        denom = mask_all.sum()
        if float(denom.item()) <= 0.0:
            return pred_coords.new_tensor(0.0)
        return (reg_all * mask_all).sum() / (denom + 1e-8)

    def forward(
        self,
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        knocked_teeth_mask: torch.Tensor,
        gt_available_teeth_mask: torch.Tensor | None = None,
        pred_curves: torch.Tensor | None = None,
        target_curves: torch.Tensor | None = None,
        curves_available_mask: torch.Tensor | None = None,
    ) -> DaeLossOutput:
        if pred_coords.shape != target_coords.shape:
            raise ValueError(
                f"Shape mismatch pred/target: {tuple(pred_coords.shape)} vs {tuple(target_coords.shape)}"
            )
        if pred_coords.ndim != 2 or pred_coords.shape[1] != 128:
            raise ValueError(f"Esperado pred_coords (B,128), recebido {tuple(pred_coords.shape)}")
        if knocked_teeth_mask.ndim != 2 or knocked_teeth_mask.shape[1] != 32:
            raise ValueError(
                f"Esperado knocked_teeth_mask (B,32), recebido {tuple(knocked_teeth_mask.shape)}"
            )

        if gt_available_teeth_mask is None:
            gt_available_teeth_mask = torch.ones_like(knocked_teeth_mask)
        if gt_available_teeth_mask.ndim != 2 or gt_available_teeth_mask.shape[1] != 32:
            raise ValueError(
                "Esperado gt_available_teeth_mask (B,32), recebido "
                f"{tuple(gt_available_teeth_mask.shape)}"
            )

        sq = (pred_coords - target_coords) ** 2
        abs_err = (pred_coords - target_coords).abs()

        available_xy = gt_available_teeth_mask.repeat_interleave(4, dim=1)
        knocked_xy = knocked_teeth_mask.repeat_interleave(4, dim=1)
        valid_knocked_xy = knocked_xy * available_xy
        valid_observed_xy = (1.0 - knocked_xy) * available_xy

        mse_all = self._masked_mean(sq, available_xy)
        mae_all = self._masked_mean(abs_err, available_xy)

        mse_knocked = self._masked_mean(sq, valid_knocked_xy)
        mse_observed = self._masked_mean(sq, valid_observed_xy)
        mae_knocked = self._masked_mean(abs_err, valid_knocked_xy)
        mae_observed = self._masked_mean(abs_err, valid_observed_xy)

        mse_curves = pred_coords.new_tensor(0.0)
        mae_curves = pred_coords.new_tensor(0.0)
        if pred_curves is not None or target_curves is not None:
            if pred_curves is None or target_curves is None:
                raise ValueError("pred_curves/target_curves devem ser ambos None ou ambos tensores")
            if pred_curves.shape != target_curves.shape:
                raise ValueError(
                    f"Shape mismatch pred_curves/target_curves: {tuple(pred_curves.shape)} vs {tuple(target_curves.shape)}"
                )
            if pred_curves.ndim != 2:
                raise ValueError(f"Esperado pred_curves (B,D), recebido {tuple(pred_curves.shape)}")
            sq_c = (pred_curves - target_curves) ** 2
            abs_c = (pred_curves - target_curves).abs()
            if curves_available_mask is None:
                c_mask = torch.ones((pred_curves.shape[0], 1), dtype=pred_curves.dtype, device=pred_curves.device)
            else:
                c_mask = curves_available_mask
                if c_mask.ndim == 1:
                    c_mask = c_mask.unsqueeze(1)
                if c_mask.ndim != 2 or c_mask.shape[1] != 1 or c_mask.shape[0] != pred_curves.shape[0]:
                    raise ValueError(
                        "Esperado curves_available_mask com shape (B,) ou (B,1), recebido "
                        f"{tuple(curves_available_mask.shape)}"
                    )
                c_mask = c_mask.to(dtype=pred_curves.dtype, device=pred_curves.device)
            c_mask = c_mask.expand(-1, pred_curves.shape[1])
            mse_curves = self._masked_mean(sq_c, c_mask)
            mae_curves = self._masked_mean(abs_c, c_mask)

        arc_spacing = self._arc_spacing_regularization(pred_coords, gt_available_teeth_mask)
        anchor_relative = self._anchor_relative_regularization(
            pred_coords=pred_coords,
            target_coords=target_coords,
            knocked_teeth_mask=knocked_teeth_mask,
            gt_available_teeth_mask=gt_available_teeth_mask,
        )

        total = (
            self.w_knocked * mse_knocked
            + self.w_observed * mse_observed
            + self.w_all * mse_all
            + self.w_curves * mse_curves
            + self.w_arc_spacing * arc_spacing
            + self.w_anchor_rel * anchor_relative
        )
        return DaeLossOutput(
            total=total,
            mse_all=mse_all,
            mse_knocked=mse_knocked,
            mse_observed=mse_observed,
            mae_all=mae_all,
            mae_knocked=mae_knocked,
            mae_observed=mae_observed,
            mse_curves=mse_curves,
            mae_curves=mae_curves,
            arc_spacing=arc_spacing,
            anchor_relative=anchor_relative,
        )


def point_distance_px(pred_coords: torch.Tensor, target_coords: torch.Tensor, grid_size: int = 256) -> torch.Tensor:
    """Distancia euclidiana por ponto (64) em pixels de um grid de referencia."""
    if pred_coords.shape != target_coords.shape:
        raise ValueError("pred_coords e target_coords precisam ter o mesmo shape")
    if pred_coords.ndim != 2 or pred_coords.shape[1] != 128:
        raise ValueError("Esperado shape (B,128)")

    scale = float(max(1, grid_size - 1))

    p = pred_coords.view(-1, 64, 2) * scale
    t = target_coords.view(-1, 64, 2) * scale
    return torch.sqrt(((p - t) ** 2).sum(dim=-1))


__all__ = [
    "CoordinateDenoisingAutoencoder",
    "DaeImputationLoss",
    "DaeLossOutput",
    "point_distance_px",
]
