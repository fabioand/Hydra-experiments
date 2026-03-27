#!/usr/bin/env python3
"""Preset utilitario para preprocess de treino U-Net 256x256 (entrada 1 canal, saida 64 canais).

Fluxo recomendado:
1) Gerar heatmaps no tamanho original.
2) Reduzir imagem e mascara juntos para 256x256.
"""

from pathlib import Path
import argparse

import cv2
import numpy as np

TARGET_SIZE = (256, 256)  # (W, H)


def load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return img


def load_stack64(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[0] != 64:
        raise ValueError(f"Esperado stack64 com shape (64,H,W). Recebido: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def preprocess_unet256(
    img_gray: np.ndarray,
    mask_stack64: np.ndarray,
    normalize_image: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Redimensiona imagem e mascara para 256x256 mantendo ranges esperados.

    Returns:
        x: (1, 256, 256), float32
        y: (64, 256, 256), float32
    """
    if img_gray.ndim != 2:
        raise ValueError(f"img_gray deve ter 2 dimensoes (H,W). Recebido: {img_gray.shape}")
    if mask_stack64.ndim != 3 or mask_stack64.shape[0] != 64:
        raise ValueError(
            f"mask_stack64 deve ter shape (64,H,W). Recebido: {mask_stack64.shape}"
        )

    # Imagem: INTER_AREA costuma preservar melhor ao reduzir.
    x = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)

    # Mascara: interpolacao suave para manter forma gaussiana.
    resized_channels = []
    for c in range(64):
        ch = cv2.resize(mask_stack64[c], TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        resized_channels.append(ch)
    y = np.stack(resized_channels, axis=0).astype(np.float32)
    np.clip(y, 0.0, 1.0, out=y)

    if normalize_image:
        x /= 255.0

    # Formato canal-primeiro para treino (PyTorch friendly).
    x = x[None, ...]
    return x, y


def save_preview_overlay(x_1xhw: np.ndarray, y_64xhw: np.ndarray, out_png: Path) -> None:
    """Cria preview rapido (entrada 256 + max dos 64 canais em vermelho)."""
    x = np.clip(x_1xhw[0] * 255.0, 0.0, 255.0).astype(np.uint8)
    y_max = np.max(y_64xhw, axis=0)

    base_bgr = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    red_layer = np.zeros_like(base_bgr, dtype=np.uint8)
    red_layer[:, :, 2] = np.clip(y_max * 255.0, 0.0, 255.0).astype(np.uint8)
    overlay = cv2.addWeighted(base_bgr, 1.0, red_layer, 0.6, 0.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess de 1 amostra para preset U-Net 256.")
    parser.add_argument("--image", type=Path, required=True, help="Caminho do JPG")
    parser.add_argument("--mask", type=Path, required=True, help="Caminho do .npy stack64")
    parser.add_argument("--out-x", type=Path, default=Path("/tmp/x_256.npy"))
    parser.add_argument("--out-y", type=Path, default=Path("/tmp/y_256.npy"))
    parser.add_argument("--out-overlay", type=Path, default=Path("/tmp/overlay_256.png"))

    args = parser.parse_args()

    img = load_gray_image(args.image)
    mask = load_stack64(args.mask)
    x, y = preprocess_unet256(img, mask, normalize_image=True)

    np.save(args.out_x, x)
    np.save(args.out_y, y)
    save_preview_overlay(x, y, args.out_overlay)

    print(f"x shape={x.shape} dtype={x.dtype} min={x.min():.4f} max={x.max():.4f}")
    print(f"y shape={y.shape} dtype={y.dtype} min={y.min():.4f} max={y.max():.4f}")
    print(f"saved: {args.out_x}")
    print(f"saved: {args.out_y}")
    print(f"saved: {args.out_overlay}")


if __name__ == "__main__":
    main()
