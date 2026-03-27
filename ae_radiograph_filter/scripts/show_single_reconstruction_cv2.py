#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from panorama_foundation.models import PanoramicResNetAutoencoder


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(ckpt_path: Path, device: torch.device) -> PanoramicResNetAutoencoder:
    model = PanoramicResNetAutoencoder(backbone="resnet34").to(device)
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw.get("model_state_dict", raw)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _read_gray(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel abrir imagem: {image_path}")
    return img


def _preprocess(img_u8: np.ndarray, size: int) -> torch.Tensor:
    img = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def _to_u8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar reconstrucao de uma radiografia com AE em janelas cv2.")
    parser.add_argument("--image", type=Path, required=True, help="Imagem de entrada (radiografia).")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/Users/fabioandrade/hydra/ae_radiograph_filter/models/ae_identity_bestE21.ckpt"),
        help="Checkpoint do autoencoder.",
    )
    parser.add_argument("--input-size", type=int, default=256, help="Tamanho quadrado de entrada do AE.")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")
    args = parser.parse_args()

    image_path = args.image.resolve()
    ckpt_path = args.ckpt.resolve()

    device = _auto_device() if args.device == "auto" else torch.device(args.device)
    model = _load_model(ckpt_path, device=device)

    orig_u8 = _read_gray(image_path)
    h0, w0 = orig_u8.shape[:2]

    x = _preprocess(orig_u8, size=args.input_size).to(device)
    with torch.no_grad():
        pred = model(x)["recon"]
    recon_256 = pred.detach().cpu().numpy()[0, 0]

    # Resize para tamanho original com interpolacao de alta qualidade.
    recon_orig = cv2.resize(recon_256, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
    recon_u8 = _to_u8(recon_orig)
    orig01 = orig_u8.astype(np.float32) / 255.0
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("AE Reconstruction (resized to original)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Enhanced (sliders)", cv2.WINDOW_NORMAL)
    cv2.imshow("Original", orig_u8)
    cv2.imshow("AE Reconstruction (resized to original)", recon_u8)

    # Sliders com alta resolução (0..1000 -> 0.000..1.000)
    max_slider = 1000
    def _redraw_from_sliders(_: int = 0) -> None:
        fs = cv2.getTrackbarPos("fs x recon", "Enhanced (sliders)") / float(max_slider)
        fa = cv2.getTrackbarPos("fa x (orig-recon)", "Enhanced (sliders)") / float(max_slider)

        # formula final: original - fs*recon + fa*(original-recon)
        enhanced01 = np.clip(orig01 - fs * recon_orig + fa * (orig01 - recon_orig), 0.0, 1.0)
        enhanced_u8 = _to_u8(enhanced01)
        cv2.imshow("Enhanced (sliders)", enhanced_u8)

    cv2.createTrackbar("fs x recon", "Enhanced (sliders)", 300, max_slider, _redraw_from_sliders)
    cv2.createTrackbar("fa x (orig-recon)", "Enhanced (sliders)", 1000, max_slider, _redraw_from_sliders)
    _redraw_from_sliders()

    print("Use os sliders fs/fa na 3a janela. Pressione 'q' ou ESC para fechar.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
