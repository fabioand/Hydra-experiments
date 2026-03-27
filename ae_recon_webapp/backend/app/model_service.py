from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch

from panorama_foundation.models import PanoramicResNetAutoencoder


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class ReconstructionService:
    def __init__(self, ckpt_path: Path):
        self.ckpt_path = ckpt_path
        self.device = self._auto_device()
        self.model = self._load_model(ckpt_path=ckpt_path, device=self.device)

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _load_model(ckpt_path: Path, device: torch.device) -> PanoramicResNetAutoencoder:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint nao encontrado: {ckpt_path}")

        model = PanoramicResNetAutoencoder(backbone="resnet34").to(device)
        raw = torch.load(str(ckpt_path), map_location="cpu")
        state = raw.get("model_state_dict", raw)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    @staticmethod
    def decode_gray(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Imagem invalida ou formato nao suportado")
        return img

    @staticmethod
    def preprocess_u8(img_u8: np.ndarray, size: int) -> torch.Tensor:
        img = cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def to_u8(img01: np.ndarray) -> np.ndarray:
        return np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def png_base64_from_u8(gray_u8: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", gray_u8)
        if not ok:
            raise ValueError("Falha ao codificar PNG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def reconstruct(self, orig_u8: np.ndarray, input_size: int) -> tuple[np.ndarray, np.ndarray]:
        h0, w0 = orig_u8.shape[:2]

        x = self.preprocess_u8(orig_u8, size=input_size).to(self.device)
        with torch.no_grad():
            pred = self.model(x)["recon"]

        recon_small = pred.detach().cpu().numpy()[0, 0]
        recon_orig = cv2.resize(recon_small, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        recon_u8 = self.to_u8(recon_orig)
        return orig_u8, recon_u8


def discover_sample_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []

    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted({path.resolve() for path in files})
