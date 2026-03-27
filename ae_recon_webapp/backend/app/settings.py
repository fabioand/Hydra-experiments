from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    project_root: Path
    ckpt_path: Path
    sample_images_dir: Path
    max_upload_bytes: int
    default_input_size: int
    allowed_origins: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _split_csv_env(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings() -> Settings:
    root = _repo_root()
    ckpt_env = os.getenv("AE_RECON_CKPT")
    sample_env = os.getenv("AE_RECON_SAMPLE_IMAGES_DIR")

    ckpt_path = Path(ckpt_env).expanduser() if ckpt_env else root / "ae_radiograph_filter/models/ae_identity_bestE21.ckpt"
    sample_images_dir = Path(sample_env).expanduser() if sample_env else root / "ae_recon_webapp/sample_images"

    default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    allowed_origins = _split_csv_env(os.getenv("AE_RECON_ALLOWED_ORIGINS")) or default_origins

    return Settings(
        project_root=root,
        ckpt_path=ckpt_path,
        sample_images_dir=sample_images_dir,
        max_upload_bytes=int(os.getenv("AE_RECON_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        default_input_size=int(os.getenv("AE_RECON_DEFAULT_INPUT_SIZE", "256")),
        allowed_origins=allowed_origins,
    )
