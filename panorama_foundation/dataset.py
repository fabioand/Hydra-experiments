from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class PanoramaSample:
    stem: str
    image_path: Path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def discover_panoramic_samples(images_dir: Path) -> List[PanoramaSample]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    uniq = sorted({f.resolve() for f in files})
    return [PanoramaSample(stem=p.stem, image_path=p) for p in uniq]


def make_or_load_split(
    samples: Sequence[PanoramaSample],
    split_path: Path,
    seed: int = 123,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    force_regen: bool = False,
) -> Dict[str, List[str]]:
    available = {s.stem for s in samples}

    if split_path.exists() and not force_regen:
        data = load_json(split_path)
        if "train" in data and "val" in data:
            train = [s for s in data["train"] if s in available]
            val = [s for s in data["val"] if s in available]
            test = [s for s in data.get("test", []) if s in available]

            train_set = set(train)
            val_set = set(val)
            test_set = set(test)
            disjoint = not (train_set & val_set or train_set & test_set or val_set & test_set)

            if "test" in data:
                current_set = train_set.union(val_set).union(test_set)
                if train and val and test and disjoint and current_set == available:
                    return {"train": train, "val": val, "test": test}
            else:
                current_set = train_set.union(val_set)
                if train and val and disjoint and current_set == available:
                    return {"train": train, "val": val}

    stems = sorted(list(available))
    rng = random.Random(seed)
    rng.shuffle(stems)

    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Ratios invalidos: requer 0 <= val_ratio, test_ratio e val_ratio+test_ratio < 1")

    n_total = len(stems)
    n_test = max(1, int(round(n_total * test_ratio))) if test_ratio > 0 and n_total > 2 else 0
    n_val = max(1, int(round(n_total * val_ratio))) if val_ratio > 0 and n_total > 1 else 0

    if n_test + n_val >= n_total:
        n_test = min(n_test, max(0, n_total - 2))
        n_val = min(n_val, max(1, n_total - n_test - 1))

    test = stems[:n_test]
    val = stems[n_test : n_test + n_val]
    train = stems[n_test + n_val :]

    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_samples": n_total,
        "train": train,
        "val": val,
    }
    if n_test > 0:
        payload["test"] = test
    save_json(split_path, payload)

    out = {"train": train, "val": val}
    if n_test > 0:
        out["test"] = test
    return out


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel abrir imagem: {path}")
    return img


def _build_aug(train: bool) -> A.Compose:
    if not train:
        return A.Compose([])
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.015,
                scale_limit=0.012,
                rotate_limit=2,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.25),
        ]
    )


class PanoramaAutoencoderDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[PanoramaSample],
        image_size_hw: Tuple[int, int] = (256, 256),
        augment: bool = False,
        pretext_cfg: Dict | None = None,
    ):
        self.samples = list(samples)
        self.image_size_hw = image_size_hw
        self.pretext_cfg = pretext_cfg or {"mode": "identity"}
        pretext_mode = str(self.pretext_cfg.get("mode", "identity")).lower()
        self.is_train = bool(augment)
        # No modo identity, desativamos augmentacoes para manter reconstrucao limpa.
        use_augment = bool(augment and pretext_mode != "identity")
        self.augment = _build_aug(train=use_augment)

    def __len__(self) -> int:
        return len(self.samples)

    def _identity_corrupt(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros_like(x, dtype=np.float32)
        return x, mask

    def _add_noise(self, x: np.ndarray, noise_cfg: Dict) -> np.ndarray:
        out = x.copy()

        std_min = float(noise_cfg.get("gaussian_std_min", 0.01))
        std_max = float(noise_cfg.get("gaussian_std_max", 0.06))
        if std_max > 0:
            std = np.random.uniform(std_min, std_max)
            out = out + np.random.normal(0.0, std, size=out.shape).astype(np.float32)

        poisson_strength = float(noise_cfg.get("poisson_strength", 0.03))
        if poisson_strength > 0:
            vals = float(np.random.uniform(24.0, 96.0))
            p = np.random.poisson(np.clip(out, 0.0, 1.0) * vals).astype(np.float32) / vals
            out = (1.0 - poisson_strength) * out + poisson_strength * p

        speckle = float(noise_cfg.get("speckle_strength", 0.03))
        if speckle > 0:
            out = out + out * np.random.normal(0.0, speckle, size=out.shape).astype(np.float32)

        return np.clip(out, 0.0, 1.0)

    def _apply_inpaint(self, x: np.ndarray, inpaint_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
        h, w = x.shape
        out = x.copy()
        mask = np.zeros_like(x, dtype=np.float32)

        cov_min = float(inpaint_cfg.get("coverage_min", 0.10))
        cov_max = float(inpaint_cfg.get("coverage_max", 0.30))
        target_coverage = float(np.random.uniform(cov_min, cov_max))
        target_pixels = int(target_coverage * h * w)

        min_holes = int(inpaint_cfg.get("min_holes", 4))
        max_holes = int(inpaint_cfg.get("max_holes", 20))
        num_holes = np.random.randint(min_holes, max_holes + 1)

        covered = 0
        max_hole_h = max(8, h // 6)
        max_hole_w = max(8, w // 6)
        min_hole_h = max(4, h // 28)
        min_hole_w = max(4, w // 28)

        for _ in range(num_holes * 3):
            if covered >= target_pixels:
                break
            hh = int(np.random.randint(min_hole_h, max_hole_h + 1))
            ww = int(np.random.randint(min_hole_w, max_hole_w + 1))
            y0 = int(np.random.randint(0, max(1, h - hh)))
            x0 = int(np.random.randint(0, max(1, w - ww)))
            y1 = min(h, y0 + hh)
            x1 = min(w, x0 + ww)

            fill_mode = np.random.choice(["zero", "mean", "noise"])
            if fill_mode == "zero":
                out[y0:y1, x0:x1] = 0.0
            elif fill_mode == "mean":
                out[y0:y1, x0:x1] = float(np.mean(out))
            else:
                out[y0:y1, x0:x1] = np.random.uniform(0.0, 1.0, size=(y1 - y0, x1 - x0)).astype(np.float32)

            new_area = 1.0 - mask[y0:y1, x0:x1]
            covered += int(new_area.sum())
            mask[y0:y1, x0:x1] = 1.0

        return np.clip(out, 0.0, 1.0), mask

    def _apply_pretext(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mode = str(self.pretext_cfg.get("mode", "identity")).lower()
        p = float(self.pretext_cfg.get("corruption_prob", 1.0))
        apply_in_eval = bool(self.pretext_cfg.get("apply_on_eval", True))
        should_apply = (self.is_train or apply_in_eval) and (np.random.uniform(0.0, 1.0) <= p)

        if mode == "identity" or not should_apply:
            return self._identity_corrupt(x)

        noise_cfg = self.pretext_cfg.get("noise", {})
        inpaint_cfg = self.pretext_cfg.get("inpaint", {})

        if mode == "denoise":
            x_noisy = self._add_noise(x, noise_cfg)
            mask = (np.abs(x_noisy - x) > 1e-4).astype(np.float32)
            return x_noisy, mask

        if mode == "inpaint":
            return self._apply_inpaint(x, inpaint_cfg)

        if mode == "hybrid":
            x_noisy = self._add_noise(x, noise_cfg)
            x_hybrid, mask_inpaint = self._apply_inpaint(x_noisy, inpaint_cfg)
            mask_noise = (np.abs(x_noisy - x) > 1e-4).astype(np.float32)
            mask = np.maximum(mask_inpaint, mask_noise).astype(np.float32)
            return x_hybrid, mask

        raise ValueError(f"pretext.mode invalido: {mode}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img = _load_gray(sample.image_path)
        tgt_h, tgt_w = self.image_size_hw
        img = cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        aug_out = self.augment(image=img)
        x_base = aug_out["image"].astype(np.float32)
        x, corr_mask = self._apply_pretext(np.clip(x_base, 0.0, 1.0))

        x = np.clip(x, 0.0, 1.0)
        y = img  # target sem augmentacao para reconstrucao

        x_t = torch.from_numpy(x).unsqueeze(0)
        y_t = torch.from_numpy(y).unsqueeze(0)
        m_t = torch.from_numpy(corr_mask.astype(np.float32)).unsqueeze(0)

        return {"x_before": y_t, "x": x_t, "y": y_t, "corruption_mask": m_t, "stem": sample.stem}
