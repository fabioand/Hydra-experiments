"""Pipeline de dados para Hydra U-Net MultiTask.

Suporta dois modos de target:
- on_the_fly: gera heatmap a partir do JSON em tempo de leitura
- precomputed: le stack64 (.npy) previamente gerado
"""

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

from hydra_multitask_model import CANONICAL_TEETH_32

DEFAULT_TARGET_SIZE_HW = (256, 256)
FIXED_WINDOW_NAMES = {"FULL", "LEFT", "CENTER", "RIGHT"}


@dataclass(frozen=True)
class HydraSample:
    stem: str
    image_path: Path
    json_path: Path
    mask_path: Path | None = None


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _canonical_index() -> Dict[str, int]:
    return {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}


def _index_from_teeth(teeth: Sequence[str]) -> Dict[str, int]:
    return {tooth: i for i, tooth in enumerate(teeth)}


def fixed_three_windows(width: int, height: int) -> Dict[str, List[int]]:
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be > 0")
    half = width // 2
    left = [0, 0, half, height]
    right = [width - half, 0, width, height]
    center_x1 = (width - half) // 2
    center = [center_x1, 0, center_x1 + half, height]
    return {"LEFT": left, "CENTER": center, "RIGHT": right}


def window_rect_for_image(window_name: str, width: int, height: int) -> List[int]:
    if window_name == "FULL":
        return [0, 0, width, height]
    rects = fixed_three_windows(width, height)
    if window_name not in rects:
        raise ValueError(f"window_name invalido: {window_name}")
    return rects[window_name]


def discover_samples(
    imgs_dir: Path,
    json_dir: Path,
    masks_dir: Path | None,
    source_mode: str = "on_the_fly",
) -> List[HydraSample]:
    if source_mode not in {"on_the_fly", "precomputed"}:
        raise ValueError(f"source_mode invalido: {source_mode}")

    imgs = {p.stem: p for p in imgs_dir.glob("*.jpg")}
    jsons = {p.stem: p for p in json_dir.glob("*.json")}

    if source_mode == "on_the_fly":
        common = sorted(set(imgs).intersection(jsons))
        return [HydraSample(stem=s, image_path=imgs[s], json_path=jsons[s], mask_path=None) for s in common]

    if masks_dir is None:
        raise ValueError("masks_dir obrigatorio quando source_mode='precomputed'")

    masks = {p.stem: p for p in masks_dir.glob("*.npy") if p.stem != "channel_order_64"}
    common = sorted(set(imgs).intersection(jsons).intersection(masks))
    return [
        HydraSample(stem=s, image_path=imgs[s], json_path=jsons[s], mask_path=masks[s])
        for s in common
    ]


def make_or_load_split(
    samples: Sequence[HydraSample],
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


def _load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
    return img


def _load_stack64_mmap(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3 or arr.shape[0] != 64:
        raise ValueError(f"Esperado stack64 (64,H,W). Recebido: {arr.shape} em {path}")
    return arr


def _build_kernel(sigma: float, radius: int) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma deve ser > 0")
    if radius < 1:
        raise ValueError("radius deve ser >= 1")

    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid)
    return np.exp(-((xx * xx + yy * yy) / (2.0 * sigma * sigma))).astype(np.float32)


def _apply_point_gaussian(
    heatmap: np.ndarray,
    kernel: np.ndarray,
    x: float,
    y: float,
    radius: int,
) -> None:
    px = int(round(x))
    py = int(round(y))

    h, w = heatmap.shape
    if px < 0 or px >= w or py < 0 or py >= h:
        return

    x0 = max(0, px - radius)
    y0 = max(0, py - radius)
    x1 = min(w, px + radius + 1)
    y1 = min(h, py + radius + 1)

    kx0 = x0 - (px - radius)
    ky0 = y0 - (py - radius)
    kx1 = kx0 + (x1 - x0)
    ky1 = ky0 + (y1 - y0)

    roi = heatmap[y0:y1, x0:x1]
    patch = kernel[ky0:ky1, kx0:kx1]
    np.maximum(roi, patch, out=roi)
    heatmap[py, px] = 1.0


def _load_points_by_label(json_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    data = load_json(json_path)
    points_by_label: Dict[str, List[Tuple[float, float]]] = {}

    for ann in data:
        label = str(ann.get("label", ""))
        pts = ann.get("pts", [])
        valid_pts: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid_pts.append((float(x), float(y)))
        if label not in points_by_label and valid_pts:
            points_by_label[label] = valid_pts

    return points_by_label


def _build_subset_stack_and_presence_from_json(
    json_path: Path,
    src_image_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    kernel: np.ndarray,
    radius: int,
    teeth_subset: Sequence[str],
    crop_xyxy: List[int],
    flip_horizontal: bool = False,
    label_remap: Dict[str, str] | None = None,
    label_remap_only_keys: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera target para subset de dentes, opcionalmente em crop e com flip."""
    tgt_h, tgt_w = target_hw
    n_teeth = len(teeth_subset)
    stack = np.zeros((2 * n_teeth, tgt_h, tgt_w), dtype=np.float32)
    presence = np.zeros((n_teeth,), dtype=np.float32)

    idx_map = _index_from_teeth(teeth_subset)
    x1, y1, x2, y2 = crop_xyxy
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    data = load_json(json_path)
    points_by_label: Dict[str, List[Tuple[float, float]]] = {}
    remap = label_remap or {}
    for ann in data:
        label_raw = str(ann.get("label", ""))
        if label_remap_only_keys and remap and label_raw not in remap:
            continue
        label = remap.get(label_raw, label_raw)
        if label not in idx_map:
            continue
        pts = ann.get("pts", [])
        valid_pts: List[Tuple[float, float]] = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            # Global -> local do crop
            lx = float(x) - float(x1)
            ly = float(y) - float(y1)
            # Mantemos apenas pontos dentro do crop.
            if lx < 0 or lx >= crop_w or ly < 0 or ly >= crop_h:
                continue
            if flip_horizontal:
                lx = float(crop_w - 1) - lx
            valid_pts.append((lx, ly))
        if label not in points_by_label and valid_pts:
            points_by_label[label] = valid_pts

    for tooth, pts in points_by_label.items():
        tooth_idx = idx_map[tooth]
        presence[tooth_idx] = 1.0
        if len(pts) >= 1:
            x_t, y_t = _project_point_to_target(pts[0][0], pts[0][1], (crop_h, crop_w), target_hw)
            _apply_point_gaussian(stack[2 * tooth_idx], kernel, x_t, y_t, radius)
        if len(pts) >= 2:
            x_t, y_t = _project_point_to_target(pts[1][0], pts[1][1], (crop_h, crop_w), target_hw)
            _apply_point_gaussian(stack[2 * tooth_idx + 1], kernel, x_t, y_t, radius)

    return stack, presence


def _project_point_to_target(
    x: float,
    y: float,
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> Tuple[float, float]:
    src_h, src_w = src_hw
    tgt_h, tgt_w = target_hw
    if src_h <= 1 or src_w <= 1:
        return 0.0, 0.0

    x_t = (x / float(src_w - 1)) * float(tgt_w - 1)
    y_t = (y / float(src_h - 1)) * float(tgt_h - 1)
    return x_t, y_t


def build_stack64_and_presence_from_json(
    json_path: Path,
    src_image_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    kernel: np.ndarray,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tgt_h, tgt_w = target_hw
    stack64 = np.zeros((64, tgt_h, tgt_w), dtype=np.float32)
    presence = np.zeros((32,), dtype=np.float32)

    points_by_label = _load_points_by_label(json_path)
    idx_map = _canonical_index()

    for label, pts in points_by_label.items():
        tooth_idx = idx_map.get(label)
        if tooth_idx is None:
            continue
        presence[tooth_idx] = 1.0

        if len(pts) >= 1:
            x_t, y_t = _project_point_to_target(pts[0][0], pts[0][1], src_image_hw, target_hw)
            _apply_point_gaussian(stack64[2 * tooth_idx], kernel, x_t, y_t, radius)
        if len(pts) >= 2:
            x_t, y_t = _project_point_to_target(pts[1][0], pts[1][1], src_image_hw, target_hw)
            _apply_point_gaussian(stack64[2 * tooth_idx + 1], kernel, x_t, y_t, radius)

    return stack64, presence


def derive_presence_from_stack64_np(stack64: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    presence = np.zeros((32,), dtype=np.float32)
    for i in range(32):
        c0 = 2 * i
        c1 = c0 + 1
        v0 = float(np.max(stack64[c0]))
        v1 = float(np.max(stack64[c1]))
        presence[i] = 1.0 if max(v0, v1) > eps else 0.0
    return presence


def preprocess_to_target(
    img_gray: np.ndarray,
    stack64: np.ndarray,
    target_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    target_h, target_w = target_hw
    x = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0

    y_channels: List[np.ndarray] = []
    for c in range(64):
        y_c = cv2.resize(stack64[c], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        y_channels.append(y_c)

    y = np.stack(y_channels, axis=0).astype(np.float32)
    np.clip(y, 0.0, 1.0, out=y)
    x = x[None, ...]
    return x, y


def preprocess_image_to_target(img_gray: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    x = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255.0
    return x[None, ...]


def build_geometric_augmentation(preset: Dict) -> A.Compose:
    aug = preset.get("augmentation", {})
    rot = float(aug.get("rotation_deg", 7.0))
    scale_range = aug.get("scale_range", [0.95, 1.05])
    trans = float(aug.get("translate_frac", 0.03))

    scale_low = float(scale_range[0])
    scale_high = float(scale_range[1])
    scale_limit = (scale_low - 1.0, scale_high - 1.0)

    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=trans,
                scale_limit=scale_limit,
                rotate_limit=rot,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=1.0,
            )
        ]
    )


def apply_intensity_and_noise(img01: np.ndarray, preset: Dict, rng: np.random.Generator) -> np.ndarray:
    out = img01.copy()
    aug = preset.get("augmentation", {})

    intensity = aug.get("intensity", {})
    bdelta = float(intensity.get("brightness_delta", 0.1))
    cmin, cmax = intensity.get("contrast_range", [0.9, 1.1])

    contrast = float(rng.uniform(float(cmin), float(cmax)))
    bright = float(rng.uniform(-bdelta, bdelta))
    out = np.clip(out * contrast + bright, 0.0, 1.0)

    noise_cfg = aug.get("noise", {})
    if not noise_cfg.get("enabled", True):
        return out

    overall_p = float(noise_cfg.get("overall_p", 0.35))
    if rng.random() > overall_p:
        return out

    patterns = noise_cfg.get("patterns", [])
    if not patterns:
        return out

    probs = np.array([float(p.get("p", 1.0)) for p in patterns], dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(patterns), p=probs))
    spec = patterns[idx]
    ntype = spec.get("type", "gaussian_additive")

    if ntype == "gaussian_additive":
        s0, s1 = spec.get("std_range", [0.005, 0.02])
        std = float(rng.uniform(float(s0), float(s1)))
        out = out + rng.normal(0.0, std, size=out.shape).astype(np.float32)
    elif ntype == "poisson":
        r0, r1 = spec.get("scale_range", [0.9, 1.1])
        scale = float(rng.uniform(float(r0), float(r1)))
        lam = max(10.0, 50.0 * scale)
        out = rng.poisson(np.clip(out, 0.0, 1.0) * lam).astype(np.float32) / lam
    elif ntype == "speckle_multiplicative":
        s0, s1 = spec.get("std_range", [0.003, 0.012])
        std = float(rng.uniform(float(s0), float(s1)))
        out = out + out * rng.normal(0.0, std, size=out.shape).astype(np.float32)

    np.clip(out, 0.0, 1.0, out=out)
    return out


class HydraTeethDataset(Dataset):
    """Dataset multitarefa Hydra.

    Retorna dict com:
    - x: (1,256,256)
    - y_heatmap: (64,256,256)
    - y_presence: (32,)
    - x_before / y_before: tensores sem augmentacao (para visual debug)
    """

    def __init__(
        self,
        samples: Sequence[HydraSample],
        preset: Dict,
        augment: bool,
        source_mode: str = "on_the_fly",
        seed: int = 123,
        teeth_subset: Sequence[str] | None = None,
        window_name: str = "FULL",
        flip_horizontal: bool = False,
        label_remap: Dict[str, str] | None = None,
        label_remap_only_keys: bool = False,
    ):
        if source_mode not in {"on_the_fly", "precomputed"}:
            raise ValueError(f"source_mode invalido: {source_mode}")
        if window_name not in FIXED_WINDOW_NAMES:
            raise ValueError(f"window_name invalido: {window_name}")

        self.samples = list(samples)
        self.preset = preset
        self.augment = augment
        self.source_mode = source_mode
        self.window_name = window_name
        self.flip_horizontal = bool(flip_horizontal)
        self.label_remap = dict(label_remap or {})
        self.label_remap_only_keys = bool(label_remap_only_keys)
        self.teeth_subset = list(teeth_subset) if teeth_subset is not None else list(CANONICAL_TEETH_32)
        if not self.teeth_subset:
            raise ValueError("teeth_subset vazio")
        self._subset_set = set(self.teeth_subset)
        self.heatmap_channels = 2 * len(self.teeth_subset)
        self.presence_channels = len(self.teeth_subset)
        self.geo_aug = build_geometric_augmentation(preset)
        self.rng = np.random.default_rng(seed)

        input_size = preset.get("input", {}).get("size", [DEFAULT_TARGET_SIZE_HW[0], DEFAULT_TARGET_SIZE_HW[1]])
        self.target_hw = (int(input_size[0]), int(input_size[1]))

        heatmap_cfg = preset.get("heatmap_generation", {})
        sigma = float(heatmap_cfg.get("sigma_px_target", heatmap_cfg.get("sigma_px_original", 7.0)))
        self.radius = int(np.ceil(3.0 * sigma))
        self.kernel = _build_kernel(sigma=sigma, radius=self.radius)

        if self.source_mode == "precomputed":
            # Modo precomputed atual so suporta stack64 inteiro sem crop/flip/remap/subset.
            incompatible = (
                self.window_name != "FULL"
                or self.flip_horizontal
                or bool(self.label_remap)
                or self.teeth_subset != list(CANONICAL_TEETH_32)
            )
            if incompatible:
                raise ValueError(
                    "source_mode='precomputed' nao suporta subset/crop/flip/remap. "
                    "Use source_mode='on_the_fly' para o modo multi-ROI."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_gray = _load_gray_image(sample.image_path)
        h, w = img_gray.shape
        rect = window_rect_for_image(self.window_name, w, h)
        x1, y1, x2, y2 = rect
        img_crop = img_gray[y1:y2, x1:x2]
        if img_crop.size == 0:
            raise RuntimeError(f"Crop vazio para {sample.stem}: rect={rect} img_shape={img_gray.shape}")
        if self.flip_horizontal:
            img_crop = np.ascontiguousarray(np.fliplr(img_crop))

        if self.source_mode == "on_the_fly":
            y_stack, y_presence = _build_subset_stack_and_presence_from_json(
                json_path=sample.json_path,
                src_image_hw=img_crop.shape,
                target_hw=self.target_hw,
                kernel=self.kernel,
                radius=self.radius,
                teeth_subset=self.teeth_subset,
                crop_xyxy=rect,
                flip_horizontal=self.flip_horizontal,
                label_remap=self.label_remap,
                label_remap_only_keys=self.label_remap_only_keys,
            )
            x_before = preprocess_image_to_target(img_crop, target_hw=self.target_hw)
            y_before = y_stack
        else:
            if sample.mask_path is None:
                raise RuntimeError("mask_path ausente para source_mode='precomputed'")
            stack64 = _load_stack64_mmap(sample.mask_path)
            y_presence = derive_presence_from_stack64_np(stack64)
            x_before, y_before = preprocess_to_target(img_crop, stack64, target_hw=self.target_hw)

        x_after = x_before[0]
        y_after = y_before

        if self.augment:
            y_hwc = np.transpose(y_before, (1, 2, 0))
            aug_out = self.geo_aug(image=x_after.astype(np.float32), mask=y_hwc.astype(np.float32))
            x_after = aug_out["image"].astype(np.float32)
            y_after = np.transpose(aug_out["mask"], (2, 0, 1)).astype(np.float32)
            x_after = apply_intensity_and_noise(x_after, self.preset, self.rng)
            np.clip(y_after, 0.0, 1.0, out=y_after)

        x_after = x_after[None, ...]

        return {
            "stem": sample.stem,
            "x": torch.from_numpy(x_after.astype(np.float32, copy=False)),
            "y_heatmap": torch.from_numpy(y_after.astype(np.float32, copy=False)),
            "y_presence": torch.from_numpy(y_presence.astype(np.float32, copy=False)),
            "x_before": torch.from_numpy(x_before.astype(np.float32, copy=False)),
            "y_before": torch.from_numpy(y_before.astype(np.float32, copy=False)),
            "roi_rect": torch.tensor(rect, dtype=torch.int32),
            "window_name": self.window_name,
            "flip_horizontal": torch.tensor(1 if self.flip_horizontal else 0, dtype=torch.int32),
            "label_remap_only_keys": torch.tensor(1 if self.label_remap_only_keys else 0, dtype=torch.int32),
        }


def quadrant_for_tooth(tooth: str) -> str:
    if tooth.startswith("1"):
        return "Q1_superior_direito"
    if tooth.startswith("2"):
        return "Q2_superior_esquerdo"
    if tooth.startswith("3"):
        return "Q3_inferior_esquerdo"
    return "Q4_inferior_direito"


def teeth_and_quadrants() -> List[Tuple[str, str]]:
    return [(t, quadrant_for_tooth(t)) for t in CANONICAL_TEETH_32]
