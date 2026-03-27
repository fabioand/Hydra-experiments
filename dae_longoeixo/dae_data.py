"""Pipeline de dados para Denoising Autoencoder de coordenadas de long-eixo.

Objetivo:
- Usar apenas exames com denticao completa (32 dentes canônicos com 2 pontos cada).
- Gerar entrada corrompida via knockout de dentes (pontos -> 0,0).
- Usar alvo sempre completo (coordenadas normalizadas de todos os dentes).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from hydra_multitask_model import CANONICAL_TEETH_32

COORDS_DIM = 128  # 64 pontos x (x,y)
POINTS_DIM = 64
THIRD_MOLARS = {"18", "28", "38", "48"}
REQUIRED_UPTO_SECOND = [t for t in CANONICAL_TEETH_32 if t not in THIRD_MOLARS]
TOOTH_TO_INDEX = {t: i for i, t in enumerate(CANONICAL_TEETH_32)}


@dataclass(frozen=True)
class DaeSample:
    stem: str
    image_path: Path
    json_path: Path
    coords_128: np.ndarray  # float32 normalizado [0,1]
    gt_available_mask_32: np.ndarray  # 1 onde ha 2 pontos validos no GT
    num_present_teeth: int
    alveolar_curves_flat: np.ndarray | None = None  # float32 [4*n_curve_points]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_image_hw(path: Path) -> Tuple[int, int]:
    """Lê apenas dimensões da imagem (h, w) sem decodificar pixels completos."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(h), int(w)
    except Exception:
        # Fallback robusto para formatos corrompidos/incomuns.
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Nao foi possivel ler imagem: {path}")
        h, w = img.shape[:2]
        return int(h), int(w)


def _extract_points_by_label(json_path: Path) -> Dict[str, List[Tuple[float, float]]]:
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


def _normalize_xy(x: float, y: float, src_hw: Tuple[int, int]) -> Tuple[float, float]:
    src_h, src_w = src_hw
    if src_h <= 1 or src_w <= 1:
        return 0.0, 0.0

    x_n = x / float(src_w - 1)
    y_n = y / float(src_h - 1)
    return float(np.clip(x_n, 0.0, 1.0)), float(np.clip(y_n, 0.0, 1.0))


def _extract_partial_coords_128_and_mask(
    json_path: Path, src_hw: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    points_by_label = _extract_points_by_label(json_path)

    coords = np.zeros((COORDS_DIM,), dtype=np.float32)
    mask = np.zeros((32,), dtype=np.float32)
    for i, tooth in enumerate(CANONICAL_TEETH_32):
        pts = points_by_label.get(tooth)
        if pts is None or len(pts) < 2:
            continue
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1n, y1n = _normalize_xy(x1, y1, src_hw)
        x2n, y2n = _normalize_xy(x2, y2, src_hw)
        base = 4 * i
        coords[base + 0] = x1n
        coords[base + 1] = y1n
        coords[base + 2] = x2n
        coords[base + 3] = y2n
        mask[i] = 1.0
    return coords, mask


def _extract_alveolar_curves_flat(
    curves_json_path: Path,
    n_curve_points: int,
    src_hw: Tuple[int, int],
) -> np.ndarray | None:
    """Carrega RebAlvSup/RebAlvInf normalizadas e concatena em vetor fixo.

    Formato esperado: JSON com chaves RebAlvSup e RebAlvInf (cada uma Nx2).
    """
    try:
        data = load_json(curves_json_path)
    except Exception:
        return None

    sup = data.get("RebAlvSup")
    inf = data.get("RebAlvInf")
    if not isinstance(sup, list) or not isinstance(inf, list):
        return None

    sup_arr = np.asarray(sup, dtype=np.float32)
    inf_arr = np.asarray(inf, dtype=np.float32)
    expected = (int(n_curve_points), 2)
    if tuple(sup_arr.shape) != expected or tuple(inf_arr.shape) != expected:
        return None

    # Se vier em pixels (tipicamente valores > 1), normaliza para [0,1] pela imagem.
    if float(np.max(np.abs(sup_arr))) > 1.5 or float(np.max(np.abs(inf_arr))) > 1.5:
        h, w = src_hw
        if h > 1 and w > 1:
            sup_arr[:, 0] = np.clip(sup_arr[:, 0] / float(w - 1), 0.0, 1.0)
            sup_arr[:, 1] = np.clip(sup_arr[:, 1] / float(h - 1), 0.0, 1.0)
            inf_arr[:, 0] = np.clip(inf_arr[:, 0] / float(w - 1), 0.0, 1.0)
            inf_arr[:, 1] = np.clip(inf_arr[:, 1] / float(h - 1), 0.0, 1.0)

    flat = np.concatenate([sup_arr.reshape(-1), inf_arr.reshape(-1)], axis=0).astype(np.float32, copy=False)
    return flat


def discover_samples(
    imgs_dir: Path,
    json_dir: Path,
    sample_filter: str = "full_32_only",
    min_teeth_present: int = 1,
    progress_interval: int = 0,
    curves_json_dir: Path | None = None,
    n_curve_points: int = 128,
    require_curves: bool = False,
) -> List[DaeSample]:
    if sample_filter not in {"full_32_only", "upto_second_molars", "any_with_min_teeth"}:
        raise ValueError(f"sample_filter invalido: {sample_filter}")

    imgs = {p.stem: p for p in imgs_dir.glob("*.jpg")}
    jsons = {p.stem: p for p in json_dir.glob("*.json")}
    stems = sorted(set(imgs).intersection(jsons))

    out: List[DaeSample] = []
    total = len(stems)
    for scanned, stem in enumerate(stems, start=1):
        image_path = imgs[stem]
        json_path = jsons[stem]

        src_hw = _load_image_hw(image_path)
        coords_128, gt_available_mask_32 = _extract_partial_coords_128_and_mask(
            json_path=json_path, src_hw=src_hw
        )
        num_present_teeth = int(gt_available_mask_32.sum())

        if progress_interval > 0:
            if scanned % progress_interval == 0 or scanned == total:
                kept = len(out)
                print(
                    "[DISCOVER] scanned={}/{} kept={} sample_filter={}".format(
                        scanned,
                        total,
                        kept,
                        sample_filter,
                    )
                )

        if sample_filter == "full_32_only" and num_present_teeth != 32:
            continue
        if sample_filter == "upto_second_molars":
            ok = True
            for tooth in REQUIRED_UPTO_SECOND:
                idx = TOOTH_TO_INDEX[tooth]
                if gt_available_mask_32[idx] < 0.5:
                    ok = False
                    break
            if not ok:
                continue
        if sample_filter == "any_with_min_teeth" and num_present_teeth < int(min_teeth_present):
            continue

        alveolar_curves_flat = None
        if curves_json_dir is not None:
            curves_json_path = curves_json_dir / f"{stem}.json"
            if curves_json_path.exists():
                alveolar_curves_flat = _extract_alveolar_curves_flat(
                    curves_json_path=curves_json_path,
                    n_curve_points=int(n_curve_points),
                    src_hw=src_hw,
                )
            if require_curves and alveolar_curves_flat is None:
                continue

        out.append(
            DaeSample(
                stem=stem,
                image_path=image_path,
                json_path=json_path,
                coords_128=coords_128,
                gt_available_mask_32=gt_available_mask_32,
                num_present_teeth=num_present_teeth,
                alveolar_curves_flat=alveolar_curves_flat,
            )
        )

    return out


def discover_complete_samples(
    imgs_dir: Path,
    json_dir: Path,
) -> List[DaeSample]:
    return discover_samples(
        imgs_dir=imgs_dir,
        json_dir=json_dir,
        sample_filter="full_32_only",
        min_teeth_present=32,
    )


def make_or_load_split(
    samples: Sequence[DaeSample],
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


def sample_knockout_mask(
    rng: np.random.Generator,
    sample_p: float,
    min_teeth: int,
    max_teeth: int,
    selectable_teeth_mask_32: np.ndarray | None = None,
) -> np.ndarray:
    sample_p = float(np.clip(sample_p, 0.0, 1.0))
    if rng.random() > sample_p:
        return np.zeros((32,), dtype=np.float32)

    if selectable_teeth_mask_32 is None:
        selectable_idx = np.arange(32, dtype=np.int64)
    else:
        if selectable_teeth_mask_32.shape != (32,):
            raise ValueError(f"selectable_teeth_mask_32 invalido: {selectable_teeth_mask_32.shape}")
        selectable_idx = np.where(selectable_teeth_mask_32 > 0.5)[0]
    if selectable_idx.size == 0:
        return np.zeros((32,), dtype=np.float32)

    max_allowed = int(selectable_idx.size)
    min_teeth = int(np.clip(min_teeth, 0, max_allowed))
    max_teeth = int(np.clip(max_teeth, 0, max_allowed))
    if max_teeth < min_teeth:
        max_teeth = min_teeth

    if max_teeth == 0:
        return np.zeros((32,), dtype=np.float32)

    k = int(rng.integers(min_teeth, max_teeth + 1))
    if k <= 0:
        return np.zeros((32,), dtype=np.float32)

    mask = np.zeros((32,), dtype=np.float32)
    chosen = rng.choice(selectable_idx, size=k, replace=False)
    mask[chosen] = 1.0
    return mask


def build_noisy_input(
    clean_coords_128: np.ndarray,
    knocked_teeth_mask_32: np.ndarray,
    include_point_mask_in_input: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if clean_coords_128.shape != (COORDS_DIM,):
        raise ValueError(f"clean_coords_128 invalido: {clean_coords_128.shape}")
    if knocked_teeth_mask_32.shape != (32,):
        raise ValueError(f"knocked_teeth_mask_32 invalido: {knocked_teeth_mask_32.shape}")

    noisy = clean_coords_128.copy()
    point_keep_64 = np.ones((POINTS_DIM,), dtype=np.float32)

    for i in range(32):
        if knocked_teeth_mask_32[i] < 0.5:
            continue
        base = 4 * i
        noisy[base : base + 4] = 0.0

        p0 = 2 * i
        p1 = p0 + 1
        point_keep_64[p0] = 0.0
        point_keep_64[p1] = 0.0

    if include_point_mask_in_input:
        x_input = np.concatenate([noisy, point_keep_64], axis=0).astype(np.float32, copy=False)
    else:
        x_input = noisy.astype(np.float32, copy=False)

    return x_input, noisy.astype(np.float32, copy=False), point_keep_64


def apply_horizontal_jitter(
    clean_coords_128: np.ndarray,
    knocked_teeth_mask_32: np.ndarray,
    gt_available_mask_32: np.ndarray,
    jitter_cfg: Dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Aplica jitter horizontal leve por dente (x1/x2) para simular migração dentária.

    Observações:
    - operação no espaço normalizado [0,1]
    - altera x1 e x2 do mesmo dente com o mesmo delta
    - opcionalmente aumenta amplitude em vizinhos de dentes nocauteados
    """
    if not bool(jitter_cfg.get("enabled", False)):
        return clean_coords_128

    p = float(jitter_cfg.get("p", 0.35))
    if rng.random() > p:
        return clean_coords_128

    r = jitter_cfg.get("range", [-0.015, 0.015])
    dx_min = float(r[0])
    dx_max = float(r[1])
    if dx_max < dx_min:
        dx_min, dx_max = dx_max, dx_min

    neighbor_boost = float(jitter_cfg.get("neighbor_boost", 1.2))
    neighbor_radius = int(jitter_cfg.get("neighbor_radius", 2))
    skip_knocked = bool(jitter_cfg.get("skip_knocked_teeth", True))

    coords = clean_coords_128.reshape(32, 4).copy()
    knocked_idx = np.where(knocked_teeth_mask_32 > 0.5)[0]

    for i in range(32):
        if gt_available_mask_32[i] < 0.5:
            continue
        if skip_knocked and knocked_teeth_mask_32[i] > 0.5:
            continue

        amp = 1.0
        if knocked_idx.size > 0 and neighbor_radius > 0:
            nearest = int(np.min(np.abs(knocked_idx - i)))
            if 0 < nearest <= neighbor_radius:
                frac = (neighbor_radius - nearest + 1) / float(neighbor_radius)
                amp += neighbor_boost * frac

        dx = float(rng.uniform(dx_min, dx_max)) * amp
        coords[i, 0] = np.clip(coords[i, 0] + dx, 0.0, 1.0)
        coords[i, 2] = np.clip(coords[i, 2] + dx, 0.0, 1.0)

    return coords.reshape(COORDS_DIM).astype(np.float32, copy=False)


class DaeCoordinateDataset(Dataset):
    """Dataset de coordenadas para DAE.

    Retorna dict com:
    - stem
    - x_input: vetor de entrada (coords ruidosas [+ mask opcional])
    - x_noisy_coords: coords ruidosas puras (128)
    - y_coords: alvo limpo (128)
    - knocked_teeth_mask: vetor (32) com 1 para dente nocauteado
    - point_keep_mask: vetor (64) com 1 para ponto preservado
    """

    def __init__(
        self,
        samples: Sequence[DaeSample],
        preset: Dict,
        stage: str,
        seed: int = 123,
    ):
        if stage not in {"train", "val", "eval"}:
            raise ValueError(f"stage invalido: {stage}")

        self.samples = list(samples)
        self.preset = preset
        self.stage = stage
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.eval_seed_offset = 0

        model_cfg = preset.get("model", {})
        self.include_point_mask_in_input = bool(model_cfg.get("include_point_mask_in_input", True))
        self.include_alveolar_curves_in_input = bool(model_cfg.get("include_alveolar_curves_in_input", False))
        self.reconstruct_target = str(model_cfg.get("reconstruct_target", "teeth_only"))
        if self.reconstruct_target not in {"teeth_only", "teeth_plus_curves"}:
            raise ValueError(f"reconstruct_target invalido: {self.reconstruct_target}")
        self.n_curve_points = int(model_cfg.get("n_curve_points", 128))
        if self.n_curve_points <= 0:
            raise ValueError("n_curve_points deve ser > 0")
        self.curves_dim = int(4 * self.n_curve_points)  # sup Nx2 + inf Nx2

        ko_cfg = preset.get("knockout", {})
        stage_cfg = ko_cfg.get(stage, ko_cfg.get("train", {}))

        self.knockout_sample_p = float(stage_cfg.get("sample_p", 1.0))
        self.knockout_min_teeth = int(stage_cfg.get("min_teeth", 2))
        self.knockout_max_teeth = int(stage_cfg.get("max_teeth", 8))
        self.knockout_deterministic = bool(stage_cfg.get("deterministic", stage != "train"))
        self.horizontal_jitter_cfg = preset.get("augmentation", {}).get("horizontal_jitter", {})

    @property
    def input_dim(self) -> int:
        d = COORDS_DIM + (POINTS_DIM if self.include_point_mask_in_input else 0)
        if self.include_alveolar_curves_in_input:
            d += self.curves_dim
        return d

    @property
    def output_dim(self) -> int:
        if self.reconstruct_target == "teeth_plus_curves":
            return COORDS_DIM + self.curves_dim
        return COORDS_DIM

    def set_eval_seed_offset(self, seed_offset: int) -> None:
        self.eval_seed_offset = int(seed_offset)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        if self.knockout_deterministic:
            return np.random.default_rng(self.seed + self.eval_seed_offset + idx * 1009)
        return self.rng

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        clean = sample.coords_128.astype(np.float32, copy=False)
        gt_available_32 = sample.gt_available_mask_32.astype(np.float32, copy=False)

        rng = self._rng_for_index(idx)
        knocked_teeth_mask = sample_knockout_mask(
            rng=rng,
            sample_p=self.knockout_sample_p,
            min_teeth=self.knockout_min_teeth,
            max_teeth=self.knockout_max_teeth,
            selectable_teeth_mask_32=gt_available_32,
        )

        if self.stage == "train":
            clean = apply_horizontal_jitter(
                clean_coords_128=clean,
                knocked_teeth_mask_32=knocked_teeth_mask,
                gt_available_mask_32=gt_available_32,
                jitter_cfg=self.horizontal_jitter_cfg,
                rng=rng,
            )

        x_input, x_noisy, point_keep_64 = build_noisy_input(
            clean_coords_128=clean,
            knocked_teeth_mask_32=knocked_teeth_mask,
            include_point_mask_in_input=self.include_point_mask_in_input,
        )

        curves_flat = sample.alveolar_curves_flat
        if curves_flat is not None and curves_flat.shape != (self.curves_dim,):
            curves_flat = None
        if curves_flat is None:
            curves_flat = np.zeros((self.curves_dim,), dtype=np.float32)
            curves_available = 0.0
        else:
            curves_flat = curves_flat.astype(np.float32, copy=False)
            curves_available = 1.0

        if self.include_alveolar_curves_in_input:
            x_input = np.concatenate([x_input, curves_flat], axis=0).astype(np.float32, copy=False)

        return {
            "stem": sample.stem,
            "x_input": torch.from_numpy(x_input.astype(np.float32, copy=False)),
            "x_noisy_coords": torch.from_numpy(x_noisy.astype(np.float32, copy=False)),
            "y_coords": torch.from_numpy(clean.astype(np.float32, copy=False)),
            "y_alveolar_curves": torch.from_numpy(curves_flat.astype(np.float32, copy=False)),
            "y_curves_available_mask": torch.tensor(curves_available, dtype=torch.float32),
            "knocked_teeth_mask": torch.from_numpy(knocked_teeth_mask.astype(np.float32, copy=False)),
            "point_keep_mask": torch.from_numpy(point_keep_64.astype(np.float32, copy=False)),
            "gt_available_teeth_mask": torch.from_numpy(gt_available_32.astype(np.float32, copy=False)),
            "gt_available_points_mask": torch.from_numpy(
                gt_available_32.repeat(2).astype(np.float32, copy=False)
            ),
            "gt_available_xy_mask": torch.from_numpy(
                gt_available_32.repeat(4).astype(np.float32, copy=False)
            ),
        }


__all__ = [
    "COORDS_DIM",
    "POINTS_DIM",
    "DaeSample",
    "DaeCoordinateDataset",
    "discover_samples",
    "discover_complete_samples",
    "make_or_load_split",
    "load_json",
    "save_json",
]
