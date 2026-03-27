#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CANONICAL_TEETH_32 = [
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


def build_kernel(sigma: float, radius: int) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if radius < 1:
        raise ValueError("radius must be >= 1")

    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid)
    kernel = np.exp(-((xx * xx + yy * yy) / (2.0 * sigma * sigma))).astype(np.float32)
    return kernel


def apply_point_gaussian(
    heatmap: np.ndarray,
    kernel: np.ndarray,
    x: float,
    y: float,
    radius: int,
    blend: str,
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

    if blend == "max":
        np.maximum(roi, patch, out=roi)
    else:
        roi += patch

    # Garante valor 1 no pixel da coordenada do ponto.
    heatmap[py, px] = 1.0


def load_points(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for annotation in data:
        pts = annotation.get("pts", [])
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            yield float(x), float(y)


def load_points_by_label(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    points_by_label = {}
    for annotation in data:
        label = str(annotation.get("label", ""))
        pts = annotation.get("pts", [])
        valid_pts = []
        for pt in pts:
            x = pt.get("x")
            y = pt.get("y")
            if x is None or y is None:
                continue
            valid_pts.append((float(x), float(y)))
        if label not in points_by_label and valid_pts:
            points_by_label[label] = valid_pts
    return points_by_label


def get_canonical_channels_64():
    channels = []
    for label in CANONICAL_TEETH_32:
        channels.append((label, 0, f"{label}_p1"))
        channels.append((label, 1, f"{label}_p2"))
    return channels


def is_valid_existing_output(path: Path, output_mode: str) -> bool:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return False

    if output_mode == "single":
        return arr.ndim == 2
    return arr.ndim == 3 and arr.shape[0] == 64


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera mapa grayscale float32 [0,1] com fundo preto e blobs gaussianos "
            "nos pontos dos JSONs."
        )
    )
    parser.add_argument("--imgs-dir", type=Path, default=Path("longoeixo/imgs"))
    parser.add_argument("--json-dir", type=Path, default=Path("longoeixo/data_longoeixo"))
    parser.add_argument("--out-dir", type=Path, default=Path("longoeixo/gaussian_maps"))
    parser.add_argument(
        "--output-mode",
        choices=["single", "stack64"],
        default="single",
        help="single: 1 mapa por imagem; stack64: tensor (64,H,W) em ordem canonica fixa",
    )
    parser.add_argument("--sigma", type=float, default=7.0, help="Desvio padrao da gaussiana em pixels")
    parser.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Raio de renderizacao em pixels (default: ceil(3*sigma))",
    )
    parser.add_argument(
        "--blend",
        choices=["max", "sum"],
        default="max",
        help="Como combinar gaussianas sobrepostas",
    )
    parser.add_argument(
        "--save-preview-png",
        action="store_true",
        help="Salva uma visualizacao PNG 16-bit (0..65535)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Numero maximo de amostras (JSON/JPG) a processar; default processa todas",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Mostra o mapa em janela OpenCV e espera tecla para avancar",
    )
    parser.add_argument(
        "--save-overlay-png",
        action="store_true",
        help="Salva imagem colorida com o JPG original + mapa gaussiano em vermelho",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Intensidade da sobreposicao vermelha (0 a 1)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pula amostras cujo .npy de saida ja existe",
    )

    args = parser.parse_args()

    radius = args.radius if args.radius is not None else int(np.ceil(3.0 * args.sigma))
    kernel = build_kernel(args.sigma, radius)
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay-alpha deve estar no intervalo [0,1]")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out_dir / "preview_png"
    overlay_dir = args.out_dir / "overlay_png"
    if args.save_preview_png:
        preview_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay_png:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    channel_spec = get_canonical_channels_64()
    if args.output_mode == "stack64":
        channel_map_file = args.out_dir / "channel_order_64.txt"
        with channel_map_file.open("w", encoding="utf-8") as f:
            for i, (_, _, name) in enumerate(channel_spec):
                f.write(f"{i:02d} {name}\n")

    json_files = sorted(args.json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Nenhum JSON encontrado em: {args.json_dir}")
    if args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("--num-samples deve ser >= 1")
        json_files = json_files[: args.num_samples]

    generated = 0
    skipped = 0

    stop_requested = False

    for json_path in tqdm(json_files, desc="Gerando mapas"):
        stem = json_path.stem
        img_path = args.imgs_dir / f"{stem}.jpg"
        out_npy_path = args.out_dir / f"{stem}.npy"

        if args.skip_existing and out_npy_path.exists():
            if is_valid_existing_output(out_npy_path, args.output_mode):
                skipped += 1
                continue

        if not img_path.exists():
            skipped += 1
            continue

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            skipped += 1
            continue

        h, w = img_gray.shape
        if args.output_mode == "single":
            output_arr = np.zeros((h, w), dtype=np.float32)
            for x, y in load_points(json_path):
                apply_point_gaussian(output_arr, kernel, x, y, radius, args.blend)
            if args.blend == "sum":
                np.clip(output_arr, 0.0, 1.0, out=output_arr)
            heatmap_for_view = output_arr
        else:
            output_arr = np.zeros((64, h, w), dtype=np.float32)
            points_by_label = load_points_by_label(json_path)
            for ch_idx, (label, point_idx, _) in enumerate(channel_spec):
                pts = points_by_label.get(label, [])
                if point_idx < len(pts):
                    x, y = pts[point_idx]
                    apply_point_gaussian(
                        output_arr[ch_idx], kernel, x, y, radius, blend="max"
                    )
            heatmap_for_view = np.max(output_arr, axis=0)

        np.save(out_npy_path, output_arr)

        if args.save_preview_png:
            preview = np.clip(heatmap_for_view * 65535.0, 0.0, 65535.0).astype(np.uint16)
            cv2.imwrite(str(preview_dir / f"{stem}.png"), preview)

        if args.save_overlay_png:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                skipped += 1
                continue
            red_layer = np.zeros_like(img_bgr, dtype=np.uint8)
            red_layer[:, :, 2] = np.clip(heatmap_for_view * 255.0, 0.0, 255.0).astype(np.uint8)
            overlay = cv2.addWeighted(img_bgr, 1.0, red_layer, args.overlay_alpha, 0.0)
            cv2.imwrite(str(overlay_dir / f"{stem}.png"), overlay)

        if args.show_window:
            view = np.clip(heatmap_for_view * 255.0, 0.0, 255.0).astype(np.uint8)
            cv2.imshow("Gaussian Map", view)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                stop_requested = True

        generated += 1
        if stop_requested:
            break

    if args.show_window:
        cv2.destroyAllWindows()

    print(f"Mapas gerados: {generated}")
    print(f"Arquivos pulados: {skipped}")
    print(f"Saida: {args.out_dir}")


if __name__ == "__main__":
    main()
