#!/usr/bin/env python3
"""Avaliacao focada em presenca/ausencia e ranking de maiores erros por radiografia."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2

from hydra_data import HydraTeethDataset, discover_samples, load_json, make_or_load_split
from hydra_multitask_model import CANONICAL_TEETH_32, HydraUNetMultiTask
from dashboard_registry import rel_to_experiment, register_record
from longoeixo.scripts.multiroi_composed_inference import (
    DEFAULT_CENTER_CKPT,
    DEFAULT_LATERAL_CKPT,
    infer_multiroi_from_image,
    latest_best_ckpt,
    load_multiroi_models,
    resolve_path as resolve_multiroi_path,
)

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_run_name(output_base_dir: Path) -> str | None:
    latest_file = output_base_dir / "latest_run.txt"
    if latest_file.exists():
        name = latest_file.read_text(encoding="utf-8").strip()
        if name:
            return name

    runs_dir = output_base_dir / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def _gt_presence_from_json(json_path: Path) -> np.ndarray:
    arr = np.zeros((32,), dtype=np.int32)
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}
    data = load_json(json_path)
    for ann in data:
        label = str(ann.get("label", ""))
        idx = idx_map.get(label)
        if idx is None:
            continue
        pts = ann.get("pts", [])
        if isinstance(pts, list) and len(pts) > 0:
            arr[idx] = 1
    return arr


def _resolve_stems_for_split(split: Dict[str, List[str]], split_name: str) -> List[str]:
    if split_name == "train":
        return list(split["train"])
    if split_name == "val":
        return list(split["val"])
    if split_name == "test":
        if "test" not in split:
            raise RuntimeError("Split 'test' indisponivel no arquivo de split atual.")
        return list(split["test"])
    return list(split["train"] + split["val"] + split.get("test", []))


def _multiroi_scores_for_image(image_path: Path, models, infer_threshold: float) -> np.ndarray:
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {image_path}")
    result = infer_multiroi_from_image(image_gray=image_gray, models=models, threshold=float(infer_threshold))
    score_floor = -1e6
    scores = np.full((32,), score_floor, dtype=np.float64)
    idx_map = {tooth: i for i, tooth in enumerate(CANONICAL_TEETH_32)}
    for pred in result.predictions:
        idx = idx_map.get(pred.tooth)
        if idx is None:
            continue
        sc = float(pred.score)
        if sc >= scores[idx]:
            scores[idx] = sc
    return scores


def _presence_bce_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Usa BCE em probabilidades para manter o ranking compatível no modo MultiROI.
    p = np.clip(y_score.astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydra eval focado em erros de presenca")
    parser.add_argument("--config", type=Path, default=Path("hydra_train_config.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Nome da run; se omitido usa latest_run.txt")
    parser.add_argument(
        "--inference-source",
        type=str,
        default="model",
        choices=["model", "multiroi_model"],
        help="Fonte das predicoes: checkpoint local (model) ou inferencia MultiROI.",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--top-k", type=int, default=300, help="Quantidade de radiografias no ranking final")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold de presenca; default vem da config")
    parser.add_argument("--fn-weight", type=float, default=3.0)
    parser.add_argument("--fp-weight", type=float, default=2.0)
    parser.add_argument("--bce-weight", type=float, default=10.0)
    parser.add_argument("--multiroi-center-ckpt", type=Path, default=DEFAULT_CENTER_CKPT)
    parser.add_argument("--multiroi-lateral-ckpt", type=Path, default=DEFAULT_LATERAL_CKPT)
    parser.add_argument(
        "--multiroi-center-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/center24_sharedflip_nopres_absenthm1"),
        help="Output dir base da center para resolver latest best.ckpt quando --multiroi-use-latest-from-output-dirs.",
    )
    parser.add_argument(
        "--multiroi-lateral-output-dir",
        type=Path,
        default=Path("longoeixo/experiments/hydra_roi_fixed_shared_lateral/lateral_shared20_nopres_absenthm1"),
        help="Output dir base da lateral para resolver latest best.ckpt quando --multiroi-use-latest-from-output-dirs.",
    )
    parser.add_argument(
        "--multiroi-use-latest-from-output-dirs",
        action="store_true",
        help="Se ligado, ignora --multiroi-center-ckpt/--multiroi-lateral-ckpt e usa latest_run.txt + best.ckpt.",
    )
    parser.add_argument(
        "--multiroi-infer-threshold",
        type=float,
        default=-1e6,
        help="Threshold interno da lib MultiROI; default muito baixo para sempre coletar score por dente.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_json(_resolve_path(repo_root, str(args.config)))

    imgs_dir = _resolve_path(repo_root, cfg["paths"]["imgs_dir"])
    json_dir = _resolve_path(repo_root, cfg["paths"]["json_dir"])
    masks_dir_cfg = cfg["paths"].get("masks_dir")
    masks_dir = _resolve_path(repo_root, masks_dir_cfg) if masks_dir_cfg else None
    split_path = _resolve_path(repo_root, cfg["paths"]["splits_path"])
    output_base_dir = _resolve_path(repo_root, cfg["paths"]["output_dir"])
    preset_path = _resolve_path(repo_root, cfg["paths"]["preset_path"])

    source_mode = str(cfg.get("data", {}).get("source_mode", "on_the_fly"))
    samples = discover_samples(imgs_dir=imgs_dir, json_dir=json_dir, masks_dir=masks_dir, source_mode=source_mode)
    if not samples:
        if source_mode == "on_the_fly":
            raise FileNotFoundError(f"Nenhum par JPG+JSON encontrado em {imgs_dir} e {json_dir}")
        raise FileNotFoundError(f"Nenhum triplo JPG+JSON+NPY encontrado em {imgs_dir}, {json_dir} e {masks_dir}")

    split = make_or_load_split(
        samples=samples,
        split_path=split_path,
        seed=int(cfg["split"].get("seed", cfg.get("seed", 123))),
        val_ratio=float(cfg["split"].get("val_ratio", 0.2)),
        test_ratio=float(cfg["split"].get("test_ratio", 0.0)),
        force_regen=False,
    )

    stems = _resolve_stems_for_split(split, args.split)

    by_stem = {s.stem: s for s in samples}
    eval_samples = [by_stem[s] for s in stems if s in by_stem]
    if not eval_samples:
        raise RuntimeError("Split selecionado sem amostras apos reconciliar com dados disponiveis.")

    if args.inference_source == "multiroi_model":
        if bool(args.multiroi_use_latest_from_output_dirs):
            center_ckpt = latest_best_ckpt(resolve_multiroi_path(repo_root, args.multiroi_center_output_dir))
            lateral_ckpt = latest_best_ckpt(resolve_multiroi_path(repo_root, args.multiroi_lateral_output_dir))
        else:
            center_ckpt = resolve_multiroi_path(repo_root, args.multiroi_center_ckpt)
            lateral_ckpt = resolve_multiroi_path(repo_root, args.multiroi_lateral_ckpt)
        run_name = args.run_name or _latest_run_name(output_base_dir) or "MULTIROI_MODEL"
        run_dir = output_base_dir / "runs" / run_name
        models = load_multiroi_models(center_ckpt=center_ckpt, lateral_ckpt=lateral_ckpt)
        loader = None
        model = None
        threshold = float(args.threshold) if args.threshold is not None else 0.1
        print(f"[MULTIROI] center_ckpt={center_ckpt}")
        print(f"[MULTIROI] lateral_ckpt={lateral_ckpt}")
        print(f"[MULTIROI] device={models.device}")
    else:
        preset = load_json(preset_path)
        ds = HydraTeethDataset(
            samples=eval_samples,
            preset=preset,
            augment=False,
            source_mode=source_mode,
            seed=int(cfg.get("seed", 123)),
        )

        batch_size = int(cfg.get("evaluation", {}).get("batch_size", 2))
        num_workers = int(cfg.get("evaluation", {}).get("num_workers", 0))
        persistent_workers = bool(cfg.get("evaluation", {}).get("persistent_workers", num_workers > 0))
        prefetch_factor = int(cfg.get("evaluation", {}).get("prefetch_factor", 2))

        def _make_loader_kwargs(nw: int) -> Dict:
            kwargs = {
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": nw,
                "pin_memory": torch.cuda.is_available(),
            }
            if nw > 0:
                kwargs["persistent_workers"] = persistent_workers
                kwargs["prefetch_factor"] = prefetch_factor
            return kwargs

        loader = DataLoader(ds, **_make_loader_kwargs(num_workers))

        device_name = str(cfg["training"].get("device", "auto"))
        device = _auto_device() if device_name == "auto" else torch.device(device_name)
        print(f"[DEVICE] using {device}")

        model = HydraUNetMultiTask(
            in_channels=1,
            heatmap_out_channels=64,
            presence_out_channels=32,
            backbone=cfg["model"].get("backbone", "resnet34"),
            presence_dropout=float(cfg["model"].get("presence_dropout", 0.1)),
        ).to(device)

        ckpt_path = args.checkpoint
        if ckpt_path is not None:
            ckpt_path = _resolve_path(repo_root, str(ckpt_path))
            run_dir = _resolve_path(repo_root, str(output_base_dir / "runs" / args.run_name)) if args.run_name else ckpt_path.parent
            run_name = args.run_name or run_dir.name
        else:
            run_name = args.run_name or _latest_run_name(output_base_dir)
            if not run_name:
                raise FileNotFoundError(
                    f"Nenhuma run encontrada em {output_base_dir / 'runs'}; passe --run-name ou --checkpoint."
                )
            run_dir = output_base_dir / "runs" / run_name
            ckpt_path = run_dir / "best.ckpt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        threshold = float(args.threshold) if args.threshold is not None else float(cfg.get("evaluation", {}).get("threshold", 0.5))

    out_dir = run_dir / ("eval_presence_multiroi" if args.inference_source == "multiroi_model" else "eval_presence")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] name={run_name}")
    print(f"[RUN] dir={run_dir}")
    print(f"[DATA] source_mode={source_mode} split={args.split} num_samples={len(eval_samples)}")

    rows: List[Dict] = []
    total_fp = 0
    total_fn = 0
    total_samples = 0
    t0 = time.time()
    if args.inference_source == "multiroi_model":
        num_samples = len(eval_samples)
        for i, sample in enumerate(eval_samples, start=1):
            yt_np = _gt_presence_from_json(sample.json_path).astype(np.int32)
            p_score_np = _multiroi_scores_for_image(
                image_path=sample.image_path,
                models=models,
                infer_threshold=float(args.multiroi_infer_threshold),
            )
            p_pred_np = (p_score_np >= threshold).astype(np.int32)
            fp_mask = (yt_np == 0) & (p_pred_np == 1)
            fn_mask = (yt_np == 1) & (p_pred_np == 0)
            fp_count = int(fp_mask.sum())
            fn_count = int(fn_mask.sum())
            err_count = int((fp_mask | fn_mask).sum())
            bce_mean = _presence_bce_from_scores(yt_np, p_score_np)
            fp_teeth = [CANONICAL_TEETH_32[j] for j in np.where(fp_mask)[0]]
            fn_teeth = [CANONICAL_TEETH_32[j] for j in np.where(fn_mask)[0]]
            suspect_score = (
                args.fn_weight * float(fn_count)
                + args.fp_weight * float(fp_count)
                + args.bce_weight * float(bce_mean)
            )
            rows.append(
                {
                    "stem": sample.stem,
                    "fn_count": fn_count,
                    "fp_count": fp_count,
                    "presence_error_count": err_count,
                    "bce_presence_mean": float(bce_mean),
                    "gt_present_count": int((yt_np == 1).sum()),
                    "pred_present_count": int((p_pred_np == 1).sum()),
                    "fn_teeth": ";".join(fn_teeth),
                    "fp_teeth": ";".join(fp_teeth),
                    "suspect_score": float(suspect_score),
                }
            )
            total_fp += fp_count
            total_fn += fn_count
            total_samples += 1
            if i == 1 or i % 25 == 0 or i == num_samples:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-8)
                eta = (num_samples - i) / max(rate, 1e-8)
                print(f"[EVAL_PROGRESS] sample={i}/{num_samples} elapsed={elapsed:.1f}s eta={eta:.1f}s")
    else:
        num_batches = len(loader)
        with torch.no_grad():
            for bi, batch in enumerate(loader, start=1):
                stems_batch = list(batch["stem"])
                batch = _to_device(batch, device)
                x = batch["x"]
                y_presence = batch["y_presence"]  # (B,32)

                pred = model(x)
                p_score = torch.sigmoid(pred["presence_logits"])  # (B,32)
                p_pred = (p_score >= threshold).to(torch.int64)
                yt = y_presence.to(torch.int64)

                bce_per_tooth = F.binary_cross_entropy_with_logits(
                    pred["presence_logits"], y_presence, reduction="none"
                )  # (B,32)
                bce_per_sample = bce_per_tooth.mean(dim=1)  # (B,)

                fp_mask = (yt == 0) & (p_pred == 1)
                fn_mask = (yt == 1) & (p_pred == 0)

                fp_count = fp_mask.sum(dim=1).cpu().numpy().astype(int)
                fn_count = fn_mask.sum(dim=1).cpu().numpy().astype(int)
                err_count = (fp_mask | fn_mask).sum(dim=1).cpu().numpy().astype(int)
                bce_mean = bce_per_sample.cpu().numpy().astype(float)

                p_score_np = p_score.cpu().numpy()
                yt_np = yt.cpu().numpy()
                p_pred_np = p_pred.cpu().numpy()

                for i, stem in enumerate(stems_batch):
                    fp_teeth = [CANONICAL_TEETH_32[j] for j in np.where((yt_np[i] == 0) & (p_pred_np[i] == 1))[0]]
                    fn_teeth = [CANONICAL_TEETH_32[j] for j in np.where((yt_np[i] == 1) & (p_pred_np[i] == 0))[0]]

                    suspect_score = (
                        args.fn_weight * float(fn_count[i])
                        + args.fp_weight * float(fp_count[i])
                        + args.bce_weight * float(bce_mean[i])
                    )

                    rows.append(
                        {
                            "stem": stem,
                            "fn_count": int(fn_count[i]),
                            "fp_count": int(fp_count[i]),
                            "presence_error_count": int(err_count[i]),
                            "bce_presence_mean": float(bce_mean[i]),
                            "gt_present_count": int((yt_np[i] == 1).sum()),
                            "pred_present_count": int((p_pred_np[i] == 1).sum()),
                            "fn_teeth": ";".join(fn_teeth),
                            "fp_teeth": ";".join(fp_teeth),
                            "suspect_score": float(suspect_score),
                        }
                    )

                total_fp += int(fp_mask.sum().item())
                total_fn += int(fn_mask.sum().item())
                total_samples += int(x.shape[0])

                if bi == 1 or bi % 50 == 0 or bi == num_batches:
                    elapsed = time.time() - t0
                    rate = bi / max(elapsed, 1e-8)
                    eta = (num_batches - bi) / max(rate, 1e-8)
                    print(f"[EVAL_PROGRESS] batch={bi}/{num_batches} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -r["suspect_score"],
            -r["fn_count"],
            -r["fp_count"],
            -r["bce_presence_mean"],
            r["stem"],
        ),
    )

    top_k = max(1, int(args.top_k))
    top_rows = rows_sorted[:top_k]

    full_csv = out_dir / "presence_errors_per_sample.csv"
    top_csv = out_dir / f"presence_top_errors_top{top_k}.csv"
    summary_json = out_dir / "presence_errors_summary.json"

    fieldnames = list(rows_sorted[0].keys()) if rows_sorted else [
        "stem",
        "fn_count",
        "fp_count",
        "presence_error_count",
        "bce_presence_mean",
        "gt_present_count",
        "pred_present_count",
        "fn_teeth",
        "fp_teeth",
        "suspect_score",
    ]

    with full_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    with top_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    # Histograma por quantidade de dentes errados (FN+FP) por radiografia.
    # Grupos exatos: 0,1,2,...,32 dentes errados.
    error_counts = np.array([int(r["presence_error_count"]) for r in rows_sorted], dtype=np.int32)
    bins = np.arange(0, 33, dtype=np.int32)
    hist_rows = []
    for b in bins:
        cnt = int(np.sum(error_counts == b))
        hist_rows.append(
            {
                "presence_error_count": int(b),
                "num_radiographs": cnt,
                "fraction": float(cnt / max(1, len(rows_sorted))),
            }
        )

    hist_csv = out_dir / "presence_error_histogram_by_teeth.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["presence_error_count", "num_radiographs", "fraction"])
        writer.writeheader()
        writer.writerows(hist_rows)

    hist_png = out_dir / "presence_error_histogram_by_teeth.png"
    hist_png_written = False
    if plt is not None:
        xs = [r["presence_error_count"] for r in hist_rows]
        ys = [r["num_radiographs"] for r in hist_rows]
        fig = plt.figure(figsize=(12, 4.5))
        ax = fig.add_subplot(111)
        ax.bar(xs, ys, width=0.85)
        ax.set_title("Presence/Absence Errors Per Radiograph")
        ax.set_xlabel("Numero de dentes errados (FN+FP)")
        ax.set_ylabel("Quantidade de radiografias")
        ax.set_xticks(list(range(0, 33, 2)))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(hist_png, dpi=140)
        plt.close(fig)
        hist_png_written = True

    summary = {
        "split": args.split,
        "threshold": threshold,
        "num_samples": total_samples,
        "num_teeth_total": int(total_samples * 32),
        "total_fn": total_fn,
        "total_fp": total_fp,
        "fn_rate_over_teeth": float(total_fn / max(1, total_samples * 32)),
        "fp_rate_over_teeth": float(total_fp / max(1, total_samples * 32)),
        "suspect_score_formula": (
            f"{args.fn_weight}*fn_count + {args.fp_weight}*fp_count + {args.bce_weight}*bce_presence_mean"
        ),
        "artifacts": {
            "full_csv": str(full_csv),
            "top_csv": str(top_csv),
            "histogram_csv": str(hist_csv),
            "histogram_png": str(hist_png) if hist_png_written else None,
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Registro no dashboard de auditorias (falhas aqui não devem quebrar a avaliação).
    try:
        register_record(
            experiment_root=output_base_dir,
            kind="audits",
            record={
                "id": f"presence_eval__{run_name}__{args.split}",
                "kind": "presence_eval",
                "experiment": output_base_dir.name,
                "run_name": run_name,
                "split": args.split,
                "summary": {
                    "num_samples": summary["num_samples"],
                    "total_fn": summary["total_fn"],
                    "total_fp": summary["total_fp"],
                    "fn_rate_over_teeth": summary["fn_rate_over_teeth"],
                    "fp_rate_over_teeth": summary["fp_rate_over_teeth"],
                },
                "artifacts": {
                    "summary_json": rel_to_experiment(summary_json, output_base_dir),
                    "full_csv": rel_to_experiment(full_csv, output_base_dir),
                    "top_csv": rel_to_experiment(top_csv, output_base_dir),
                    "histogram_csv": rel_to_experiment(hist_csv, output_base_dir),
                    "histogram_png": rel_to_experiment(hist_png, output_base_dir) if hist_png_written else None,
                },
            },
        )
    except Exception as e:
        print(f"[WARN] dashboard registry skipped in eval_presence_top_errors.py: {e}")

    print(f"[PRESENCE_EVAL] split={args.split} num_samples={total_samples} threshold={threshold:.3f}")
    print(f"[PRESENCE_EVAL] total_fn={total_fn} total_fp={total_fp}")
    print(f"[PRESENCE_EVAL] top_csv={top_csv}")
    print(f"[PRESENCE_EVAL] summary={summary_json}")


if __name__ == "__main__":
    main()
