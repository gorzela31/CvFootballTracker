#!/usr/bin/env python3
"""
Ewaluacja 32 punktów charakterystycznych boiska na połączonym zbiorze:
    data/keypoints_evaluation_combined/test

Główne metryki:
- recall widocznych punktów GT przy confidence >= 0.60,
- średni błąd lokalizacji [px],
- mediana błędu lokalizacji [px],
- średni błąd lokalizacji [m] dla części SoccerNet-Calibration,
- mediana błędu lokalizacji [m] dla części SoccerNet-Calibration.

Błędy [px] są liczone w przestrzeni wejściowej modelu po fizycznym
STRETCH do 640x640. Zapewnia to wspólną skalę dla obu źródeł danych.

Błędy [m] są liczone wyłącznie dla obrazów SoccerNet, dla których
manifest wskazuje niezależną referencyjną H_GT. Predykcja keypointu jest
skalowana z 640x640 z powrotem do oryginalnej rozdzielczości obrazu,
a następnie rzutowana przez H_GT IMAGE -> PITCH i porównywana ze znaną
pozycją danego punktu w PITCH_KEYPOINTS_TEMPLATE_M.

Uruchomienie:
    python scripts/evaluate_pitch_keypoints_combined.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from ultralytics import YOLO

from src.calibration.keypoints_homography import PITCH_KEYPOINTS_TEMPLATE_M


TEST_ROOT = PROJECT_ROOT / "data" / "keypoints_evaluation_combined" / "test"
MANIFEST = PROJECT_ROOT / "data" / "keypoints_evaluation_combined" / "manifest.csv"
WEIGHTS = PROJECT_ROOT / "models" / "pitch_keypoints" / "trained_keypoints.pt"
RESULTS_ROOT = PROJECT_ROOT / "results" / "keypoints_evaluation_combined"

NUM_KEYPOINTS = 32
CONF_THRESHOLD = 0.60
STRETCH_SIZE = 640
SOCCERNET_SOURCE = "soccerNet_hgt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", type=Path, default=TEST_ROOT)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    p.add_argument("--stretch-size", type=int, default=STRETCH_SIZE)
    p.add_argument(
        "--source",
        choices=["all", "soccerNet_hgt", "football_field_detection_16"],
        default="all",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-visualizations", action="store_true")
    p.add_argument("--max-visualizations", type=int, default=None)
    return p.parse_args()


def ensure(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"Brak {what}: {path}")


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.stem)]


def mean_or_nan(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def median_or_nan(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def normalize_h(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"Homografia ma shape {h.shape}, oczekiwano (3, 3)")
    if not np.isfinite(h).all():
        raise ValueError("Homografia zawiera NaN/Inf")
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    else:
        norm = np.linalg.norm(h)
        if norm <= 1e-12:
            raise ValueError("Nieprawidłowa homografia")
        h = h / norm
    return h


def transform_points(h: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points_xy, np.ones(len(points_xy), dtype=np.float64)])
    projected = (h @ homogeneous.T).T
    valid = np.abs(projected[:, 2]) > 1e-12
    result = np.full((len(points_xy), 2), np.nan, dtype=np.float64)
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def load_source_map(manifest_path: Path):
    if not manifest_path.exists():
        print(f"WARNING: brak manifestu: {manifest_path}")
        return {}

    df = pd.read_csv(manifest_path)
    required = {"source", "output_image"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv nie ma kolumn: {sorted(missing)}")

    out = {}
    for row in df.to_dict("records"):
        out[Path(str(row["output_image"])).name] = {
            "source": str(row["source"]),
            "combined_id": row.get("combined_id", np.nan),
            "eval_id": row.get("eval_id", np.nan),
            "source_label_or_gt": row.get("source_label_or_gt", np.nan),
        }
    return out


def resolve_gt_path(source: str, source_label_or_gt) -> Optional[Path]:
    if source != SOCCERNET_SOURCE or pd.isna(source_label_or_gt):
        return None
    path = Path(str(source_label_or_gt))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_h_image_to_pitch(gt_path: Optional[Path]) -> Optional[np.ndarray]:
    if gt_path is None:
        return None
    ensure(gt_path, "referencyjnej H_GT")
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "H_image_to_pitch" not in data:
        raise ValueError(f"{gt_path.name} nie zawiera H_image_to_pitch")
    return normalize_h(np.asarray(data["H_image_to_pitch"], dtype=np.float64))


def collect_samples(test_root: Path, source_map: dict, source_filter: str):
    images_dir = test_root / "images"
    labels_dir = test_root / "labels"
    ensure(images_dir, "test/images")
    ensure(labels_dir, "test/labels")

    images = sorted(
        [x for x in images_dir.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )

    samples = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"WARNING: brak labela dla {image_path.name}")
            continue

        info = source_map.get(image_path.name, {})
        source = str(info.get("source", "UNKNOWN"))
        if source_filter != "all" and source != source_filter:
            continue

        samples.append(
            {
                "image": image_path,
                "label": label_path,
                "source": source,
                "combined_id": info.get("combined_id", np.nan),
                "eval_id": info.get("eval_id", np.nan),
                "gt_h_path": resolve_gt_path(source, info.get("source_label_or_gt", np.nan)),
            }
        )
    return samples


def parse_gt(label_path: Path, orig_w: int, orig_h: int, stretch_size: int):
    lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label_path.name}: oczekiwano 1 obiektu pitch")

    values = [float(x) for x in lines[0].split()]
    expected = 5 + NUM_KEYPOINTS * 3
    if len(values) != expected:
        raise ValueError(f"{label_path.name}: oczekiwano {expected} wartości, jest {len(values)}")

    kp = np.asarray(values[5:], dtype=np.float64).reshape(NUM_KEYPOINTS, 3)
    norm_xy = kp[:, :2]
    visibility = kp[:, 2]

    gt_input = norm_xy * float(stretch_size)
    gt_original = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
    gt_original[:, 0] = norm_xy[:, 0] * orig_w
    gt_original[:, 1] = norm_xy[:, 1] * orig_h
    return gt_input, gt_original, visibility, int(values[0])


class StretchDetector:
    def __init__(self, weights: Path, stretch_size: int):
        self.model = YOLO(str(weights))
        self.stretch_size = int(stretch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, image_bgr):
        orig_h, orig_w = image_bgr.shape[:2]
        stretched = cv2.resize(
            image_bgr,
            (self.stretch_size, self.stretch_size),
            interpolation=cv2.INTER_LINEAR,
        )

        results = self.model(
            stretched,
            verbose=False,
            device=self.device,
            imgsz=self.stretch_size,
        )

        if not results or results[0].keypoints is None:
            return None, None, stretched

        xy = results[0].keypoints.xy
        conf = results[0].keypoints.conf
        if xy is None or conf is None or len(xy) == 0:
            return None, None, stretched

        best_idx = int(conf.mean(dim=1).argmax().item())
        pred_xy = xy[best_idx].detach().cpu().numpy().astype(np.float64)
        pred_conf = conf[best_idx].detach().cpu().numpy().reshape(-1, 1).astype(np.float64)

        if pred_xy.shape != (NUM_KEYPOINTS, 2) or pred_conf.shape != (NUM_KEYPOINTS, 1):
            return None, None, stretched

        pred_input = np.hstack([pred_xy, pred_conf])
        pred_original = pred_input.copy()
        pred_original[:, 0] *= orig_w / float(self.stretch_size)
        pred_original[:, 1] *= orig_h / float(self.stretch_size)
        return pred_input, pred_original, stretched


def evaluate_one(
    gt_input,
    visibility,
    pred_input: Optional[np.ndarray],
    pred_original: Optional[np.ndarray],
    conf_threshold: float,
    h_image_to_pitch: Optional[np.ndarray],
):
    visible = visibility > 0
    n_visible = int(visible.sum())
    rows = []

    if pred_input is None:
        for k in range(NUM_KEYPOINTS):
            if not visible[k]:
                continue
            rows.append(
                {
                    "keypoint_id": k,
                    "confidence": np.nan,
                    "accepted": False,
                    "error_640_px": np.nan,
                    "error_m": np.nan,
                }
            )

        return (
            {
                "object_detected": False,
                "gt_visible_count": n_visible,
                "accepted_visible_count": 0,
                "visible_keypoint_recall": 0.0 if n_visible else np.nan,
                "mean_error_640_px": np.nan,
                "median_error_640_px": np.nan,
                "mean_error_m": np.nan,
                "median_error_m": np.nan,
            },
            pd.DataFrame(rows),
        )

    conf = pred_input[:, 2]
    accepted = conf >= conf_threshold
    matched = visible & accepted

    err_640 = np.linalg.norm(pred_input[:, :2] - gt_input, axis=1)
    accepted_errors_640 = err_640[matched]

    err_m = np.full(NUM_KEYPOINTS, np.nan, dtype=np.float64)
    if h_image_to_pitch is not None:
        pred_pitch = transform_points(h_image_to_pitch, pred_original[:, :2])
        template = np.asarray(PITCH_KEYPOINTS_TEMPLATE_M, dtype=np.float64)
        valid_metric = np.isfinite(pred_pitch).all(axis=1)
        err_m[valid_metric] = np.linalg.norm(pred_pitch[valid_metric] - template[valid_metric], axis=1)

    accepted_errors_m = err_m[matched & np.isfinite(err_m)]

    for k in range(NUM_KEYPOINTS):
        if not visible[k]:
            continue
        is_accepted = bool(accepted[k])
        rows.append(
            {
                "keypoint_id": k,
                "confidence": float(conf[k]),
                "accepted": is_accepted,
                "gt_x_640_px": float(gt_input[k, 0]),
                "gt_y_640_px": float(gt_input[k, 1]),
                "pred_x_640_px": float(pred_input[k, 0]),
                "pred_y_640_px": float(pred_input[k, 1]),
                "error_640_px": float(err_640[k]) if is_accepted else np.nan,
                "error_m": float(err_m[k]) if is_accepted and np.isfinite(err_m[k]) else np.nan,
            }
        )

    return (
        {
            "object_detected": True,
            "gt_visible_count": n_visible,
            "accepted_visible_count": int(matched.sum()),
            "visible_keypoint_recall": float(matched.sum() / n_visible) if n_visible else np.nan,
            "mean_error_640_px": mean_or_nan(accepted_errors_640),
            "median_error_640_px": median_or_nan(accepted_errors_640),
            "mean_error_m": mean_or_nan(accepted_errors_m),
            "median_error_m": median_or_nan(accepted_errors_m),
        },
        pd.DataFrame(rows),
    )


def visualize(
    stretched,
    image_name,
    source,
    gt_input,
    visibility,
    pred_input,
    conf_threshold,
    output_path,
):
    rgb = cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.axis("off")
    visible = visibility > 0

    for k in range(NUM_KEYPOINTS):
        if not visible[k]:
            continue
        gt = gt_input[k]
        ax.scatter([gt[0]], [gt[1]], marker="o", s=30, color="lime", edgecolors="black", linewidths=0.4, zorder=5)
        ax.annotate(
            f"K{k:02d}",
            (gt[0], gt[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    if pred_input is not None:
        for k in range(NUM_KEYPOINTS):
            if not visible[k]:
                continue
            pred = pred_input[k, :2]
            conf = float(pred_input[k, 2])
            if conf < conf_threshold:
                continue

            gt = gt_input[k]
            error = float(np.linalg.norm(pred - gt))
            ax.scatter([pred[0]], [pred[1]], marker="x", s=42, color="red", linewidths=1.4, zorder=6)
            ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="yellow", linewidth=0.8, alpha=0.85, zorder=4)
            ax.annotate(
                f"K{k:02d}\nc={conf:.2f}\ne={error:.1f}px",
                (pred[0], pred[1]),
                xytext=(4, -8),
                textcoords="offset points",
                fontsize=7,
                color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.60, edgecolor="none"),
            )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="lime", markeredgecolor="black", markersize=7, label="Punkt referencyjny GT"),
        Line2D([0], [0], marker="x", linestyle="None", color="red", markersize=7, label=f"Predykcja (confidence ≥ {conf_threshold:.2f})"),
        Line2D([0], [0], color="yellow", linewidth=1.2, label="Błąd lokalizacji"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(
        f"{image_name} | {source}\n"
        f"GT vs predykcja | STRETCH {stretched.shape[1]}x{stretched.shape[0]}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_row(per_image: pd.DataFrame, per_kp: pd.DataFrame, source: str):
    total_images = len(per_image)
    total_visible = int(per_image["gt_visible_count"].sum())
    total_accepted = int(per_image["accepted_visible_count"].sum())

    errors_px = (
        per_kp["error_640_px"].dropna().to_numpy(dtype=np.float64)
        if not per_kp.empty else np.array([], dtype=np.float64)
    )
    errors_m = (
        per_kp["error_m"].dropna().to_numpy(dtype=np.float64)
        if not per_kp.empty and "error_m" in per_kp.columns else np.array([], dtype=np.float64)
    )

    return {
        "source": source,
        "total_images": total_images,
        "total_gt_visible_keypoints": total_visible,
        "total_detected_visible_keypoints": total_accepted,
        "visible_keypoint_recall": total_accepted / total_visible if total_visible else np.nan,
        "mean_localization_error_640_px": mean_or_nan(errors_px),
        "median_localization_error_640_px": median_or_nan(errors_px),
        "mean_localization_error_m": mean_or_nan(errors_m),
        "median_localization_error_m": median_or_nan(errors_m),
        "meter_error_keypoints_count": int(len(errors_m)),
    }


def build_summaries(per_image, per_kp):
    rows = [summary_row(per_image, per_kp, "ALL")]
    for source in sorted(per_image["source"].dropna().unique()):
        source = str(source)
        if source == "UNKNOWN":
            continue
        img = per_image[per_image["source"] == source]
        kp = per_kp[per_kp["source"] == source] if not per_kp.empty else pd.DataFrame()
        rows.append(summary_row(img, kp, source))
    return pd.DataFrame(rows)


def build_per_keypoint_summary(per_kp):
    if per_kp.empty:
        return pd.DataFrame()

    rows = []
    groups = [("ALL", per_kp)]
    for source in sorted(per_kp["source"].dropna().unique()):
        source = str(source)
        if source != "UNKNOWN":
            groups.append((source, per_kp[per_kp["source"] == source]))

    for source, source_df in groups:
        for k in range(NUM_KEYPOINTS):
            df = source_df[source_df["keypoint_id"] == k]
            if df.empty:
                continue
            accepted = df[df["accepted"] == True]
            rows.append(
                {
                    "source": source,
                    "keypoint_id": k,
                    "gt_visible_count": len(df),
                    "detected_count": len(accepted),
                    "visible_keypoint_recall": len(accepted) / len(df),
                    "mean_localization_error_640_px": mean_or_nan(accepted["error_640_px"]),
                    "median_localization_error_640_px": median_or_nan(accepted["error_640_px"]),
                    "mean_localization_error_m": mean_or_nan(accepted["error_m"]),
                    "median_localization_error_m": median_or_nan(accepted["error_m"]),
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if not 0.0 <= args.conf_threshold <= 1.0:
        raise ValueError("--conf-threshold musi należeć do [0,1]")
    if args.stretch_size < 32:
        raise ValueError("--stretch-size jest za mały")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit musi być >= 1")

    ensure(args.test_root, "test root")
    ensure(args.weights, "wag modelu")

    source_map = load_source_map(args.manifest)
    samples = collect_samples(args.test_root, source_map, args.source)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("Brak próbek do ewaluacji")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = RESULTS_ROOT / timestamp
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=False)
    if not args.no_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EWALUACJA KEYPOINTÓW — COMBINED")
    print("=" * 80)
    print(f"Test:       {args.test_root}")
    print(f"Weights:    {args.weights}")
    print(f"Samples:    {len(samples)}")
    print(f"Input:      {args.stretch_size}x{args.stretch_size} STRETCH")
    print(f"Threshold:  {args.conf_threshold}")
    print(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Błąd [m]:  tylko SoccerNet z referencyjną H_GT")
    print(f"Results:    {output_dir}\n")

    detector = StretchDetector(args.weights, args.stretch_size)
    image_rows = []
    kp_tables = []
    vis_count = 0

    for idx, sample in enumerate(samples, 1):
        image_path = sample["image"]
        label_path = sample["label"]
        source = sample["source"]

        print(f"[{idx:03d}/{len(samples):03d}] {image_path.name} | {source} ...", end=" ", flush=True)
        image = cv2.imread(str(image_path))
        if image is None:
            print("FAILED: image")
            continue

        h, w = image.shape[:2]

        try:
            gt_input, _, visibility, class_id = parse_gt(label_path, w, h, args.stretch_size)
            h_image_to_pitch = load_h_image_to_pitch(sample["gt_h_path"])
            pred_input, pred_original, stretched = detector.detect(image)

            metrics, kp_df = evaluate_one(
                gt_input,
                visibility,
                pred_input,
                pred_original,
                args.conf_threshold,
                h_image_to_pitch,
            )

            metrics.update(
                {
                    "image_index": idx,
                    "image_name": image_path.name,
                    "source": source,
                    "combined_id": sample["combined_id"],
                    "eval_id": sample["eval_id"],
                    "class_id": class_id,
                    "original_width": w,
                    "original_height": h,
                }
            )
            image_rows.append(metrics)

            if not kp_df.empty:
                kp_df.insert(0, "source", source)
                kp_df.insert(0, "image_name", image_path.name)
                kp_df.insert(0, "image_index", idx)
                kp_tables.append(kp_df)

            should_vis = (
                not args.no_visualizations
                and (args.max_visualizations is None or vis_count < args.max_visualizations)
            )
            if should_vis:
                visualize(
                    stretched,
                    image_path.name,
                    source,
                    gt_input,
                    visibility,
                    pred_input,
                    args.conf_threshold,
                    vis_dir / f"{image_path.stem}_gt_vs_pred.png",
                )
                vis_count += 1

            if pred_input is None:
                print(f"NO DETECTION | GT={metrics['gt_visible_count']}")
            else:
                msg = (
                    f"GT={metrics['gt_visible_count']} | "
                    f"detected={metrics['accepted_visible_count']} | "
                    f"recall={100 * metrics['visible_keypoint_recall']:.1f}% | "
                    f"mean={metrics['mean_error_640_px']:.2f}px | "
                    f"median={metrics['median_error_640_px']:.2f}px"
                )
                if np.isfinite(metrics["mean_error_m"]):
                    msg += (
                        f" | mean_m={metrics['mean_error_m']:.3f}m | "
                        f"median_m={metrics['median_error_m']:.3f}m"
                    )
                print(msg)

        except Exception as exc:
            print(f"FAILED | {type(exc).__name__}: {exc}")

    if not image_rows:
        raise RuntimeError("Nie udało się ocenić żadnej klatki")

    per_image = pd.DataFrame(image_rows).sort_values("image_index").reset_index(drop=True)
    per_kp = (
        pd.concat(kp_tables, ignore_index=True)
        .sort_values(["image_index", "keypoint_id"])
        .reset_index(drop=True)
        if kp_tables else pd.DataFrame()
    )

    per_source = build_summaries(per_image, per_kp)
    global_summary = per_source[per_source["source"] == "ALL"].reset_index(drop=True)
    per_kp_summary = build_per_keypoint_summary(per_kp)

    global_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    per_source.to_csv(output_dir / "per_source_summary.csv", index=False)
    per_image.to_csv(output_dir / "per_image_metrics.csv", index=False)
    per_kp.to_csv(output_dir / "per_keypoint_metrics.csv", index=False)
    per_kp_summary.to_csv(output_dir / "per_keypoint_summary.csv", index=False)

    config = {
        "test_root": str(args.test_root),
        "manifest": str(args.manifest),
        "weights": str(args.weights),
        "confidence_threshold": args.conf_threshold,
        "stretch_size": args.stretch_size,
        "num_samples": len(samples),
        "source_filter": args.source,
        "pixel_metric_space": f"{args.stretch_size}x{args.stretch_size} STRETCH input",
        "meter_metric_scope": "soccerNet_hgt only; predicted original-image keypoint projected by H_GT image_to_pitch",
        "meter_reference_points": "PITCH_KEYPOINTS_TEMPLATE_M",
        "best_detection_rule": "highest mean confidence across 32 keypoints",
        "reported_metrics": [
            "visible_keypoint_recall",
            "mean_localization_error_640_px",
            "median_localization_error_640_px",
            "mean_localization_error_m",
            "median_localization_error_m",
        ],
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)

    for row in per_source.itertuples(index=False):
        print(f"\n[{row.source}] n={int(row.total_images)}")
        print(f"Recall @ conf>={args.conf_threshold:.2f}: {100 * row.visible_keypoint_recall:.2f}%")
        print(
            f"Error {args.stretch_size}x{args.stretch_size} [px]: "
            f"mean={row.mean_localization_error_640_px:.2f} | "
            f"median={row.median_localization_error_640_px:.2f}"
        )
        if int(row.meter_error_keypoints_count) > 0:
            print(
                f"Error pitch [m] (SoccerNet/H_GT): "
                f"mean={row.mean_localization_error_m:.3f} | "
                f"median={row.median_localization_error_m:.3f} | "
                f"n={int(row.meter_error_keypoints_count)}"
            )
        else:
            print("Error pitch [m]: n/a — brak niezależnej H_GT dla tego źródła")

    print(f"\nWyniki: {output_dir}")


if __name__ == "__main__":
    main()
