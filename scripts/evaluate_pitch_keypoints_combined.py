#!/usr/bin/env python3
"""
Ewaluacja 32 keypointów na data/keypoints_evaluation_combined/test.

Główne metryki:
- recall widocznych GT przy confidence >= 0.60,
- mean / median / P90 błędu lokalizacji,
- PCK@5 / PCK@10 / PCK@20,
- średnia liczba confident keypointów na klatkę,
- odsetek klatek z >= 4 confident keypointami.

Wszystkie główne błędy [px] liczone są w przestrzeni wejściowej modelu
po fizycznym STRETCH do 640x640. Dzięki temu SoccerNet i
football-field-detection można porównywać mimo różnych rozdzielczości.

Uruchomienie:
    python scripts/evaluate_pitch_keypoints.py
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
from ultralytics import YOLO

try:
    from src.calibration.keypoints_homography import MIN_KEYPOINTS_FOR_HOMOGRAPHY
except Exception:
    MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4


TEST_ROOT = PROJECT_ROOT / "data" / "keypoints_evaluation_combined" / "test"
MANIFEST = PROJECT_ROOT / "data" / "keypoints_evaluation_combined" / "manifest.csv"
WEIGHTS = PROJECT_ROOT / "models" / "pitch_keypoints" / "trained_keypoints.pt"
RESULTS_ROOT = PROJECT_ROOT / "results" / "keypoints_evaluation_combined"

NUM_KEYPOINTS = 32
CONF_THRESHOLD = 0.60
STRETCH_SIZE = 640
PCK_THRESHOLDS = (5.0, 10.0, 20.0)
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
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.stem)
    ]


def mean_or_nan(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def median_or_nan(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def p90_or_nan(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 90)) if x.size else np.nan


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
        }
    return out


def collect_samples(test_root: Path, source_map: dict, source_filter: str):
    images_dir = test_root / "images"
    labels_dir = test_root / "labels"
    ensure(images_dir, "test/images")
    ensure(labels_dir, "test/labels")

    images = sorted(
        [
            x for x in images_dir.iterdir()
            if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS
        ],
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
            }
        )
    return samples


def parse_gt(label_path: Path, orig_w: int, orig_h: int, stretch_size: int):
    lines = [
        x.strip()
        for x in label_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"{label_path.name}: oczekiwano 1 obiektu pitch")

    values = [float(x) for x in lines[0].split()]
    expected = 5 + NUM_KEYPOINTS * 3
    if len(values) != expected:
        raise ValueError(
            f"{label_path.name}: oczekiwano {expected} wartości, jest {len(values)}"
        )

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
        pred_conf = (
            conf[best_idx]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
            .astype(np.float64)
        )

        if pred_xy.shape != (32, 2) or pred_conf.shape != (32, 1):
            return None, None, stretched

        pred_input = np.hstack([pred_xy, pred_conf])

        pred_original = pred_input.copy()
        pred_original[:, 0] *= orig_w / float(self.stretch_size)
        pred_original[:, 1] *= orig_h / float(self.stretch_size)

        return pred_input, pred_original, stretched


def evaluate_one(
    gt_input,
    gt_original,
    visibility,
    pred_input: Optional[np.ndarray],
    pred_original: Optional[np.ndarray],
    conf_threshold: float,
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
                    "error_original_px": np.nan,
                    "correct_5px": False,
                    "correct_10px": False,
                    "correct_20px": False,
                }
            )

        return (
            {
                "object_detected": False,
                "gt_visible_count": n_visible,
                "confident_keypoints_all_count": 0,
                "accepted_visible_count": 0,
                "visible_keypoint_recall": 0.0 if n_visible else np.nan,
                "mean_error_640_px": np.nan,
                "median_error_640_px": np.nan,
                "p90_error_640_px": np.nan,
                "pck_5px": 0.0 if n_visible else np.nan,
                "pck_10px": 0.0 if n_visible else np.nan,
                "pck_20px": 0.0 if n_visible else np.nan,
                "enough_confident_for_homography": False,
            },
            pd.DataFrame(rows),
        )

    conf = pred_input[:, 2]
    accepted = conf >= conf_threshold
    err_640 = np.linalg.norm(pred_input[:, :2] - gt_input, axis=1)
    err_orig = np.linalg.norm(pred_original[:, :2] - gt_original, axis=1)

    matched = visible & accepted
    accepted_errors = err_640[matched]

    def pck(t):
        if n_visible == 0:
            return np.nan
        ok = visible & accepted & (err_640 <= t)
        return float(ok.sum() / n_visible)

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
                "error_640_px_raw": float(err_640[k]),
                "error_original_px": float(err_orig[k]) if is_accepted else np.nan,
                "correct_5px": bool(is_accepted and err_640[k] <= 5.0),
                "correct_10px": bool(is_accepted and err_640[k] <= 10.0),
                "correct_20px": bool(is_accepted and err_640[k] <= 20.0),
            }
        )

    return (
        {
            "object_detected": True,
            "gt_visible_count": n_visible,
            "confident_keypoints_all_count": int(accepted.sum()),
            "accepted_visible_count": int(matched.sum()),
            "visible_keypoint_recall": (
                float(matched.sum() / n_visible) if n_visible else np.nan
            ),
            "mean_error_640_px": mean_or_nan(accepted_errors),
            "median_error_640_px": median_or_nan(accepted_errors),
            "p90_error_640_px": p90_or_nan(accepted_errors),
            "pck_5px": pck(5.0),
            "pck_10px": pck(10.0),
            "pck_20px": pck(20.0),
            "enough_confident_for_homography": bool(
                int(accepted.sum()) >= MIN_KEYPOINTS_FOR_HOMOGRAPHY
            ),
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
        ax.scatter([gt[0]], [gt[1]], marker="o", s=28, zorder=5)
        ax.annotate(
            f"K{k:02d}",
            (gt[0], gt[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
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

            ax.scatter([pred[0]], [pred[1]], marker="x", s=34, zorder=6)
            ax.plot([gt[0], pred[0]], [gt[1], pred[1]], linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"K{k:02d}\nc={conf:.2f}\ne={error:.1f}px",
                (pred[0], pred[1]),
                xytext=(4, -8),
                textcoords="offset points",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", alpha=0.55),
            )

    ax.set_title(
        f"{image_name} | {source}\n"
        f"GT vs prediction | STRETCH {stretched.shape[1]}x{stretched.shape[0]}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_row(per_image: pd.DataFrame, per_kp: pd.DataFrame, source: str):
    total_images = len(per_image)
    total_visible = int(per_image["gt_visible_count"].sum())
    total_accepted = int(per_image["accepted_visible_count"].sum())

    accepted_errors = (
        per_kp["error_640_px"].dropna().to_numpy(dtype=np.float64)
        if not per_kp.empty
        else np.array([], dtype=np.float64)
    )

    return {
        "source": source,
        "total_images": total_images,
        "object_detection_success_rate": float(per_image["object_detected"].mean()),
        "total_gt_visible_keypoints": total_visible,
        "total_accepted_visible_keypoints": total_accepted,
        "visible_keypoint_recall": (
            total_accepted / total_visible if total_visible else np.nan
        ),
        "mean_localization_error_640_px": mean_or_nan(accepted_errors),
        "median_localization_error_640_px": median_or_nan(accepted_errors),
        "p90_localization_error_640_px": p90_or_nan(accepted_errors),
        "pck_5px": float(per_kp["correct_5px"].mean()) if not per_kp.empty else np.nan,
        "pck_10px": float(per_kp["correct_10px"].mean()) if not per_kp.empty else np.nan,
        "pck_20px": float(per_kp["correct_20px"].mean()) if not per_kp.empty else np.nan,
        "mean_confident_keypoints_per_frame": float(
            per_image["confident_keypoints_all_count"].mean()
        ),
        "median_confident_keypoints_per_frame": float(
            per_image["confident_keypoints_all_count"].median()
        ),
        "rate_frames_with_at_least_4_confident": float(
            per_image["enough_confident_for_homography"].mean()
        ),
    }


def build_summaries(per_image, per_kp):
    rows = [summary_row(per_image, per_kp, "ALL")]

    for source in sorted(per_image["source"].dropna().unique()):
        source = str(source)
        if source == "UNKNOWN":
            continue
        img = per_image[per_image["source"] == source]
        kp = (
            per_kp[per_kp["source"] == source]
            if not per_kp.empty
            else pd.DataFrame()
        )
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
                    "accepted_count": len(accepted),
                    "visible_keypoint_recall": len(accepted) / len(df),
                    "mean_confidence": (
                        float(df["confidence"].mean())
                        if df["confidence"].notna().any()
                        else np.nan
                    ),
                    "mean_error_640_px": (
                        float(accepted["error_640_px"].mean())
                        if not accepted.empty
                        else np.nan
                    ),
                    "median_error_640_px": (
                        float(accepted["error_640_px"].median())
                        if not accepted.empty
                        else np.nan
                    ),
                    "pck_5px": float(df["correct_5px"].mean()),
                    "pck_10px": float(df["correct_10px"].mean()),
                    "pck_20px": float(df["correct_20px"].mean()),
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
    print(f"Results:    {output_dir}")
    print()

    detector = StretchDetector(args.weights, args.stretch_size)

    image_rows = []
    kp_tables = []
    vis_count = 0

    for idx, sample in enumerate(samples, 1):
        image_path = sample["image"]
        label_path = sample["label"]
        source = sample["source"]

        print(
            f"[{idx:03d}/{len(samples):03d}] {image_path.name} | {source} ...",
            end=" ",
            flush=True,
        )

        image = cv2.imread(str(image_path))
        if image is None:
            print("FAILED: image")
            continue

        h, w = image.shape[:2]

        try:
            gt_input, gt_original, visibility, class_id = parse_gt(
                label_path, w, h, args.stretch_size
            )
            pred_input, pred_original, stretched = detector.detect(image)

            metrics, kp_df = evaluate_one(
                gt_input,
                gt_original,
                visibility,
                pred_input,
                pred_original,
                args.conf_threshold,
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
                and (
                    args.max_visualizations is None
                    or vis_count < args.max_visualizations
                )
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
                print(
                    f"GT={metrics['gt_visible_count']} | "
                    f"accepted={metrics['accepted_visible_count']} | "
                    f"recall={100*metrics['visible_keypoint_recall']:.1f}% | "
                    f"median={metrics['median_error_640_px']:.2f}px | "
                    f"PCK20={100*metrics['pck_20px']:.1f}%"
                )

        except Exception as exc:
            print(f"FAILED | {type(exc).__name__}: {exc}")

    if not image_rows:
        raise RuntimeError("Nie udało się ocenić żadnej klatki")

    per_image = pd.DataFrame(image_rows).sort_values("image_index").reset_index(drop=True)

    per_kp = (
        pd.concat(kp_tables, ignore_index=True)
        .sort_values(["image_index", "keypoint_id"])
        .reset_index(drop=True)
        if kp_tables
        else pd.DataFrame()
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
        "metric_space": f"{args.stretch_size}x{args.stretch_size} STRETCH input",
        "pck_thresholds_px": list(PCK_THRESHOLDS),
        "pck_denominator": (
            "all GT-visible keypoints; missing/low-confidence prediction = incorrect"
        ),
        "best_detection_rule": "highest mean confidence across 32 keypoints",
        "min_keypoints_for_homography": MIN_KEYPOINTS_FOR_HOMOGRAPHY,
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)

    for row in per_source.itertuples(index=False):
        print(f"\n[{row.source}] n={int(row.total_images)}")
        print(
            f"Recall @ conf>={args.conf_threshold:.2f}: "
            f"{100*row.visible_keypoint_recall:.2f}%"
        )
        print(
            f"Error 640 [px]: mean={row.mean_localization_error_640_px:.2f} | "
            f"median={row.median_localization_error_640_px:.2f} | "
            f"P90={row.p90_localization_error_640_px:.2f}"
        )
        print(
            f"PCK@5={100*row.pck_5px:.2f}% | "
            f"PCK@10={100*row.pck_10px:.2f}% | "
            f"PCK@20={100*row.pck_20px:.2f}%"
        )
        print(
            f"Confident points/frame: "
            f"mean={row.mean_confident_keypoints_per_frame:.2f} | "
            f"median={row.median_confident_keypoints_per_frame:.2f}"
        )
        print(
            f"Frames >= {MIN_KEYPOINTS_FOR_HOMOGRAPHY} confident: "
            f"{100*row.rate_frames_with_at_least_4_confident:.2f}%"
        )

    print(f"\nWyniki: {output_dir}")


if __name__ == "__main__":
    main()
