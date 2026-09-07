#!/usr/bin/env python3
"""
Ewaluacja homografii TVCalib względem referencyjnej H_GT.

Umieść plik jako:
    CvFootballTracker/scripts/evaluate_homography_tvcalib.py

Skrypt wykorzystuje tę samą klasę TVCalibHomography co pipeline:
    src/calibration/homography.py

Domyślna konfiguracja TVCalib:
    OPTIM_STEPS = 500
    lens_dist = False

Wejście:
    data/data_homography_evaluation/
        images/
            eval_homo_001.jpg ...
        manifest.csv
        ground_truth/
            eval_homo_001_gt.json ...

Wagi TVCalib:
    src/calibration/tvcalib/data/segment_localization/train_59.pt

Domyślne uruchomienie — jedna klatka:
    python scripts/evaluate_homography_tvcalib.py

Inna klatka:
    python scripts/evaluate_homography_tvcalib.py --mode single --eval-id 27

Cała pula:
    python scripts/evaluate_homography_tvcalib.py --mode all

Wyniki:
    results/homography_evaluation/<timestamp>/
        metrics_summary.csv
        per_frame_metrics.csv
        point_errors.csv
        config.json
        estimated_homographies/
            eval_homo_001_tvcalib.json
            ...
        visualizations/
            eval_homo_001_tvcalib_vs_gt.png
            ...

Metryki:
    1. pitch error [m]
       Dla referencyjnego punktu boiska P:
           P --H_GT_pitch_to_image--> p_ref na obrazie
           p_ref --H_TVCalib_image_to_pitch--> P_est
       Błąd:
           ||P_est - P|| [m]

    2. reprojection error [px]
       Dla tego samego P:
           P --H_GT_pitch_to_image--> p_ref
           P --H_TVCalib_pitch_to_image--> p_est
       Błąd:
           ||p_est - p_ref|| [px]

Punkty testowe:
    - regularna siatka na płaszczyźnie boiska,
    - tylko punkty widoczne na danej klatce według H_GT,
    - odsunięte od krawędzi boiska o PITCH_MARGIN_M,
    - dzięki temu metryki nie są liczone na liniach użytych do estymacji H_GT.

WAŻNE:
    - każda klatka jest kalibrowana NIEZALEŻNIE;
    - nie stosujemy mechanizmu "ostatniej poprawnej H" z pipeline'u,
      ponieważ zawyżałoby to success rate ewaluacji;
    - inicjalizacja modelu TVCalib nie wchodzi do czasu estymacji;
    - punkty TVCalib wypadające poza boisko NIE są odrzucane przy liczeniu błędu.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.calibration.homography import TVCalibHomography

from generate_homography_gt import (
    draw_top_down_pitch,
    pitch_drawing_polylines,
    transform_points,
)


# =============================================================================
# KONFIGURACJA
# =============================================================================

EVAL_ROOT = PROJECT_ROOT / "data" / "data_homography_evaluation"
EVAL_IMAGES = EVAL_ROOT / "images"
EVAL_MANIFEST = EVAL_ROOT / "manifest.csv"
GT_DIR = EVAL_ROOT / "ground_truth"

TVCALIB_WEIGHTS = (
    PROJECT_ROOT
    / "src"
    / "calibration"
    / "tvcalib"
    / "data"
    / "segment_localization"
    / "train_59.pt"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "homography_evaluation"

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Tak samo jak w pipeline_yolo_bytetrack_tvcalib.py.
OPTIM_STEPS = 500

GRID_STEP_M = 5.0
PITCH_MARGIN_M = 5.0
IMAGE_MARGIN_PX = 5.0
NUM_VIS_POINTS = 2


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ewaluacja TVCalib względem referencyjnej homografii H_GT."
    )

    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        default="single",
        help="single = jedna klatka, all = cała zamrożona pula. Domyślnie single.",
    )
    parser.add_argument(
        "--eval-id",
        type=int,
        default=1,
        help="Numer eval_id dla --mode single. Domyślnie 1.",
    )
    parser.add_argument(
        "--optim-steps",
        type=int,
        default=OPTIM_STEPS,
        help=f"Liczba kroków optymalizacji TVCalib. Domyślnie {OPTIM_STEPS}.",
    )
    parser.add_argument(
        "--grid-step-m",
        type=float,
        default=GRID_STEP_M,
        help=f"Odstęp siatki punktów testowych [m]. Domyślnie {GRID_STEP_M}.",
    )
    parser.add_argument(
        "--pitch-margin-m",
        type=float,
        default=PITCH_MARGIN_M,
        help=f"Margines punktów od krawędzi boiska [m]. Domyślnie {PITCH_MARGIN_M}.",
    )
    parser.add_argument(
        "--image-margin-px",
        type=float,
        default=IMAGE_MARGIN_PX,
        help=f"Margines widoczności punktu na obrazie [px]. Domyślnie {IMAGE_MARGIN_PX}.",
    )
    parser.add_argument(
        "--num-vis-points",
        type=int,
        default=NUM_VIS_POINTS,
        help=f"Liczba par punktów na wizualizacji. Domyślnie {NUM_VIS_POINTS}.",
    )
    parser.add_argument(
        "--lens-dist",
        action="store_true",
        help="Włącz model dystorsji TVCalib. Pipeline domyślnie go nie używa.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Nie generuj PNG z porównaniem TVCalib vs GT.",
    )
    return parser.parse_args()


# =============================================================================
# HELPERY
# =============================================================================

def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Brak {description}: {path}")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def normalize_h(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)

    if h.shape != (3, 3):
        raise ValueError(f"Homografia ma nieprawidłowy kształt: {h.shape}")

    if not np.isfinite(h).all():
        raise ValueError("Homografia zawiera NaN/Inf.")

    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    else:
        norm = np.linalg.norm(h)
        if norm <= 1e-12:
            raise ValueError("Homografia ma zerową normę.")
        h = h / norm

    if abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("Homografia jest osobliwa.")

    return h


def flatten_h(prefix: str, h: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_{r}{c}": float(h[r, c])
        for r in range(3)
        for c in range(3)
    }


def percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def load_manifest() -> pd.DataFrame:
    ensure_exists(EVAL_MANIFEST, "manifest.csv")
    ensure_exists(EVAL_IMAGES, "katalogu obrazów ewaluacyjnych")
    ensure_exists(GT_DIR, "katalogu ground_truth")

    manifest = pd.read_csv(EVAL_MANIFEST)

    required = {"eval_id", "eval_stem", "eval_image"}
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"manifest.csv nie zawiera wymaganych kolumn: {sorted(missing)}"
        )

    return manifest.sort_values("eval_id").reset_index(drop=True)


def select_samples(
    manifest: pd.DataFrame,
    mode: str,
    eval_id: int,
) -> List[dict]:

    if mode == "single":
        rows = manifest[manifest["eval_id"] == eval_id]
        if rows.empty:
            raise ValueError(
                f"Nie znaleziono eval_id={eval_id}. "
                f"Dostępny zakres: "
                f"{int(manifest['eval_id'].min())}-"
                f"{int(manifest['eval_id'].max())}."
            )
        selected = rows
    else:
        selected = manifest

    samples: List[dict] = []

    for row in selected.itertuples(index=False):
        eval_stem = str(row.eval_stem)
        image_path = EVAL_IMAGES / str(row.eval_image)
        gt_path = GT_DIR / f"{eval_stem}_gt.json"

        ensure_exists(image_path, f"obrazu {eval_stem}")
        ensure_exists(gt_path, f"H_GT dla {eval_stem}")

        samples.append(
            {
                "eval_id": int(row.eval_id),
                "eval_stem": eval_stem,
                "image_path": image_path,
                "gt_path": gt_path,
            }
        )

    return samples


def load_gt(gt_path: Path) -> dict:
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "H_image_to_pitch" not in data or "H_pitch_to_image" not in data:
        raise ValueError(
            f"{gt_path.name} nie zawiera H_image_to_pitch / H_pitch_to_image."
        )

    h_img2pitch = normalize_h(
        np.asarray(data["H_image_to_pitch"], dtype=np.float64)
    )
    h_pitch2img = normalize_h(
        np.asarray(data["H_pitch_to_image"], dtype=np.float64)
    )

    return {
        **data,
        "_H_image_to_pitch": h_img2pitch,
        "_H_pitch_to_image": h_pitch2img,
    }


def build_canonical_pitch_grid(
    pitch_length: float,
    pitch_width: float,
    step_m: float,
    margin_m: float,
) -> np.ndarray:
    """
    Niezależna regularna siatka punktów na boisku.
    """
    if step_m <= 0:
        raise ValueError("--grid-step-m musi być > 0.")

    if margin_m < 0:
        raise ValueError("--pitch-margin-m nie może być ujemny.")

    x_min = -pitch_length / 2.0 + margin_m
    x_max = +pitch_length / 2.0 - margin_m
    y_min = -pitch_width / 2.0 + margin_m
    y_max = +pitch_width / 2.0 - margin_m

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Margines boiska jest zbyt duży.")

    xs = np.arange(x_min, x_max + 1e-9, step_m, dtype=np.float64)
    ys = np.arange(y_min, y_max + 1e-9, step_m, dtype=np.float64)

    xx, yy = np.meshgrid(xs, ys)

    return np.column_stack([xx.ravel(), yy.ravel()])


def visible_reference_points(
    pitch_grid: np.ndarray,
    h_gt_pitch2img: np.ndarray,
    image_width: int,
    image_height: int,
    image_margin_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z siatki wybiera tylko punkty widoczne na klatce według H_GT.

    Zwraca:
        pitch_points_ref [N,2]
        image_points_ref [N,2]
    """
    image_points = transform_points(h_gt_pitch2img, pitch_grid)

    finite = np.isfinite(image_points).all(axis=1)

    visible = (
        finite
        & (image_points[:, 0] >= image_margin_px)
        & (image_points[:, 0] < image_width - image_margin_px)
        & (image_points[:, 1] >= image_margin_px)
        & (image_points[:, 1] < image_height - image_margin_px)
    )

    return pitch_grid[visible], image_points[visible]


def calculate_point_errors(
    pitch_points_ref: np.ndarray,
    image_points_ref: np.ndarray,
    h_tv_img2pitch: np.ndarray,
    h_tv_pitch2img: np.ndarray,
) -> pd.DataFrame:
    """
    Liczy błędy TVCalib względem H_GT.

    Nie używamy calibrator.project_point_to_pitch(), bo ten helper odrzuca
    punkty poza boiskiem. W ewaluacji chcemy zachować duże błędy.
    """
    tv_pitch_points = transform_points(
        h_tv_img2pitch,
        image_points_ref,
    )

    tv_image_points = transform_points(
        h_tv_pitch2img,
        pitch_points_ref,
    )

    valid = (
        np.isfinite(tv_pitch_points).all(axis=1)
        & np.isfinite(tv_image_points).all(axis=1)
    )

    pitch_ref_valid = pitch_points_ref[valid]
    image_ref_valid = image_points_ref[valid]
    tv_pitch_valid = tv_pitch_points[valid]
    tv_image_valid = tv_image_points[valid]

    pitch_errors = np.linalg.norm(
        tv_pitch_valid - pitch_ref_valid,
        axis=1,
    )

    reprojection_errors = np.linalg.norm(
        tv_image_valid - image_ref_valid,
        axis=1,
    )

    return pd.DataFrame(
        {
            "ref_pitch_x_m": pitch_ref_valid[:, 0],
            "ref_pitch_y_m": pitch_ref_valid[:, 1],
            "ref_image_x_px": image_ref_valid[:, 0],
            "ref_image_y_px": image_ref_valid[:, 1],
            "tvcalib_pitch_x_m": tv_pitch_valid[:, 0],
            "tvcalib_pitch_y_m": tv_pitch_valid[:, 1],
            "tvcalib_image_x_px": tv_image_valid[:, 0],
            "tvcalib_image_y_px": tv_image_valid[:, 1],
            "pitch_error_m": pitch_errors,
            "reprojection_error_px": reprojection_errors,
        }
    )


def choose_visualization_indices(
    image_points_ref: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Wybiera kilka możliwie dobrze rozdzielonych punktów do PNG.
    """
    n = len(image_points_ref)

    if n == 0 or count <= 0:
        return np.empty((0,), dtype=int)

    count = min(count, n)

    center = image_points_ref.mean(axis=0)
    first = int(
        np.argmin(
            np.linalg.norm(image_points_ref - center, axis=1)
        )
    )

    selected = [first]

    while len(selected) < count:
        candidates = [i for i in range(n) if i not in selected]

        best_idx = None
        best_distance = -1.0

        for idx in candidates:
            distance = min(
                np.linalg.norm(
                    image_points_ref[idx] - image_points_ref[s]
                )
                for s in selected
            )

            if distance > best_distance:
                best_distance = distance
                best_idx = idx

        selected.append(int(best_idx))

    return np.asarray(selected, dtype=int)


def draw_projected_pitch(
    ax,
    h_pitch2img: np.ndarray,
    pitch_length: float,
    pitch_width: float,
    linestyle: str,
    label: str,
) -> None:
    """
    Rysuje model boiska na klatce.
    """
    first = True

    for poly_world in pitch_drawing_polylines(
        pitch_length,
        pitch_width,
    ):
        poly_img = transform_points(h_pitch2img, poly_world)
        finite = np.isfinite(poly_img).all(axis=1)

        if not finite.any():
            continue

        ax.plot(
            poly_img[finite, 0],
            poly_img[finite, 1],
            linestyle=linestyle,
            linewidth=1.0,
            alpha=0.70,
            label=label if first else None,
        )

        first = False


def create_visualization(
    image_bgr: np.ndarray,
    eval_stem: str,
    h_gt_pitch2img: np.ndarray,
    h_tv_pitch2img: np.ndarray,
    point_df: pd.DataFrame,
    output_path: Path,
    pitch_length: float,
    pitch_width: float,
    num_vis_points: int,
) -> None:
    """
    Lewo:
        obraz + projekcja H_GT i TVCalib,
        mały punkt referencyjny i TVCalib reprojection.

    Prawo:
        punkt referencyjny na boisku oraz punkt otrzymany z TVCalib.

    Markery są celowo mniejsze niż w generate_homography_gt.py.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_bgr.shape[:2]

    pitch_ref = point_df[
        ["ref_pitch_x_m", "ref_pitch_y_m"]
    ].to_numpy(dtype=np.float64)

    image_ref = point_df[
        ["ref_image_x_px", "ref_image_y_px"]
    ].to_numpy(dtype=np.float64)

    tv_pitch = point_df[
        ["tvcalib_pitch_x_m", "tvcalib_pitch_y_m"]
    ].to_numpy(dtype=np.float64)

    tv_image = point_df[
        ["tvcalib_image_x_px", "tvcalib_image_y_px"]
    ].to_numpy(dtype=np.float64)

    pitch_errors = point_df[
        "pitch_error_m"
    ].to_numpy(dtype=np.float64)

    reprojection_errors = point_df[
        "reprojection_error_px"
    ].to_numpy(dtype=np.float64)

    vis_ids = choose_visualization_indices(
        image_ref,
        num_vis_points,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_img, ax_pitch = axes

    # LEWY PANEL ------------------------------------------------------
    ax_img.imshow(image_rgb)
    ax_img.set_xlim(0, w_img)
    ax_img.set_ylim(h_img, 0)
    ax_img.set_title(f"{eval_stem}: H_GT vs TVCalib")
    ax_img.axis("off")

    draw_projected_pitch(
        ax_img,
        h_gt_pitch2img,
        pitch_length,
        pitch_width,
        linestyle="-",
        label="H_GT pitch projection",
    )

    draw_projected_pitch(
        ax_img,
        h_tv_pitch2img,
        pitch_length,
        pitch_width,
        linestyle="--",
        label="TVCalib pitch projection",
    )

    if len(vis_ids):
        ref_uv = image_ref[vis_ids]
        tv_uv = tv_image[vis_ids]

        ax_img.scatter(
            ref_uv[:, 0],
            ref_uv[:, 1],
            s=24,
            marker="o",
            label="reference pixel (GT)",
            zorder=5,
        )

        ax_img.scatter(
            tv_uv[:, 0],
            tv_uv[:, 1],
            s=28,
            marker="x",
            linewidths=1.4,
            label="TVCalib reprojection",
            zorder=6,
        )

        for local_i, point_idx in enumerate(vis_ids, start=1):
            ref = image_ref[point_idx]
            est = tv_image[point_idx]

            ax_img.plot(
                [ref[0], est[0]],
                [ref[1], est[1]],
                linewidth=0.9,
                alpha=0.7,
            )

            ax_img.annotate(
                f"P{local_i}: {reprojection_errors[point_idx]:.1f}px",
                (ref[0], ref[1]),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.18", alpha=0.65),
            )

    ax_img.legend(loc="lower left", fontsize=8)

    # PRAWY PANEL -----------------------------------------------------
    draw_top_down_pitch(
        ax_pitch,
        pitch_length,
        pitch_width,
    )

    ax_pitch.set_title("Reference pitch point vs TVCalib result")

    if len(vis_ids):
        ref_xy = pitch_ref[vis_ids]
        tv_xy = tv_pitch[vis_ids]

        ax_pitch.scatter(
            ref_xy[:, 0],
            ref_xy[:, 1],
            s=26,
            marker="o",
            label="GT reference",
            zorder=5,
        )

        ax_pitch.scatter(
            tv_xy[:, 0],
            tv_xy[:, 1],
            s=30,
            marker="x",
            linewidths=1.5,
            label="TVCalib",
            zorder=6,
        )

        for local_i, point_idx in enumerate(vis_ids, start=1):
            ref = pitch_ref[point_idx]
            est = tv_pitch[point_idx]

            ax_pitch.plot(
                [ref[0], est[0]],
                [ref[1], est[1]],
                linewidth=0.9,
                alpha=0.7,
            )

            ax_pitch.annotate(
                f"P{local_i}: {pitch_errors[point_idx]:.2f} m",
                (ref[0], ref[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.18", alpha=0.65),
            )

    ax_pitch.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "TVCalib homography evaluation against annotation-derived H_GT",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EWALUACJA JEDNEJ KLATKI
# =============================================================================

def evaluate_one(
    sample: dict,
    calibrator: TVCalibHomography,
    pitch_grid: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[dict, pd.DataFrame]:

    eval_id = sample["eval_id"]
    eval_stem = sample["eval_stem"]
    image_path = sample["image_path"]
    gt_path = sample["gt_path"]

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(f"OpenCV nie może odczytać: {image_path}")

    image_height, image_width = image_bgr.shape[:2]

    if (
        image_width != calibrator.image_width
        or image_height != calibrator.image_height
    ):
        raise ValueError(
            f"{eval_stem} ma rozmiar {image_width}x{image_height}, "
            f"a kalibrator jest dla "
            f"{calibrator.image_width}x{calibrator.image_height}."
        )

    gt = load_gt(gt_path)
    h_gt_img2pitch = gt["_H_image_to_pitch"]
    h_gt_pitch2img = gt["_H_pitch_to_image"]

    pitch_points_ref, image_points_ref = visible_reference_points(
        pitch_grid=pitch_grid,
        h_gt_pitch2img=h_gt_pitch2img,
        image_width=image_width,
        image_height=image_height,
        image_margin_px=args.image_margin_px,
    )

    if len(pitch_points_ref) < 4:
        raise RuntimeError(
            f"{eval_stem}: tylko {len(pitch_points_ref)} widoczne punkty testowe."
        )

    # TVCalib — ta sama publiczna metoda co w pipeline.
    # Mierzymy segmentację + kalibrację, bez inicjalizacji modelu.
    sync_cuda()
    t0 = time.perf_counter()

    h_tv_img2pitch_raw = calibrator.get_homography(str(image_path))

    sync_cuda()
    estimation_time_ms = (time.perf_counter() - t0) * 1000.0

    if h_tv_img2pitch_raw is None:
        return (
            {
                "eval_id": eval_id,
                "eval_stem": eval_stem,
                "status": "failed",
                "estimation_time_ms": estimation_time_ms,
                "n_reference_points": len(pitch_points_ref),
                "n_valid_points": 0,
                "mean_pitch_error_m": np.nan,
                "median_pitch_error_m": np.nan,
                "p90_pitch_error_m": np.nan,
                "max_pitch_error_m": np.nan,
                "mean_reprojection_error_px": np.nan,
                "median_reprojection_error_px": np.nan,
                "p90_reprojection_error_px": np.nan,
                "max_reprojection_error_px": np.nan,
                "error": "TVCalibHomography.get_homography() returned None",
            },
            pd.DataFrame(),
        )

    h_tv_img2pitch = normalize_h(h_tv_img2pitch_raw)
    h_tv_pitch2img = normalize_h(np.linalg.inv(h_tv_img2pitch))

    point_df = calculate_point_errors(
        pitch_points_ref=pitch_points_ref,
        image_points_ref=image_points_ref,
        h_tv_img2pitch=h_tv_img2pitch,
        h_tv_pitch2img=h_tv_pitch2img,
    )

    if point_df.empty:
        raise RuntimeError(
            f"{eval_stem}: TVCalib zwrócił H, ale brak skończonych punktów."
        )

    point_df.insert(0, "point_id", np.arange(1, len(point_df) + 1))
    point_df.insert(0, "eval_stem", eval_stem)
    point_df.insert(0, "eval_id", eval_id)
    point_df.insert(3, "method", "tvcalib")

    pitch_errors = point_df["pitch_error_m"].to_numpy(dtype=np.float64)
    reproj_errors = point_df["reprojection_error_px"].to_numpy(dtype=np.float64)

    row = {
        "eval_id": eval_id,
        "eval_stem": eval_stem,
        "status": "ok",
        "estimation_time_ms": estimation_time_ms,
        "n_reference_points": len(pitch_points_ref),
        "n_valid_points": len(point_df),

        "mean_pitch_error_m": float(np.mean(pitch_errors)),
        "median_pitch_error_m": float(np.median(pitch_errors)),
        "p90_pitch_error_m": percentile(pitch_errors, 90),
        "max_pitch_error_m": float(np.max(pitch_errors)),

        "mean_reprojection_error_px": float(np.mean(reproj_errors)),
        "median_reprojection_error_px": float(np.median(reproj_errors)),
        "p90_reprojection_error_px": percentile(reproj_errors, 90),
        "max_reprojection_error_px": float(np.max(reproj_errors)),

        "error": "",
    }

    row.update(flatten_h("H_tv_img2pitch", h_tv_img2pitch))
    row.update(flatten_h("H_gt_img2pitch", h_gt_img2pitch))

    # Zapis oszacowanej H --------------------------------------------
    homography_dir = output_dir / "estimated_homographies"
    homography_dir.mkdir(parents=True, exist_ok=True)

    h_json_path = homography_dir / f"{eval_stem}_tvcalib.json"

    with h_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_id": eval_id,
                "eval_stem": eval_stem,
                "method": "tvcalib",
                "optim_steps": args.optim_steps,
                "lens_dist": bool(args.lens_dist),
                "estimation_time_ms": estimation_time_ms,
                "H_image_to_pitch": h_tv_img2pitch.tolist(),
                "H_pitch_to_image": h_tv_pitch2img.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Wizualizacja ----------------------------------------------------
    if not args.no_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        vis_path = vis_dir / f"{eval_stem}_tvcalib_vs_gt.png"

        create_visualization(
            image_bgr=image_bgr,
            eval_stem=eval_stem,
            h_gt_pitch2img=h_gt_pitch2img,
            h_tv_pitch2img=h_tv_pitch2img,
            point_df=point_df,
            output_path=vis_path,
            pitch_length=PITCH_LENGTH_M,
            pitch_width=PITCH_WIDTH_M,
            num_vis_points=args.num_vis_points,
        )

    return row, point_df


# =============================================================================
# PODSUMOWANIE
# =============================================================================

def build_summary(
    per_frame: pd.DataFrame,
    point_errors: pd.DataFrame,
) -> pd.DataFrame:

    total_frames = len(per_frame)
    successful = per_frame[per_frame["status"] == "ok"].copy()

    summary = {
        "method": "tvcalib",
        "total_frames": total_frames,
        "successful_frames": len(successful),
        "failed_frames": total_frames - len(successful),
        "success_rate": (
            len(successful) / total_frames
            if total_frames
            else 0.0
        ),
    }

    if not successful.empty:
        pitch_values = point_errors[
            "pitch_error_m"
        ].to_numpy(dtype=np.float64)

        reproj_values = point_errors[
            "reprojection_error_px"
        ].to_numpy(dtype=np.float64)

        time_values = successful[
            "estimation_time_ms"
        ].to_numpy(dtype=np.float64)

        summary.update(
            {
                "mean_pitch_error_m": float(np.mean(pitch_values)),
                "median_pitch_error_m": float(np.median(pitch_values)),
                "p90_pitch_error_m": percentile(pitch_values, 90),

                "mean_reprojection_error_px": float(np.mean(reproj_values)),
                "median_reprojection_error_px": float(np.median(reproj_values)),
                "p90_reprojection_error_px": percentile(reproj_values, 90),

                "mean_estimation_time_ms": float(np.mean(time_values)),
                "median_estimation_time_ms": float(np.median(time_values)),

                "total_evaluation_points": int(len(point_errors)),
            }
        )

    else:
        summary.update(
            {
                "mean_pitch_error_m": np.nan,
                "median_pitch_error_m": np.nan,
                "p90_pitch_error_m": np.nan,
                "mean_reprojection_error_px": np.nan,
                "median_reprojection_error_px": np.nan,
                "p90_reprojection_error_px": np.nan,
                "mean_estimation_time_ms": np.nan,
                "median_estimation_time_ms": np.nan,
                "total_evaluation_points": 0,
            }
        )

    return pd.DataFrame([summary])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.optim_steps <= 0:
        raise ValueError("--optim-steps musi być > 0.")
    if args.grid_step_m <= 0:
        raise ValueError("--grid-step-m musi być > 0.")
    if args.pitch_margin_m < 0:
        raise ValueError("--pitch-margin-m nie może być ujemny.")
    if args.image_margin_px < 0:
        raise ValueError("--image-margin-px nie może być ujemny.")
    if args.num_vis_points < 1:
        raise ValueError("--num-vis-points musi być >= 1.")

    ensure_exists(TVCALIB_WEIGHTS, "wag TVCalib train_59.pt")

    manifest = load_manifest()
    samples = select_samples(
        manifest=manifest,
        mode=args.mode,
        eval_id=args.eval_id,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    first_image = cv2.imread(str(samples[0]["image_path"]))
    if first_image is None:
        raise RuntimeError(
            f"Nie udało się odczytać {samples[0]['image_path']}"
        )

    image_height, image_width = first_image.shape[:2]

    print("========================================")
    print("EWALUACJA HOMOGRAFII — TVCALIB")
    print("========================================")
    print(f"Tryb: {args.mode}")
    print(f"Liczba klatek: {len(samples)}")
    print(f"Rozdzielczość: {image_width}x{image_height}")
    print(f"OPTIM_STEPS: {args.optim_steps}")
    print(f"lens_dist: {args.lens_dist}")
    print(f"Grid step: {args.grid_step_m} m")
    print(f"Pitch margin: {args.pitch_margin_m} m")
    print(f"Wyniki: {output_dir}")
    print()

    # Identycznie jak w pipeline:
    calibrator = TVCalibHomography(
        model_weights=str(TVCALIB_WEIGHTS),
        image_width=image_width,
        image_height=image_height,
        optim_steps=args.optim_steps,
        lens_dist=args.lens_dist,
    )

    pitch_grid = build_canonical_pitch_grid(
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        step_m=args.grid_step_m,
        margin_m=args.pitch_margin_m,
    )

    per_frame_rows: List[dict] = []
    point_tables: List[pd.DataFrame] = []

    for index, sample in enumerate(samples, start=1):
        eval_stem = sample["eval_stem"]

        print(
            f"[{index:03d}/{len(samples):03d}] {eval_stem} ...",
            end=" ",
            flush=True,
        )

        try:
            row, point_df = evaluate_one(
                sample=sample,
                calibrator=calibrator,
                pitch_grid=pitch_grid,
                output_dir=output_dir,
                args=args,
            )

        except Exception as exc:
            row = {
                "eval_id": sample["eval_id"],
                "eval_stem": eval_stem,
                "status": "failed",
                "estimation_time_ms": np.nan,
                "n_reference_points": np.nan,
                "n_valid_points": 0,
                "mean_pitch_error_m": np.nan,
                "median_pitch_error_m": np.nan,
                "p90_pitch_error_m": np.nan,
                "max_pitch_error_m": np.nan,
                "mean_reprojection_error_px": np.nan,
                "median_reprojection_error_px": np.nan,
                "p90_reprojection_error_px": np.nan,
                "max_reprojection_error_px": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
            }
            point_df = pd.DataFrame()

        per_frame_rows.append(row)

        if not point_df.empty:
            point_tables.append(point_df)

        if row["status"] == "ok":
            print(
                f"OK | "
                f"pitch={row['mean_pitch_error_m']:.3f} m | "
                f"reproj={row['mean_reprojection_error_px']:.2f} px | "
                f"time={row['estimation_time_ms']:.1f} ms"
            )
        else:
            print(f"FAILED | {row.get('error', '')}")

    per_frame_df = (
        pd.DataFrame(per_frame_rows)
        .sort_values("eval_id")
        .reset_index(drop=True)
    )

    if point_tables:
        point_errors_df = (
            pd.concat(point_tables, ignore_index=True)
            .sort_values(["eval_id", "point_id"])
            .reset_index(drop=True)
        )
    else:
        point_errors_df = pd.DataFrame(
            columns=[
                "eval_id",
                "eval_stem",
                "point_id",
                "method",
                "ref_pitch_x_m",
                "ref_pitch_y_m",
                "ref_image_x_px",
                "ref_image_y_px",
                "tvcalib_pitch_x_m",
                "tvcalib_pitch_y_m",
                "tvcalib_image_x_px",
                "tvcalib_image_y_px",
                "pitch_error_m",
                "reprojection_error_px",
            ]
        )

    summary_df = build_summary(
        per_frame=per_frame_df,
        point_errors=point_errors_df,
    )

    per_frame_path = output_dir / "per_frame_metrics.csv"
    point_errors_path = output_dir / "point_errors.csv"
    summary_path = output_dir / "metrics_summary.csv"

    per_frame_df.to_csv(per_frame_path, index=False)
    point_errors_df.to_csv(point_errors_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    config = {
        "timestamp": timestamp,
        "method": "tvcalib",
        "mode": args.mode,
        "eval_id": args.eval_id if args.mode == "single" else None,
        "dataset_root": str(EVAL_ROOT),
        "tvcalib_weights": str(TVCALIB_WEIGHTS),
        "image_width": image_width,
        "image_height": image_height,
        "optim_steps": args.optim_steps,
        "lens_dist": bool(args.lens_dist),
        "pitch_length_m": PITCH_LENGTH_M,
        "pitch_width_m": PITCH_WIDTH_M,
        "grid_step_m": args.grid_step_m,
        "pitch_margin_m": args.pitch_margin_m,
        "image_margin_px": args.image_margin_px,
        "num_visualization_points": args.num_vis_points,
        "visualizations": not args.no_visualizations,
        "success_definition": (
            "TVCalib independently returns a finite, invertible homography "
            "for the evaluated frame; no previous-frame fallback is used."
        ),
        "timing_definition": (
            "TVCalib get_homography() only: segmentation + calibration. "
            "Model initialization is excluded."
        ),
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    summary = summary_df.iloc[0]

    print("\n========================================")
    print("PODSUMOWANIE TVCALIB")
    print("========================================")
    print(
        f"Success rate: "
        f"{100.0 * summary['success_rate']:.2f}% "
        f"({int(summary['successful_frames'])}/"
        f"{int(summary['total_frames'])})"
    )

    if int(summary["successful_frames"]) > 0:
        print(
            f"Pitch error: mean="
            f"{summary['mean_pitch_error_m']:.3f} m, "
            f"median={summary['median_pitch_error_m']:.3f} m"
        )
        print(
            f"Reprojection error: mean="
            f"{summary['mean_reprojection_error_px']:.2f} px, "
            f"median={summary['median_reprojection_error_px']:.2f} px"
        )
        print(
            f"Estimation time: mean="
            f"{summary['mean_estimation_time_ms']:.1f} ms, "
            f"median={summary['median_estimation_time_ms']:.1f} ms"
        )

    print(f"\nSummary:   {summary_path}")
    print(f"Per-frame: {per_frame_path}")
    print(f"Points:    {point_errors_path}")

    if not args.no_visualizations:
        print(f"Visualizations: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
