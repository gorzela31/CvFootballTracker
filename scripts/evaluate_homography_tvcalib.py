#!/usr/bin/env python3
"""
Ewaluacja homografii TVCalib wzgledem referencyjnej H_GT.

Najwazniejsze zalozenia:
1. TVCalib otrzymuje ORYGINALNA klatke - bez zewnetrznego STRETCH do 640x640.
   Preprocessing wymagany przez model segmentacji jest realizowany wewnetrznie przez
   TVCalibHomography.
2. Homografia jest estymowana w kierunku IMAGE [px] -> PITCH [m].
3. Kazda klatka jest estymowana niezaleznie, bez previous-H fallback.
4. Jakosc H jest oceniana na tych samych dwoch punktach P1/P2 co w ewaluacji
   metody keypointowej. Punkty sa wybierane niezaleznie od badanej metody na
   podstawie H_GT i maja reprezentowac przykladowe polozenia zawodnikow.
5. Dla P1/P2 liczone sa:
   - blad reprojekcji [px] w ORYGINALNEJ rozdzielczosci obrazu,
   - blad odwzorowania na plaszczyznie boiska [m].
6. P1/P2 sa wybierane deterministycznie z regularnej siatki boiska:
   - P1: punkt najblizszy srodkowi widocznego obszaru,
   - P2: punkt najbardziej oddalony od P1 na obrazie.
7. Czas estymacji obejmuje get_homography(), czyli segmentacje i kalibracje.
   Inicjalizacja modelu i warm-up sa poza pomiarem. W podsumowaniu czas jest
   liczony dla wszystkich prob z dostepnym pomiarem, rowniez nieudanych.

Uruchomienie jednej klatki:
    python scripts/evaluate_homography_tvcalib.py --mode single --eval-id 1

Uruchomienie calej puli:
    python scripts/evaluate_homography_tvcalib.py --mode all

Wyniki:
    results/homography_evaluation/<timestamp>_tvcalib_p1p2/
        metrics_summary.csv
        per_frame_metrics.csv
        point_errors.csv
        config.json
        estimated_homographies/
        visualizations/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

# Tak samo jak w pipeline.
OPTIM_STEPS = 500

# Siatka sluzy WYLACZNIE do deterministycznego wyboru P1/P2.
GRID_STEP_M = 5.0
PITCH_MARGIN_M = 5.0
IMAGE_MARGIN_PX = 5.0
NUM_EVAL_POINTS = 2


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ewaluacja homografii TVCalib wzgledem H_GT "
            "na punktach testowych P1/P2."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        default="single",
        help="single = jedna klatka, all = cala pula. Domyslnie single.",
    )
    parser.add_argument(
        "--eval-id",
        type=int,
        default=1,
        help="Numer eval_id dla --mode single. Domyslnie 1.",
    )
    parser.add_argument(
        "--optim-steps",
        type=int,
        default=OPTIM_STEPS,
        help=f"Liczba krokow optymalizacji TVCalib. Domyslnie {OPTIM_STEPS}.",
    )
    parser.add_argument(
        "--grid-step-m",
        type=float,
        default=GRID_STEP_M,
        help=(
            "Krok pomocniczej siatki, z ktorej wybierane sa P1/P2 [m]. "
            f"Domyslnie {GRID_STEP_M}."
        ),
    )
    parser.add_argument(
        "--pitch-margin-m",
        type=float,
        default=PITCH_MARGIN_M,
        help=(
            "Margines pomocniczej siatki od krawedzi boiska [m]. "
            f"Domyslnie {PITCH_MARGIN_M}."
        ),
    )
    parser.add_argument(
        "--image-margin-px",
        type=float,
        default=IMAGE_MARGIN_PX,
        help=(
            "Margines widocznosci kandydata P1/P2 na obrazie [px]. "
            f"Domyslnie {IMAGE_MARGIN_PX}."
        ),
    )
    parser.add_argument(
        "--lens-dist",
        action="store_true",
        help="Wlacz model dystorsji TVCalib. Pipeline domyslnie go nie uzywa.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Nie generuj PNG z porownaniem H_est vs H_GT i P1/P2.",
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
        raise ValueError(f"Homografia ma nieprawidlowy ksztalt: {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError("Homografia zawiera NaN/Inf.")

    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    else:
        norm = np.linalg.norm(h)
        if norm <= 1e-12:
            raise ValueError("Homografia ma zerowa norme.")
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


def load_manifest() -> pd.DataFrame:
    ensure_exists(EVAL_MANIFEST, "manifest.csv")
    ensure_exists(EVAL_IMAGES, "katalogu obrazow ewaluacyjnych")
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
        selected = manifest[manifest["eval_id"] == eval_id]
        if selected.empty:
            raise ValueError(
                f"Nie znaleziono eval_id={eval_id}. "
                f"Dostepny zakres: {int(manifest['eval_id'].min())}-"
                f"{int(manifest['eval_id'].max())}."
            )
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

    return {
        **data,
        "_H_image_to_pitch": normalize_h(
            np.asarray(data["H_image_to_pitch"], dtype=np.float64)
        ),
        "_H_pitch_to_image": normalize_h(
            np.asarray(data["H_pitch_to_image"], dtype=np.float64)
        ),
    }


def build_canonical_pitch_grid(
    pitch_length: float,
    pitch_width: float,
    step_m: float,
    margin_m: float,
) -> np.ndarray:
    """Pomocnicza regularna siatka, z ktorej wybierane sa P1/P2."""
    if step_m <= 0:
        raise ValueError("--grid-step-m musi byc > 0.")
    if margin_m < 0:
        raise ValueError("--pitch-margin-m nie moze byc ujemny.")

    x_min = -pitch_length / 2.0 + margin_m
    x_max = +pitch_length / 2.0 - margin_m
    y_min = -pitch_width / 2.0 + margin_m
    y_max = +pitch_width / 2.0 - margin_m
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Margines boiska jest zbyt duzy.")

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
    """Zwraca punkty siatki, ktore wedlug H_GT sa widoczne na obrazie."""
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


def choose_evaluation_indices(image_points_ref: np.ndarray) -> np.ndarray:
    """
    Deterministyczny wybor tych samych dwoch punktow P1/P2 co dla keypoints.

    P1: kandydat najblizszy centroidowi widocznych punktow na obrazie.
    P2: kandydat najbardziej oddalony od P1 w przestrzeni obrazu.

    Wybor wykorzystuje wylacznie punkty referencyjne z H_GT, a nie H_est.
    """
    n = len(image_points_ref)
    if n < NUM_EVAL_POINTS:
        raise ValueError(
            f"Za malo widocznych kandydatow do wyboru P1/P2: {n}."
        )

    center = image_points_ref.mean(axis=0)
    p1 = int(np.argmin(np.linalg.norm(image_points_ref - center, axis=1)))

    distances = np.linalg.norm(image_points_ref - image_points_ref[p1], axis=1)
    distances[p1] = -np.inf
    p2 = int(np.argmax(distances))

    return np.asarray([p1, p2], dtype=int)


# =============================================================================
# METRYKI P1/P2
# =============================================================================

def calculate_point_errors(
    pitch_points_ref: np.ndarray,
    image_points_ref: np.ndarray,
    h_est_img2pitch: np.ndarray,
    h_est_pitch2img: np.ndarray,
) -> pd.DataFrame:
    """Liczy bledy dla P1/P2 w metrach i w oryginalnych pikselach obrazu."""
    est_pitch_points = transform_points(h_est_img2pitch, image_points_ref)
    est_image_points = transform_points(h_est_pitch2img, pitch_points_ref)

    valid = (
        np.isfinite(est_pitch_points).all(axis=1)
        & np.isfinite(est_image_points).all(axis=1)
    )

    pitch_ref_valid = pitch_points_ref[valid]
    image_ref_valid = image_points_ref[valid]
    est_pitch_valid = est_pitch_points[valid]
    est_image_valid = est_image_points[valid]

    pitch_errors = np.linalg.norm(est_pitch_valid - pitch_ref_valid, axis=1)
    reprojection_errors = np.linalg.norm(est_image_valid - image_ref_valid, axis=1)

    return pd.DataFrame(
        {
            "ref_pitch_x_m": pitch_ref_valid[:, 0],
            "ref_pitch_y_m": pitch_ref_valid[:, 1],
            "ref_image_x_px": image_ref_valid[:, 0],
            "ref_image_y_px": image_ref_valid[:, 1],
            "estimated_pitch_x_m": est_pitch_valid[:, 0],
            "estimated_pitch_y_m": est_pitch_valid[:, 1],
            "estimated_image_x_px": est_image_valid[:, 0],
            "estimated_image_y_px": est_image_valid[:, 1],
            "pitch_error_m": pitch_errors,
            "reprojection_error_px": reprojection_errors,
        }
    )


# =============================================================================
# WIZUALIZACJA
# =============================================================================

def draw_projected_pitch(
    ax,
    h_pitch2img: np.ndarray,
    pitch_length: float,
    pitch_width: float,
    linestyle: str,
    label: str,
) -> None:
    first = True
    for poly_world in pitch_drawing_polylines(pitch_length, pitch_width):
        poly_img = transform_points(h_pitch2img, poly_world)
        finite = np.isfinite(poly_img).all(axis=1)
        if not finite.any():
            continue
        ax.plot(
            poly_img[finite, 0],
            poly_img[finite, 1],
            linestyle=linestyle,
            linewidth=1.0,
            alpha=0.75,
            label=label if first else None,
        )
        first = False


def create_visualization(
    image_bgr: np.ndarray,
    eval_stem: str,
    h_gt_pitch2img: np.ndarray,
    h_est_pitch2img: np.ndarray,
    point_df: pd.DataFrame,
    output_path: Path,
    pitch_length: float,
    pitch_width: float,
) -> None:
    """
    Prosta wizualizacja identyczna metodologicznie jak dla keypoints.

    Legenda:
    - lewy panel: H_GT, H_est, P1/P2 - GT, P1/P2 - H_est,
    - prawy panel: GT, H_est.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_bgr.shape[:2]

    ref_pitch = point_df[["ref_pitch_x_m", "ref_pitch_y_m"]].to_numpy(float)
    ref_image = point_df[["ref_image_x_px", "ref_image_y_px"]].to_numpy(float)
    est_pitch = point_df[["estimated_pitch_x_m", "estimated_pitch_y_m"]].to_numpy(float)
    est_image = point_df[["estimated_image_x_px", "estimated_image_y_px"]].to_numpy(float)
    pitch_errors = point_df["pitch_error_m"].to_numpy(float)
    reproj_errors = point_df["reprojection_error_px"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_img, ax_pitch = axes

    ax_img.imshow(image_rgb)
    ax_img.set_xlim(0, w_img)
    ax_img.set_ylim(h_img, 0)
    ax_img.set_title("Porownanie odwzorowania na obrazie")
    ax_img.axis("off")

    draw_projected_pitch(
        ax_img,
        h_gt_pitch2img,
        pitch_length,
        pitch_width,
        linestyle="-",
        label="H_GT",
    )
    draw_projected_pitch(
        ax_img,
        h_est_pitch2img,
        pitch_length,
        pitch_width,
        linestyle="--",
        label="H_est",
    )

    ax_img.scatter(
        ref_image[:, 0],
        ref_image[:, 1],
        s=34,
        marker="o",
        label="P1/P2 - GT",
        zorder=6,
    )
    ax_img.scatter(
        est_image[:, 0],
        est_image[:, 1],
        s=42,
        marker="x",
        linewidths=1.5,
        label="P1/P2 - H_est",
        zorder=7,
    )

    for i in range(len(point_df)):
        label = f"P{i + 1}"
        ax_img.plot(
            [ref_image[i, 0], est_image[i, 0]],
            [ref_image[i, 1], est_image[i, 1]],
            linewidth=0.9,
            alpha=0.8,
        )
        ax_img.annotate(
            f"{label}: {reproj_errors[i]:.1f} px",
            (ref_image[i, 0], ref_image[i, 1]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", alpha=0.65),
        )

    ax_img.legend(loc="lower left", fontsize=8, ncol=2)

    draw_top_down_pitch(ax_pitch, pitch_length, pitch_width)
    ax_pitch.set_title("Blad odwzorowania na plaszczyznie boiska")

    ax_pitch.scatter(
        ref_pitch[:, 0],
        ref_pitch[:, 1],
        s=34,
        marker="o",
        label="GT",
        zorder=6,
    )
    ax_pitch.scatter(
        est_pitch[:, 0],
        est_pitch[:, 1],
        s=42,
        marker="x",
        linewidths=1.5,
        label="H_est",
        zorder=7,
    )

    for i in range(len(point_df)):
        label = f"P{i + 1}"
        ax_pitch.plot(
            [ref_pitch[i, 0], est_pitch[i, 0]],
            [ref_pitch[i, 1], est_pitch[i, 1]],
            linewidth=0.9,
            alpha=0.8,
        )
        ax_pitch.annotate(
            f"{label}: {pitch_errors[i]:.2f} m",
            (ref_pitch[i, 0], ref_pitch[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", alpha=0.65),
        )

    ax_pitch.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{eval_stem} - ewaluacja TVCalib dla P1 i P2", fontsize=13)
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
        raise RuntimeError(f"OpenCV nie moze odczytac: {image_path}")

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

    pitch_candidates, image_candidates = visible_reference_points(
        pitch_grid=pitch_grid,
        h_gt_pitch2img=h_gt_pitch2img,
        image_width=image_width,
        image_height=image_height,
        image_margin_px=args.image_margin_px,
    )

    if len(pitch_candidates) < NUM_EVAL_POINTS:
        raise RuntimeError(
            f"{eval_stem}: tylko {len(pitch_candidates)} widoczne punkty-kandydaci."
        )

    eval_indices = choose_evaluation_indices(image_candidates)
    pitch_points_ref = pitch_candidates[eval_indices]
    image_points_ref = image_candidates[eval_indices]

    # Pelny etap TVCalib: segmentacja + kalibracja.
    # Inicjalizacja i warm-up sa poza pomiarem.
    sync_cuda()
    t0 = time.perf_counter()
    h_est_img2pitch_raw = calibrator.get_homography(str(image_path))
    sync_cuda()
    estimation_time_ms = (time.perf_counter() - t0) * 1000.0

    base_row = {
        "eval_id": eval_id,
        "eval_stem": eval_stem,
        "estimation_time_ms": estimation_time_ms,
        "n_candidate_points": len(pitch_candidates),
        "n_evaluation_points": NUM_EVAL_POINTS,
    }

    if h_est_img2pitch_raw is None:
        return (
            {
                **base_row,
                "status": "failed",
                "failure_reason": "tvcalib_returned_none",
                "mean_pitch_error_m": np.nan,
                "median_pitch_error_m": np.nan,
                "mean_reprojection_error_px": np.nan,
                "median_reprojection_error_px": np.nan,
                "p1_pitch_error_m": np.nan,
                "p2_pitch_error_m": np.nan,
                "p1_reprojection_error_px": np.nan,
                "p2_reprojection_error_px": np.nan,
                "error": "TVCalibHomography.get_homography() returned None",
            },
            pd.DataFrame(),
        )

    h_est_img2pitch = normalize_h(h_est_img2pitch_raw)
    h_est_pitch2img = normalize_h(np.linalg.inv(h_est_img2pitch))

    point_df = calculate_point_errors(
        pitch_points_ref=pitch_points_ref,
        image_points_ref=image_points_ref,
        h_est_img2pitch=h_est_img2pitch,
        h_est_pitch2img=h_est_pitch2img,
    )

    if len(point_df) != NUM_EVAL_POINTS:
        raise RuntimeError(
            f"{eval_stem}: oczekiwano {NUM_EVAL_POINTS} poprawnych P1/P2, "
            f"otrzymano {len(point_df)}."
        )

    point_df.insert(0, "point_label", ["P1", "P2"])
    point_df.insert(0, "point_id", [1, 2])
    point_df.insert(0, "method", "tvcalib")
    point_df.insert(0, "eval_stem", eval_stem)
    point_df.insert(0, "eval_id", eval_id)

    pitch_errors = point_df["pitch_error_m"].to_numpy(dtype=np.float64)
    reproj_errors = point_df["reprojection_error_px"].to_numpy(dtype=np.float64)

    row = {
        **base_row,
        "status": "ok",
        "failure_reason": "",
        "mean_pitch_error_m": float(np.mean(pitch_errors)),
        "median_pitch_error_m": float(np.median(pitch_errors)),
        "mean_reprojection_error_px": float(np.mean(reproj_errors)),
        "median_reprojection_error_px": float(np.median(reproj_errors)),
        "p1_pitch_error_m": float(pitch_errors[0]),
        "p2_pitch_error_m": float(pitch_errors[1]),
        "p1_reprojection_error_px": float(reproj_errors[0]),
        "p2_reprojection_error_px": float(reproj_errors[1]),
        "error": "",
    }
    row.update(flatten_h("H_tvcalib_img2pitch", h_est_img2pitch))
    row.update(flatten_h("H_gt_img2pitch", h_gt_img2pitch))

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
                "H_image_to_pitch": h_est_img2pitch.tolist(),
                "H_pitch_to_image": h_est_pitch2img.tolist(),
                "evaluation_points": point_df[
                    [
                        "point_label",
                        "ref_pitch_x_m",
                        "ref_pitch_y_m",
                        "ref_image_x_px",
                        "ref_image_y_px",
                        "pitch_error_m",
                        "reprojection_error_px",
                    ]
                ].to_dict("records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.no_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"{eval_stem}_tvcalib_vs_gt_p1p2.png"
        create_visualization(
            image_bgr=image_bgr,
            eval_stem=eval_stem,
            h_gt_pitch2img=h_gt_pitch2img,
            h_est_pitch2img=h_est_pitch2img,
            point_df=point_df,
            output_path=vis_path,
            pitch_length=PITCH_LENGTH_M,
            pitch_width=PITCH_WIDTH_M,
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

    finite_times = per_frame["estimation_time_ms"].to_numpy(dtype=np.float64)
    finite_times = finite_times[np.isfinite(finite_times)]

    summary = {
        "method": "tvcalib",
        "total_frames": total_frames,
        "successful_frames": len(successful),
        "failed_frames": total_frames - len(successful),
        "success_rate": len(successful) / total_frames if total_frames else 0.0,
        "total_evaluation_points": int(len(point_errors)),
        "mean_pitch_error_m": np.nan,
        "median_pitch_error_m": np.nan,
        "mean_reprojection_error_px": np.nan,
        "median_reprojection_error_px": np.nan,
        # Tak samo jak dla keypoints: czas po wszystkich probach z dostepnym timingiem.
        "mean_estimation_time_ms": (
            float(np.mean(finite_times)) if finite_times.size else np.nan
        ),
        "median_estimation_time_ms": (
            float(np.median(finite_times)) if finite_times.size else np.nan
        ),
    }

    if not point_errors.empty:
        pitch_values = point_errors["pitch_error_m"].to_numpy(dtype=np.float64)
        reproj_values = point_errors["reprojection_error_px"].to_numpy(dtype=np.float64)
        summary.update(
            {
                "mean_pitch_error_m": float(np.mean(pitch_values)),
                "median_pitch_error_m": float(np.median(pitch_values)),
                "mean_reprojection_error_px": float(np.mean(reproj_values)),
                "median_reprojection_error_px": float(np.median(reproj_values)),
            }
        )

    return pd.DataFrame([summary])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.optim_steps <= 0:
        raise ValueError("--optim-steps musi byc > 0.")
    if args.grid_step_m <= 0:
        raise ValueError("--grid-step-m musi byc > 0.")
    if args.pitch_margin_m < 0:
        raise ValueError("--pitch-margin-m nie moze byc ujemny.")
    if args.image_margin_px < 0:
        raise ValueError("--image-margin-px nie moze byc ujemny.")

    ensure_exists(TVCALIB_WEIGHTS, "wag TVCalib train_59.pt")

    manifest = load_manifest()
    samples = select_samples(manifest, args.mode, args.eval_id)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = RESULTS_ROOT / f"{timestamp}_tvcalib_p1p2"
    output_dir.mkdir(parents=True, exist_ok=False)

    first_image = cv2.imread(str(samples[0]["image_path"]))
    if first_image is None:
        raise RuntimeError(f"Nie udalo sie odczytac {samples[0]['image_path']}")
    image_height, image_width = first_image.shape[:2]

    print("========================================")
    print("EWALUACJA HOMOGRAFII - TVCALIB - P1/P2")
    print("========================================")
    print(f"Tryb: {args.mode}")
    print(f"Liczba klatek: {len(samples)}")
    print(f"Oryginalna rozdzielczosc: {image_width}x{image_height}")
    print("Zewnetrzny resize: brak (TVCalib obsluguje preprocessing wewnetrznie)")
    print(f"OPTIM_STEPS: {args.optim_steps}")
    print(f"lens_dist: {args.lens_dist}")
    print(f"Punkty ewaluacyjne: P1/P2 ({NUM_EVAL_POINTS} na udana klatke)")
    print("Blad reprojekcji [px]: oryginalna przestrzen obrazu")
    print("Blad pitch [m]: plaszczyzna boiska")
    print(f"Wyniki: {output_dir}\n")

    # Ta sama klasa i parametry jak w pipeline. Nie wykonujemy stretchu 640x640.
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

    # Dla porownywalnosci z keypoints pierwszy pelny przebieg jest warm-upem
    # i nie wchodzi do pomiaru czasu.
    print("Warm-up TVCalib...")
    _ = calibrator.get_homography(str(samples[0]["image_path"]))
    sync_cuda()
    print("Warm-up gotowy.\n")

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
                "failure_reason": "exception",
                "estimation_time_ms": np.nan,
                "n_candidate_points": np.nan,
                "n_evaluation_points": NUM_EVAL_POINTS,
                "mean_pitch_error_m": np.nan,
                "median_pitch_error_m": np.nan,
                "mean_reprojection_error_px": np.nan,
                "median_reprojection_error_px": np.nan,
                "p1_pitch_error_m": np.nan,
                "p2_pitch_error_m": np.nan,
                "p1_reprojection_error_px": np.nan,
                "p2_reprojection_error_px": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
            }
            point_df = pd.DataFrame()

        per_frame_rows.append(row)
        if not point_df.empty:
            point_tables.append(point_df)

        if row["status"] == "ok":
            print(
                f"OK | P1/P2 pitch mean={row['mean_pitch_error_m']:.3f} m | "
                f"reproj mean={row['mean_reprojection_error_px']:.2f} px | "
                f"time={row['estimation_time_ms']:.1f} ms"
            )
        else:
            if np.isfinite(row.get("estimation_time_ms", np.nan)):
                print(
                    f"FAILED | reason={row['failure_reason']} | "
                    f"time={row['estimation_time_ms']:.1f} ms"
                )
            else:
                print(
                    f"FAILED | reason={row['failure_reason']} | "
                    f"{row.get('error', '')}"
                )

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
                "method",
                "point_id",
                "point_label",
                "ref_pitch_x_m",
                "ref_pitch_y_m",
                "ref_image_x_px",
                "ref_image_y_px",
                "estimated_pitch_x_m",
                "estimated_pitch_y_m",
                "estimated_image_x_px",
                "estimated_image_y_px",
                "pitch_error_m",
                "reprojection_error_px",
            ]
        )

    summary_df = build_summary(per_frame_df, point_errors_df)

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
        "original_first_image_width": image_width,
        "original_first_image_height": image_height,
        "external_resize": False,
        "preprocessing": "handled internally by TVCalibHomography",
        "optim_steps": int(args.optim_steps),
        "lens_dist": bool(args.lens_dist),
        "homography_direction": "image pixels -> pitch meters",
        "pitch_length_m": PITCH_LENGTH_M,
        "pitch_width_m": PITCH_WIDTH_M,
        "candidate_grid_step_m": args.grid_step_m,
        "candidate_pitch_margin_m": args.pitch_margin_m,
        "candidate_image_margin_px": args.image_margin_px,
        "evaluation_points_per_successful_frame": NUM_EVAL_POINTS,
        "evaluation_points_selection": (
            "P1 = visible H_GT grid point nearest centroid of visible candidate "
            "points in the original image; P2 = visible candidate farthest from "
            "P1 in the original image. Selection uses H_GT only and is independent "
            "of the evaluated homography method."
        ),
        "reprojection_error_space": "original image pixel coordinates",
        "pitch_error_space": "pitch coordinates in meters",
        "success_definition": (
            "TVCalib independently returns a finite, invertible image-to-pitch H "
            "and both P1/P2 can be evaluated. No previous-frame fallback."
        ),
        "timing_definition": (
            "TVCalib get_homography(): segmentation + calibration. Model "
            "initialization and warm-up excluded. Summary timing uses all frames "
            "with a finite timing value, including failed estimation attempts."
        ),
        "main_error_aggregation": (
            "Mean and median over P1/P2 from all successful frames. "
            "Each successful frame contributes exactly two error observations."
        ),
        "visualizations": not args.no_visualizations,
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    summary = summary_df.iloc[0]
    print("\n========================================")
    print("PODSUMOWANIE TVCALIB - P1/P2")
    print("========================================")
    print(
        f"Success rate: {100.0 * summary['success_rate']:.2f}% "
        f"({int(summary['successful_frames'])}/{int(summary['total_frames'])})"
    )
    print(
        f"Liczba punktow P1/P2 w metrykach: "
        f"{int(summary['total_evaluation_points'])}"
    )

    if int(summary["successful_frames"]) > 0:
        print(
            f"Pitch error P1/P2: mean={summary['mean_pitch_error_m']:.3f} m, "
            f"median={summary['median_pitch_error_m']:.3f} m"
        )
        print(
            f"Reprojection error P1/P2: "
            f"mean={summary['mean_reprojection_error_px']:.2f} px, "
            f"median={summary['median_reprojection_error_px']:.2f} px"
        )
        print(
            f"Estimation time (all attempts): "
            f"mean={summary['mean_estimation_time_ms']:.1f} ms, "
            f"median={summary['median_estimation_time_ms']:.1f} ms"
        )

    print(f"\nSummary:   {summary_path}")
    print(f"Per-frame: {per_frame_path}")
    print(f"P1/P2:     {point_errors_path}")
    if not args.no_visualizations:
        print(f"Visualizations: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
