#!/usr/bin/env python3
"""
Diagnostyczna ewaluacja homografii keypointowej na ORYGINALNYM zbiorze
football-field-detection-16/test.

Cel:
    sprawdzić, czy metoda homografii działa poprawnie w domenie zbliżonej
    do danych treningowych.

GT homografii:
    H_GT jest estymowana z ręcznie oznaczonych keypointów YOLO Pose
    (visibility > 0) oraz PITCH_KEYPOINTS_TEMPLATE_M.

Predykowana homografia:
    obraz -> stretch 640x640 -> YOLOv8x-pose -> keypointy -> powrót do
    oryginalnego rozmiaru -> conf >= 0.6 -> RANSAC IMAGE->PITCH, 10 m.

UWAGA METODOLOGICZNA:
    Ten skrypt traktuj jako sanity check / analizę in-domain.
    H_GT powstaje z tych samych typów punktów charakterystycznych, które
    przewiduje model, dlatego nie jest to tak niezależny benchmark jak
    główna ewaluacja na SoccerNet z H_GT wyznaczoną z linii boiska.

Umieść jako:
    scripts/evaluate_homography_keypoints_original_dataset.py

Uruchomienie:
    python scripts/evaluate_homography_keypoints_original_dataset.py --limit 5
    python scripts/evaluate_homography_keypoints_original_dataset.py

Domyślne ścieżki:
    data/football-field-detection-16/test/images
    data/football-field-detection-16/test/labels
    models/pitch_keypoints/trained_keypoints.pt

Wyniki:
    results/homography_keypoints_original_dataset/<timestamp>/
        metrics_summary.csv
        per_image_metrics.csv
        point_errors.csv
        keypoint_diagnostics.csv
        config.json
        visualizations/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.calibration.keypoints_homography import (
    MIN_KEYPOINTS_FOR_HOMOGRAPHY,
    PITCH_KEYPOINTS_TEMPLATE_M,
)

from generate_homography_gt import (
    draw_top_down_pitch,
    pitch_drawing_polylines,
    transform_points,
)


# =============================================================================
# KONFIGURACJA
# =============================================================================

IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "football-field-detection-16"
    / "test"
    / "images"
)

LABELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "football-field-detection-16"
    / "test"
    / "labels"
)

WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "pitch_keypoints"
    / "trained_keypoints.pt"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "homography_keypoints_original_dataset"
)

NUM_KEYPOINTS = 32
STRETCH_SIZE = 640
CONF_THRESHOLD = 0.60
RANSAC_THRESHOLD_M = 10.0

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

GRID_STEP_M = 5.0
PITCH_MARGIN_M = 5.0
IMAGE_MARGIN_PX = 5.0
NUM_VIS_POINTS = 2

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Opcjonalnie oceniaj tylko pierwsze N obrazów.",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONF_THRESHOLD,
    )

    parser.add_argument(
        "--ransac-threshold-m",
        type=float,
        default=RANSAC_THRESHOLD_M,
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# HELPERY
# =============================================================================

def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Brak {description}: {path}"
        )


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def normalize_h(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)

    if h.shape != (3, 3):
        raise ValueError(
            f"Nieprawidłowy shape H: {h.shape}"
        )

    if not np.isfinite(h).all():
        raise ValueError("H zawiera NaN/Inf.")

    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    else:
        norm = np.linalg.norm(h)
        if norm <= 1e-12:
            raise ValueError("Zerowa homografia.")
        h = h / norm

    if abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("Homografia osobliwa.")

    return h


def collect_pairs() -> List[Tuple[Path, Path]]:
    images = sorted(
        [
            p
            for p in IMAGES_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )

    pairs = []

    for image_path in images:
        label_path = (
            LABELS_DIR
            / f"{image_path.stem}.txt"
        )

        if label_path.exists():
            pairs.append(
                (image_path, label_path)
            )
        else:
            print(
                f"WARNING: brak labela dla "
                f"{image_path.name}"
            )

    return pairs


def parse_yolo_pose_gt(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lines = [
        x.strip()
        for x in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if not lines:
        raise ValueError("Pusty label.")

    values = [
        float(v)
        for v in lines[0].split()
    ]

    expected = 5 + NUM_KEYPOINTS * 3

    if len(values) != expected:
        raise ValueError(
            f"Oczekiwano {expected} wartości, "
            f"otrzymano {len(values)}."
        )

    kp = np.asarray(
        values[5:],
        dtype=np.float64,
    ).reshape(NUM_KEYPOINTS, 3)

    xy = np.empty(
        (NUM_KEYPOINTS, 2),
        dtype=np.float64,
    )

    xy[:, 0] = kp[:, 0] * image_width
    xy[:, 1] = kp[:, 1] * image_height

    visibility = kp[:, 2]

    return xy, visibility


def build_pitch_grid() -> np.ndarray:
    x_min = (
        -PITCH_LENGTH_M / 2
        + PITCH_MARGIN_M
    )
    x_max = (
        PITCH_LENGTH_M / 2
        - PITCH_MARGIN_M
    )
    y_min = (
        -PITCH_WIDTH_M / 2
        + PITCH_MARGIN_M
    )
    y_max = (
        PITCH_WIDTH_M / 2
        - PITCH_MARGIN_M
    )

    xs = np.arange(
        x_min,
        x_max + 1e-9,
        GRID_STEP_M,
    )
    ys = np.arange(
        y_min,
        y_max + 1e-9,
        GRID_STEP_M,
    )

    xx, yy = np.meshgrid(xs, ys)

    return np.column_stack(
        [xx.ravel(), yy.ravel()]
    )


def visible_grid_points(
    pitch_grid: np.ndarray,
    h_gt_pitch2img: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:

    uv = transform_points(
        h_gt_pitch2img,
        pitch_grid,
    )

    valid = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= IMAGE_MARGIN_PX)
        & (
            uv[:, 0]
            < image_width - IMAGE_MARGIN_PX
        )
        & (uv[:, 1] >= IMAGE_MARGIN_PX)
        & (
            uv[:, 1]
            < image_height - IMAGE_MARGIN_PX
        )
    )

    return pitch_grid[valid], uv[valid]


# =============================================================================
# MODEL
# =============================================================================

class Detector:
    def __init__(
        self,
        conf_threshold: float,
        ransac_threshold_m: float,
    ):
        self.conf_threshold = (
            float(conf_threshold)
        )
        self.ransac_threshold_m = (
            float(ransac_threshold_m)
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[OriginalDataset] device="
            f"{self.device}"
        )

        self.model = YOLO(
            str(WEIGHTS)
        )

    def detect(
        self,
        image_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:

        orig_h, orig_w = image_bgr.shape[:2]

        resized = cv2.resize(
            image_bgr,
            (STRETCH_SIZE, STRETCH_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        results = self.model(
            resized,
            imgsz=STRETCH_SIZE,
            verbose=False,
            device=self.device,
        )

        if (
            not results
            or results[0].keypoints is None
            or len(results[0].keypoints) == 0
            or results[0].keypoints.conf is None
        ):
            return None

        kp = results[0].keypoints

        best_idx = int(
            kp.conf.mean(dim=1).argmax()
        )

        xy = (
            kp.xy[best_idx]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        conf = (
            kp.conf[best_idx]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if (
            xy.shape != (32, 2)
            or conf.shape != (32,)
        ):
            return None

        xy[:, 0] *= (
            orig_w / float(STRETCH_SIZE)
        )
        xy[:, 1] *= (
            orig_h / float(STRETCH_SIZE)
        )

        return np.column_stack(
            [xy, conf]
        )

    def estimate_pred_h(
        self,
        keypoints: np.ndarray,
    ) -> Tuple[
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:

        accepted = (
            keypoints[:, 2]
            >= self.conf_threshold
        )

        inliers32 = np.zeros(
            NUM_KEYPOINTS,
            dtype=bool,
        )

        if (
            int(accepted.sum())
            < MIN_KEYPOINTS_FOR_HOMOGRAPHY
        ):
            return None, accepted, inliers32

        src = keypoints[
            accepted,
            :2,
        ]

        dst = np.asarray(
            PITCH_KEYPOINTS_TEMPLATE_M[
                accepted
            ],
            dtype=np.float64,
        )

        h, mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            cv2.RANSAC,
            self.ransac_threshold_m,
        )

        if h is None or mask is None:
            return None, accepted, inliers32

        mask = (
            mask.reshape(-1).astype(bool)
        )

        ids = np.flatnonzero(accepted)
        inliers32[ids] = mask

        if (
            int(mask.sum())
            < MIN_KEYPOINTS_FOR_HOMOGRAPHY
        ):
            return None, accepted, inliers32

        try:
            h = normalize_h(h)
        except Exception:
            return None, accepted, inliers32

        return h, accepted, inliers32


# =============================================================================
# GT HOMOGRAFIA Z RĘCZNYCH KEYPOINTÓW
# =============================================================================

def estimate_gt_h(
    gt_xy: np.ndarray,
    visibility: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    visible = visibility > 0

    if (
        int(visible.sum())
        < MIN_KEYPOINTS_FOR_HOMOGRAPHY
    ):
        raise ValueError(
            "Za mało widocznych GT keypointów."
        )

    src = gt_xy[visible]
    dst = np.asarray(
        PITCH_KEYPOINTS_TEMPLATE_M[
            visible
        ],
        dtype=np.float64,
    )

    # GT: używamy wszystkich ręcznie oznaczonych punktów.
    # Bez RANSAC, aby nie odrzucać adnotacji GT.
    h, _ = cv2.findHomography(
        src.astype(np.float32),
        dst.astype(np.float32),
        method=0,
    )

    if h is None:
        raise ValueError(
            "Nie udało się wyznaczyć H_GT."
        )

    return normalize_h(h), visible


# =============================================================================
# METRYKI
# =============================================================================

def calculate_errors(
    pitch_ref: np.ndarray,
    image_ref: np.ndarray,
    h_pred_img2pitch: np.ndarray,
) -> pd.DataFrame:

    h_pred_pitch2img = normalize_h(
        np.linalg.inv(
            h_pred_img2pitch
        )
    )

    est_pitch = transform_points(
        h_pred_img2pitch,
        image_ref,
    )

    est_image = transform_points(
        h_pred_pitch2img,
        pitch_ref,
    )

    valid = (
        np.isfinite(est_pitch).all(axis=1)
        & np.isfinite(est_image).all(axis=1)
    )

    pitch_ref = pitch_ref[valid]
    image_ref = image_ref[valid]
    est_pitch = est_pitch[valid]
    est_image = est_image[valid]

    return pd.DataFrame(
        {
            "ref_pitch_x_m": pitch_ref[:, 0],
            "ref_pitch_y_m": pitch_ref[:, 1],
            "ref_image_x_px": image_ref[:, 0],
            "ref_image_y_px": image_ref[:, 1],
            "pred_pitch_x_m": est_pitch[:, 0],
            "pred_pitch_y_m": est_pitch[:, 1],
            "pred_image_x_px": est_image[:, 0],
            "pred_image_y_px": est_image[:, 1],
            "pitch_error_m": np.linalg.norm(
                est_pitch - pitch_ref,
                axis=1,
            ),
            "reprojection_error_px": (
                np.linalg.norm(
                    est_image - image_ref,
                    axis=1,
                )
            ),
        }
    )


# =============================================================================
# CZYSTA WIZUALIZACJA
# =============================================================================

def choose_vis_points(
    image_ref: np.ndarray,
) -> np.ndarray:

    if len(image_ref) == 0:
        return np.empty(0, dtype=int)

    if len(image_ref) <= NUM_VIS_POINTS:
        return np.arange(len(image_ref))

    center = image_ref.mean(axis=0)
    first = int(
        np.argmin(
            np.linalg.norm(
                image_ref - center,
                axis=1,
            )
        )
    )

    distances = np.linalg.norm(
        image_ref - image_ref[first],
        axis=1,
    )
    second = int(
        np.argmax(distances)
    )

    return np.asarray(
        [first, second],
        dtype=int,
    )


def draw_projected_pitch(
    ax,
    h_pitch2img: np.ndarray,
    linestyle: str,
    label: str,
) -> None:

    first = True
    projection_color = None

    for poly_world in pitch_drawing_polylines(
        PITCH_LENGTH_M,
        PITCH_WIDTH_M,
    ):
        poly_img = transform_points(
            h_pitch2img,
            poly_world,
        )

        finite = np.isfinite(
            poly_img
        ).all(axis=1)

        if not finite.any():
            continue

        kwargs = dict(
            linestyle=linestyle,
            linewidth=1.25,
            alpha=0.85,
            label=label if first else None,
        )

        if projection_color is not None:
            kwargs["color"] = (
                projection_color
            )

        line = ax.plot(
            poly_img[finite, 0],
            poly_img[finite, 1],
            **kwargs,
        )[0]

        if projection_color is None:
            projection_color = (
                line.get_color()
            )

        first = False


def create_visualization(
    image_bgr: np.ndarray,
    stem: str,
    h_gt_img2pitch: np.ndarray,
    h_pred_img2pitch: np.ndarray,
    gt_xy: np.ndarray,
    gt_visible: np.ndarray,
    pred_keypoints: np.ndarray,
    accepted: np.ndarray,
    inliers: np.ndarray,
    point_df: pd.DataFrame,
    output_path: Path,
) -> None:

    h_gt_pitch2img = normalize_h(
        np.linalg.inv(
            h_gt_img2pitch
        )
    )

    h_pred_pitch2img = normalize_h(
        np.linalg.inv(
            h_pred_img2pitch
        )
    )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    h_img, w_img = (
        image_bgr.shape[:2]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.5),
    )

    ax_img, ax_pitch = axes

    # Dane punktów testowych P1/P2 wykorzystywane na obu panelach
    image_ref = point_df[
        [
            "ref_image_x_px",
            "ref_image_y_px",
        ]
    ].to_numpy(dtype=np.float64)

    pred_image = point_df[
        [
            "pred_image_x_px",
            "pred_image_y_px",
        ]
    ].to_numpy(dtype=np.float64)

    ref_pitch = point_df[
        [
            "ref_pitch_x_m",
            "ref_pitch_y_m",
        ]
    ].to_numpy(dtype=np.float64)

    pred_pitch = point_df[
        [
            "pred_pitch_x_m",
            "pred_pitch_y_m",
        ]
    ].to_numpy(dtype=np.float64)

    pitch_errors = point_df[
        "pitch_error_m"
    ].to_numpy(dtype=np.float64)

    reproj_errors = point_df[
        "reprojection_error_px"
    ].to_numpy(dtype=np.float64)

    vis_ids = choose_vis_points(
        image_ref
    )

    # LEWO ------------------------------------------------------------
    ax_img.imshow(image_rgb)
    ax_img.set_xlim(0, w_img)
    ax_img.set_ylim(h_img, 0)
    ax_img.axis("off")
    ax_img.set_title(
        f"{stem}: H_GT vs H_keypoints"
    )

    draw_projected_pitch(
        ax_img,
        h_gt_pitch2img,
        "-",
        "H_GT z adnotacji",
    )

    draw_projected_pitch(
        ax_img,
        h_pred_pitch2img,
        "--",
        "H_keypoints",
    )

    visible_pts = gt_xy[
        gt_visible
    ]

    if len(visible_pts):
        ax_img.scatter(
            visible_pts[:, 0],
            visible_pts[:, 1],
            s=18,
            marker=".",
            label="GT keypoint",
            zorder=5,
        )

    pred_in = (
        accepted & inliers
    )
    pred_out = (
        accepted & ~inliers
    )

    if pred_in.any():
        ax_img.scatter(
            pred_keypoints[pred_in, 0],
            pred_keypoints[pred_in, 1],
            s=26,
            marker="o",
            label="RANSAC inlier",
            zorder=7,
        )

    if pred_out.any():
        ax_img.scatter(
            pred_keypoints[pred_out, 0],
            pred_keypoints[pred_out, 1],
            s=30,
            marker="x",
            linewidths=1.4,
            label="RANSAC outlier",
            zorder=7,
        )

    for kp_id in np.flatnonzero(
        accepted
    ):
        ax_img.annotate(
            f"K{kp_id:02d}",
            (
                pred_keypoints[kp_id, 0],
                pred_keypoints[kp_id, 1],
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    # Dodane P1/P2 na obrazie po lewej:
    if len(vis_ids):
        ax_img.scatter(
            image_ref[vis_ids, 0],
            image_ref[vis_ids, 1],
            s=24,
            marker="o",
            label="P ref (GT)",
            zorder=8,
        )

        ax_img.scatter(
            pred_image[vis_ids, 0],
            pred_image[vis_ids, 1],
            s=28,
            marker="x",
            linewidths=1.3,
            label="P reproj (H_keypoints)",
            zorder=9,
        )

        for local_i, idx in enumerate(
            vis_ids,
            start=1,
        ):
            a = image_ref[idx]
            b = pred_image[idx]

            ax_img.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                linewidth=0.9,
                alpha=0.75,
            )

            ax_img.annotate(
                f"P{local_i}: {reproj_errors[idx]:.1f}px",
                (a[0], a[1]),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    alpha=0.60,
                ),
            )

    ax_img.legend(
        loc="lower left",
        fontsize=8,
    )

    # PRAWO -----------------------------------------------------------
    draw_top_down_pitch(
        ax_pitch,
        PITCH_LENGTH_M,
        PITCH_WIDTH_M,
    )

    ax_pitch.set_title(
        "Błąd na płaszczyźnie boiska"
    )

    if len(vis_ids):
        ax_pitch.scatter(
            ref_pitch[vis_ids, 0],
            ref_pitch[vis_ids, 1],
            s=28,
            marker="o",
            label="GT",
            zorder=5,
        )

        ax_pitch.scatter(
            pred_pitch[vis_ids, 0],
            pred_pitch[vis_ids, 1],
            s=32,
            marker="x",
            linewidths=1.4,
            label="H_keypoints",
            zorder=6,
        )

        for local_i, idx in enumerate(
            vis_ids,
            start=1,
        ):
            a = ref_pitch[idx]
            b = pred_pitch[idx]

            ax_pitch.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                linewidth=0.9,
                alpha=0.75,
            )

            middle = (
                a + b
            ) / 2.0

            ax_pitch.annotate(
                f"P{local_i}: "
                f"{pitch_errors[idx]:.2f} m",
                (
                    middle[0],
                    middle[1],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    boxstyle=(
                        "round,pad=0.18"
                    ),
                    alpha=0.60,
                ),
            )

    ax_pitch.legend(
        loc="upper right",
        fontsize=8,
    )

    fig.suptitle(
        "Original dataset: "
        "GT homography vs keypoint homography",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    ensure_exists(
        IMAGES_DIR,
        "images",
    )
    ensure_exists(
        LABELS_DIR,
        "labels",
    )
    ensure_exists(
        WEIGHTS,
        "wag modelu",
    )

    template = np.asarray(
        PITCH_KEYPOINTS_TEMPLATE_M,
        dtype=np.float64,
    )

    if template.shape != (32, 2):
        raise ValueError(
            f"Template ma shape "
            f"{template.shape}, "
            "oczekiwano (32,2)."
        )

    pairs = collect_pairs()

    if args.limit is not None:
        pairs = pairs[
            : args.limit
        ]

    if not pairs:
        raise RuntimeError(
            "Brak image+label pairs."
        )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_dir = (
        RESULTS_ROOT
        / timestamp
    )

    vis_dir = (
        output_dir
        / "visualizations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if not args.no_visualizations:
        vis_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    detector = Detector(
        conf_threshold=(
            args.conf_threshold
        ),
        ransac_threshold_m=(
            args.ransac_threshold_m
        ),
    )

    # warm-up
    warm = cv2.imread(
        str(pairs[0][0])
    )

    if warm is not None:
        _ = detector.detect(warm)
        sync_cuda()

    pitch_grid = build_pitch_grid()

    per_image_rows = []
    point_tables = []
    kp_rows = []

    for idx, (
        image_path,
        label_path,
    ) in enumerate(
        pairs,
        start=1,
    ):

        print(
            f"[{idx:03d}/{len(pairs):03d}] "
            f"{image_path.name} ...",
            end=" ",
            flush=True,
        )

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            print("FAILED image")
            continue

        h_img, w_img = (
            frame.shape[:2]
        )

        try:
            gt_xy, visibility = (
                parse_yolo_pose_gt(
                    label_path,
                    w_img,
                    h_img,
                )
            )

            h_gt_img2pitch, gt_visible = (
                estimate_gt_h(
                    gt_xy,
                    visibility,
                )
            )

            h_gt_pitch2img = normalize_h(
                np.linalg.inv(
                    h_gt_img2pitch
                )
            )

            sync_cuda()
            t0 = time.perf_counter()

            pred_kp = detector.detect(
                frame
            )

            if pred_kp is None:
                sync_cuda()
                elapsed = (
                    time.perf_counter()
                    - t0
                ) * 1000.0

                per_image_rows.append(
                    {
                        "image_name": (
                            image_path.name
                        ),
                        "status": "failed",
                        "failure_reason": (
                            "no_pose_detection"
                        ),
                        "n_gt_visible": int(
                            gt_visible.sum()
                        ),
                        "n_confident": 0,
                        "n_inliers": 0,
                        "inlier_ratio": np.nan,
                        "mean_pitch_error_m": np.nan,
                        "median_pitch_error_m": np.nan,
                        "mean_reprojection_error_px": np.nan,
                        "median_reprojection_error_px": np.nan,
                        "estimation_time_ms": elapsed,
                    }
                )

                print(
                    "FAILED no pose"
                )
                continue

            (
                h_pred_img2pitch,
                accepted,
                inliers,
            ) = detector.estimate_pred_h(
                pred_kp
            )

            sync_cuda()

            elapsed = (
                time.perf_counter()
                - t0
            ) * 1000.0

            n_conf = int(
                accepted.sum()
            )
            n_in = int(
                inliers.sum()
            )

            for kp_id in range(32):
                kp_rows.append(
                    {
                        "image_name": (
                            image_path.name
                        ),
                        "keypoint_id": kp_id,
                        "gt_visible": bool(
                            gt_visible[kp_id]
                        ),
                        "gt_x_px": float(
                            gt_xy[kp_id, 0]
                        ),
                        "gt_y_px": float(
                            gt_xy[kp_id, 1]
                        ),
                        "pred_x_px": float(
                            pred_kp[kp_id, 0]
                        ),
                        "pred_y_px": float(
                            pred_kp[kp_id, 1]
                        ),
                        "confidence": float(
                            pred_kp[kp_id, 2]
                        ),
                        "accepted_by_conf": bool(
                            accepted[kp_id]
                        ),
                        "ransac_inlier": bool(
                            inliers[kp_id]
                        ),
                    }
                )

            if h_pred_img2pitch is None:
                per_image_rows.append(
                    {
                        "image_name": (
                            image_path.name
                        ),
                        "status": "failed",
                        "failure_reason": (
                            "homography_failed"
                        ),
                        "n_gt_visible": int(
                            gt_visible.sum()
                        ),
                        "n_confident": n_conf,
                        "n_inliers": n_in,
                        "inlier_ratio": (
                            n_in / n_conf
                            if n_conf
                            else np.nan
                        ),
                        "mean_pitch_error_m": np.nan,
                        "median_pitch_error_m": np.nan,
                        "mean_reprojection_error_px": np.nan,
                        "median_reprojection_error_px": np.nan,
                        "estimation_time_ms": elapsed,
                    }
                )

                print(
                    f"FAILED H | "
                    f"kp={n_conf}"
                )
                continue

            pitch_ref, image_ref = (
                visible_grid_points(
                    pitch_grid,
                    h_gt_pitch2img,
                    w_img,
                    h_img,
                )
            )

            point_df = calculate_errors(
                pitch_ref,
                image_ref,
                h_pred_img2pitch,
            )

            point_df.insert(
                0,
                "image_name",
                image_path.name,
            )

            point_tables.append(
                point_df
            )

            pitch_errors = point_df[
                "pitch_error_m"
            ].to_numpy()

            reproj_errors = point_df[
                "reprojection_error_px"
            ].to_numpy()

            row = {
                "image_name": (
                    image_path.name
                ),
                "status": "ok",
                "failure_reason": "",
                "n_gt_visible": int(
                    gt_visible.sum()
                ),
                "n_confident": n_conf,
                "n_inliers": n_in,
                "inlier_ratio": (
                    n_in / n_conf
                    if n_conf
                    else np.nan
                ),
                "mean_pitch_error_m": float(
                    np.mean(
                        pitch_errors
                    )
                ),
                "median_pitch_error_m": float(
                    np.median(
                        pitch_errors
                    )
                ),
                "p90_pitch_error_m": float(
                    np.percentile(
                        pitch_errors,
                        90,
                    )
                ),
                "mean_reprojection_error_px": float(
                    np.mean(
                        reproj_errors
                    )
                ),
                "median_reprojection_error_px": float(
                    np.median(
                        reproj_errors
                    )
                ),
                "p90_reprojection_error_px": float(
                    np.percentile(
                        reproj_errors,
                        90,
                    )
                ),
                "estimation_time_ms": elapsed,
            }

            per_image_rows.append(row)

            if not args.no_visualizations:
                create_visualization(
                    image_bgr=frame,
                    stem=image_path.stem,
                    h_gt_img2pitch=(
                        h_gt_img2pitch
                    ),
                    h_pred_img2pitch=(
                        h_pred_img2pitch
                    ),
                    gt_xy=gt_xy,
                    gt_visible=(
                        gt_visible
                    ),
                    pred_keypoints=pred_kp,
                    accepted=accepted,
                    inliers=inliers,
                    point_df=point_df,
                    output_path=(
                        vis_dir
                        / (
                            f"{image_path.stem}"
                            "_homography.png"
                        )
                    ),
                )

            print(
                f"OK | "
                f"GTkp={int(gt_visible.sum())} | "
                f"predkp={n_conf} | "
                f"in={n_in} | "
                f"pitch="
                f"{row['mean_pitch_error_m']:.3f} m"
            )

        except Exception as exc:
            print(
                f"FAILED "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            per_image_rows.append(
                {
                    "image_name": (
                        image_path.name
                    ),
                    "status": "failed",
                    "failure_reason": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    "n_gt_visible": np.nan,
                    "n_confident": np.nan,
                    "n_inliers": np.nan,
                    "inlier_ratio": np.nan,
                    "mean_pitch_error_m": np.nan,
                    "median_pitch_error_m": np.nan,
                    "mean_reprojection_error_px": np.nan,
                    "median_reprojection_error_px": np.nan,
                    "estimation_time_ms": np.nan,
                }
            )

    per_image_df = pd.DataFrame(
        per_image_rows
    )

    if point_tables:
        point_df_all = pd.concat(
            point_tables,
            ignore_index=True,
        )
    else:
        point_df_all = pd.DataFrame()

    kp_df_all = pd.DataFrame(
        kp_rows
    )

    successful = per_image_df[
        per_image_df["status"] == "ok"
    ]

    if (
        not successful.empty
        and not point_df_all.empty
    ):
        summary = {
            "total_images": int(
                len(per_image_df)
            ),
            "successful_images": int(
                len(successful)
            ),
            "success_rate": float(
                len(successful)
                / len(per_image_df)
            ),
            "mean_pitch_error_m": float(
                point_df_all[
                    "pitch_error_m"
                ].mean()
            ),
            "median_pitch_error_m": float(
                point_df_all[
                    "pitch_error_m"
                ].median()
            ),
            "p90_pitch_error_m": float(
                point_df_all[
                    "pitch_error_m"
                ].quantile(0.9)
            ),
            "mean_reprojection_error_px": float(
                point_df_all[
                    "reprojection_error_px"
                ].mean()
            ),
            "median_reprojection_error_px": float(
                point_df_all[
                    "reprojection_error_px"
                ].median()
            ),
            "p90_reprojection_error_px": float(
                point_df_all[
                    "reprojection_error_px"
                ].quantile(0.9)
            ),
            "mean_estimation_time_ms": float(
                successful[
                    "estimation_time_ms"
                ].mean()
            ),
            "mean_confident_keypoints": float(
                successful[
                    "n_confident"
                ].mean()
            ),
            "mean_ransac_inliers": float(
                successful[
                    "n_inliers"
                ].mean()
            ),
        }
    else:
        summary = {
            "total_images": int(
                len(per_image_df)
            ),
            "successful_images": 0,
            "success_rate": 0.0,
        }

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir
        / "metrics_summary.csv",
        index=False,
    )

    per_image_df.to_csv(
        output_dir
        / "per_image_metrics.csv",
        index=False,
    )

    point_df_all.to_csv(
        output_dir
        / "point_errors.csv",
        index=False,
    )

    kp_df_all.to_csv(
        output_dir
        / "keypoint_diagnostics.csv",
        index=False,
    )

    config = {
        "dataset": (
            "football-field-detection-16/test"
        ),
        "images_dir": str(
            IMAGES_DIR
        ),
        "labels_dir": str(
            LABELS_DIR
        ),
        "weights": str(
            WEIGHTS
        ),
        "stretch_size": (
            STRETCH_SIZE
        ),
        "keypoint_conf_threshold": float(
            args.conf_threshold
        ),
        "ransac_threshold_m": float(
            args.ransac_threshold_m
        ),
        "gt_homography_definition": (
            "Direct homography fitted from all "
            "manual visible GT keypoints "
            "(visibility > 0) to "
            "PITCH_KEYPOINTS_TEMPLATE_M."
        ),
        "note": (
            "Diagnostic in-domain sanity check. "
            "Do not treat as equally independent "
            "to the SoccerNet line-based H_GT benchmark."
        ),
    }

    with (
        output_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n========================================")
    print("PODSUMOWANIE")
    print("========================================")
    print(
        f"Success: "
        f"{100 * summary['success_rate']:.2f}% "
        f"({summary['successful_images']}/"
        f"{summary['total_images']})"
    )

    if (
        summary.get(
            "successful_images",
            0,
        ) > 0
    ):
        print(
            f"Pitch error: mean="
            f"{summary['mean_pitch_error_m']:.3f} m | "
            f"median="
            f"{summary['median_pitch_error_m']:.3f} m"
        )

        print(
            f"Reprojection: mean="
            f"{summary['mean_reprojection_error_px']:.2f} px | "
            f"median="
            f"{summary['median_reprojection_error_px']:.2f} px"
        )

    print(
        f"Wyniki: {output_dir}"
    )


if __name__ == "__main__":
    main()
