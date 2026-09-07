#!/usr/bin/env python3
"""
Ewaluacja homografii estymowanej z 32 keypointów YOLOv8x-pose
względem referencyjnej H_GT.

Umieść plik jako:
    CvFootballTracker/scripts/evaluate_homography_keypoints.py

Najważniejsze założenia:
    1. Klatka jest najpierw STRETCHowana do 640x640.
    2. YOLOv8x-pose wykonuje inferencję na obrazie 640x640.
    3. Współrzędne keypointów są skalowane z powrotem do oryginalnego
       rozmiaru klatki.
    4. Do homografii trafiają keypointy z confidence >= 0.60.
    5. Homografia jest liczona dokładnie w kierunku:
           IMAGE [px] -> PITCH [m]
       przez cv2.findHomography(..., cv2.RANSAC, 10.0),
       więc próg RANSAC = 10.0 jest liczony w METRACH.
    6. Każda klatka jest estymowana niezależnie — bez previous-H fallback.
    7. Metryki H są liczone na tej samej niezależnej siatce punktów boiska
       co w evaluate_homography_tvcalib.py.

Domyślne wejście:
    data/data_homography_evaluation/
        images/
        manifest.csv
        ground_truth/

Wagi:
    models/pitch_keypoints/trained_keypoints.pt

Domyślne uruchomienie — jedna klatka:
    python scripts/evaluate_homography_keypoints.py

Inna klatka:
    python scripts/evaluate_homography_keypoints.py --mode single --eval-id 27

Cała pula:
    python scripts/evaluate_homography_keypoints.py --mode all

Wyniki:
    results/homography_evaluation/<timestamp>/
        metrics_summary.csv
        per_frame_metrics.csv
        point_errors.csv
        keypoint_diagnostics.csv
        config.json
        estimated_homographies/
        visualizations/

Metryki homografii:
    pitch error [m]:
        P_pitch --H_GT--> p_ref_image --H_keypoints--> P_est_pitch

    reprojection error [px]:
        P_pitch --H_GT--> p_ref_image
        P_pitch --inv(H_keypoints)--> p_est_image

Dodatkowo per frame:
    - n_confident_keypoints
    - n_ransac_inliers
    - n_ransac_outliers
    - ransac_inlier_ratio
    - średni confidence użytych keypointów
    - czas estymacji

UWAGA:
    Ten skrypt importuje PITCH_KEYPOINTS_TEMPLATE_M z:
        src/calibration/keypoints_homography.py

    Czyli korzysta z aktualnej wersji szablonu w Twoim projekcie,
    w tym z poprawek indeksów K30/K31 oraz punktów rzutów karnych.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

EVAL_ROOT = PROJECT_ROOT / "data" / "data_homography_evaluation"
EVAL_IMAGES = EVAL_ROOT / "images"
EVAL_MANIFEST = EVAL_ROOT / "manifest.csv"
GT_DIR = EVAL_ROOT / "ground_truth"

KEYPOINTS_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "pitch_keypoints"
    / "trained_keypoints.pt"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "homography_evaluation"

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

STRETCH_SIZE = 640
KEYPOINT_CONF_THRESHOLD = 0.60

# IMAGE -> PITCH, więc jednostką progu są METRY.
RANSAC_REPROJ_THRESHOLD_M = 10.0

GRID_STEP_M = 5.0
PITCH_MARGIN_M = 5.0
IMAGE_MARGIN_PX = 5.0
NUM_VIS_POINTS = 2


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ewaluacja homografii keypointowej względem H_GT "
            "z wejściem stretch 640x640."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        default="single",
        help="single = jedna klatka, all = cała pula. Domyślnie single.",
    )
    parser.add_argument(
        "--eval-id",
        type=int,
        default=1,
        help="Numer eval_id dla --mode single. Domyślnie 1.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=KEYPOINT_CONF_THRESHOLD,
        help=(
            "Minimalny confidence keypointu używanego do H. "
            f"Domyślnie {KEYPOINT_CONF_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--ransac-threshold-m",
        type=float,
        default=RANSAC_REPROJ_THRESHOLD_M,
        help=(
            "Próg RANSAC w metrach, ponieważ estymujemy IMAGE -> PITCH. "
            f"Domyślnie {RANSAC_REPROJ_THRESHOLD_M} m."
        ),
    )
    parser.add_argument(
        "--stretch-size",
        type=int,
        default=STRETCH_SIZE,
        help=f"Rozmiar wejścia modelu po stretch. Domyślnie {STRETCH_SIZE}.",
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
        "--no-visualizations",
        action="store_true",
        help="Nie generuj PNG z porównaniem H_keypoints vs H_GT.",
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
    Niezależna regularna siatka punktów boiska.
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


# =============================================================================
# DETEKCJA KEYPOINTÓW + RANSAC
# =============================================================================

class StretchKeypointsHomography:
    """
    Diagnostyczno-ewaluacyjna wersja metody keypointowej.

    Różnica względem starego wrappera:
        obraz jest jawnie stretchowany do NxN przed modelem.

    Kierunek H i RANSAC pozostają zgodne z dotychczasowym pipeline:
        IMAGE [px] -> PITCH [m]
        ransacReprojThreshold = wartość w metrach
    """

    def __init__(
        self,
        model_weights: Path,
        conf_threshold: float,
        ransac_threshold_m: float,
        stretch_size: int,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.ransac_threshold_m = float(ransac_threshold_m)
        self.stretch_size = int(stretch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[KeypointsEval] Urządzenie: {self.device}")
        print(f"[KeypointsEval] Wczytywanie modelu: {model_weights}")

        self.model = YOLO(str(model_weights))

        print("[KeypointsEval] Model gotowy")

    def detect_keypoints(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Zwraca keypointy (32,3) już w ORYGINALNYCH współrzędnych obrazu.

        Proces:
            original W x H
            -> cv2.resize(..., stretch_size x stretch_size)
            -> YOLO
            -> współrzędne x,y z powrotem do W x H
        """
        orig_h, orig_w = image_bgr.shape[:2]

        resized = cv2.resize(
            image_bgr,
            (self.stretch_size, self.stretch_size),
            interpolation=cv2.INTER_LINEAR,
        )

        results = self.model(
            resized,
            verbose=False,
            device=self.device,
            imgsz=self.stretch_size,
        )

        if not results:
            return None, {
                "orig_width": orig_w,
                "orig_height": orig_h,
                "input_width": self.stretch_size,
                "input_height": self.stretch_size,
            }

        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None, {
                "orig_width": orig_w,
                "orig_height": orig_h,
                "input_width": self.stretch_size,
                "input_height": self.stretch_size,
            }

        kp_data = result.keypoints

        if kp_data.conf is None or len(kp_data.conf) == 0:
            return None, {
                "orig_width": orig_w,
                "orig_height": orig_h,
                "input_width": self.stretch_size,
                "input_height": self.stretch_size,
            }

        # Tak jak w istniejącym KeypointsHomography:
        # wybór detekcji z najwyższą średnią pewnością keypointów.
        best_idx = int(
            kp_data.conf.mean(dim=1).argmax()
        )

        xy = kp_data.xy[best_idx].cpu().numpy().astype(np.float64)
        conf = kp_data.conf[best_idx].cpu().numpy().astype(np.float64)

        if xy.shape != (32, 2) or conf.shape != (32,):
            raise RuntimeError(
                f"Nieoczekiwany shape keypointów: xy={xy.shape}, conf={conf.shape}"
            )

        scale_x = orig_w / float(self.stretch_size)
        scale_y = orig_h / float(self.stretch_size)

        xy[:, 0] *= scale_x
        xy[:, 1] *= scale_y

        keypoints = np.column_stack([xy, conf])

        return keypoints, {
            "orig_width": orig_w,
            "orig_height": orig_h,
            "input_width": self.stretch_size,
            "input_height": self.stretch_size,
            "scale_x_back": scale_x,
            "scale_y_back": scale_y,
        }

    def estimate(
        self,
        image_bgr: np.ndarray,
    ) -> dict:
        """
        Zwraca pełną diagnostykę estymacji jednej klatki.
        """
        keypoints, meta = self.detect_keypoints(image_bgr)

        result = {
            "H_image_to_pitch": None,
            "keypoints": keypoints,
            "confident_mask": None,
            "ransac_inlier_mask_32": None,
            "n_confident": 0,
            "n_inliers": 0,
            "failure_reason": "",
            **meta,
        }

        if keypoints is None:
            result["failure_reason"] = "no_pose_detection"
            return result

        confident_mask = keypoints[:, 2] >= self.conf_threshold
        n_confident = int(confident_mask.sum())

        result["confident_mask"] = confident_mask
        result["n_confident"] = n_confident

        if n_confident < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            result["failure_reason"] = "too_few_confident_keypoints"
            return result

        src_pts = keypoints[confident_mask, :2]
        dst_pts = np.asarray(
            PITCH_KEYPOINTS_TEMPLATE_M[confident_mask],
            dtype=np.float64,
        )

        h_raw, ransac_mask = cv2.findHomography(
            src_pts.astype(np.float32),
            dst_pts.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_m,
        )

        if h_raw is None:
            result["failure_reason"] = "ransac_failed"
            return result

        if ransac_mask is None:
            result["failure_reason"] = "ransac_mask_missing"
            return result

        ransac_mask = ransac_mask.reshape(-1).astype(bool)
        n_inliers = int(ransac_mask.sum())

        result["n_inliers"] = n_inliers

        inlier_mask_32 = np.zeros(32, dtype=bool)
        confident_indices = np.flatnonzero(confident_mask)
        inlier_mask_32[confident_indices] = ransac_mask
        result["ransac_inlier_mask_32"] = inlier_mask_32

        if n_inliers < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            result["failure_reason"] = "too_few_ransac_inliers"
            return result

        try:
            h = normalize_h(h_raw)
        except Exception:
            result["failure_reason"] = "invalid_homography"
            return result

        result["H_image_to_pitch"] = h
        return result


# =============================================================================
# METRYKI
# =============================================================================

def calculate_point_errors(
    pitch_points_ref: np.ndarray,
    image_points_ref: np.ndarray,
    h_kp_img2pitch: np.ndarray,
    h_kp_pitch2img: np.ndarray,
) -> pd.DataFrame:

    kp_pitch_points = transform_points(
        h_kp_img2pitch,
        image_points_ref,
    )

    kp_image_points = transform_points(
        h_kp_pitch2img,
        pitch_points_ref,
    )

    valid = (
        np.isfinite(kp_pitch_points).all(axis=1)
        & np.isfinite(kp_image_points).all(axis=1)
    )

    pitch_ref_valid = pitch_points_ref[valid]
    image_ref_valid = image_points_ref[valid]
    kp_pitch_valid = kp_pitch_points[valid]
    kp_image_valid = kp_image_points[valid]

    pitch_errors = np.linalg.norm(
        kp_pitch_valid - pitch_ref_valid,
        axis=1,
    )

    reprojection_errors = np.linalg.norm(
        kp_image_valid - image_ref_valid,
        axis=1,
    )

    return pd.DataFrame(
        {
            "ref_pitch_x_m": pitch_ref_valid[:, 0],
            "ref_pitch_y_m": pitch_ref_valid[:, 1],
            "ref_image_x_px": image_ref_valid[:, 0],
            "ref_image_y_px": image_ref_valid[:, 1],
            "keypoints_pitch_x_m": kp_pitch_valid[:, 0],
            "keypoints_pitch_y_m": kp_pitch_valid[:, 1],
            "keypoints_image_x_px": kp_image_valid[:, 0],
            "keypoints_image_y_px": kp_image_valid[:, 1],
            "pitch_error_m": pitch_errors,
            "reprojection_error_px": reprojection_errors,
        }
    )


def build_keypoint_diagnostics(
    eval_id: int,
    eval_stem: str,
    estimation: dict,
) -> pd.DataFrame:

    keypoints = estimation["keypoints"]

    if keypoints is None:
        return pd.DataFrame(
            [
                {
                    "eval_id": eval_id,
                    "eval_stem": eval_stem,
                    "keypoint_id": np.nan,
                    "x_px": np.nan,
                    "y_px": np.nan,
                    "confidence": np.nan,
                    "accepted_by_conf": False,
                    "ransac_inlier": False,
                    "pitch_x_m": np.nan,
                    "pitch_y_m": np.nan,
                }
            ]
        )

    confident_mask = estimation["confident_mask"]
    inlier_mask = estimation["ransac_inlier_mask_32"]

    if confident_mask is None:
        confident_mask = np.zeros(32, dtype=bool)

    if inlier_mask is None:
        inlier_mask = np.zeros(32, dtype=bool)

    rows = []

    for kp_id in range(32):
        rows.append(
            {
                "eval_id": eval_id,
                "eval_stem": eval_stem,
                "keypoint_id": kp_id,
                "x_px": float(keypoints[kp_id, 0]),
                "y_px": float(keypoints[kp_id, 1]),
                "confidence": float(keypoints[kp_id, 2]),
                "accepted_by_conf": bool(confident_mask[kp_id]),
                "ransac_inlier": bool(inlier_mask[kp_id]),
                "pitch_x_m": float(PITCH_KEYPOINTS_TEMPLATE_M[kp_id, 0]),
                "pitch_y_m": float(PITCH_KEYPOINTS_TEMPLATE_M[kp_id, 1]),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# WIZUALIZACJA
# =============================================================================

def choose_visualization_indices(
    image_points_ref: np.ndarray,
    count: int,
) -> np.ndarray:

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
        candidates = [
            i for i in range(n)
            if i not in selected
        ]

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

    first = True

    for poly_world in pitch_drawing_polylines(
        pitch_length,
        pitch_width,
    ):
        poly_img = transform_points(
            h_pitch2img,
            poly_world,
        )

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
    h_kp_pitch2img: np.ndarray,
    point_df: pd.DataFrame,
    keypoint_df: pd.DataFrame,
    output_path: Path,
    pitch_length: float,
    pitch_width: float,
    num_vis_points: int,
    conf_threshold: float,
) -> None:

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = image_bgr.shape[:2]

    pitch_ref = point_df[
        ["ref_pitch_x_m", "ref_pitch_y_m"]
    ].to_numpy(dtype=np.float64)

    image_ref = point_df[
        ["ref_image_x_px", "ref_image_y_px"]
    ].to_numpy(dtype=np.float64)

    kp_pitch = point_df[
        ["keypoints_pitch_x_m", "keypoints_pitch_y_m"]
    ].to_numpy(dtype=np.float64)

    kp_image = point_df[
        ["keypoints_image_x_px", "keypoints_image_y_px"]
    ].to_numpy(dtype=np.float64)

    pitch_errors = point_df[
        "pitch_error_m"
    ].to_numpy(dtype=np.float64)

    reproj_errors = point_df[
        "reprojection_error_px"
    ].to_numpy(dtype=np.float64)

    vis_ids = choose_visualization_indices(
        image_ref,
        num_vis_points,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_img, ax_pitch = axes

    # -----------------------------------------------------------------
    # LEWY PANEL: klatka + H_GT + H_keypoints + detekcje keypointów
    # -----------------------------------------------------------------
    ax_img.imshow(image_rgb)
    ax_img.set_xlim(0, w_img)
    ax_img.set_ylim(h_img, 0)
    ax_img.set_title(f"{eval_stem}: H_GT vs keypoints")
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
        h_kp_pitch2img,
        pitch_length,
        pitch_width,
        linestyle="--",
        label="Keypoints pitch projection",
    )

    accepted = keypoint_df[
        keypoint_df["accepted_by_conf"] == True
    ]

    inliers = accepted[
        accepted["ransac_inlier"] == True
    ]

    outliers = accepted[
        accepted["ransac_inlier"] == False
    ]

    if not inliers.empty:
        ax_img.scatter(
            inliers["x_px"],
            inliers["y_px"],
            s=24,
            marker="o",
            label="RANSAC inlier keypoint",
            zorder=7,
        )

    if not outliers.empty:
        ax_img.scatter(
            outliers["x_px"],
            outliers["y_px"],
            s=28,
            marker="x",
            linewidths=1.4,
            label="RANSAC outlier keypoint",
            zorder=7,
        )

    for row in accepted.itertuples(index=False):
        ax_img.annotate(
            f"K{int(row.keypoint_id):02d}",
            (row.x_px, row.y_px),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    if len(vis_ids):
        ref_uv = image_ref[vis_ids]
        kp_uv = kp_image[vis_ids]

        ax_img.scatter(
            ref_uv[:, 0],
            ref_uv[:, 1],
            s=20,
            marker="o",
            label="reference pixel (GT)",
            zorder=5,
        )

        ax_img.scatter(
            kp_uv[:, 0],
            kp_uv[:, 1],
            s=24,
            marker="x",
            linewidths=1.2,
            label="keypoints reprojection",
            zorder=6,
        )

        for local_i, point_idx in enumerate(vis_ids, start=1):
            ref = image_ref[point_idx]
            est = kp_image[point_idx]

            ax_img.plot(
                [ref[0], est[0]],
                [ref[1], est[1]],
                linewidth=0.8,
                alpha=0.7,
            )

            ax_img.annotate(
                f"P{local_i}: {reproj_errors[point_idx]:.1f}px",
                (ref[0], ref[1]),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    alpha=0.65,
                ),
            )

    ax_img.legend(loc="lower left", fontsize=7)

    # -----------------------------------------------------------------
    # PRAWY PANEL: błąd na płaszczyźnie boiska
    # -----------------------------------------------------------------
    draw_top_down_pitch(
        ax_pitch,
        pitch_length,
        pitch_width,
    )

    ax_pitch.set_title(
        "Reference pitch point vs keypoints result"
    )

    if len(vis_ids):
        ref_xy = pitch_ref[vis_ids]
        kp_xy = kp_pitch[vis_ids]

        ax_pitch.scatter(
            ref_xy[:, 0],
            ref_xy[:, 1],
            s=24,
            marker="o",
            label="GT reference",
            zorder=5,
        )

        ax_pitch.scatter(
            kp_xy[:, 0],
            kp_xy[:, 1],
            s=28,
            marker="x",
            linewidths=1.4,
            label="Keypoints",
            zorder=6,
        )

        for local_i, point_idx in enumerate(vis_ids, start=1):
            ref = pitch_ref[point_idx]
            est = kp_pitch[point_idx]

            ax_pitch.plot(
                [ref[0], est[0]],
                [ref[1], est[1]],
                linewidth=0.8,
                alpha=0.7,
            )

            ax_pitch.annotate(
                f"P{local_i}: {pitch_errors[point_idx]:.2f} m",
                (ref[0], ref[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    alpha=0.65,
                ),
            )

    ax_pitch.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Keypoints homography evaluation | "
        f"stretch 640x640 | conf >= {conf_threshold:.2f}",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# EWALUACJA JEDNEJ KLATKI
# =============================================================================

def evaluate_one(
    sample: dict,
    calibrator: StretchKeypointsHomography,
    pitch_grid: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:

    eval_id = sample["eval_id"]
    eval_stem = sample["eval_stem"]
    image_path = sample["image_path"]
    gt_path = sample["gt_path"]

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(
            f"OpenCV nie może odczytać: {image_path}"
        )

    image_height, image_width = image_bgr.shape[:2]

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
            f"{eval_stem}: tylko {len(pitch_points_ref)} "
            "widoczne punkty testowe."
        )

    # Mierzymy pełny etap metody keypointowej:
    # stretch + YOLO pose + confidence filter + RANSAC.
    # Inicjalizacja modelu jest poza timingiem.
    sync_cuda()
    t0 = time.perf_counter()

    estimation = calibrator.estimate(image_bgr)

    sync_cuda()
    estimation_time_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    keypoint_df = build_keypoint_diagnostics(
        eval_id=eval_id,
        eval_stem=eval_stem,
        estimation=estimation,
    )

    n_confident = int(estimation["n_confident"])
    n_inliers = int(estimation["n_inliers"])
    n_outliers = max(n_confident - n_inliers, 0)

    inlier_ratio = (
        n_inliers / n_confident
        if n_confident > 0
        else np.nan
    )

    if estimation["keypoints"] is not None and n_confident > 0:
        confident_confs = estimation["keypoints"][
            estimation["confident_mask"], 2
        ]
        mean_confident_conf = float(
            np.mean(confident_confs)
        )
        median_confident_conf = float(
            np.median(confident_confs)
        )
    else:
        mean_confident_conf = np.nan
        median_confident_conf = np.nan

    h_kp_img2pitch_raw = estimation["H_image_to_pitch"]

    if h_kp_img2pitch_raw is None:
        return (
            {
                "eval_id": eval_id,
                "eval_stem": eval_stem,
                "status": "failed",
                "failure_reason": estimation["failure_reason"],
                "estimation_time_ms": estimation_time_ms,
                "n_reference_points": len(pitch_points_ref),
                "n_valid_points": 0,
                "n_confident_keypoints": n_confident,
                "n_ransac_inliers": n_inliers,
                "n_ransac_outliers": n_outliers,
                "ransac_inlier_ratio": inlier_ratio,
                "mean_confident_keypoint_conf": mean_confident_conf,
                "median_confident_keypoint_conf": median_confident_conf,
                "mean_pitch_error_m": np.nan,
                "median_pitch_error_m": np.nan,
                "p90_pitch_error_m": np.nan,
                "max_pitch_error_m": np.nan,
                "mean_reprojection_error_px": np.nan,
                "median_reprojection_error_px": np.nan,
                "p90_reprojection_error_px": np.nan,
                "max_reprojection_error_px": np.nan,
                "error": estimation["failure_reason"],
            },
            pd.DataFrame(),
            keypoint_df,
        )

    h_kp_img2pitch = normalize_h(
        h_kp_img2pitch_raw
    )

    h_kp_pitch2img = normalize_h(
        np.linalg.inv(h_kp_img2pitch)
    )

    point_df = calculate_point_errors(
        pitch_points_ref=pitch_points_ref,
        image_points_ref=image_points_ref,
        h_kp_img2pitch=h_kp_img2pitch,
        h_kp_pitch2img=h_kp_pitch2img,
    )

    if point_df.empty:
        raise RuntimeError(
            f"{eval_stem}: H_keypoints istnieje, "
            "ale brak skończonych punktów ewaluacyjnych."
        )

    point_df.insert(
        0,
        "point_id",
        np.arange(1, len(point_df) + 1),
    )
    point_df.insert(0, "eval_stem", eval_stem)
    point_df.insert(0, "eval_id", eval_id)
    point_df.insert(3, "method", "keypoints")

    pitch_errors = point_df[
        "pitch_error_m"
    ].to_numpy(dtype=np.float64)

    reproj_errors = point_df[
        "reprojection_error_px"
    ].to_numpy(dtype=np.float64)

    row = {
        "eval_id": eval_id,
        "eval_stem": eval_stem,
        "status": "ok",
        "failure_reason": "",
        "estimation_time_ms": estimation_time_ms,
        "n_reference_points": len(pitch_points_ref),
        "n_valid_points": len(point_df),

        "n_confident_keypoints": n_confident,
        "n_ransac_inliers": n_inliers,
        "n_ransac_outliers": n_outliers,
        "ransac_inlier_ratio": inlier_ratio,
        "mean_confident_keypoint_conf": mean_confident_conf,
        "median_confident_keypoint_conf": median_confident_conf,

        "mean_pitch_error_m": float(
            np.mean(pitch_errors)
        ),
        "median_pitch_error_m": float(
            np.median(pitch_errors)
        ),
        "p90_pitch_error_m": percentile(
            pitch_errors,
            90,
        ),
        "max_pitch_error_m": float(
            np.max(pitch_errors)
        ),

        "mean_reprojection_error_px": float(
            np.mean(reproj_errors)
        ),
        "median_reprojection_error_px": float(
            np.median(reproj_errors)
        ),
        "p90_reprojection_error_px": percentile(
            reproj_errors,
            90,
        ),
        "max_reprojection_error_px": float(
            np.max(reproj_errors)
        ),

        "error": "",
    }

    row.update(
        flatten_h(
            "H_keypoints_img2pitch",
            h_kp_img2pitch,
        )
    )
    row.update(
        flatten_h(
            "H_gt_img2pitch",
            h_gt_img2pitch,
        )
    )

    # Zapis H ---------------------------------------------------------
    homography_dir = (
        output_dir
        / "estimated_homographies"
    )
    homography_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    h_json_path = (
        homography_dir
        / f"{eval_stem}_keypoints.json"
    )

    with h_json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "eval_id": eval_id,
                "eval_stem": eval_stem,
                "method": "keypoints",
                "stretch_size": args.stretch_size,
                "keypoint_conf_threshold": args.conf_threshold,
                "ransac_threshold_m": args.ransac_threshold_m,
                "n_confident_keypoints": n_confident,
                "n_ransac_inliers": n_inliers,
                "ransac_inlier_ratio": inlier_ratio,
                "estimation_time_ms": estimation_time_ms,
                "H_image_to_pitch": h_kp_img2pitch.tolist(),
                "H_pitch_to_image": h_kp_pitch2img.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Wizualizacja ----------------------------------------------------
    if not args.no_visualizations:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        vis_path = (
            vis_dir
            / f"{eval_stem}_keypoints_vs_gt.png"
        )

        create_visualization(
            image_bgr=image_bgr,
            eval_stem=eval_stem,
            h_gt_pitch2img=h_gt_pitch2img,
            h_kp_pitch2img=h_kp_pitch2img,
            point_df=point_df,
            keypoint_df=keypoint_df,
            output_path=vis_path,
            pitch_length=PITCH_LENGTH_M,
            pitch_width=PITCH_WIDTH_M,
            num_vis_points=args.num_vis_points,
            conf_threshold=args.conf_threshold,
        )

    return row, point_df, keypoint_df


# =============================================================================
# PODSUMOWANIE
# =============================================================================

def build_summary(
    per_frame: pd.DataFrame,
    point_errors: pd.DataFrame,
) -> pd.DataFrame:

    total_frames = len(per_frame)
    successful = (
        per_frame[
            per_frame["status"] == "ok"
        ]
        .copy()
    )

    summary = {
        "method": "keypoints",
        "total_frames": total_frames,
        "successful_frames": len(successful),
        "failed_frames": (
            total_frames - len(successful)
        ),
        "success_rate": (
            len(successful) / total_frames
            if total_frames
            else 0.0
        ),
    }

    # Diagnostyka liczby punktów liczona po wszystkich klatkach.
    if total_frames:
        summary.update(
            {
                "mean_confident_keypoints_per_frame": float(
                    per_frame[
                        "n_confident_keypoints"
                    ].fillna(0).mean()
                ),
                "median_confident_keypoints_per_frame": float(
                    per_frame[
                        "n_confident_keypoints"
                    ].fillna(0).median()
                ),
                "frames_with_at_least_4_confident": int(
                    (
                        per_frame[
                            "n_confident_keypoints"
                        ].fillna(0)
                        >= MIN_KEYPOINTS_FOR_HOMOGRAPHY
                    ).sum()
                ),
            }
        )

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
                "mean_pitch_error_m": float(
                    np.mean(pitch_values)
                ),
                "median_pitch_error_m": float(
                    np.median(pitch_values)
                ),
                "p90_pitch_error_m": percentile(
                    pitch_values,
                    90,
                ),

                "mean_reprojection_error_px": float(
                    np.mean(reproj_values)
                ),
                "median_reprojection_error_px": float(
                    np.median(reproj_values)
                ),
                "p90_reprojection_error_px": percentile(
                    reproj_values,
                    90,
                ),

                "mean_estimation_time_ms": float(
                    np.mean(time_values)
                ),
                "median_estimation_time_ms": float(
                    np.median(time_values)
                ),

                "mean_ransac_inliers": float(
                    successful[
                        "n_ransac_inliers"
                    ].mean()
                ),
                "median_ransac_inliers": float(
                    successful[
                        "n_ransac_inliers"
                    ].median()
                ),
                "mean_ransac_inlier_ratio": float(
                    successful[
                        "ransac_inlier_ratio"
                    ].mean()
                ),

                "total_evaluation_points": int(
                    len(point_errors)
                ),
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
                "mean_ransac_inliers": np.nan,
                "median_ransac_inliers": np.nan,
                "mean_ransac_inlier_ratio": np.nan,
                "total_evaluation_points": 0,
            }
        )

    return pd.DataFrame([summary])


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if not (0.0 <= args.conf_threshold <= 1.0):
        raise ValueError("--conf-threshold musi należeć do [0,1].")

    if args.ransac_threshold_m <= 0:
        raise ValueError("--ransac-threshold-m musi być > 0.")

    if args.stretch_size <= 0:
        raise ValueError("--stretch-size musi być > 0.")

    if args.grid_step_m <= 0:
        raise ValueError("--grid-step-m musi być > 0.")

    if args.pitch_margin_m < 0:
        raise ValueError("--pitch-margin-m nie może być ujemny.")

    if args.image_margin_px < 0:
        raise ValueError("--image-margin-px nie może być ujemny.")

    if args.num_vis_points < 1:
        raise ValueError("--num-vis-points musi być >= 1.")

    ensure_exists(
        KEYPOINTS_WEIGHTS,
        "wag modelu keypointów",
    )

    template = np.asarray(
        PITCH_KEYPOINTS_TEMPLATE_M,
        dtype=np.float64,
    )

    if template.shape != (32, 2):
        raise ValueError(
            "PITCH_KEYPOINTS_TEMPLATE_M musi mieć shape (32,2), "
            f"a ma {template.shape}."
        )

    manifest = load_manifest()

    samples = select_samples(
        manifest=manifest,
        mode=args.mode,
        eval_id=args.eval_id,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_dir = (
        RESULTS_ROOT
        / f"{timestamp}_keypoints"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    first_image = cv2.imread(
        str(samples[0]["image_path"])
    )

    if first_image is None:
        raise RuntimeError(
            f"Nie udało się odczytać "
            f"{samples[0]['image_path']}"
        )

    image_height, image_width = (
        first_image.shape[:2]
    )

    print("========================================")
    print("EWALUACJA HOMOGRAFII — KEYPOINTS")
    print("========================================")
    print(f"Tryb: {args.mode}")
    print(f"Liczba klatek: {len(samples)}")
    print(
        f"Oryginalna rozdzielczość pierwszej klatki: "
        f"{image_width}x{image_height}"
    )
    print(
        f"Wejście modelu: "
        f"{args.stretch_size}x{args.stretch_size} STRETCH"
    )
    print(
        f"Keypoint confidence threshold: "
        f"{args.conf_threshold}"
    )
    print(
        f"RANSAC threshold: "
        f"{args.ransac_threshold_m} m"
    )
    print(f"Grid step: {args.grid_step_m} m")
    print(
        f"Pitch margin: "
        f"{args.pitch_margin_m} m"
    )
    print(f"Wyniki: {output_dir}")
    print()

    calibrator = StretchKeypointsHomography(
        model_weights=KEYPOINTS_WEIGHTS,
        conf_threshold=args.conf_threshold,
        ransac_threshold_m=args.ransac_threshold_m,
        stretch_size=args.stretch_size,
    )

    pitch_grid = build_canonical_pitch_grid(
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        step_m=args.grid_step_m,
        margin_m=args.pitch_margin_m,
    )

    # Warm-up poza pomiarem czasu.
    print("Warm-up modelu...")

    warm_image = cv2.imread(
        str(samples[0]["image_path"])
    )

    if warm_image is None:
        raise RuntimeError(
            "Nie można wykonać warm-up."
        )

    _ = calibrator.detect_keypoints(
        warm_image
    )

    sync_cuda()
    print("Warm-up gotowy.\n")

    per_frame_rows: List[dict] = []
    point_tables: List[pd.DataFrame] = []
    keypoint_tables: List[pd.DataFrame] = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        eval_stem = sample["eval_stem"]

        print(
            f"[{index:03d}/{len(samples):03d}] "
            f"{eval_stem} ...",
            end=" ",
            flush=True,
        )

        try:
            row, point_df, keypoint_df = (
                evaluate_one(
                    sample=sample,
                    calibrator=calibrator,
                    pitch_grid=pitch_grid,
                    output_dir=output_dir,
                    args=args,
                )
            )

        except Exception as exc:
            row = {
                "eval_id": sample["eval_id"],
                "eval_stem": eval_stem,
                "status": "failed",
                "failure_reason": "exception",
                "estimation_time_ms": np.nan,
                "n_reference_points": np.nan,
                "n_valid_points": 0,
                "n_confident_keypoints": 0,
                "n_ransac_inliers": 0,
                "n_ransac_outliers": 0,
                "ransac_inlier_ratio": np.nan,
                "mean_confident_keypoint_conf": np.nan,
                "median_confident_keypoint_conf": np.nan,
                "mean_pitch_error_m": np.nan,
                "median_pitch_error_m": np.nan,
                "p90_pitch_error_m": np.nan,
                "max_pitch_error_m": np.nan,
                "mean_reprojection_error_px": np.nan,
                "median_reprojection_error_px": np.nan,
                "p90_reprojection_error_px": np.nan,
                "max_reprojection_error_px": np.nan,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

            point_df = pd.DataFrame()

            keypoint_df = pd.DataFrame(
                [
                    {
                        "eval_id": sample["eval_id"],
                        "eval_stem": eval_stem,
                        "keypoint_id": np.nan,
                        "x_px": np.nan,
                        "y_px": np.nan,
                        "confidence": np.nan,
                        "accepted_by_conf": False,
                        "ransac_inlier": False,
                        "pitch_x_m": np.nan,
                        "pitch_y_m": np.nan,
                    }
                ]
            )

        per_frame_rows.append(row)

        if not point_df.empty:
            point_tables.append(point_df)

        if not keypoint_df.empty:
            keypoint_tables.append(keypoint_df)

        if row["status"] == "ok":
            print(
                f"OK | "
                f"kp={row['n_confident_keypoints']} | "
                f"inliers={row['n_ransac_inliers']} | "
                f"pitch={row['mean_pitch_error_m']:.3f} m | "
                f"reproj={row['mean_reprojection_error_px']:.2f} px | "
                f"time={row['estimation_time_ms']:.1f} ms"
            )
        else:
            print(
                f"FAILED | "
                f"kp={row['n_confident_keypoints']} | "
                f"reason={row['failure_reason']} | "
                f"time={row['estimation_time_ms']:.1f} ms"
                if np.isfinite(row["estimation_time_ms"])
                else (
                    f"FAILED | "
                    f"reason={row['failure_reason']} | "
                    f"{row.get('error', '')}"
                )
            )

    per_frame_df = (
        pd.DataFrame(per_frame_rows)
        .sort_values("eval_id")
        .reset_index(drop=True)
    )

    if point_tables:
        point_errors_df = (
            pd.concat(
                point_tables,
                ignore_index=True,
            )
            .sort_values(
                ["eval_id", "point_id"]
            )
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
                "keypoints_pitch_x_m",
                "keypoints_pitch_y_m",
                "keypoints_image_x_px",
                "keypoints_image_y_px",
                "pitch_error_m",
                "reprojection_error_px",
            ]
        )

    if keypoint_tables:
        keypoints_df = (
            pd.concat(
                keypoint_tables,
                ignore_index=True,
            )
            .sort_values(
                ["eval_id", "keypoint_id"],
                na_position="last",
            )
            .reset_index(drop=True)
        )
    else:
        keypoints_df = pd.DataFrame()

    summary_df = build_summary(
        per_frame=per_frame_df,
        point_errors=point_errors_df,
    )

    per_frame_path = (
        output_dir
        / "per_frame_metrics.csv"
    )
    point_errors_path = (
        output_dir
        / "point_errors.csv"
    )
    keypoints_path = (
        output_dir
        / "keypoint_diagnostics.csv"
    )
    summary_path = (
        output_dir
        / "metrics_summary.csv"
    )

    per_frame_df.to_csv(
        per_frame_path,
        index=False,
    )

    point_errors_df.to_csv(
        point_errors_path,
        index=False,
    )

    keypoints_df.to_csv(
        keypoints_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    config = {
        "timestamp": timestamp,
        "method": "keypoints",
        "mode": args.mode,
        "eval_id": (
            args.eval_id
            if args.mode == "single"
            else None
        ),
        "dataset_root": str(EVAL_ROOT),
        "keypoints_weights": str(
            KEYPOINTS_WEIGHTS
        ),
        "original_first_image_width": image_width,
        "original_first_image_height": image_height,
        "stretch_resize": True,
        "stretch_size": int(
            args.stretch_size
        ),
        "keypoint_conf_threshold": float(
            args.conf_threshold
        ),
        "min_keypoints_for_homography": int(
            MIN_KEYPOINTS_FOR_HOMOGRAPHY
        ),
        "homography_direction": (
            "image pixels -> pitch meters"
        ),
        "ransac_threshold_value": float(
            args.ransac_threshold_m
        ),
        "ransac_threshold_unit": "meters",
        "pitch_length_m": PITCH_LENGTH_M,
        "pitch_width_m": PITCH_WIDTH_M,
        "grid_step_m": args.grid_step_m,
        "pitch_margin_m": args.pitch_margin_m,
        "image_margin_px": args.image_margin_px,
        "num_visualization_points": args.num_vis_points,
        "visualizations": (
            not args.no_visualizations
        ),
        "success_definition": (
            "For the frame independently: YOLO pose returns keypoints, "
            "at least 4 keypoints satisfy the confidence threshold, "
            "RANSAC returns a finite invertible image-to-pitch homography "
            "with at least 4 inliers. No previous-frame fallback is used."
        ),
        "timing_definition": (
            "Stretch resize + YOLO pose inference + confidence filtering "
            "+ RANSAC homography estimation. Model initialization and warm-up "
            "are excluded."
        ),
        "evaluation_points_definition": (
            "Regular pitch-plane grid, filtered only by H_GT visibility "
            "inside the original image. Predicted points outside the pitch "
            "are retained and contribute to error."
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

    summary = summary_df.iloc[0]

    print("\n========================================")
    print("PODSUMOWANIE KEYPOINTS")
    print("========================================")
    print(
        f"Success rate: "
        f"{100.0 * summary['success_rate']:.2f}% "
        f"({int(summary['successful_frames'])}/"
        f"{int(summary['total_frames'])})"
    )

    print(
        f"Klatki z >=4 confident kp: "
        f"{int(summary['frames_with_at_least_4_confident'])}/"
        f"{int(summary['total_frames'])}"
    )

    if int(summary["successful_frames"]) > 0:
        print(
            f"Pitch error: mean="
            f"{summary['mean_pitch_error_m']:.3f} m, "
            f"median="
            f"{summary['median_pitch_error_m']:.3f} m"
        )
        print(
            f"Reprojection error: mean="
            f"{summary['mean_reprojection_error_px']:.2f} px, "
            f"median="
            f"{summary['median_reprojection_error_px']:.2f} px"
        )
        print(
            f"RANSAC: mean inliers="
            f"{summary['mean_ransac_inliers']:.2f}, "
            f"mean ratio="
            f"{100.0 * summary['mean_ransac_inlier_ratio']:.1f}%"
        )
        print(
            f"Estimation time: mean="
            f"{summary['mean_estimation_time_ms']:.1f} ms, "
            f"median="
            f"{summary['median_estimation_time_ms']:.1f} ms"
        )

    print(f"\nSummary:    {summary_path}")
    print(f"Per-frame:  {per_frame_path}")
    print(f"Points:     {point_errors_path}")
    print(f"Keypoints:  {keypoints_path}")

    if not args.no_visualizations:
        print(
            f"Visualizations: "
            f"{output_dir / 'visualizations'}"
        )


if __name__ == "__main__":
    main()
