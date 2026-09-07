#!/usr/bin/env python3
"""
Wizualizacja detekcji 32 keypointów boiska na połączonym zbiorze
ewaluacyjnym 128 klatek:

    - 100 klatek SoccerNet-Calibration
    - 28 klatek football-field-detection

Domyślnie wykonywany jest jawny STRETCH obrazu do 640x640 przed
inferencją modelu YOLOv8x-pose.

Schemat:
    original W x H
        -> cv2.resize(..., 640x640)
        -> YOLOv8x-pose
        -> 32 keypointy w układzie 640x640
        -> mapowanie współrzędnych z powrotem do W x H
        -> wizualizacja na oryginalnym obrazie

Plik:
    scripts/visualize_pitch_keypoints_resized.py

Wagi:
    models/pitch_keypoints/trained_keypoints.pt

Dataset:
    data/keypoints_evaluation_combined/
        manifest.csv
        test/
            images/
            labels/

Przykłady:

Jedna klatka:
    python scripts/visualize_pitch_keypoints_resized.py --eval-id 1

Wszystkie 128 klatek:
    python scripts/visualize_pitch_keypoints_resized.py --mode all

Wszystkie klatki, próg 0.6:
    python scripts/visualize_pitch_keypoints_resized.py --mode all --conf-threshold 0.6

Bez pokazywania słabych predykcji:
    python scripts/visualize_pitch_keypoints_resized.py --mode all --conf-threshold 0.6 --hide-low-confidence

Wyłączenie stretchu tylko diagnostycznie:
    python scripts/visualize_pitch_keypoints_resized.py --mode all --disable-stretch

Wyniki:
    results/keypoints_detection_combined/<timestamp>/
        keypoints.csv
        config.json
        visualizations/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.calibration.keypoints_homography import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    MIN_KEYPOINTS_FOR_HOMOGRAPHY,
    PITCH_KEYPOINTS_TEMPLATE_M,
)


# =============================================================================
# KONFIGURACJA
# =============================================================================

EVAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "keypoints_evaluation_combined"
)

EVAL_IMAGES = (
    EVAL_ROOT
    / "test"
    / "images"
)

EVAL_MANIFEST = (
    EVAL_ROOT
    / "manifest.csv"
)

KEYPOINTS_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "pitch_keypoints"
    / "trained_keypoints.pt"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "keypoints_detection_combined"
)

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

DISPLAY_MIN_CONF = 0.05
DEFAULT_STRETCH_SIZE = 640


KEYPOINT_NAMES: Dict[int, str] = {
    0: "left top pitch corner",
    1: "left goal line / penalty top",
    2: "left goal line / goal-area top",
    3: "left goal line / goal-area bottom",
    4: "left goal line / penalty bottom",
    5: "left bottom pitch corner",
    6: "left penalty far top",
    7: "left penalty far bottom",
    8: "left penalty far center",
    9: "left goal-area far top",
    10: "left goal-area far bottom",
    11: "left penalty arc top",
    12: "left penalty arc bottom",
    13: "halfway line top",
    14: "center circle / halfway top",
    15: "center circle / halfway bottom",
    16: "halfway line bottom",
    17: "right goal-area far top",
    18: "right goal-area far bottom",
    19: "right penalty arc top",
    20: "right penalty arc bottom",
    21: "right penalty far center",
    22: "right penalty far top",
    23: "right penalty far bottom",
    24: "right top pitch corner",
    25: "right goal line / penalty top",
    26: "right goal line / goal-area top",
    27: "right goal line / goal-area bottom",
    28: "right goal line / penalty bottom",
    29: "right bottom pitch corner",
    30: "left center-circle point",
    31: "right center-circle point",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wizualizacja detekcji 32 keypointów na połączonym "
            "zbiorze ewaluacyjnym 128 klatek."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        default="single",
        help=(
            "single = jedna klatka, "
            "all = cały zbiór. "
            "Domyślnie single."
        ),
    )

    parser.add_argument(
        "--eval-id",
        type=int,
        default=1,
        help=(
            "Numer obrazu 1..128 dla trybu single. "
            "Domyślnie 1."
        ),
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=KEYPOINT_CONFIDENCE_THRESHOLD,
        help=(
            "Próg confidence uznania keypointu za użyteczny. "
            f"Domyślnie {KEYPOINT_CONFIDENCE_THRESHOLD}."
        ),
    )

    parser.add_argument(
        "--display-min-conf",
        type=float,
        default=DISPLAY_MIN_CONF,
        help=(
            "Minimalny confidence słabego punktu pokazywanego "
            f"na wizualizacji. Domyślnie {DISPLAY_MIN_CONF}."
        ),
    )

    parser.add_argument(
        "--hide-low-confidence",
        action="store_true",
        help=(
            "Nie pokazuj punktów poniżej --conf-threshold."
        ),
    )

    parser.add_argument(
        "--stretch-size",
        type=int,
        default=DEFAULT_STRETCH_SIZE,
        help=(
            "Rozmiar wejścia modelu po stretch resize. "
            f"Domyślnie {DEFAULT_STRETCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--disable-stretch",
        action="store_true",
        help=(
            "Wyłącz jawny stretch do 640x640. "
            "Opcja wyłącznie diagnostyczna."
        ),
    )

    return parser.parse_args()


# =============================================================================
# HELPERY
# =============================================================================

def ensure_exists(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Brak {description}: {path}"
        )


def load_manifest() -> pd.DataFrame:
    ensure_exists(
        EVAL_MANIFEST,
        "manifest.csv",
    )

    ensure_exists(
        EVAL_IMAGES,
        "katalogu obrazów ewaluacyjnych",
    )

    manifest = pd.read_csv(
        EVAL_MANIFEST
    )

    required = {
        "combined_id",
        "source",
        "output_image",
    }

    missing = required - set(
        manifest.columns
    )

    if missing:
        raise ValueError(
            "manifest.csv nie zawiera wymaganych kolumn: "
            f"{sorted(missing)}"
        )

    manifest = manifest.reset_index(
        drop=True
    )

    # Numer techniczny 1..128 do wygodnego --eval-id.
    manifest["sample_id"] = np.arange(
        1,
        len(manifest) + 1,
    )

    return manifest


def select_samples(
    manifest: pd.DataFrame,
    mode: str,
    eval_id: int,
) -> List[dict]:

    if mode == "single":
        selected = manifest[
            manifest["sample_id"] == eval_id
        ]

        if selected.empty:
            raise ValueError(
                f"Nie znaleziono eval-id={eval_id}. "
                f"Dostępny zakres: 1-{len(manifest)}."
            )
    else:
        selected = manifest

    samples: List[dict] = []

    for row in selected.itertuples(
        index=False
    ):
        # Builder zapisuje output_image względem PROJECT_ROOT.
        image_path = (
            PROJECT_ROOT
            / str(row.output_image)
        )

        ensure_exists(
            image_path,
            f"obrazu {row.combined_id}",
        )

        samples.append(
            {
                "eval_id": int(
                    row.sample_id
                ),
                "eval_stem": str(
                    row.combined_id
                ),
                "source": str(
                    row.source
                ),
                "image_path": image_path,
            }
        )

    return samples


def point_is_drawable(
    x: float,
    y: float,
    conf: float,
    image_width: int,
    image_height: int,
    display_min_conf: float,
) -> bool:

    return (
        np.isfinite(x)
        and np.isfinite(y)
        and np.isfinite(conf)
        and conf >= display_min_conf
        and 0.0 <= x < image_width
        and 0.0 <= y < image_height
    )


# =============================================================================
# WIZUALIZACJA MODELU BOISKA
# =============================================================================

def draw_pitch_template(
    ax,
) -> None:

    x_min = -PITCH_LENGTH_M / 2.0
    x_max = PITCH_LENGTH_M / 2.0

    y_min = -PITCH_WIDTH_M / 2.0
    y_max = PITCH_WIDTH_M / 2.0

    # Obrys boiska.
    ax.plot(
        [
            x_min,
            x_max,
            x_max,
            x_min,
            x_min,
        ],
        [
            y_min,
            y_min,
            y_max,
            y_max,
            y_min,
        ],
        linewidth=1.3,
    )

    # Linia środkowa.
    ax.plot(
        [0.0, 0.0],
        [y_min, y_max],
        linewidth=1.0,
    )

    # Koło środkowe.
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        240,
    )

    ax.plot(
        9.15 * np.cos(theta),
        9.15 * np.sin(theta),
        linewidth=1.0,
    )

    # Pola karne.
    for x_goal, x_far in [
        (-52.5, -36.0),
        (52.5, 36.0),
    ]:
        ax.plot(
            [
                x_goal,
                x_far,
                x_far,
                x_goal,
            ],
            [
                -20.16,
                -20.16,
                20.16,
                20.16,
            ],
            linewidth=1.0,
        )

    # Pola bramkowe.
    for x_goal, x_far in [
        (-52.5, -47.0),
        (52.5, 47.0),
    ]:
        ax.plot(
            [
                x_goal,
                x_far,
                x_far,
                x_goal,
            ],
            [
                -9.16,
                -9.16,
                9.16,
                9.16,
            ],
            linewidth=1.0,
        )

    pts = np.asarray(
        PITCH_KEYPOINTS_TEMPLATE_M,
        dtype=np.float64,
    )

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=22,
        marker="o",
        zorder=4,
    )

    for idx, (
        x_m,
        y_m,
    ) in enumerate(pts):

        ax.annotate(
            f"K{idx:02d}",
            (x_m, y_m),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlim(
        x_min - 3.0,
        x_max + 3.0,
    )

    ax.set_ylim(
        y_max + 3.0,
        y_min - 3.0,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.set_title(
        "Referencyjne indeksy 32 keypointów"
    )

    ax.grid(
        alpha=0.2
    )


# =============================================================================
# DETEKTOR KEYPOINTÓW
# =============================================================================

class ResizedKeypointsDetector:
    """
    Detekcja keypointów z opcjonalnym jawnym stretch resize.

    Tryb domyślny:

        original W x H
        -> resize 640x640
        -> YOLO
        -> keypointy 640x640
        -> mapowanie do W x H
    """

    def __init__(
        self,
        model_weights: str,
        conf_threshold: float,
        stretch_size: int,
        use_stretch: bool,
    ) -> None:

        self.model = YOLO(
            model_weights
        )

        self.conf_threshold = float(
            conf_threshold
        )

        self.stretch_size = int(
            stretch_size
        )

        self.use_stretch = bool(
            use_stretch
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[Keypoints] Urządzenie: {self.device}"
        )

        print(
            f"[Keypoints] Model: {model_weights}"
        )

    def _extract_best_detection_keypoints(
        self,
        results,
    ) -> Optional[np.ndarray]:
        """
        Jeżeli YOLO zwróci więcej niż jedną detekcję pitch,
        wybieramy tę o najwyższej ŚREDNIEJ pewności 32 keypointów.

        Jest to ta sama reguła, której używamy podczas właściwej
        ewaluacji metody keypointowej.
        """

        if not results:
            return None

        result = results[0]

        if (
            result.keypoints is None
            or len(result.keypoints) == 0
        ):
            return None

        kpts_xy = (
            result.keypoints.xy
        )

        kpts_conf = (
            result.keypoints.conf
        )

        if (
            kpts_xy is None
            or kpts_conf is None
        ):
            return None

        if (
            len(kpts_xy) == 0
            or len(kpts_conf) == 0
        ):
            return None

        # Najlepsza detekcja według średniego confidence keypointów.
        best_idx = int(
            kpts_conf
            .mean(dim=1)
            .argmax()
        )

        xy = (
            kpts_xy[
                best_idx
            ]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        conf = (
            kpts_conf[
                best_idx
            ]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if (
            xy.shape != (32, 2)
            or conf.shape != (32,)
        ):
            raise RuntimeError(
                "Nieoczekiwany shape keypointów: "
                f"xy={xy.shape}, conf={conf.shape}"
            )

        return np.column_stack(
            [
                xy,
                conf,
            ]
        )

    def detect(
        self,
        image_path: str,
    ) -> Tuple[
        Optional[np.ndarray],
        dict,
    ]:

        image_bgr = cv2.imread(
            image_path
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Nie można odczytać obrazu: {image_path}"
            )

        orig_h, orig_w = (
            image_bgr.shape[:2]
        )

        # ---------------------------------------------------------------------
        # TRYB BEZ STRETCH — tylko diagnostycznie
        # ---------------------------------------------------------------------

        if not self.use_stretch:

            results = self.model(
                image_bgr,
                verbose=False,
                device=self.device,
            )

            keypoints = (
                self._extract_best_detection_keypoints(
                    results
                )
            )

            meta = {
                "use_stretch": False,
                "orig_width": orig_w,
                "orig_height": orig_h,
                "input_width": orig_w,
                "input_height": orig_h,
                "scale_x_back": 1.0,
                "scale_y_back": 1.0,
            }

            return (
                keypoints,
                meta,
            )

        # ---------------------------------------------------------------------
        # WŁAŚCIWY TRYB — STRETCH 640x640
        # ---------------------------------------------------------------------

        resized_bgr = cv2.resize(
            image_bgr,
            (
                self.stretch_size,
                self.stretch_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        results = self.model(
            resized_bgr,
            verbose=False,
            device=self.device,
            imgsz=self.stretch_size,
        )

        keypoints = (
            self._extract_best_detection_keypoints(
                results
            )
        )

        scale_x = (
            orig_w
            / float(
                self.stretch_size
            )
        )

        scale_y = (
            orig_h
            / float(
                self.stretch_size
            )
        )

        meta = {
            "use_stretch": True,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "input_width": self.stretch_size,
            "input_height": self.stretch_size,
            "scale_x_back": scale_x,
            "scale_y_back": scale_y,
        }

        if keypoints is None:
            return (
                None,
                meta,
            )

        # Współrzędne modelu 640x640
        # -> oryginalny rozmiar obrazu.
        keypoints_back = (
            keypoints.copy()
        )

        keypoints_back[
            :,
            0,
        ] *= scale_x

        keypoints_back[
            :,
            1,
        ] *= scale_y

        return (
            keypoints_back,
            meta,
        )


# =============================================================================
# WIZUALIZACJA JEDNEJ KLATKI
# =============================================================================

def create_visualization(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    eval_stem: str,
    source: str,
    conf_threshold: float,
    display_min_conf: float,
    hide_low_confidence: bool,
    output_path: Path,
    input_mode_text: str,
) -> None:

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image_height, image_width = (
        image_bgr.shape[:2]
    )

    confident_mask = (
        keypoints[:, 2]
        >= conf_threshold
    )

    n_confident = int(
        confident_mask.sum()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
    )

    ax_img, ax_pitch = axes

    ax_img.imshow(
        image_rgb
    )

    ax_img.set_xlim(
        0,
        image_width,
    )

    ax_img.set_ylim(
        image_height,
        0,
    )

    ax_img.axis(
        "off"
    )

    ax_img.set_title(
        f"{eval_stem} | {source}\n"
        f"wykryte keypointy: "
        f"{n_confident}/32 "
        f"z conf >= {conf_threshold:.2f}"
    )

    label_confident_used = False
    label_low_used = False

    for idx, (
        x,
        y,
        conf,
    ) in enumerate(
        keypoints
    ):

        if not point_is_drawable(
            x=x,
            y=y,
            conf=conf,
            image_width=image_width,
            image_height=image_height,
            display_min_conf=display_min_conf,
        ):
            continue

        accepted = (
            conf
            >= conf_threshold
        )

        if accepted:

            ax_img.scatter(
                [x],
                [y],
                s=30,
                marker="o",
                zorder=5,
                label=(
                    f"conf >= {conf_threshold:.2f}"
                    if not label_confident_used
                    else None
                ),
            )

            label_confident_used = True

        else:

            if hide_low_confidence:
                continue

            ax_img.scatter(
                [x],
                [y],
                s=25,
                marker="x",
                linewidths=1.2,
                alpha=0.65,
                zorder=4,
                label=(
                    f"{display_min_conf:.2f} <= conf < "
                    f"{conf_threshold:.2f}"
                    if not label_low_used
                    else None
                ),
            )

            label_low_used = True

        ax_img.annotate(
            f"K{idx:02d}\n{conf:.2f}",
            (x, y),
            xytext=(4, -5),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(
                boxstyle="round,pad=0.15",
                alpha=0.55,
            ),
        )

    if (
        label_confident_used
        or label_low_used
    ):
        ax_img.legend(
            loc="lower left",
            fontsize=8,
        )

    draw_pitch_template(
        ax_pitch
    )

    enough_text = (
        "TAK"
        if (
            n_confident
            >= MIN_KEYPOINTS_FOR_HOMOGRAPHY
        )
        else "NIE"
    )

    fig.suptitle(
        "YOLOv8x-pose keypoints | "
        f"punktów do H: {n_confident} | "
        f">= {MIN_KEYPOINTS_FOR_HOMOGRAPHY}? "
        f"{enough_text} | "
        f"wejście: {input_mode_text}",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    if not (
        0.0
        <= args.conf_threshold
        <= 1.0
    ):
        raise ValueError(
            "--conf-threshold musi należeć do [0,1]."
        )

    if not (
        0.0
        <= args.display_min_conf
        <= 1.0
    ):
        raise ValueError(
            "--display-min-conf musi należeć do [0,1]."
        )

    if (
        args.display_min_conf
        > args.conf_threshold
    ):
        raise ValueError(
            "--display-min-conf nie powinien być większy "
            "od --conf-threshold."
        )

    if args.stretch_size < 32:
        raise ValueError(
            "--stretch-size ma niepoprawną wartość."
        )

    ensure_exists(
        KEYPOINTS_WEIGHTS,
        "wag modelu keypointów",
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
        / timestamp
    )

    vis_dir = (
        output_dir
        / "visualizations"
    )

    vis_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    use_stretch = (
        not args.disable_stretch
    )

    print(
        "=" * 70
    )

    print(
        "WIZUALIZACJA DETEKCJI KEYPOINTÓW"
    )

    print(
        "=" * 70
    )

    print(
        f"Tryb: {args.mode}"
    )

    print(
        f"Liczba klatek: {len(samples)}"
    )

    print(
        f"Dataset: {EVAL_ROOT}"
    )

    print(
        f"Wagi: {KEYPOINTS_WEIGHTS}"
    )

    print(
        f"Próg confidence: {args.conf_threshold}"
    )

    print(
        f"Minimalny conf pokazywany: "
        f"{args.display_min_conf}"
    )

    print(
        f"Wymuszony stretch resize: "
        f"{use_stretch}"
    )

    if use_stretch:
        print(
            f"Wejście modelu: "
            f"{args.stretch_size}x"
            f"{args.stretch_size}"
        )

    print(
        f"Wyniki: {output_dir}"
    )

    print()

    detector = ResizedKeypointsDetector(
        model_weights=str(
            KEYPOINTS_WEIGHTS
        ),
        conf_threshold=(
            args.conf_threshold
        ),
        stretch_size=(
            args.stretch_size
        ),
        use_stretch=(
            use_stretch
        ),
    )

    rows: List[dict] = []

    for sample_idx, sample in enumerate(
        samples,
        start=1,
    ):

        eval_id = sample[
            "eval_id"
        ]

        eval_stem = sample[
            "eval_stem"
        ]

        source = sample[
            "source"
        ]

        image_path = sample[
            "image_path"
        ]

        print(
            f"[{sample_idx:03d}/"
            f"{len(samples):03d}] "
            f"{eval_stem} "
            f"({source}) ...",
            end=" ",
            flush=True,
        )

        image_bgr = cv2.imread(
            str(image_path)
        )

        if image_bgr is None:
            print(
                "FAILED: nie można odczytać obrazu"
            )
            continue

        keypoints, meta = (
            detector.detect(
                str(image_path)
            )
        )

        if keypoints is None:

            print(
                "BRAK DETEKCJI"
            )

            rows.append(
                {
                    "eval_id": eval_id,
                    "eval_stem": eval_stem,
                    "source": source,
                    "keypoint_id": None,
                    "keypoint_name": None,
                    "x_px": None,
                    "y_px": None,
                    "confidence": None,
                    "accepted_for_homography": False,
                    "pitch_x_m": None,
                    "pitch_y_m": None,
                    "input_width": meta[
                        "input_width"
                    ],
                    "input_height": meta[
                        "input_height"
                    ],
                    "orig_width": meta[
                        "orig_width"
                    ],
                    "orig_height": meta[
                        "orig_height"
                    ],
                    "scale_x_back": meta[
                        "scale_x_back"
                    ],
                    "scale_y_back": meta[
                        "scale_y_back"
                    ],
                    "stretch_resize_used": meta[
                        "use_stretch"
                    ],
                }
            )

            continue

        if keypoints.shape != (
            32,
            3,
        ):
            raise RuntimeError(
                f"{eval_stem}: "
                "oczekiwano shape (32,3), "
                f"otrzymano {keypoints.shape}."
            )

        n_confident = int(
            np.sum(
                keypoints[:, 2]
                >= args.conf_threshold
            )
        )

        for kp_id, (
            x,
            y,
            conf,
        ) in enumerate(
            keypoints
        ):

            pitch_xy = (
                PITCH_KEYPOINTS_TEMPLATE_M[
                    kp_id
                ]
            )

            rows.append(
                {
                    "eval_id": eval_id,
                    "eval_stem": eval_stem,
                    "source": source,
                    "keypoint_id": kp_id,
                    "keypoint_name": (
                        KEYPOINT_NAMES[
                            kp_id
                        ]
                    ),
                    "x_px": float(x),
                    "y_px": float(y),
                    "confidence": float(
                        conf
                    ),
                    "accepted_for_homography": bool(
                        conf
                        >= args.conf_threshold
                    ),
                    "pitch_x_m": float(
                        pitch_xy[0]
                    ),
                    "pitch_y_m": float(
                        pitch_xy[1]
                    ),
                    "input_width": meta[
                        "input_width"
                    ],
                    "input_height": meta[
                        "input_height"
                    ],
                    "orig_width": meta[
                        "orig_width"
                    ],
                    "orig_height": meta[
                        "orig_height"
                    ],
                    "scale_x_back": meta[
                        "scale_x_back"
                    ],
                    "scale_y_back": meta[
                        "scale_y_back"
                    ],
                    "stretch_resize_used": meta[
                        "use_stretch"
                    ],
                }
            )

        vis_path = (
            vis_dir
            / f"{eval_stem}_keypoints.png"
        )

        input_mode_text = (
            f"{meta['input_width']}x"
            f"{meta['input_height']} "
            "(stretch)"
            if meta["use_stretch"]
            else
            f"{meta['input_width']}x"
            f"{meta['input_height']} "
            "(bez stretch)"
        )

        create_visualization(
            image_bgr=image_bgr,
            keypoints=keypoints,
            eval_stem=eval_stem,
            source=source,
            conf_threshold=(
                args.conf_threshold
            ),
            display_min_conf=(
                args.display_min_conf
            ),
            hide_low_confidence=(
                args.hide_low_confidence
            ),
            output_path=vis_path,
            input_mode_text=(
                input_mode_text
            ),
        )

        status = (
            "WYSTARCZY DO H"
            if (
                n_confident
                >= MIN_KEYPOINTS_FOR_HOMOGRAPHY
            )
            else
            "ZA MAŁO DO H"
        )

        print(
            f"{n_confident}/32 "
            f">= {args.conf_threshold:.2f} "
            f"| {status}"
        )

    # =========================================================================
    # CSV
    # =========================================================================

    csv_path = (
        output_dir
        / "keypoints.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        csv_path,
        index=False,
    )

    # =========================================================================
    # CONFIG
    # =========================================================================

    source_counts = (
        manifest[
            "source"
        ]
        .value_counts()
        .to_dict()
    )

    config = {
        "timestamp": timestamp,
        "mode": args.mode,
        "eval_id": (
            args.eval_id
            if args.mode == "single"
            else None
        ),
        "weights": str(
            KEYPOINTS_WEIGHTS
        ),
        "dataset_root": str(
            EVAL_ROOT
        ),
        "total_dataset_images": int(
            len(manifest)
        ),
        "source_counts": {
            str(k): int(v)
            for k, v
            in source_counts.items()
        },
        "confidence_threshold": float(
            args.conf_threshold
        ),
        "display_min_confidence": float(
            args.display_min_conf
        ),
        "min_keypoints_for_homography": int(
            MIN_KEYPOINTS_FOR_HOMOGRAPHY
        ),
        "stretch_resize_used": bool(
            use_stretch
        ),
        "stretch_size": int(
            args.stretch_size
        ),
        "note": (
            "Original image -> explicit square stretch -> "
            "YOLO pose -> keypoints mapped back to original "
            "image coordinates. If multiple pitch detections "
            "are returned, the detection with the highest mean "
            "keypoint confidence is selected."
        ),
    }

    config_path = (
        output_dir
        / "config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=" * 70
    )
    print(
        "GOTOWE"
    )
    print(
        "=" * 70
    )
    print(
        f"CSV: {csv_path}"
    )
    print(
        f"Config: {config_path}"
    )
    print(
        f"Wizualizacje: {vis_dir}"
    )


if __name__ == "__main__":
    main()