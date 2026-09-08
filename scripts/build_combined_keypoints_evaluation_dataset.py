#!/usr/bin/env python3
"""
Budowa połączonego zbioru testowego keypointów boiska.

Źródła:
1) 100 klatek z:
       data/data_homography_evaluation/
   Dla każdej klatki 32 referencyjne punkty boiska są rzutowane przez
   H_GT (pitch -> image). Do labela YOLO Pose trafiają jako widoczne
   tylko te punkty, których projekcja leży WEWNĄTRZ oryginalnego obrazu.

2) Oryginalny split test z:
       data/football-field-detection-16/test/
   Obrazy i ręczne etykiety YOLO Pose są kopiowane bez zmian.

Wynik:
    data/keypoints_evaluation_combined/
        test/
            images/
            labels/
        visualizations/
            pitch_template_32_keypoints.png
            sn_*.png
            ffd_*.png
        manifest.csv
        data.yaml
        summary.json

WAŻNE:
- Nie zmieniamy rozdzielczości obrazów SoccerNet.
- GT keypointów SoccerNet jest wyznaczane w ORYGINALNYM układzie obrazu.
- Późniejszy evaluator może robić:
      original -> stretch 640x640 -> model -> back to original size
  i dopiero wtedy porównywać predykcję z tym GT.
- Punkty poza kadrem dostają w YOLO Pose:
      x=0, y=0, visibility=0
- Punkty wewnątrz kadru:
      visibility=2

Umieść jako:
    scripts/build_combined_keypoints_evaluation_dataset.py

Uruchomienie:
    python scripts/build_combined_keypoints_evaluation_dataset.py

Bez wizualizacji:
    python scripts/build_combined_keypoints_evaluation_dataset.py --no-visualizations

Tylko np. pierwsze 10 wizualizacji:
    python scripts/build_combined_keypoints_evaluation_dataset.py --max-visualizations 10

Nadpisanie wcześniej utworzonego zbioru:
    python scripts/build_combined_keypoints_evaluation_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from src.calibration.keypoints_homography import (
    PITCH_KEYPOINTS_TEMPLATE_M,
)


# =============================================================================
# ŚCIEŻKI
# =============================================================================

HOMO_EVAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "data_homography_evaluation"
)

HOMO_IMAGES_DIR = (
    HOMO_EVAL_ROOT
    / "images"
)

HOMO_MANIFEST = (
    HOMO_EVAL_ROOT
    / "manifest.csv"
)

HOMO_GT_DIR = (
    HOMO_EVAL_ROOT
    / "ground_truth"
)

FFD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "football-field-detection-16"
)

FFD_TEST_IMAGES = (
    FFD_ROOT
    / "test"
    / "images"
)

FFD_TEST_LABELS = (
    FFD_ROOT
    / "test"
    / "labels"
)

FFD_DATA_YAML = (
    FFD_ROOT
    / "data.yaml"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "keypoints_evaluation_combined"
)

OUTPUT_IMAGES = (
    OUTPUT_ROOT
    / "test"
    / "images"
)

OUTPUT_LABELS = (
    OUTPUT_ROOT
    / "test"
    / "labels"
)

OUTPUT_VIS = (
    OUTPUT_ROOT
    / "visualizations"
)


# =============================================================================
# KONFIGURACJA
# =============================================================================

NUM_KEYPOINTS = 32
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Punkt referencyjny musi leżeć trochę wewnątrz obrazu.
KEYPOINT_IMAGE_MARGIN_PX = 5.0

# Padding bbox obiektu "pitch".
BBOX_PADDING_PX = 8.0

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tworzy połączony test set keypointów z 100 klatek "
            "SoccerNet/H_GT oraz football-field-detection-16/test."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Katalog wynikowy. Domyślnie: {OUTPUT_ROOT}",
    )

    parser.add_argument(
        "--keypoint-margin-px",
        type=float,
        default=KEYPOINT_IMAGE_MARGIN_PX,
        help=(
            "Margines od krawędzi obrazu używany przy kwalifikacji "
            "punktu jako znajdującego się w kadrze. "
            f"Domyślnie {KEYPOINT_IMAGE_MARGIN_PX}px."
        ),
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Nie zapisuj wizualizacji GT.",
    )

    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=None,
        help=(
            "Opcjonalnie ogranicz liczbę wizualizacji obrazów. "
            "Pitch template jest zapisywany niezależnie."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Usuń istniejący katalog wynikowy i zbuduj go od nowa.",
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


def normalize_h(
    h: np.ndarray,
) -> np.ndarray:
    h = np.asarray(
        h,
        dtype=np.float64,
    )

    if h.shape != (3, 3):
        raise ValueError(
            f"Homografia ma shape {h.shape}, oczekiwano (3,3)."
        )

    if not np.isfinite(h).all():
        raise ValueError(
            "Homografia zawiera NaN/Inf."
        )

    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    else:
        norm = np.linalg.norm(h)
        if norm <= 1e-12:
            raise ValueError(
                "Nieprawidłowa homografia."
            )
        h = h / norm

    return h


def transform_points(
    h: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    points_xy = np.asarray(
        points_xy,
        dtype=np.float64,
    ).reshape(-1, 2)

    homogeneous = np.column_stack(
        [
            points_xy,
            np.ones(
                len(points_xy),
                dtype=np.float64,
            ),
        ]
    )

    projected = (
        h
        @ homogeneous.T
    ).T

    valid = (
        np.abs(
            projected[:, 2]
        )
        > 1e-12
    )

    result = np.full(
        (len(points_xy), 2),
        np.nan,
        dtype=np.float64,
    )

    result[valid] = (
        projected[valid, :2]
        / projected[valid, 2:3]
    )

    return result


def load_h_gt(
    gt_path: Path,
) -> np.ndarray:
    """
    Zwraca H_pitch_to_image.
    """
    with gt_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if "H_pitch_to_image" in data:
        h = data[
            "H_pitch_to_image"
        ]
    elif "H_image_to_pitch" in data:
        h_img2pitch = normalize_h(
            np.asarray(
                data["H_image_to_pitch"],
                dtype=np.float64,
            )
        )
        h = np.linalg.inv(
            h_img2pitch
        )
    else:
        raise KeyError(
            f"{gt_path.name}: brak H_pitch_to_image "
            "i H_image_to_pitch."
        )

    return normalize_h(
        np.asarray(
            h,
            dtype=np.float64,
        )
    )


def projected_keypoints_from_h(
    h_pitch_to_image: np.ndarray,
    image_width: int,
    image_height: int,
    margin_px: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Rzutuje wszystkie 32 keypointy na obraz.

    Zwraca:
        xy_px     [32,2]
        visible   [32] bool

    visible oznacza tylko:
        "punkt geometrycznie wypada wewnątrz obrazu"
    a nie:
        "punkt jest na pewno wizualnie niezasłonięty".
    """
    template = np.asarray(
        PITCH_KEYPOINTS_TEMPLATE_M,
        dtype=np.float64,
    )

    if template.shape != (
        NUM_KEYPOINTS,
        2,
    ):
        raise ValueError(
            "PITCH_KEYPOINTS_TEMPLATE_M musi mieć shape (32,2), "
            f"a ma {template.shape}."
        )

    xy_px = transform_points(
        h_pitch_to_image,
        template,
    )

    visible = (
        np.isfinite(
            xy_px
        ).all(axis=1)
        & (
            xy_px[:, 0]
            >= margin_px
        )
        & (
            xy_px[:, 0]
            < image_width - margin_px
        )
        & (
            xy_px[:, 1]
            >= margin_px
        )
        & (
            xy_px[:, 1]
            < image_height - margin_px
        )
    )

    return (
        xy_px,
        visible,
    )


def dense_visible_pitch_points(
    h_pitch_to_image: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Gęsta siatka boiska służąca wyłącznie do oszacowania bbox obiektu pitch.
    """
    xs = np.linspace(
        -PITCH_LENGTH_M / 2.0,
        PITCH_LENGTH_M / 2.0,
        54,
    )

    ys = np.linspace(
        -PITCH_WIDTH_M / 2.0,
        PITCH_WIDTH_M / 2.0,
        36,
    )

    xx, yy = np.meshgrid(
        xs,
        ys,
    )

    pitch_points = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
        ]
    )

    uv = transform_points(
        h_pitch_to_image,
        pitch_points,
    )

    inside = (
        np.isfinite(
            uv
        ).all(axis=1)
        & (
            uv[:, 0] >= 0
        )
        & (
            uv[:, 0] < image_width
        )
        & (
            uv[:, 1] >= 0
        )
        & (
            uv[:, 1] < image_height
        )
    )

    return uv[inside]


def bbox_from_reference_geometry(
    h_pitch_to_image: np.ndarray,
    keypoints_px: np.ndarray,
    visible_mask: np.ndarray,
    image_width: int,
    image_height: int,
    padding_px: float = BBOX_PADDING_PX,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Zwraca bbox YOLO:
        cx_norm, cy_norm, w_norm, h_norm

    Najpierw próbuje wyznaczyć bbox z widocznej części gęstej siatki boiska.
    Gdy to niemożliwe, używa widocznych keypointów.
    Ostateczny fallback to cały obraz.
    """
    candidates = dense_visible_pitch_points(
        h_pitch_to_image,
        image_width,
        image_height,
    )

    if len(candidates) < 4:
        candidates = (
            keypoints_px[
                visible_mask
            ]
        )

    if len(candidates) == 0:
        return (
            0.5,
            0.5,
            1.0,
            1.0,
        )

    x1 = max(
        0.0,
        float(
            np.min(
                candidates[:, 0]
            )
        )
        - padding_px,
    )

    y1 = max(
        0.0,
        float(
            np.min(
                candidates[:, 1]
            )
        )
        - padding_px,
    )

    x2 = min(
        float(image_width),
        float(
            np.max(
                candidates[:, 0]
            )
        )
        + padding_px,
    )

    y2 = min(
        float(image_height),
        float(
            np.max(
                candidates[:, 1]
            )
        )
        + padding_px,
    )

    width = max(
        x2 - x1,
        1.0,
    )

    height = max(
        y2 - y1,
        1.0,
    )

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    return (
        cx / image_width,
        cy / image_height,
        width / image_width,
        height / image_height,
    )


def write_yolo_pose_label(
    output_path: Path,
    bbox_yolo: Tuple[
        float,
        float,
        float,
        float,
    ],
    keypoints_px: np.ndarray,
    visible_mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    """
    Format:
        class cx cy w h
        kp0_x kp0_y visibility
        ...
        kp31_x kp31_y visibility
    """
    cx, cy, bw, bh = (
        bbox_yolo
    )

    values: List[str] = [
        "0",
        f"{cx:.10f}",
        f"{cy:.10f}",
        f"{bw:.10f}",
        f"{bh:.10f}",
    ]

    for kp_id in range(
        NUM_KEYPOINTS
    ):
        if visible_mask[
            kp_id
        ]:
            x_norm = (
                keypoints_px[
                    kp_id,
                    0,
                ]
                / image_width
            )

            y_norm = (
                keypoints_px[
                    kp_id,
                    1,
                ]
                / image_height
            )

            # Ochrona numeryczna.
            x_norm = float(
                np.clip(
                    x_norm,
                    0.0,
                    1.0,
                )
            )

            y_norm = float(
                np.clip(
                    y_norm,
                    0.0,
                    1.0,
                )
            )

            values.extend(
                [
                    f"{x_norm:.10f}",
                    f"{y_norm:.10f}",
                    "2",
                ]
            )
        else:
            values.extend(
                [
                    "0",
                    "0",
                    "0",
                ]
            )

    output_path.write_text(
        " ".join(values) + "\n",
        encoding="utf-8",
    )


def parse_yolo_pose_label(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Odczyt istniejącego labela football-field-detection.
    Zwraca:
        xy_px [32,2]
        visibility [32]
    """
    lines = [
        x.strip()
        for x in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if not lines:
        raise ValueError(
            f"Pusty label: {label_path}"
        )

    values = [
        float(v)
        for v in lines[0].split()
    ]

    expected = (
        5
        + NUM_KEYPOINTS * 3
    )

    if len(values) != expected:
        raise ValueError(
            f"{label_path.name}: oczekiwano {expected} wartości, "
            f"otrzymano {len(values)}."
        )

    kp = np.asarray(
        values[5:],
        dtype=np.float64,
    ).reshape(
        NUM_KEYPOINTS,
        3,
    )

    xy_px = np.empty(
        (
            NUM_KEYPOINTS,
            2,
        ),
        dtype=np.float64,
    )

    xy_px[:, 0] = (
        kp[:, 0]
        * image_width
    )

    xy_px[:, 1] = (
        kp[:, 1]
        * image_height
    )

    visibility = (
        kp[:, 2]
    )

    return (
        xy_px,
        visibility,
    )


# =============================================================================
# WIZUALIZACJE
# =============================================================================

def visualize_keypoints_on_image(
    image_path: Path,
    keypoints_px: np.ndarray,
    visible_mask: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Nie można odczytać obrazu: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(12, 7),
    )

    ax.imshow(
        image_rgb
    )

    ax.axis(
        "off"
    )

    ids = np.flatnonzero(
        visible_mask
    )

    if len(ids):
        pts = keypoints_px[
            ids
        ]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=34,
            marker="o",
            zorder=5,
            label=(
                "referencyjny keypoint"
            ),
        )

        for kp_id in ids:
            x, y = (
                keypoints_px[
                    kp_id
                ]
            )

            ax.annotate(
                f"K{kp_id:02d}",
                (
                    x,
                    y,
                ),
                xytext=(
                    4,
                    4,
                ),
                textcoords=(
                    "offset points"
                ),
                fontsize=8,
                bbox=dict(
                    boxstyle=(
                        "round,pad=0.15"
                    ),
                    alpha=0.55,
                ),
            )

    ax.set_title(
        f"{title} | "
        f"punkty w kadrze: "
        f"{len(ids)}/32"
    )

    if len(ids):
        ax.legend(
            loc="lower left",
            fontsize=8,
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


def visualize_pitch_template(
    output_path: Path,
) -> None:
    template = np.asarray(
        PITCH_KEYPOINTS_TEMPLATE_M,
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(12, 8),
    )

    x_min = (
        -PITCH_LENGTH_M
        / 2.0
    )

    x_max = (
        PITCH_LENGTH_M
        / 2.0
    )

    y_min = (
        -PITCH_WIDTH_M
        / 2.0
    )

    y_max = (
        PITCH_WIDTH_M
        / 2.0
    )

    # Obrys
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
        linewidth=1.2,
    )

    # Połowa
    ax.plot(
        [
            0.0,
            0.0,
        ],
        [
            y_min,
            y_max,
        ],
        linewidth=1.0,
    )

    # Koło środkowe
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        300,
    )

    ax.plot(
        9.15
        * np.cos(theta),
        9.15
        * np.sin(theta),
        linewidth=1.0,
    )

    # Pola karne
    for goal_x, far_x in [
        (
            -52.5,
            -36.0,
        ),
        (
            52.5,
            36.0,
        ),
    ]:
        ax.plot(
            [
                goal_x,
                far_x,
                far_x,
                goal_x,
            ],
            [
                -20.16,
                -20.16,
                20.16,
                20.16,
            ],
            linewidth=1.0,
        )

    # Pola bramkowe
    for goal_x, far_x in [
        (
            -52.5,
            -47.0,
        ),
        (
            52.5,
            47.0,
        ),
    ]:
        ax.plot(
            [
                goal_x,
                far_x,
                far_x,
                goal_x,
            ],
            [
                -9.16,
                -9.16,
                9.16,
                9.16,
            ],
            linewidth=1.0,
        )

    ax.scatter(
        template[:, 0],
        template[:, 1],
        s=42,
        marker="o",
        zorder=5,
    )

    for kp_id, (
        x,
        y,
    ) in enumerate(
        template
    ):
        ax.annotate(
            f"K{kp_id:02d}",
            (
                x,
                y,
            ),
            xytext=(
                4,
                4,
            ),
            textcoords=(
                "offset points"
            ),
            fontsize=8,
        )

    ax.set_xlim(
        x_min - 3,
        x_max + 3,
    )

    ax.set_ylim(
        y_max + 3,
        y_min - 3,
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
        "32 punkty charakterystyczne boiska"
    )

    ax.grid(
        alpha=0.2
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
# DATA.YAML
# =============================================================================

DEFAULT_FLIP_IDX = [
    24,
    25,
    26,
    27,
    28,
    29,
    22,
    23,
    21,
    17,
    18,
    19,
    20,
    13,
    14,
    15,
    16,
    9,
    10,
    11,
    12,
    8,
    6,
    7,
    0,
    1,
    2,
    3,
    4,
    5,
    31,
    30,
]


def load_dataset_metadata() -> Dict:
    metadata = {
        "kpt_shape": [
            32,
            3,
        ],
        "flip_idx": (
            DEFAULT_FLIP_IDX
        ),
        "names": [
            "pitch",
        ],
        "nc": 1,
    }

    if (
        yaml is not None
        and FFD_DATA_YAML.exists()
    ):
        try:
            source = yaml.safe_load(
                FFD_DATA_YAML.read_text(
                    encoding="utf-8"
                )
            )

            for key in [
                "kpt_shape",
                "flip_idx",
                "names",
                "nc",
            ]:
                if key in source:
                    metadata[
                        key
                    ] = source[
                        key
                    ]

        except Exception as exc:
            print(
                f"WARNING: nie udało się odczytać "
                f"{FFD_DATA_YAML}: {exc}"
            )

    return metadata


def write_output_yaml(
    output_root: Path,
) -> None:
    metadata = (
        load_dataset_metadata()
    )

    config = {
        "path": str(
            output_root.resolve()
        ),
        "test": "test/images",
        "kpt_shape": (
            metadata[
                "kpt_shape"
            ]
        ),
        "flip_idx": (
            metadata[
                "flip_idx"
            ]
        ),
        "names": (
            metadata[
                "names"
            ]
        ),
        "nc": (
            metadata[
                "nc"
            ]
        ),
    }

    yaml_path = (
        output_root
        / "data.yaml"
    )

    if yaml is not None:
        yaml_path.write_text(
            yaml.safe_dump(
                config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    else:
        # Minimalny fallback bez PyYAML.
        lines = [
            f"path: {config['path']}",
            "test: test/images",
            "kpt_shape: [32, 3]",
            "flip_idx: ["
            + ", ".join(
                str(x)
                for x in config[
                    "flip_idx"
                ]
            )
            + "]",
            "names: [pitch]",
            "nc: 1",
        ]

        yaml_path.write_text(
            "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )


# =============================================================================
# BUDOWA ZBIORU
# =============================================================================

def build_soccer_net_part(
    output_images: Path,
    output_labels: Path,
    output_vis: Path,
    margin_px: float,
    save_visualizations: bool,
    max_visualizations: Optional[int],
    visualization_counter: List[int],
) -> List[dict]:

    ensure_exists(
        HOMO_MANIFEST,
        "manifest.csv homography evaluation",
    )

    ensure_exists(
        HOMO_IMAGES_DIR,
        "katalogu obrazów homography evaluation",
    )

    ensure_exists(
        HOMO_GT_DIR,
        "katalogu H_GT",
    )

    manifest = pd.read_csv(
        HOMO_MANIFEST
    )

    required = {
        "eval_id",
        "eval_stem",
        "eval_image",
    }

    missing = (
        required
        - set(
            manifest.columns
        )
    )

    if missing:
        raise ValueError(
            f"Brak kolumn w manifest.csv: {sorted(missing)}"
        )

    rows: List[dict] = []

    for row in (
        manifest
        .sort_values(
            "eval_id"
        )
        .itertuples(
            index=False
        )
    ):
        eval_id = int(
            row.eval_id
        )

        eval_stem = str(
            row.eval_stem
        )

        source_image = (
            HOMO_IMAGES_DIR
            / str(
                row.eval_image
            )
        )

        gt_path = (
            HOMO_GT_DIR
            / f"{eval_stem}_gt.json"
        )

        ensure_exists(
            source_image,
            f"obrazu {eval_stem}",
        )

        ensure_exists(
            gt_path,
            f"H_GT {eval_stem}",
        )

        image_bgr = cv2.imread(
            str(
                source_image
            )
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Nie można odczytać: {source_image}"
            )

        height, width = (
            image_bgr.shape[:2]
        )

        h_pitch2image = (
            load_h_gt(
                gt_path
            )
        )

        (
            keypoints_px,
            visible,
        ) = (
            projected_keypoints_from_h(
                h_pitch_to_image=h_pitch2image,
                image_width=width,
                image_height=height,
                margin_px=margin_px,
            )
        )

        bbox = (
            bbox_from_reference_geometry(
                h_pitch_to_image=h_pitch2image,
                keypoints_px=keypoints_px,
                visible_mask=visible,
                image_width=width,
                image_height=height,
            )
        )

        output_stem = (
            f"sn_{eval_stem}"
        )

        output_image = (
            output_images
            / (
                output_stem
                + source_image.suffix.lower()
            )
        )

        output_label = (
            output_labels
            / f"{output_stem}.txt"
        )

        shutil.copy2(
            source_image,
            output_image,
        )

        write_yolo_pose_label(
            output_path=output_label,
            bbox_yolo=bbox,
            keypoints_px=keypoints_px,
            visible_mask=visible,
            image_width=width,
            image_height=height,
        )

        do_vis = (
            save_visualizations
            and (
                max_visualizations is None
                or visualization_counter[
                    0
                ]
                < max_visualizations
            )
        )

        if do_vis:
            visualize_keypoints_on_image(
                image_path=output_image,
                keypoints_px=keypoints_px,
                visible_mask=visible,
                title=(
                    f"SoccerNet {eval_stem} | "
                    "GT z H_GT"
                ),
                output_path=(
                    output_vis
                    / (
                        output_stem
                        + "_gt_keypoints.png"
                    )
                ),
            )

            visualization_counter[
                0
            ] += 1

        rows.append(
            {
                "combined_id": (
                    output_stem
                ),
                "source": (
                    "soccerNet_hgt"
                ),
                "source_split": (
                    "homography_evaluation"
                ),
                "source_image": str(
                    source_image.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "source_label_or_gt": str(
                    gt_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "output_image": str(
                    output_image.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "output_label": str(
                    output_label.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "eval_id": eval_id,
                "width": width,
                "height": height,
                "n_visible_keypoints": int(
                    visible.sum()
                ),
                "gt_origin": (
                    "PITCH_KEYPOINTS_TEMPLATE_M "
                    "projected by H_GT pitch_to_image"
                ),
            }
        )

        print(
            f"[SoccerNet {eval_id:03d}] "
            f"{output_stem} | "
            f"visible={int(visible.sum())}/32"
        )

    return rows


def build_football_field_detection_part(
    output_images: Path,
    output_labels: Path,
    output_vis: Path,
    save_visualizations: bool,
    max_visualizations: Optional[int],
    visualization_counter: List[int],
) -> List[dict]:

    ensure_exists(
        FFD_TEST_IMAGES,
        "football-field-detection-16/test/images",
    )

    ensure_exists(
        FFD_TEST_LABELS,
        "football-field-detection-16/test/labels",
    )

    source_images = sorted(
        [
            p
            for p in FFD_TEST_IMAGES.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=lambda p: (
            p.name.lower()
        ),
    )

    rows: List[dict] = []

    for index, source_image in enumerate(
        source_images,
        start=1,
    ):
        source_label = (
            FFD_TEST_LABELS
            / f"{source_image.stem}.txt"
        )

        if not source_label.exists():
            print(
                f"WARNING: brak labela dla "
                f"{source_image.name}; pomijam."
            )
            continue

        image_bgr = cv2.imread(
            str(
                source_image
            )
        )

        if image_bgr is None:
            print(
                f"WARNING: nie można odczytać "
                f"{source_image}; pomijam."
            )
            continue

        height, width = (
            image_bgr.shape[:2]
        )

        (
            keypoints_px,
            visibility,
        ) = (
            parse_yolo_pose_label(
                label_path=source_label,
                image_width=width,
                image_height=height,
            )
        )

        visible = (
            visibility > 0
        )

        output_stem = (
            f"ffd_test_{source_image.stem}"
        )

        output_image = (
            output_images
            / (
                output_stem
                + source_image.suffix.lower()
            )
        )

        output_label = (
            output_labels
            / f"{output_stem}.txt"
        )

        shutil.copy2(
            source_image,
            output_image,
        )

        # Kopiujemy ręczny label bez zmian.
        shutil.copy2(
            source_label,
            output_label,
        )

        do_vis = (
            save_visualizations
            and (
                max_visualizations is None
                or visualization_counter[
                    0
                ]
                < max_visualizations
            )
        )

        if do_vis:
            visualize_keypoints_on_image(
                image_path=output_image,
                keypoints_px=keypoints_px,
                visible_mask=visible,
                title=(
                    "football-field-detection-16/test | "
                    "ręczne GT"
                ),
                output_path=(
                    output_vis
                    / (
                        output_stem
                        + "_gt_keypoints.png"
                    )
                ),
            )

            visualization_counter[
                0
            ] += 1

        rows.append(
            {
                "combined_id": (
                    output_stem
                ),
                "source": (
                    "football_field_detection_16"
                ),
                "source_split": (
                    "test"
                ),
                "source_image": str(
                    source_image.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "source_label_or_gt": str(
                    source_label.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "output_image": str(
                    output_image.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "output_label": str(
                    output_label.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "eval_id": np.nan,
                "width": width,
                "height": height,
                "n_visible_keypoints": int(
                    visible.sum()
                ),
                "gt_origin": (
                    "manual YOLO Pose label"
                ),
            }
        )

        print(
            f"[FFD {index:03d}] "
            f"{output_stem} | "
            f"visible={int(visible.sum())}/32"
        )

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.keypoint_margin_px < 0:
        raise ValueError(
            "--keypoint-margin-px nie może być ujemny."
        )

    if (
        args.max_visualizations is not None
        and args.max_visualizations < 1
    ):
        raise ValueError(
            "--max-visualizations musi być >= 1."
        )

    output_root = (
        args.output_root
    )

    output_images = (
        output_root
        / "test"
        / "images"
    )

    output_labels = (
        output_root
        / "test"
        / "labels"
    )

    output_vis = (
        output_root
        / "visualizations"
    )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Katalog wynikowy już istnieje:\n"
                f"{output_root}\n"
                "Użyj --overwrite, jeśli chcesz go zbudować od nowa."
            )

        shutil.rmtree(
            output_root
        )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.no_visualizations:
        output_vis.mkdir(
            parents=True,
            exist_ok=True,
        )

        visualize_pitch_template(
            output_vis
            / "pitch_template_32_keypoints.png"
        )

    print("=" * 80)
    print("BUDOWA POŁĄCZONEGO KEYPOINT TEST SET")
    print("=" * 80)
    print(
        f"Output: {output_root}"
    )
    print(
        f"SoccerNet H_GT: {HOMO_GT_DIR}"
    )
    print(
        f"FFD test: {FFD_TEST_IMAGES}"
    )
    print(
        f"Keypoint image margin: "
        f"{args.keypoint_margin_px}px"
    )
    print()

    visualization_counter = [
        0
    ]

    rows: List[dict] = []

    rows.extend(
        build_soccer_net_part(
            output_images=output_images,
            output_labels=output_labels,
            output_vis=output_vis,
            margin_px=args.keypoint_margin_px,
            save_visualizations=(
                not args.no_visualizations
            ),
            max_visualizations=(
                args.max_visualizations
            ),
            visualization_counter=(
                visualization_counter
            ),
        )
    )

    rows.extend(
        build_football_field_detection_part(
            output_images=output_images,
            output_labels=output_labels,
            output_vis=output_vis,
            save_visualizations=(
                not args.no_visualizations
            ),
            max_visualizations=(
                args.max_visualizations
            ),
            visualization_counter=(
                visualization_counter
            ),
        )
    )

    manifest_df = pd.DataFrame(
        rows
    )

    manifest_path = (
        output_root
        / "manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    write_output_yaml(
        output_root
    )

    source_counts = (
        manifest_df[
            "source"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "total_images": int(
            len(
                manifest_df
            )
        ),
        "source_counts": {
            str(k): int(v)
            for k, v in source_counts.items()
        },
        "soccerNet_images": int(
            (
                manifest_df[
                    "source"
                ]
                == "soccerNet_hgt"
            ).sum()
        ),
        "football_field_detection_test_images": int(
            (
                manifest_df[
                    "source"
                ]
                == "football_field_detection_16"
            ).sum()
        ),
        "soccerNet_total_visible_keypoints": int(
            manifest_df.loc[
                manifest_df[
                    "source"
                ]
                == "soccerNet_hgt",
                "n_visible_keypoints",
            ].sum()
        ),
        "football_field_detection_total_visible_keypoints": int(
            manifest_df.loc[
                manifest_df[
                    "source"
                ]
                == "football_field_detection_16",
                "n_visible_keypoints",
            ].sum()
        ),
        "keypoint_visibility_rule_soccerNet": (
            "Projected PITCH_KEYPOINTS_TEMPLATE_M point is finite and "
            f"lies at least {args.keypoint_margin_px}px inside the original image."
        ),
        "soccerNet_gt_type": (
            "reference keypoints derived geometrically from line-based H_GT"
        ),
        "football_field_detection_gt_type": (
            "manual YOLO Pose keypoint labels"
        ),
    }

    with (
        output_root
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("GOTOWE")
    print("=" * 80)
    print(
        f"Łącznie obrazów: "
        f"{summary['total_images']}"
    )
    print(
        f"SoccerNet/H_GT: "
        f"{summary['soccerNet_images']}"
    )
    print(
        f"FFD16/test: "
        f"{summary['football_field_detection_test_images']}"
    )
    print(
        f"Manifest: {manifest_path}"
    )
    print(
        f"Images:   {output_images}"
    )
    print(
        f"Labels:   {output_labels}"
    )

    if not args.no_visualizations:
        print(
            f"Visuals:  {output_vis}"
        )
        print(
            f"Liczba wizualizacji obrazów: "
            f"{visualization_counter[0]}"
        )

    print(
        f"YAML:     {output_root / 'data.yaml'}"
    )
    print(
        f"Summary:  {output_root / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
