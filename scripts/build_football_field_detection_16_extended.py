#!/usr/bin/env python3
"""
Buduje:
    data/football-field-detection-16_extended

na bazie:
    data/football-field-detection-16
oraz pierwszych 500 poprawnie przygotowanych klatek z:
    data/calibration_dataset/calibration/train

Założenia:
- kopiujemy istniejące train/ i valid/ bez modyfikacji,
- zachowujemy proporcję train:valid istniejącego zbioru,
- dla SoccerNet-Calibration wyznaczamy H_GT z ręcznych adnotacji linii,
- 32 punkty PITCH_KEYPOINTS_TEMPLATE_M rzutujemy H_GT na klatkę,
- punkty wypadające poza obrazem zapisujemy jako visibility=0,
- punkty znajdujące się w obrazie zapisujemy jako visibility=2,
- nowe obrazy są FIZYCZNIE stretchowane do 640x640, zgodnie z
  preprocessingiem oryginalnego football-field-detection v16,
- współrzędne YOLO są znormalizowane, więc po stretchu pozostają poprawne,
- tworzymy nowe data.yaml wskazujące na train/images i valid/images,
- zapisujemy manifest oraz preview nowych automatycznych etykiet.

Skrypt NIE korzysta z 100 klatek ewaluacyjnych z
data/data_homography_evaluation. Dzięki temu pozostają one czystym test setem.

Umieść jako:
    scripts/build_football_field_detection_16_extended.py

Uruchom:
    python scripts/build_football_field_detection_16_extended.py

Ponowne zbudowanie:
    python scripts/build_football_field_detection_16_extended.py --overwrite

Np. 30 preview:
    python scripts/build_football_field_detection_16_extended.py --preview 30 --overwrite

Domyślnie skrypt zbiera pierwsze 500 POPRAWNIE przetworzonych przykładów.
Jeżeli któraś z pierwszych klatek nie pozwala wyznaczyć H_GT lub nie ma
żadnego z 32 punktów w obrazie, jest pomijana i skrypt idzie dalej,
aż uzyska 500 przykładów albo wyczerpie dane.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

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

from generate_homography_gt import (
    collect_line_data,
    estimate_homography_from_lines,
    line_residuals_px,
    refine_homography,
)


# =============================================================================
# ŚCIEŻKI
# =============================================================================

SOURCE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "football-field-detection-16"
)

SOURCE_TRAIN_IMAGES = (
    SOURCE_DATASET
    / "train"
    / "images"
)

SOURCE_TRAIN_LABELS = (
    SOURCE_DATASET
    / "train"
    / "labels"
)

SOURCE_VALID_IMAGES = (
    SOURCE_DATASET
    / "valid"
    / "images"
)

SOURCE_VALID_LABELS = (
    SOURCE_DATASET
    / "valid"
    / "labels"
)

SOURCE_YAML = (
    SOURCE_DATASET
    / "data.yaml"
)

SOCCERNET_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "calibration_dataset"
    / "calibration"
    / "train"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "football-field-detection-16_extended"
)


# =============================================================================
# KONFIGURACJA
# =============================================================================

NUM_NEW_IMAGES = 500
OUTPUT_SIZE = 640

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

NUM_KEYPOINTS = 32

# Nie odcinamy keypointów przy samej krawędzi z dużym marginesem.
# 1 px chroni jedynie przed błędami numerycznymi.
KEYPOINT_IMAGE_MARGIN_PX = 1.0

# Do treningu dopuszczamy także wąskie kadry z małą liczbą punktów.
# To celowe: właśnie takich ujęć brakowało w domenie treningowej.
MIN_VISIBLE_KEYPOINTS = 1

# BBox klasy "pitch" budujemy z widocznej części rzutowanej powierzchni boiska.
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
            "Rozszerzenie football-field-detection-16 o automatycznie "
            "etykietowane klatki SoccerNet-Calibration/train."
        )
    )

    parser.add_argument(
        "--num-new",
        type=int,
        default=NUM_NEW_IMAGES,
        help=f"Liczba nowych przykładów. Domyślnie {NUM_NEW_IMAGES}.",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help=(
            "Liczba wizualizacji nowych pseudo-labeli. "
            "Domyślnie 20; 0 wyłącza."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DATASET,
        help=f"Katalog wynikowy. Domyślnie {OUTPUT_DATASET}",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Usuń istniejący katalog wynikowy i zbuduj go od nowa.",
    )

    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Nie wykonuj nieliniowego refinementu H_GT przez SciPy.",
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
            f"H ma shape {h.shape}, oczekiwano (3,3)."
        )

    if not np.isfinite(h).all():
        raise ValueError(
            "H zawiera NaN/Inf."
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

    if abs(np.linalg.det(h)) < 1e-12:
        raise ValueError(
            "Homografia jest osobliwa."
        )

    return h


def transform_points(
    h: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    points_xy = np.asarray(
        points_xy,
        dtype=np.float64,
    ).reshape(-1, 2)

    hp = np.column_stack(
        [
            points_xy,
            np.ones(
                len(points_xy),
                dtype=np.float64,
            ),
        ]
    )

    out = (
        h
        @ hp.T
    ).T

    valid = (
        np.abs(
            out[:, 2]
        )
        > 1e-12
    )

    result = np.full(
        (len(points_xy), 2),
        np.nan,
        dtype=np.float64,
    )

    result[valid] = (
        out[valid, :2]
        / out[valid, 2:3]
    )

    return result


def count_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> int:
    count = 0

    for image_path in images_dir.iterdir():
        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            if label_path.exists():
                count += 1

    return count


def copy_original_dataset(
    output_root: Path,
) -> Tuple[int, int]:
    """
    Kopiuje train, valid i data.yaml.
    """
    ensure_exists(
        SOURCE_TRAIN_IMAGES,
        "oryginalnego train/images",
    )
    ensure_exists(
        SOURCE_TRAIN_LABELS,
        "oryginalnego train/labels",
    )
    ensure_exists(
        SOURCE_VALID_IMAGES,
        "oryginalnego valid/images",
    )
    ensure_exists(
        SOURCE_VALID_LABELS,
        "oryginalnego valid/labels",
    )
    ensure_exists(
        SOURCE_YAML,
        "oryginalnego data.yaml",
    )

    shutil.copytree(
        SOURCE_DATASET / "train",
        output_root / "train",
    )

    shutil.copytree(
        SOURCE_DATASET / "valid",
        output_root / "valid",
    )

    # Zachowujemy oryginalny YAML do audytu.
    shutil.copy2(
        SOURCE_YAML,
        output_root / "data_source.yaml",
    )

    n_train = (
        count_image_label_pairs(
            output_root / "train" / "images",
            output_root / "train" / "labels",
        )
    )

    n_valid = (
        count_image_label_pairs(
            output_root / "valid" / "images",
            output_root / "valid" / "labels",
        )
    )

    return (
        n_train,
        n_valid,
    )


# =============================================================================
# ODKRYWANIE PAR SOCCERNET: IMAGE + JSON
# =============================================================================

def discover_soccer_net_pairs(
    root: Path,
) -> List[Tuple[Path, Path]]:
    """
    Rekurencyjnie wyszukuje obrazy i próbuje znaleźć odpowiadający JSON.

    Priorytet:
      1) JSON o tym samym stemie w tym samym katalogu,
      2) JSON o tym samym stemie w katalogu nadrzędnym / annotation(s),
      3) jednoznaczny JSON o tym samym stemie w całym train.

    Dzięki temu skrypt jest odporny na kilka popularnych wariantów
    organizacji SoccerNet-Calibration.
    """
    ensure_exists(
        root,
        "SoccerNet calibration/train",
    )

    images = sorted(
        [
            p
            for p in root.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=lambda p: str(
            p.relative_to(root)
        ).lower(),
    )

    json_files = sorted(
        root.rglob("*.json"),
        key=lambda p: str(
            p.relative_to(root)
        ).lower(),
    )

    json_by_stem: Dict[
        str,
        List[Path],
    ] = {}

    for json_path in json_files:
        json_by_stem.setdefault(
            json_path.stem,
            [],
        ).append(
            json_path
        )

    pairs: List[
        Tuple[Path, Path]
    ] = []

    for image_path in images:
        candidates: List[Path] = []

        same_dir = (
            image_path.with_suffix(
                ".json"
            )
        )

        if same_dir.exists():
            candidates.append(
                same_dir
            )

        for dirname in [
            "annotation",
            "annotations",
            "label",
            "labels",
        ]:
            candidate = (
                image_path.parent
                / dirname
                / f"{image_path.stem}.json"
            )

            if candidate.exists():
                candidates.append(
                    candidate
                )

            candidate_parent = (
                image_path.parent.parent
                / dirname
                / f"{image_path.stem}.json"
            )

            if candidate_parent.exists():
                candidates.append(
                    candidate_parent
                )

        global_candidates = (
            json_by_stem.get(
                image_path.stem,
                [],
            )
        )

        if len(
            global_candidates
        ) == 1:
            candidates.append(
                global_candidates[0]
            )

        # Unikalizacja przy zachowaniu kolejności.
        unique_candidates: List[Path] = []

        seen = set()

        for candidate in candidates:
            key = str(
                candidate.resolve()
            )

            if key not in seen:
                seen.add(
                    key
                )
                unique_candidates.append(
                    candidate
                )

        if not unique_candidates:
            continue

        # Preferujemy pierwszy najbardziej lokalny.
        pairs.append(
            (
                image_path,
                unique_candidates[0],
            )
        )

    return pairs


# =============================================================================
# H_GT I KEYPOINTY
# =============================================================================

def estimate_h_gt(
    image_bgr: np.ndarray,
    annotation_path: Path,
    use_refine: bool,
) -> Tuple[
    np.ndarray,
    int,
    bool,
    float,
    float,
    float,
]:
    """
    Wyznacza H_pitch_to_image dokładnie tym samym mechanizmem co
    generate_homography_gt.py.
    """
    h_img, w_img = (
        image_bgr.shape[:2]
    )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        annotations = json.load(
            f
        )

    (
        correspondences,
        line_data,
        world_norm_pts,
        image_norm_pts,
    ) = collect_line_data(
        annotations,
        w_img,
        h_img,
        PITCH_LENGTH_M,
        PITCH_WIDTH_M,
    )

    if len(
        correspondences
    ) < 4:
        raise ValueError(
            f"Za mało użytecznych klas linii: "
            f"{len(correspondences)}"
        )

    h_pitch_to_image = (
        estimate_homography_from_lines(
            correspondences,
            world_norm_pts,
            image_norm_pts,
        )
    )

    refined = False

    if use_refine:
        (
            h_pitch_to_image,
            refined,
        ) = refine_homography(
            h_pitch_to_image,
            line_data,
        )

    h_pitch_to_image = normalize_h(
        h_pitch_to_image
    )

    residuals = (
        line_residuals_px(
            h_pitch_to_image,
            line_data,
        )
    )

    if (
        residuals.size == 0
        or not np.isfinite(
            residuals
        ).all()
    ):
        raise ValueError(
            "Nieprawidłowe residuale H_GT."
        )

    rms_px = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    mean_abs_px = float(
        np.mean(
            np.abs(
                residuals
            )
        )
    )

    max_abs_px = float(
        np.max(
            np.abs(
                residuals
            )
        )
    )

    return (
        h_pitch_to_image,
        len(line_data),
        bool(refined),
        rms_px,
        mean_abs_px,
        max_abs_px,
    )


def project_template(
    h_pitch_to_image: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
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

    m = (
        KEYPOINT_IMAGE_MARGIN_PX
    )

    visible = (
        np.isfinite(
            xy_px
        ).all(axis=1)
        & (
            xy_px[:, 0]
            >= m
        )
        & (
            xy_px[:, 0]
            < image_width - m
        )
        & (
            xy_px[:, 1]
            >= m
        )
        & (
            xy_px[:, 1]
            < image_height - m
        )
    )

    return (
        xy_px,
        visible,
    )


def sample_projected_pitch_surface(
    h_pitch_to_image: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Gęsta siatka całej płaszczyzny boiska, wyłącznie do wyznaczenia bbox.
    """
    xs = np.linspace(
        -PITCH_LENGTH_M / 2.0,
        PITCH_LENGTH_M / 2.0,
        81,
    )

    ys = np.linspace(
        -PITCH_WIDTH_M / 2.0,
        PITCH_WIDTH_M / 2.0,
        55,
    )

    xx, yy = np.meshgrid(
        xs,
        ys,
    )

    pitch_xy = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
        ]
    )

    uv = transform_points(
        h_pitch_to_image,
        pitch_xy,
    )

    inside = (
        np.isfinite(
            uv
        ).all(axis=1)
        & (
            uv[:, 0] >= 0.0
        )
        & (
            uv[:, 0] < image_width
        )
        & (
            uv[:, 1] >= 0.0
        )
        & (
            uv[:, 1] < image_height
        )
    )

    return uv[
        inside
    ]


def build_pitch_bbox(
    h_pitch_to_image: np.ndarray,
    image_width: int,
    image_height: int,
    keypoints_px: np.ndarray,
    visible_mask: np.ndarray,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Zwraca YOLO bbox:
        cx_norm, cy_norm, width_norm, height_norm

    Box obejmuje widoczną część powierzchni boiska, nie całe zdjęcie.
    """
    points = (
        sample_projected_pitch_surface(
            h_pitch_to_image,
            image_width,
            image_height,
        )
    )

    if len(points) < 4:
        points = (
            keypoints_px[
                visible_mask
            ]
        )

    if len(points) == 0:
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
                points[:, 0]
            )
        )
        - BBOX_PADDING_PX,
    )

    y1 = max(
        0.0,
        float(
            np.min(
                points[:, 1]
            )
        )
        - BBOX_PADDING_PX,
    )

    x2 = min(
        float(
            image_width
        ),
        float(
            np.max(
                points[:, 0]
            )
        )
        + BBOX_PADDING_PX,
    )

    y2 = min(
        float(
            image_height
        ),
        float(
            np.max(
                points[:, 1]
            )
        )
        + BBOX_PADDING_PX,
    )

    bw = max(
        x2 - x1,
        1.0,
    )

    bh = max(
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
        cx
        / image_width,
        cy
        / image_height,
        bw
        / image_width,
        bh
        / image_height,
    )


def write_pose_label(
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
    cx, cy, bw, bh = (
        bbox_yolo
    )

    values = [
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
        " ".join(
            values
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# PREVIEW
# =============================================================================

def save_preview(
    resized_bgr: np.ndarray,
    keypoints_px_original: np.ndarray,
    visible_mask: np.ndarray,
    original_width: int,
    original_height: int,
    bbox_yolo: Tuple[
        float,
        float,
        float,
        float,
    ],
    split: str,
    source_name: str,
    n_lines: int,
    rms_px: float,
    output_path: Path,
) -> None:
    """
    Wizualizacja na faktycznym obrazie treningowym 640x640.
    """
    image_rgb = cv2.cvtColor(
        resized_bgr,
        cv2.COLOR_BGR2RGB,
    )

    scale_x = (
        OUTPUT_SIZE
        / float(
            original_width
        )
    )

    scale_y = (
        OUTPUT_SIZE
        / float(
            original_height
        )
    )

    kp640 = (
        keypoints_px_original.copy()
    )

    kp640[:, 0] *= (
        scale_x
    )

    kp640[:, 1] *= (
        scale_y
    )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(
            8,
            8,
        ),
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
        ax.scatter(
            kp640[
                ids,
                0,
            ],
            kp640[
                ids,
                1,
            ],
            s=36,
            marker="o",
            zorder=5,
            label="pseudo-GT keypoint",
        )

        for kp_id in ids:
            x = kp640[
                kp_id,
                0,
            ]

            y = kp640[
                kp_id,
                1,
            ]

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
                    alpha=0.60,
                ),
            )

    cx, cy, bw, bh = (
        bbox_yolo
    )

    x1 = (
        cx - bw / 2.0
    ) * OUTPUT_SIZE

    y1 = (
        cy - bh / 2.0
    ) * OUTPUT_SIZE

    box_w = (
        bw
        * OUTPUT_SIZE
    )

    box_h = (
        bh
        * OUTPUT_SIZE
    )

    rect = plt.Rectangle(
        (
            x1,
            y1,
        ),
        box_w,
        box_h,
        fill=False,
        linewidth=1.2,
        label="pitch bbox",
    )

    ax.add_patch(
        rect
    )

    ax.set_title(
        f"{split} | {source_name}\n"
        f"visible={len(ids)}/32 | "
        f"used lines={n_lines} | "
        f"H_GT line RMS={rms_px:.2f}px"
    )

    ax.legend(
        loc="lower left",
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=170,
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


def write_extended_yaml(
    output_root: Path,
) -> None:
    """
    Zachowuje konfigurację keypointów źródłowego data.yaml,
    ale poprawia ścieżki dla nowego datasetu.
    """
    config = {
        "path": str(
            output_root.resolve()
        ),
        "train": "train/images",
        "val": "valid/images",
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
        and SOURCE_YAML.exists()
    ):
        try:
            source_config = (
                yaml.safe_load(
                    SOURCE_YAML.read_text(
                        encoding="utf-8"
                    )
                )
                or {}
            )

            for key in [
                "kpt_shape",
                "flip_idx",
                "names",
                "nc",
                "roboflow",
            ]:
                if key in (
                    source_config
                ):
                    config[
                        key
                    ] = (
                        source_config[
                            key
                        ]
                    )

        except Exception as exc:
            print(
                f"WARNING: nie udało się sparsować data.yaml: {exc}"
            )

    output_yaml = (
        output_root
        / "data.yaml"
    )

    if yaml is not None:
        output_yaml.write_text(
            yaml.safe_dump(
                config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    else:
        # Minimalny YAML bez zależności PyYAML.
        lines = [
            f"path: {output_root.resolve()}",
            "train: train/images",
            "val: valid/images",
            "kpt_shape: [32, 3]",
            "flip_idx: ["
            + ", ".join(
                str(x)
                for x in DEFAULT_FLIP_IDX
            )
            + "]",
            "names: [pitch]",
            "nc: 1",
        ]

        output_yaml.write_text(
            "\n".join(
                lines
            )
            + "\n",
            encoding="utf-8",
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.num_new < 1:
        raise ValueError(
            "--num-new musi być >= 1."
        )

    if args.preview < 0:
        raise ValueError(
            "--preview nie może być ujemne."
        )

    ensure_exists(
        SOURCE_DATASET,
        "football-field-detection-16",
    )

    ensure_exists(
        SOCCERNET_TRAIN,
        "SoccerNet calibration/train",
    )

    output_root = (
        args.output.resolve()
    )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Katalog już istnieje:\n"
                f"{output_root}\n"
                f"Użyj --overwrite."
            )

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 80)
    print("FOOTBALL-FIELD-DETECTION-16 EXTENDED")
    print("=" * 80)

    print(
        "\nKopiowanie oryginalnego train/ i valid/..."
    )

    (
        original_train,
        original_valid,
    ) = copy_original_dataset(
        output_root
    )

    original_total = (
        original_train
        + original_valid
    )

    if original_total <= 0:
        raise RuntimeError(
            "Oryginalny dataset nie zawiera żadnych par image+label."
        )

    train_ratio = (
        original_train
        / original_total
    )

    new_train_target = int(
        round(
            args.num_new
            * train_ratio
        )
    )

    new_train_target = min(
        max(
            new_train_target,
            1,
        ),
        args.num_new,
    )

    new_valid_target = (
        args.num_new
        - new_train_target
    )

    print(
        f"Oryginalny train: {original_train}"
    )

    print(
        f"Oryginalny valid: {original_valid}"
    )

    print(
        f"Proporcja train:valid = "
        f"{100 * train_ratio:.2f}% : "
        f"{100 * (1 - train_ratio):.2f}%"
    )

    print(
        f"Nowe SoccerNet -> train: "
        f"{new_train_target}"
    )

    print(
        f"Nowe SoccerNet -> valid: "
        f"{new_valid_target}"
    )

    pairs = (
        discover_soccer_net_pairs(
            SOCCERNET_TRAIN
        )
    )

    print(
        f"\nZnaleziono kandydatów image+JSON: "
        f"{len(pairs)}"
    )

    if len(pairs) < args.num_new:
        print(
            "WARNING: liczba par jest mniejsza "
            "niż żądana liczba nowych przykładów."
        )

    preview_dir = (
        output_root
        / "pseudo_gt_preview"
    )

    if args.preview > 0:
        preview_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_rows: List[dict] = []
    rejected_rows: List[dict] = []

    accepted = 0
    added_train = 0
    added_valid = 0
    preview_saved = 0

    use_refine = (
        not args.no_refine
    )

    for candidate_index, (
        image_path,
        annotation_path,
    ) in enumerate(
        pairs,
        start=1,
    ):
        if accepted >= (
            args.num_new
        ):
            break

        try:
            image_bgr = cv2.imread(
                str(
                    image_path
                )
            )

            if image_bgr is None:
                raise RuntimeError(
                    "OpenCV nie może odczytać obrazu."
                )

            orig_h, orig_w = (
                image_bgr.shape[:2]
            )

            (
                h_pitch_to_image,
                n_lines,
                refined,
                rms_px,
                mean_abs_px,
                max_abs_px,
            ) = estimate_h_gt(
                image_bgr=image_bgr,
                annotation_path=annotation_path,
                use_refine=use_refine,
            )

            (
                keypoints_px,
                visible_mask,
            ) = project_template(
                h_pitch_to_image=(
                    h_pitch_to_image
                ),
                image_width=orig_w,
                image_height=orig_h,
            )

            n_visible = int(
                visible_mask.sum()
            )

            if (
                n_visible
                < MIN_VISIBLE_KEYPOINTS
            ):
                raise ValueError(
                    f"Brak widocznych template keypointów "
                    f"(visible={n_visible})."
                )

            bbox_yolo = (
                build_pitch_bbox(
                    h_pitch_to_image=(
                        h_pitch_to_image
                    ),
                    image_width=orig_w,
                    image_height=orig_h,
                    keypoints_px=(
                        keypoints_px
                    ),
                    visible_mask=(
                        visible_mask
                    ),
                )
            )

            # Proporcjonalny przydział:
            # najpierw wypełniamy dokładnie target train,
            # potem target valid.
            if (
                added_train
                < new_train_target
            ):
                split = (
                    "train"
                )
                added_train += 1
            else:
                split = (
                    "valid"
                )
                added_valid += 1

            accepted += 1

            output_stem = (
                f"sncal_auto_{accepted:04d}"
            )

            output_image = (
                output_root
                / split
                / "images"
                / f"{output_stem}.jpg"
            )

            output_label = (
                output_root
                / split
                / "labels"
                / f"{output_stem}.txt"
            )

            # Fizyczny STRETCH 640x640.
            resized_bgr = cv2.resize(
                image_bgr,
                (
                    OUTPUT_SIZE,
                    OUTPUT_SIZE,
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                ),
            )

            ok = cv2.imwrite(
                str(
                    output_image
                ),
                resized_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95,
                ],
            )

            if not ok:
                raise RuntimeError(
                    f"Nie udało się zapisać: {output_image}"
                )

            # Label liczymy w oryginalnym układzie,
            # ale zapisujemy normalized coordinates.
            # Po stretchu x_norm/y_norm są identyczne.
            write_pose_label(
                output_path=(
                    output_label
                ),
                bbox_yolo=(
                    bbox_yolo
                ),
                keypoints_px=(
                    keypoints_px
                ),
                visible_mask=(
                    visible_mask
                ),
                image_width=(
                    orig_w
                ),
                image_height=(
                    orig_h
                ),
            )

            if (
                preview_saved
                < args.preview
            ):
                save_preview(
                    resized_bgr=(
                        resized_bgr
                    ),
                    keypoints_px_original=(
                        keypoints_px
                    ),
                    visible_mask=(
                        visible_mask
                    ),
                    original_width=(
                        orig_w
                    ),
                    original_height=(
                        orig_h
                    ),
                    bbox_yolo=(
                        bbox_yolo
                    ),
                    split=(
                        split
                    ),
                    source_name=(
                        image_path.name
                    ),
                    n_lines=(
                        n_lines
                    ),
                    rms_px=(
                        rms_px
                    ),
                    output_path=(
                        preview_dir
                        / (
                            f"{output_stem}"
                            "_pseudo_gt.png"
                        )
                    ),
                )

                preview_saved += 1

            manifest_rows.append(
                {
                    "new_id": (
                        output_stem
                    ),
                    "split": (
                        split
                    ),
                    "source_image": str(
                        image_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "source_annotation": str(
                        annotation_path.relative_to(
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
                    "original_width": (
                        orig_w
                    ),
                    "original_height": (
                        orig_h
                    ),
                    "stored_width": (
                        OUTPUT_SIZE
                    ),
                    "stored_height": (
                        OUTPUT_SIZE
                    ),
                    "resize_mode": (
                        "stretch"
                    ),
                    "n_used_line_classes": (
                        n_lines
                    ),
                    "h_gt_refined": (
                        refined
                    ),
                    "h_gt_rms_line_residual_px": (
                        rms_px
                    ),
                    "h_gt_mean_abs_line_residual_px": (
                        mean_abs_px
                    ),
                    "h_gt_max_abs_line_residual_px": (
                        max_abs_px
                    ),
                    "n_visible_keypoints": (
                        n_visible
                    ),
                    "bbox_cx_norm": (
                        bbox_yolo[0]
                    ),
                    "bbox_cy_norm": (
                        bbox_yolo[1]
                    ),
                    "bbox_w_norm": (
                        bbox_yolo[2]
                    ),
                    "bbox_h_norm": (
                        bbox_yolo[3]
                    ),
                    "gt_type": (
                        "geometry-derived pseudo-GT "
                        "from SoccerNet line annotations via H_GT"
                    ),
                }
            )

            print(
                f"[OK {accepted:03d}/{args.num_new}] "
                f"{split:5s} | "
                f"{image_path.name} | "
                f"lines={n_lines:2d} | "
                f"kp={n_visible:2d}/32 | "
                f"RMS={rms_px:.2f}px"
            )

        except Exception as exc:
            rejected_rows.append(
                {
                    "candidate_index": (
                        candidate_index
                    ),
                    "source_image": str(
                        image_path
                    ),
                    "source_annotation": str(
                        annotation_path
                    ),
                    "reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

            print(
                f"[SKIP] "
                f"{image_path.name} | "
                f"{type(exc).__name__}: {exc}"
            )

    if accepted < (
        args.num_new
    ):
        print(
            "\nWARNING: nie udało się uzyskać "
            f"{args.num_new} poprawnych przykładów. "
            f"Utworzono {accepted}."
        )

    # Jeżeli zabrakło przykładów, split może nie osiągnąć targetu valid.
    # To jest jawnie zapisane w summary.

    write_extended_yaml(
        output_root
    )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    rejected_df = pd.DataFrame(
        rejected_rows
    )

    manifest_df.to_csv(
        output_root
        / "soccerNet_extension_manifest.csv",
        index=False,
    )

    rejected_df.to_csv(
        output_root
        / "soccerNet_extension_rejected.csv",
        index=False,
    )

    final_train = (
        count_image_label_pairs(
            output_root
            / "train"
            / "images",
            output_root
            / "train"
            / "labels",
        )
    )

    final_valid = (
        count_image_label_pairs(
            output_root
            / "valid"
            / "images",
            output_root
            / "valid"
            / "labels",
        )
    )

    summary = {
        "source_dataset": str(
            SOURCE_DATASET
        ),
        "soccerNet_source": str(
            SOCCERNET_TRAIN
        ),
        "output_dataset": str(
            output_root
        ),
        "original_train": (
            original_train
        ),
        "original_valid": (
            original_valid
        ),
        "original_train_ratio": (
            train_ratio
        ),
        "requested_new_images": (
            args.num_new
        ),
        "target_new_train": (
            new_train_target
        ),
        "target_new_valid": (
            new_valid_target
        ),
        "actual_new_total": (
            accepted
        ),
        "actual_new_train": (
            added_train
        ),
        "actual_new_valid": (
            added_valid
        ),
        "final_train": (
            final_train
        ),
        "final_valid": (
            final_valid
        ),
        "output_image_size": [
            OUTPUT_SIZE,
            OUTPUT_SIZE,
        ],
        "resize_mode": (
            "640x640 stretch"
        ),
        "min_visible_keypoints": (
            MIN_VISIBLE_KEYPOINTS
        ),
        "keypoint_visibility_rule": (
            "Template keypoint projected by H_GT must lie inside "
            "the original image; in-frame -> visibility 2, "
            "out-of-frame -> 0 0 0."
        ),
        "h_gt_method": (
            "same line-based dual-DLT + optional nonlinear refinement "
            "as generate_homography_gt.py"
        ),
        "refinement_enabled": (
            use_refine
        ),
        "rejected_candidates": (
            len(
                rejected_rows
            )
        ),
        "important_note": (
            "The 100 frozen homography evaluation frames are not used. "
            "Only calibration/train is used for this extension."
        ),
    }

    if not manifest_df.empty:
        summary.update(
            {
                "new_visible_keypoints_mean": float(
                    manifest_df[
                        "n_visible_keypoints"
                    ].mean()
                ),
                "new_visible_keypoints_median": float(
                    manifest_df[
                        "n_visible_keypoints"
                    ].median()
                ),
                "h_gt_line_rms_mean_px": float(
                    manifest_df[
                        "h_gt_rms_line_residual_px"
                    ].mean()
                ),
                "h_gt_line_rms_median_px": float(
                    manifest_df[
                        "h_gt_rms_line_residual_px"
                    ].median()
                ),
            }
        )

    with (
        output_root
        / "extension_summary.json"
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
        f"Output: {output_root}"
    )
    print(
        f"Nowe train: {added_train}"
    )
    print(
        f"Nowe valid: {added_valid}"
    )
    print(
        f"Final train: {final_train}"
    )
    print(
        f"Final valid: {final_valid}"
    )
    print(
        f"Odrzucone kandydaty: "
        f"{len(rejected_rows)}"
    )
    print(
        f"Data YAML: "
        f"{output_root / 'data.yaml'}"
    )
    print(
        f"Manifest: "
        f"{output_root / 'soccerNet_extension_manifest.csv'}"
    )

    if args.preview > 0:
        print(
            f"Preview: "
            f"{preview_dir}"
        )


if __name__ == "__main__":
    main()
