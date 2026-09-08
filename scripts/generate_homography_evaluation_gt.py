#!/usr/bin/env python3
"""
Batchowe wyznaczanie referencyjnych macierzy homografii H_GT
dla zamrożonej puli ewaluacyjnej SoccerNet-Calibration.

Umieść plik jako:
    CvFootballTracker/scripts/generate_homography_evaluation_gt.py

WYMAGANE:
    W tym samym katalogu scripts/ powinien znajdować się wcześniej przygotowany:
        generate_homography_gt.py

Wejście:
    data/data_homography_evaluation/
        images/
        annotations/
        manifest.csv

Wyjście:
    data/data_homography_evaluation/
        ground_truth/
            eval_homo_001_gt.json
            ...
        visualizations_gt/
            eval_homo_001_gt.png
            ...
        homography_ground_truth.csv
        homography_gt_summary.json
        homography_gt_failures.csv   # tylko jeśli wystąpią błędy

Każdy *_gt.json zawiera:
    - H_pitch_to_image
    - H_image_to_pitch
    - wykorzystane klasy linii
    - diagnostykę dopasowania linii
    - deterministyczne punkty demonstracyjne image -> pitch

Wizualizacja jest zgodna z wcześniejszym generate_homography_gt.py:
    lewa strona  = obraz transmisji + adnotacje + projekcja modelu boiska,
    prawa strona = rzut boiska z odpowiadającymi punktami w metrach.

Przykłady:
    python scripts/generate_homography_evaluation_gt.py

    python scripts/generate_homography_evaluation_gt.py --no-refine

    python scripts/generate_homography_evaluation_gt.py --limit 1

    python scripts/generate_homography_evaluation_gt.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "data" / "data_homography_evaluation"

# Importujemy sprawdzoną logikę pojedynczego H_GT zamiast ją dublować.
try:
    from generate_homography_gt import (
        collect_line_data,
        create_visualization,
        estimate_homography_from_lines,
        least_squares,
        line_residuals_px,
        refine_homography,
        sample_demo_image_points,
        transform_points,
    )
except ImportError as exc:
    raise ImportError(
        "Nie udało się zaimportować generate_homography_gt.py. "
        "Umieść ten plik w katalogu CvFootballTracker/scripts/ "
        "obok generate_homography_evaluation_gt.py."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wyznacz referencyjne homografie H_GT dla zamrożonej puli ewaluacyjnej."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help=f"Katalog puli ewaluacyjnej. Domyślnie: {DEFAULT_EVAL_ROOT}",
    )
    parser.add_argument(
        "--pitch-length",
        type=float,
        default=105.0,
        help="Długość boiska [m]. Domyślnie: 105.",
    )
    parser.add_argument(
        "--pitch-width",
        type=float,
        default=68.0,
        help="Szerokość boiska [m]. Domyślnie: 68.",
    )
    parser.add_argument(
        "--num-demo-points",
        type=int,
        default=2,
        help="Liczba punktów demonstracyjnych na wizualizacji. Domyślnie: 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bazowy seed punktów demonstracyjnych. Domyślnie: 42.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Opcjonalnie przetwórz tylko pierwsze N próbek, np. --limit 1 do testu.",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Wyłącz robust nonlinear refinement z SciPy.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Nie generuj plików PNG z wizualizacją.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Usuń wcześniejsze ground_truth/ i visualizations_gt/ przed uruchomieniem.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Brak {description}: {path}")


def normalize_h(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    if abs(h[2, 2]) > 1e-12:
        return h / h[2, 2]
    norm = np.linalg.norm(h)
    if norm <= 1e-12:
        raise ValueError("Macierz homografii ma zerową normę.")
    return h / norm


def flatten_h(prefix: str, h: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_{r}{c}": float(h[r, c])
        for r in range(3)
        for c in range(3)
    }


def process_one(
    row,
    eval_root: Path,
    gt_dir: Path,
    vis_dir: Path,
    pitch_length: float,
    pitch_width: float,
    num_demo_points: int,
    seed: int,
    use_refine: bool,
    make_visualization: bool,
) -> dict:
    eval_id = int(row.eval_id)
    eval_stem = str(row.eval_stem)

    image_path = eval_root / "images" / str(row.eval_image)
    annotation_path = eval_root / "annotations" / str(row.eval_annotation)

    ensure_exists(image_path, f"obrazu {eval_stem}")
    ensure_exists(annotation_path, f"adnotacji {eval_stem}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"OpenCV nie może odczytać obrazu: {image_path}")

    image_height, image_width = image_bgr.shape[:2]

    with annotation_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    correspondences, line_data, world_norm_pts, image_norm_pts = collect_line_data(
        annotations=annotations,
        image_width=image_width,
        image_height=image_height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    used_classes = [entry[0] for entry in line_data]

    # generate_homography_gt.py sam wymaga >= 4 prostych odpowiadających liniom murawy.
    if len(correspondences) < 4:
        raise ValueError(
            f"Za mało użytecznych prostych linii boiska: {len(correspondences)}. "
            "Do jednoznacznej estymacji wymagane są co najmniej 4."
        )

    h_pitch_to_image = estimate_homography_from_lines(
        correspondences=correspondences,
        world_points_for_normalization=world_norm_pts,
        image_points_for_normalization=image_norm_pts,
    )

    refined = False
    if use_refine:
        h_pitch_to_image, refined = refine_homography(
            h_initial=h_pitch_to_image,
            line_data=line_data,
        )

    h_pitch_to_image = normalize_h(h_pitch_to_image)

    h_image_to_pitch = np.linalg.inv(h_pitch_to_image)
    h_image_to_pitch = normalize_h(h_image_to_pitch)

    residuals = line_residuals_px(h_pitch_to_image, line_data)

    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Nie udało się poprawnie policzyć diagnostyki błędu linii.")

    rms_px = float(np.sqrt(np.mean(residuals ** 2)))
    mean_abs_px = float(np.mean(np.abs(residuals)))
    median_abs_px = float(np.median(np.abs(residuals)))
    max_abs_px = float(np.max(np.abs(residuals)))

    # Deterministyczne punkty tylko do wizualizacji.
    image_points = sample_demo_image_points(
        h_image_to_pitch=h_image_to_pitch,
        image_width=image_width,
        image_height=image_height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        count=num_demo_points,
        seed=seed + eval_id,
    )
    pitch_points = transform_points(h_image_to_pitch, image_points)

    gt_json_path = gt_dir / f"{eval_stem}_gt.json"
    vis_path = vis_dir / f"{eval_stem}_gt.png"

    output = {
        "eval_id": eval_id,
        "eval_stem": eval_stem,
        "image": str(image_path.relative_to(PROJECT_ROOT)),
        "annotation": str(annotation_path.relative_to(PROJECT_ROOT)),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "pitch_length_m": float(pitch_length),
        "pitch_width_m": float(pitch_width),
        "coordinate_system": {
            "origin": "pitch center",
            "x_axis": "negative toward left goal, positive toward right goal",
            "y_axis": "negative toward top touchline, positive toward bottom touchline",
            "units": "meters",
        },
        "used_line_classes": used_classes,
        "used_line_count": len(used_classes),
        "refined_with_scipy": bool(refined),
        "quality": {
            "rms_point_to_projected_line_px": rms_px,
            "mean_abs_point_to_projected_line_px": mean_abs_px,
            "median_abs_point_to_projected_line_px": median_abs_px,
            "max_abs_point_to_projected_line_px": max_abs_px,
        },
        "H_pitch_to_image": h_pitch_to_image.tolist(),
        "H_image_to_pitch": h_image_to_pitch.tolist(),
        "demo_points": [
            {
                "label": f"P{i}",
                "image_px": [float(uv[0]), float(uv[1])],
                "pitch_m": [float(xy[0]), float(xy[1])],
            }
            for i, (uv, xy) in enumerate(zip(image_points, pitch_points), start=1)
        ],
    }

    with gt_json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if make_visualization:
        create_visualization(
            image_bgr=image_bgr,
            h_pitch_to_image=h_pitch_to_image,
            line_data=line_data,
            image_points=image_points,
            pitch_points=pitch_points,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            output_path=vis_path,
        )

    row_out = {
        "eval_id": eval_id,
        "eval_stem": eval_stem,
        "status": "ok",
        "used_line_count": len(used_classes),
        "used_line_classes": "|".join(used_classes),
        "refined_with_scipy": bool(refined),
        "rms_line_error_px": rms_px,
        "mean_abs_line_error_px": mean_abs_px,
        "median_abs_line_error_px": median_abs_px,
        "max_abs_line_error_px": max_abs_px,
        "gt_json": str(gt_json_path.relative_to(PROJECT_ROOT)),
        "visualization": (
            str(vis_path.relative_to(PROJECT_ROOT))
            if make_visualization
            else ""
        ),
    }
    row_out.update(flatten_h("H_img2pitch", h_image_to_pitch))
    row_out.update(flatten_h("H_pitch2img", h_pitch_to_image))

    return row_out


def main() -> None:
    args = parse_args()

    if args.pitch_length <= 0 or args.pitch_width <= 0:
        raise ValueError("Wymiary boiska muszą być dodatnie.")
    if args.num_demo_points < 1:
        raise ValueError("--num-demo-points musi być >= 1.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit musi być > 0.")

    eval_root = args.eval_root.resolve()

    images_dir = eval_root / "images"
    annotations_dir = eval_root / "annotations"
    manifest_path = eval_root / "manifest.csv"
    gt_dir = eval_root / "ground_truth"
    vis_dir = eval_root / "visualizations_gt"

    ensure_exists(eval_root, "katalogu ewaluacyjnego")
    ensure_exists(images_dir, "katalogu images")
    ensure_exists(annotations_dir, "katalogu annotations")
    ensure_exists(manifest_path, "manifest.csv")

    if args.overwrite:
        if gt_dir.exists():
            shutil.rmtree(gt_dir)
        if vis_dir.exists():
            shutil.rmtree(vis_dir)

        for old_file in (
            eval_root / "homography_ground_truth.csv",
            eval_root / "homography_gt_summary.json",
            eval_root / "homography_gt_failures.csv",
        ):
            if old_file.exists():
                old_file.unlink()

    gt_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    required = {
        "eval_id",
        "eval_stem",
        "eval_image",
        "eval_annotation",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest.csv nie ma kolumn: {sorted(missing)}")

    manifest = manifest.sort_values("eval_id").reset_index(drop=True)

    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    print(f"Pula: {eval_root}")
    print(f"Liczba próbek do przetworzenia: {len(manifest)}")
    print(f"Refinement SciPy: {'OFF' if args.no_refine else 'ON'}")
    print(f"Wizualizacje: {'OFF' if args.no_visualizations else 'ON'}")
    print()

    results: List[dict] = []
    failures: List[dict] = []

    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        eval_id = int(row.eval_id)
        eval_stem = str(row.eval_stem)

        try:
            result = process_one(
                row=row,
                eval_root=eval_root,
                gt_dir=gt_dir,
                vis_dir=vis_dir,
                pitch_length=args.pitch_length,
                pitch_width=args.pitch_width,
                num_demo_points=args.num_demo_points,
                seed=args.seed,
                use_refine=not args.no_refine,
                make_visualization=not args.no_visualizations,
            )
            results.append(result)

            print(
                f"[{idx:03d}/{len(manifest):03d}] {eval_stem}: OK | "
                f"lines={result['used_line_count']} | "
                f"RMS={result['rms_line_error_px']:.3f}px"
            )

        except Exception as exc:
            failure = {
                "eval_id": eval_id,
                "eval_stem": eval_stem,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)

            print(
                f"[{idx:03d}/{len(manifest):03d}] {eval_stem}: FAILED | "
                f"{failure['error']}"
            )

    results_path = eval_root / "homography_ground_truth.csv"
    failures_path = eval_root / "homography_gt_failures.csv"
    summary_path = eval_root / "homography_gt_summary.json"

    if results:
        results_df = pd.DataFrame(results).sort_values("eval_id")
        results_df.to_csv(results_path, index=False)

        summary_quality = {
            "rms_line_error_px_mean": float(results_df["rms_line_error_px"].mean()),
            "rms_line_error_px_median": float(results_df["rms_line_error_px"].median()),
            "mean_abs_line_error_px_mean": float(results_df["mean_abs_line_error_px"].mean()),
            "median_abs_line_error_px_median": float(results_df["median_abs_line_error_px"].median()),
        }
    else:
        summary_quality = {}

    if failures:
        pd.DataFrame(failures).sort_values("eval_id").to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()

    total = len(manifest)
    success = len(results)
    failed = len(failures)

    summary = {
        "eval_root": str(eval_root),
        "total_requested": total,
        "success_count": success,
        "failure_count": failed,
        "success_rate": (success / total) if total else 0.0,
        "pitch_length_m": float(args.pitch_length),
        "pitch_width_m": float(args.pitch_width),
        "refinement_requested": not args.no_refine,
        "scipy_available": least_squares is not None,
        "visualizations_generated": not args.no_visualizations,
        "quality_diagnostics": summary_quality,
        "important_note": (
            "The line-error values describe how H_GT fits the same SoccerNet line annotations "
            "used to derive it. Treat them as GT diagnostics, not as an independent evaluation "
            "metric for the tested homography methods."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========================================")
    print("PODSUMOWANIE H_GT")
    print("========================================")
    print(f"Przetworzono: {total}")
    print(f"Poprawne H_GT: {success}")
    print(f"Niepowodzenia: {failed}")
    print(f"Success rate: {(100.0 * success / total) if total else 0.0:.2f}%")
    print(f"\nZbiorczy GT: {results_path}")
    print(f"Podsumowanie: {summary_path}")
    if failures:
        print(f"Błędy: {failures_path}")
    if not args.no_visualizations:
        print(f"Wizualizacje: {vis_dir}")


if __name__ == "__main__":
    main()
