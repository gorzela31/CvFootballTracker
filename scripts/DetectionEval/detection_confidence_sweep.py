"""
Confidence sweep dla wyników ewaluacji detekcji BEZ ponownej inferencji.

Umieść plik jako:
    CvFootballTracker/scripts/confidence_sweep.py

Domyślne uruchomienie:
    python scripts/confidence_sweep.py

Skrypt:
    - automatycznie znajduje najnowszy katalog:
          results/detection_evaluation/<timestamp>/
      zawierający predictions.csv,
    - korzysta z zamrożonego GT:
          data/data_detection_evaluation/ground_truth.csv,
    - liczy Precision, Recall i F1 dla progów confidence 0.05...0.90,
    - robi to osobno dla:
          ball, player, referee
      oraz osobno dla:
          yolo, faster_rcnn,
    - zapisuje CSV i wykresy,
    - NIE uruchamia ponownie modeli.

UWAGA METODOLOGICZNA:
    best_thresholds.csv traktuj jako analizę wrażliwości na próg,
    a nie jako podstawę do dostrojenia modelu na zbiorze testowym.

Dodatkowe przykłady:
    python scripts/confidence_sweep.py --start 0.05 --end 0.95 --step 0.05

    python scripts/confidence_sweep.py ^
        --results-dir results/detection_evaluation/2026-09-06_12-34-56
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = PROJECT_ROOT / "results" / "detection_evaluation"
GT_PATH = (
    PROJECT_ROOT
    / "data"
    / "data_detection_evaluation"
    / "ground_truth.csv"
)

MATCH_IOU = 0.50

CLASS_NAMES = {
    0: "ball",
    1: "player",
    2: "referee",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza wpływu progu confidence na Precision, Recall i F1 "
            "bez ponownej inferencji."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Konkretny katalog results/detection_evaluation/<timestamp>. "
            "Jeśli pominięty, skrypt wybierze najnowszy zawierający predictions.csv."
        ),
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.05,
        help="Pierwszy próg confidence. Domyślnie 0.05.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=0.90,
        help="Ostatni próg confidence. Domyślnie 0.90.",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Krok confidence. Domyślnie 0.05.",
    )

    return parser.parse_args()


def find_latest_results_dir() -> Path:
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu wyników: {RESULTS_ROOT}"
        )

    candidates = [
        p
        for p in RESULTS_ROOT.iterdir()
        if p.is_dir() and (p / "predictions.csv").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Nie znaleziono żadnego predictions.csv w {RESULTS_ROOT}"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def box_iou_numpy(
    box: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(
        0.0,
        float(box[2] - box[0]),
    ) * max(
        0.0,
        float(box[3] - box[1]),
    )

    area_b = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_a + area_b - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int]:

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    order = np.argsort(-pred_scores)

    matched_gt = np.zeros(
        len(gt_boxes),
        dtype=bool,
    )

    tp = 0
    fp = 0

    for pred_idx in order:
        ious = box_iou_numpy(
            pred_boxes[pred_idx],
            gt_boxes,
        )

        ious[matched_gt] = -1.0

        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_threshold:
            matched_gt[best_gt_idx] = True
            tp += 1
        else:
            fp += 1

    fn = int((~matched_gt).sum())

    return tp, fp, fn


def load_gt(gt_path: Path) -> Dict[Tuple[int, str], np.ndarray]:
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono ground truth: {gt_path}"
        )

    gt_df = pd.read_csv(gt_path)

    required = {
        "eval_id",
        "class_id",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = required - set(gt_df.columns)

    if missing:
        raise ValueError(
            f"ground_truth.csv nie ma kolumn: {sorted(missing)}"
        )

    gt_lookup: Dict[Tuple[int, str], np.ndarray] = {}

    for eval_id in sorted(gt_df["eval_id"].unique()):
        eval_rows = gt_df[
            gt_df["eval_id"] == eval_id
        ]

        for class_id, class_name in CLASS_NAMES.items():
            rows = eval_rows[
                eval_rows["class_id"] == class_id
            ]

            gt_lookup[
                (int(eval_id), class_name)
            ] = rows[
                ["x1", "y1", "x2", "y2"]
            ].to_numpy(dtype=np.float32)

    return gt_lookup


def build_thresholds(
    start: float,
    end: float,
    step: float,
) -> np.ndarray:

    if step <= 0:
        raise ValueError("--step musi być > 0.")

    if not (0 <= start <= 1):
        raise ValueError("--start musi należeć do [0, 1].")

    if not (0 <= end <= 1):
        raise ValueError("--end musi należeć do [0, 1].")

    if end < start:
        raise ValueError("--end nie może być mniejsze od --start.")

    count = int(
        round((end - start) / step)
    ) + 1

    thresholds = start + np.arange(count) * step

    thresholds = thresholds[
        thresholds <= end + 1e-9
    ]

    return np.round(
        thresholds,
        10,
    )


def calculate_sweep(
    predictions: pd.DataFrame,
    gt_lookup: Dict[Tuple[int, str], np.ndarray],
    thresholds: np.ndarray,
) -> pd.DataFrame:

    required = {
        "model",
        "eval_id",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"predictions.csv nie ma kolumn: {sorted(missing)}"
        )

    models = sorted(
        predictions["model"].dropna().unique()
    )

    eval_ids = sorted(
        {
            eval_id
            for eval_id, _class_name in gt_lookup.keys()
        }
    )

    rows: List[dict] = []

    for model_name in models:
        model_preds = predictions[
            predictions["model"] == model_name
        ]

        for class_name in CLASS_NAMES.values():
            class_preds = model_preds[
                model_preds["class"] == class_name
            ]

            for threshold in thresholds:
                tp_total = 0
                fp_total = 0
                fn_total = 0

                for eval_id in eval_ids:
                    gt_boxes = gt_lookup.get(
                        (eval_id, class_name),
                        np.empty((0, 4), dtype=np.float32),
                    )

                    frame_preds = class_preds[
                        (class_preds["eval_id"] == eval_id)
                        & (class_preds["confidence"] >= threshold)
                    ]

                    pred_boxes = frame_preds[
                        ["x1", "y1", "x2", "y2"]
                    ].to_numpy(dtype=np.float32)

                    pred_scores = frame_preds[
                        "confidence"
                    ].to_numpy(dtype=np.float32)

                    tp, fp, fn = greedy_match(
                        pred_boxes=pred_boxes,
                        pred_scores=pred_scores,
                        gt_boxes=gt_boxes,
                        iou_threshold=MATCH_IOU,
                    )

                    tp_total += tp
                    fp_total += fp
                    fn_total += fn

                precision = (
                    tp_total / (tp_total + fp_total)
                    if (tp_total + fp_total) > 0
                    else 0.0
                )

                recall = (
                    tp_total / (tp_total + fn_total)
                    if (tp_total + fn_total) > 0
                    else 0.0
                )

                f1 = (
                    2.0 * precision * recall
                    / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                rows.append(
                    {
                        "model": model_name,
                        "class": class_name,
                        "confidence": float(threshold),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "tp": tp_total,
                        "fp": fp_total,
                        "fn": fn_total,
                    }
                )

    return pd.DataFrame(rows)


def best_thresholds(
    sweep_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Maksymalne F1 tylko jako analiza opisowa.
    Nie należy traktować tego wyniku jako strojenia na zbiorze testowym.
    """
    rows = []

    for (model_name, class_name), group in sweep_df.groupby(
        ["model", "class"]
    ):
        max_f1 = group["f1"].max()

        # Jeśli kilka progów ma dokładnie taki sam F1,
        # bierzemy najniższy z nich, żeby zachować większy Recall.
        best = (
            group[
                np.isclose(
                    group["f1"],
                    max_f1,
                    atol=1e-12,
                )
            ]
            .sort_values("confidence")
            .iloc[0]
        )

        rows.append(
            {
                "model": model_name,
                "class": class_name,
                "best_confidence_by_f1": float(
                    best["confidence"]
                ),
                "precision": float(
                    best["precision"]
                ),
                "recall": float(
                    best["recall"]
                ),
                "f1": float(
                    best["f1"]
                ),
                "tp": int(best["tp"]),
                "fp": int(best["fp"]),
                "fn": int(best["fn"]),
            }
        )

    return pd.DataFrame(rows)


def plot_class(
    sweep_df: pd.DataFrame,
    class_name: str,
    output_path: Path,
) -> None:

    subset = sweep_df[
        sweep_df["class"] == class_name
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model_name in sorted(
        subset["model"].unique()
    ):
        model_data = subset[
            subset["model"] == model_name
        ].sort_values("confidence")

        ax.plot(
            model_data["confidence"],
            model_data["precision"],
            marker="o",
            label=f"{model_name} Precision",
        )

        ax.plot(
            model_data["confidence"],
            model_data["recall"],
            marker="o",
            label=f"{model_name} Recall",
        )

        ax.plot(
            model_data["confidence"],
            model_data["f1"],
            marker="o",
            label=f"{model_name} F1",
        )

    ax.set_title(
        f"Wpływ progu confidence na metryki - {class_name}"
    )
    ax.set_xlabel(
        "Próg confidence"
    )
    ax.set_ylabel(
        "Wartość metryki"
    )
    ax.set_ylim(
        0.0,
        1.02,
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_player_separate(
    sweep_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Dodatkowe prostsze wykresy dla klasy player:
      - Precision vs confidence
      - Recall vs confidence
      - F1 vs confidence

    Są wygodniejsze do późniejszego użycia w pracy niż jeden zatłoczony wykres.
    """
    player_df = sweep_df[
        sweep_df["class"] == "player"
    ]

    for metric in [
        "precision",
        "recall",
        "f1",
    ]:
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for model_name in sorted(
            player_df["model"].unique()
        ):
            model_data = player_df[
                player_df["model"] == model_name
            ].sort_values("confidence")

            ax.plot(
                model_data["confidence"],
                model_data[metric],
                marker="o",
                label=model_name,
            )

        ax.set_title(
            f"Player - {metric.capitalize()} w funkcji confidence"
        )
        ax.set_xlabel(
            "Próg confidence"
        )
        ax.set_ylabel(
            metric.capitalize()
        )
        ax.set_ylim(
            0.0,
            1.02,
        )
        ax.grid(
            True,
            alpha=0.25,
        )
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            output_dir
            / f"player_{metric}_vs_confidence.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)


def main() -> None:
    args = parse_args()

    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else find_latest_results_dir()
    )

    predictions_path = (
        results_dir
        / "predictions.csv"
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono: {predictions_path}"
        )

    thresholds = build_thresholds(
        start=args.start,
        end=args.end,
        step=args.step,
    )

    print("=" * 78)
    print("CONFIDENCE SWEEP")
    print("=" * 78)
    print(f"Wyniki źródłowe: {results_dir}")
    print(f"Predykcje:       {predictions_path}")
    print(f"Ground truth:    {GT_PATH}")
    print(
        "Progi:           "
        + ", ".join(
            f"{t:.2f}"
            for t in thresholds
        )
    )
    print(f"IoU matching:    {MATCH_IOU:.2f}")
    print()
    print(
        "UWAGA: analiza najlepszego F1 ma charakter opisowy. "
        "Nie używaj zbioru testowego do strojenia finalnego progu modelu."
    )
    print()

    predictions = pd.read_csv(
        predictions_path
    )

    gt_lookup = load_gt(
        GT_PATH
    )

    sweep_df = calculate_sweep(
        predictions=predictions,
        gt_lookup=gt_lookup,
        thresholds=thresholds,
    )

    best_df = best_thresholds(
        sweep_df
    )

    output_dir = (
        results_dir
        / "confidence_sweep"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sweep_path = (
        output_dir
        / "confidence_sweep.csv"
    )

    best_path = (
        output_dir
        / "best_thresholds.csv"
    )

    sweep_df.to_csv(
        sweep_path,
        index=False,
    )

    best_df.to_csv(
        best_path,
        index=False,
    )

    for class_name in CLASS_NAMES.values():
        plot_class(
            sweep_df=sweep_df,
            class_name=class_name,
            output_path=(
                output_dir
                / f"{class_name}_precision_recall_f1_vs_confidence.png"
            ),
        )

    plot_player_separate(
        sweep_df=sweep_df,
        output_dir=output_dir,
    )

    print("=" * 78)
    print("NAJLEPSZE F1 W BADANYM ZAKRESIE - TYLKO ANALIZA OPISOWA")
    print("=" * 78)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        160,
    ):
        print(
            best_df
            .round(4)
            .to_string(index=False)
        )

    print()
    print("Zapisano:")
    print(f"  {sweep_path}")
    print(f"  {best_path}")
    print(
        f"  wykresy PNG -> {output_dir}"
    )

    print()
    print(
        "Do magisterki w pierwszej kolejności analizuj klasę 'player'."
    )


if __name__ == "__main__":
    main()
