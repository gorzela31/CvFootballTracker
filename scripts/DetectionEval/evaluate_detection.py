"""
Ewaluacja YOLOv8n i Faster R-CNN na ZAMROZONEJ puli badawczej detekcji.

Umiesc plik jako:
    CvFootballTracker/scripts/evaluate_detection.py

Zrodlo danych:
    data/data_detection_evaluation/
        images/eval_001.jpg ... eval_100.jpg
        manifest.csv
        ground_truth.csv

Tryby:
    1) Jedna klatka:
       python scripts/evaluate_detection.py

       Domyslnie:
       eval_001.jpg

       Inna klatka:
       python scripts/evaluate_detection.py --mode single --eval-id 37

    2) Cala pula badawcza:
       python scripts/evaluate_detection.py --mode all

    3) Tylko jeden model:
       python scripts/evaluate_detection.py --mode all --models yolo
       python scripts/evaluate_detection.py --mode all --models faster_rcnn

Metryki:
    - Precision / Recall przy OPERATING_CONF i IoU=0.5
    - AP@0.5
    - AP@0.5:0.95, COCO-style, 101-point interpolation
    - mean / median IoU dla true positives
    - TP / FP / FN
    - sredni czas inferencji [ms]
    - FPS

Wyniki:
    results/detection_evaluation/<timestamp>/
        metrics_summary.csv
        per_frame_metrics.csv
        predictions.csv
        sample_manifest.csv
        config.json
        annotations/<model>/pred/*.jpg
        annotations/<model>/compare_gt/*.jpg

Klasy:
    0 = ball
    1 = player (w tym goalkeeper)
    2 = referee

Najwazniejszym wierszem do pracy jest class='player'.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from ultralytics import YOLO


# =============================================================================
# KONFIGURACJA
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVAL_ROOT = PROJECT_ROOT / "data" / "data_detection_evaluation"
EVAL_IMAGES = EVAL_ROOT / "images"
EVAL_MANIFEST = EVAL_ROOT / "manifest.csv"
EVAL_GT = EVAL_ROOT / "ground_truth.csv"

YOLO_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "yolov8n"
    / "trained_detection_yolov8n.pt"
)

FRCNN_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "faster_rcnn"
    / "trained_fasterrcnn_resnet50.pt"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "detection_evaluation"

CLASS_NAMES = {
    0: "ball",
    1: "player",
    2: "referee",
}
CLASS_IDS = tuple(CLASS_NAMES.keys())

# Precision / Recall / TP / FP / FN / IoU / wizualizacja
OPERATING_CONF = 0.25
#OPERATING_CONF = 0.5

# Do AP potrzebujemy predykcji od bardzo niskiego confidence.
AP_MIN_CONF = 0.001

MATCH_IOU = 0.50

# YOLO
YOLO_IMGSZ = 960

# Faster R-CNN - konfiguracja finalnego treningu v4.
FRCNN_MIN_SIZE = 1080
FRCNN_MAX_SIZE = 1920
FRCNN_ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))
FRCNN_ANCHOR_RATIOS = (0.5, 1.0, 2.0)
FRCNN_RPN_PRE_NMS_TEST = 2000
FRCNN_RPN_POST_NMS_TEST = 1000


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


def box_iou_numpy(
    box: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """IoU jednego boxa [4] z tablica boxow [N, 4]."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = (
        max(0.0, float(box[2] - box[0]))
        * max(0.0, float(box[3] - box[1]))
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


def empty_gt() -> Dict[str, np.ndarray]:
    return {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "classes": np.empty((0,), dtype=np.int64),
        "track_ids": np.empty((0,), dtype=np.int64),
    }


# =============================================================================
# ZAMROZONA PULA BADAWCZA
# =============================================================================

def load_evaluation_dataset():
    """
    Wczytuje manifest i ground_truth.csv wygenerowane przez
    prepare_detection_evaluation_dataset.py.

    Zwraca:
        manifest_df
        gt_by_eval_id
    """
    ensure_exists(EVAL_ROOT, "katalogu data_detection_evaluation")
    ensure_exists(EVAL_IMAGES, "katalogu images")
    ensure_exists(EVAL_MANIFEST, "manifest.csv")
    ensure_exists(EVAL_GT, "ground_truth.csv")

    manifest = pd.read_csv(EVAL_MANIFEST)
    gt_df = pd.read_csv(EVAL_GT)

    required_manifest = {
        "eval_id",
        "eval_image",
        "source_sequence",
        "source_frame",
    }
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(
            f"manifest.csv nie ma kolumn: {sorted(missing_manifest)}"
        )

    required_gt = {
        "eval_id",
        "class_id",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    missing_gt = required_gt - set(gt_df.columns)
    if missing_gt:
        raise ValueError(
            f"ground_truth.csv nie ma kolumn: {sorted(missing_gt)}"
        )

    manifest = (
        manifest
        .sort_values("eval_id")
        .reset_index(drop=True)
    )

    gt_by_eval_id: Dict[int, Dict[str, np.ndarray]] = {}

    for eval_id, group in gt_df.groupby("eval_id"):
        boxes = group[
            ["x1", "y1", "x2", "y2"]
        ].to_numpy(dtype=np.float32)

        classes = group["class_id"].to_numpy(dtype=np.int64)
        track_ids = group["track_id"].to_numpy(dtype=np.int64)

        gt_by_eval_id[int(eval_id)] = {
            "boxes": boxes,
            "classes": classes,
            "track_ids": track_ids,
        }

    return manifest, gt_by_eval_id


def select_samples(
    manifest: pd.DataFrame,
    mode: str,
    eval_id: int,
) -> List[dict]:
    """
    single -> jedna klatka eval_XXX
    all    -> cala zamrozona pula z manifest.csv
    """

    if mode == "single":
        rows = manifest[
            manifest["eval_id"] == eval_id
        ]

        if rows.empty:
            available_min = int(manifest["eval_id"].min())
            available_max = int(manifest["eval_id"].max())

            raise ValueError(
                f"Nie znaleziono eval_id={eval_id}. "
                f"Dostepny zakres: {available_min}-{available_max}"
            )

        manifest_selected = rows.copy()

    elif mode == "all":
        manifest_selected = manifest.copy()

    else:
        raise ValueError(
            f"Nieznany mode={mode}"
        )

    samples: List[dict] = []

    for row in manifest_selected.itertuples(index=False):
        image_path = EVAL_IMAGES / str(row.eval_image)
        ensure_exists(
            image_path,
            f"obrazu {row.eval_image}",
        )

        samples.append(
            {
                "eval_id": int(row.eval_id),
                "eval_image": str(row.eval_image),
                "image_path": image_path,
                "source_sequence": str(row.source_sequence),
                "source_frame": int(row.source_frame),
            }
        )

    return samples


# =============================================================================
# MODELE
# =============================================================================

def create_frcnn_model(
    weights_path: Path,
    device: torch.device,
    score_thresh: float,
):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        min_size=FRCNN_MIN_SIZE,
        max_size=FRCNN_MAX_SIZE,
        rpn_pre_nms_top_n_test=FRCNN_RPN_PRE_NMS_TEST,
        rpn_post_nms_top_n_test=FRCNN_RPN_POST_NMS_TEST,
        box_score_thresh=score_thresh,
    )

    model.rpn.anchor_generator = AnchorGenerator(
        FRCNN_ANCHOR_SIZES,
        (FRCNN_ANCHOR_RATIOS,)
        * len(FRCNN_ANCHOR_SIZES),
    )

    in_features = (
        model.roi_heads.box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            4,  # background + ball/player/referee
        )
    )

    state = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model


def infer_yolo(
    model: YOLO,
    frame_bgr: np.ndarray,
    device: torch.device,
):
    yolo_device = (
        0
        if device.type == "cuda"
        else "cpu"
    )

    sync_cuda()
    t0 = time.perf_counter()

    result = model.predict(
        source=frame_bgr,
        conf=AP_MIN_CONF,
        imgsz=YOLO_IMGSZ,
        device=yolo_device,
        verbose=False,
    )[0]

    sync_cuda()

    elapsed_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            elapsed_ms,
        )

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    scores = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    valid = np.isin(
        classes,
        CLASS_IDS,
    )

    return (
        boxes[valid],
        scores[valid],
        classes[valid],
        elapsed_ms,
    )


@torch.no_grad()
def infer_frcnn(
    model,
    frame_bgr: np.ndarray,
    device: torch.device,
):
    sync_cuda()
    t0 = time.perf_counter()

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        torch.from_numpy(frame_rgb)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .to(device)
    )

    pred = model([tensor])[0]

    sync_cuda()

    elapsed_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    boxes = (
        pred["boxes"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    scores = (
        pred["scores"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Faster R-CNN:
    # 0 background
    # 1 ball
    # 2 player
    # 3 referee
    # Konwertujemy do wspolnego 0..2.
    classes = (
        pred["labels"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
        - 1
    )

    valid = np.isin(
        classes,
        CLASS_IDS,
    )

    return (
        boxes[valid],
        scores[valid],
        classes[valid],
        elapsed_ms,
    )


# =============================================================================
# MATCHING + METRYKI
# =============================================================================

def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int, List[float]]:

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    order = np.argsort(
        -pred_scores
    )

    matched_gt = np.zeros(
        len(gt_boxes),
        dtype=bool,
    )

    tp = 0
    fp = 0
    matched_ious: List[float] = []

    for idx in order:
        ious = box_iou_numpy(
            pred_boxes[idx],
            gt_boxes,
        )

        if len(ious) == 0:
            fp += 1
            continue

        ious_masked = ious.copy()
        ious_masked[matched_gt] = -1.0

        best_idx = int(
            np.argmax(
                ious_masked
            )
        )

        best_iou = float(
            ious_masked[best_idx]
        )

        if best_iou >= iou_threshold:
            matched_gt[best_idx] = True
            tp += 1
            matched_ious.append(
                best_iou
            )
        else:
            fp += 1

    fn = int(
        (~matched_gt).sum()
    )

    return (
        tp,
        fp,
        fn,
        matched_ious,
    )


def compute_ap_101(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:

    if len(recalls) == 0:
        return 0.0

    mrec = np.concatenate(
        ([0.0], recalls, [1.0])
    )

    mpre = np.concatenate(
        ([1.0], precisions, [0.0])
    )

    for i in range(
        len(mpre) - 2,
        -1,
        -1,
    ):
        mpre[i] = max(
            mpre[i],
            mpre[i + 1],
        )

    recall_points = np.linspace(
        0.0,
        1.0,
        101,
    )

    p_interp = np.zeros_like(
        recall_points
    )

    for i, r in enumerate(
        recall_points
    ):
        mask = mrec >= r

        p_interp[i] = (
            np.max(mpre[mask])
            if np.any(mask)
            else 0.0
        )

    return float(
        p_interp.mean()
    )


def ap_for_class(
    records: Sequence[dict],
    class_id: int,
    iou_threshold: float,
) -> float:

    all_preds = []
    gt_by_image = {}
    n_gt = 0

    for image_idx, rec in enumerate(records):
        gt_mask = (
            rec["gt_classes"]
            == class_id
        )

        gt_boxes = (
            rec["gt_boxes"][gt_mask]
        )

        gt_by_image[image_idx] = (
            gt_boxes
        )

        n_gt += len(
            gt_boxes
        )

        pred_mask = (
            rec["pred_classes"]
            == class_id
        )

        for box, score in zip(
            rec["pred_boxes"][pred_mask],
            rec["pred_scores"][pred_mask],
        ):
            all_preds.append(
                (
                    image_idx,
                    float(score),
                    box,
                )
            )

    if n_gt == 0:
        return float("nan")

    all_preds.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    matched = {
        i: np.zeros(
            len(gt_by_image[i]),
            dtype=bool,
        )
        for i in gt_by_image
    }

    tp = np.zeros(
        len(all_preds),
        dtype=np.float64,
    )

    fp = np.zeros(
        len(all_preds),
        dtype=np.float64,
    )

    for i, (
        image_idx,
        _score,
        box,
    ) in enumerate(all_preds):

        gt_boxes = (
            gt_by_image[image_idx]
        )

        if len(gt_boxes) == 0:
            fp[i] = 1.0
            continue

        ious = box_iou_numpy(
            box,
            gt_boxes,
        )

        ious_masked = ious.copy()

        ious_masked[
            matched[image_idx]
        ] = -1.0

        best_idx = int(
            np.argmax(
                ious_masked
            )
        )

        best_iou = float(
            ious_masked[best_idx]
        )

        if best_iou >= iou_threshold:
            tp[i] = 1.0
            matched[image_idx][best_idx] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = (
        cum_tp / max(n_gt, 1)
    )

    precisions = (
        cum_tp
        / np.maximum(
            cum_tp + cum_fp,
            1e-12,
        )
    )

    return compute_ap_101(
        recalls,
        precisions,
    )


def aggregate_metrics(
    records: Sequence[dict],
    inference_times_ms: Sequence[float],
) -> pd.DataFrame:

    rows = []

    ap_thresholds = np.arange(
        0.50,
        1.00,
        0.05,
    )

    for (
        class_id,
        class_name,
    ) in CLASS_NAMES.items():

        total_tp = 0
        total_fp = 0
        total_fn = 0

        total_gt = 0
        total_pred_at_conf = 0

        all_matched_ious: List[float] = []

        for rec in records:
            gt_mask = (
                rec["gt_classes"]
                == class_id
            )

            gt_boxes = (
                rec["gt_boxes"][gt_mask]
            )

            pred_mask = (
                (rec["pred_classes"] == class_id)
                & (
                    rec["pred_scores"]
                    >= OPERATING_CONF
                )
            )

            pred_boxes = (
                rec["pred_boxes"][pred_mask]
            )

            pred_scores = (
                rec["pred_scores"][pred_mask]
            )

            tp, fp, fn, ious = (
                greedy_match(
                    pred_boxes,
                    pred_scores,
                    gt_boxes,
                    MATCH_IOU,
                )
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            all_matched_ious.extend(
                ious
            )

            total_gt += len(
                gt_boxes
            )

            total_pred_at_conf += len(
                pred_boxes
            )

        precision = (
            total_tp
            / (total_tp + total_fp)
            if (total_tp + total_fp)
            else 0.0
        )

        recall = (
            total_tp
            / (total_tp + total_fn)
            if (total_tp + total_fn)
            else 0.0
        )

        ap50 = ap_for_class(
            records,
            class_id,
            0.50,
        )

        aps = [
            ap_for_class(
                records,
                class_id,
                float(t),
            )
            for t in ap_thresholds
        ]

        aps_valid = [
            x
            for x in aps
            if not math.isnan(x)
        ]

        ap5095 = (
            float(np.mean(aps_valid))
            if aps_valid
            else float("nan")
        )

        mean_ms = (
            float(np.mean(inference_times_ms))
            if inference_times_ms
            else float("nan")
        )

        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap5095,
                "mean_iou_tp": (
                    float(
                        np.mean(
                            all_matched_ious
                        )
                    )
                    if all_matched_ious
                    else float("nan")
                ),
                "median_iou_tp": (
                    float(
                        np.median(
                            all_matched_ious
                        )
                    )
                    if all_matched_ious
                    else float("nan")
                ),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "n_gt": total_gt,
                "n_predictions_at_conf": (
                    total_pred_at_conf
                ),
                "mean_inference_ms": mean_ms,
                "fps": (
                    1000.0 / mean_ms
                    if (
                        not math.isnan(mean_ms)
                        and mean_ms > 0
                    )
                    else float("nan")
                ),
            }
        )

    class_df = pd.DataFrame(
        rows
    )

    mean_ms = (
        float(np.mean(inference_times_ms))
        if inference_times_ms
        else float("nan")
    )

    macro = {
        "class": "macro",
        "precision": class_df["precision"].mean(),
        "recall": class_df["recall"].mean(),
        "ap50": class_df["ap50"].mean(),
        "ap50_95": class_df["ap50_95"].mean(),
        "mean_iou_tp": class_df["mean_iou_tp"].mean(),
        "median_iou_tp": class_df["median_iou_tp"].mean(),
        "tp": int(class_df["tp"].sum()),
        "fp": int(class_df["fp"].sum()),
        "fn": int(class_df["fn"].sum()),
        "n_gt": int(class_df["n_gt"].sum()),
        "n_predictions_at_conf": int(
            class_df[
                "n_predictions_at_conf"
            ].sum()
        ),
        "mean_inference_ms": mean_ms,
        "fps": (
            1000.0 / mean_ms
            if (
                not math.isnan(mean_ms)
                and mean_ms > 0
            )
            else float("nan")
        ),
    }

    return pd.concat(
        [
            class_df,
            pd.DataFrame([macro]),
        ],
        ignore_index=True,
    )


def per_frame_metrics(
    model_name: str,
    rec: dict,
) -> List[dict]:

    rows = []

    for (
        class_id,
        class_name,
    ) in CLASS_NAMES.items():

        gt_mask = (
            rec["gt_classes"]
            == class_id
        )

        gt_boxes = (
            rec["gt_boxes"][gt_mask]
        )

        pred_mask = (
            (rec["pred_classes"] == class_id)
            & (
                rec["pred_scores"]
                >= OPERATING_CONF
            )
        )

        pred_boxes = (
            rec["pred_boxes"][pred_mask]
        )

        pred_scores = (
            rec["pred_scores"][pred_mask]
        )

        tp, fp, fn, ious = (
            greedy_match(
                pred_boxes,
                pred_scores,
                gt_boxes,
                MATCH_IOU,
            )
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        rows.append(
            {
                "model": model_name,
                "eval_id": rec["eval_id"],
                "eval_image": rec["eval_image"],
                "source_sequence": rec["source_sequence"],
                "source_frame": rec["source_frame"],
                "class": class_name,
                "n_gt": len(gt_boxes),
                "n_pred": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "mean_iou_tp": (
                    float(np.mean(ious))
                    if ious
                    else float("nan")
                ),
                "median_iou_tp": (
                    float(np.median(ious))
                    if ious
                    else float("nan")
                ),
                "inference_ms": rec["inference_ms"],
            }
        )

    return rows


# =============================================================================
# WIZUALIZACJA
# =============================================================================

def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray | None = None,
    prefix: str = "",
    color: Tuple[int, int, int] = (0, 0, 255),
    show_labels: bool = True,
) -> np.ndarray:

    out = image.copy()

    for i, (
        box,
        class_id,
    ) in enumerate(
        zip(
            boxes,
            classes,
        )
    ):
        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        if show_labels:
            name = CLASS_NAMES.get(
                int(class_id),
                str(class_id),
            )

            label = (
                f"{prefix}{name}"
            )

            if scores is not None:
                label += (
                    f" {float(scores[i]):.3f}"
                )

            cv2.putText(
                out,
                label,
                (
                    x1,
                    max(18, y1 - 5),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return out


def save_annotations(
    output_dir: Path,
    model_name: str,
    rec: dict,
    frame_bgr: np.ndarray,
) -> None:

    pred_mask = (
        rec["pred_scores"]
        >= OPERATING_CONF
    )

    pred_boxes = (
        rec["pred_boxes"][pred_mask]
    )

    pred_scores = (
        rec["pred_scores"][pred_mask]
    )

    pred_classes = (
        rec["pred_classes"][pred_mask]
    )

    base_name = (
        rec["eval_image"]
    )

    pred_dir = (
        output_dir
        / "annotations"
        / model_name
        / "pred"
    )

    cmp_dir = (
        output_dir
        / "annotations"
        / model_name
        / "compare_gt"
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_img = draw_boxes(
        frame_bgr,
        pred_boxes,
        pred_classes,
        pred_scores,
        prefix="PRED ",
        color=(0, 0, 255),
    )

    cv2.imwrite(
        str(
            pred_dir
            / base_name
        ),
        pred_img,
    )

    # GT = zielony, PRED = czerwony.
    cmp_img = draw_boxes(
        frame_bgr,
        rec["gt_boxes"],
        rec["gt_classes"],
        None,
        prefix="GT ",
        color=(0, 255, 0),
        show_labels=False,
    )

    cmp_img = draw_boxes(
        cmp_img,
        pred_boxes,
        pred_classes,
        pred_scores,
        prefix="PRED ",
        color=(0, 0, 255),
    )

    cv2.imwrite(
        str(
            cmp_dir
            / base_name
        ),
        cmp_img,
    )


# =============================================================================
# EWALUACJA MODELU
# =============================================================================

def evaluate_model(
    model_name: str,
    samples: Sequence[dict],
    gt_by_eval_id: Dict[int, Dict[str, np.ndarray]],
    output_dir: Path,
    device: torch.device,
    save_images: bool,
):

    print(
        f"\n{'=' * 80}"
        f"\nMODEL: {model_name}"
        f"\n{'=' * 80}"
    )

    if model_name == "yolo":
        ensure_exists(
            YOLO_WEIGHTS,
            "wag YOLO",
        )

        model = YOLO(
            str(YOLO_WEIGHTS)
        )

        infer_fn = (
            lambda frame:
            infer_yolo(
                model,
                frame,
                device,
            )
        )

    elif model_name == "faster_rcnn":
        ensure_exists(
            FRCNN_WEIGHTS,
            "wag Faster R-CNN",
        )

        model = create_frcnn_model(
            FRCNN_WEIGHTS,
            device,
            score_thresh=AP_MIN_CONF,
        )

        infer_fn = (
            lambda frame:
            infer_frcnn(
                model,
                frame,
                device,
            )
        )

    else:
        raise ValueError(
            model_name
        )

    # Warm-up nie jest wliczany do czasu.
    warm_frame = cv2.imread(
        str(
            samples[0]["image_path"]
        )
    )

    if warm_frame is None:
        raise RuntimeError(
            "Nie mozna wczytac "
            f'{samples[0]["image_path"]}'
        )

    print(
        "Warm-up modelu..."
    )

    _ = infer_fn(
        warm_frame
    )

    records = []
    times = []
    per_frame_rows = []
    prediction_rows = []

    for idx, sample in enumerate(
        samples,
        start=1,
    ):
        image_path = (
            sample["image_path"]
        )

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            print(
                "[WARN] Pomijam - "
                f"nie mozna wczytac: "
                f"{image_path}"
            )
            continue

        gt = gt_by_eval_id.get(
            sample["eval_id"],
            empty_gt(),
        )

        (
            pred_boxes,
            pred_scores,
            pred_classes,
            inference_ms,
        ) = infer_fn(
            frame
        )

        rec = {
            "eval_id": sample["eval_id"],
            "eval_image": sample["eval_image"],
            "source_sequence": sample["source_sequence"],
            "source_frame": sample["source_frame"],
            "image_path": str(image_path),
            "gt_boxes": gt["boxes"],
            "gt_classes": gt["classes"],
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_classes": pred_classes,
            "inference_ms": inference_ms,
        }

        records.append(
            rec
        )

        times.append(
            inference_ms
        )

        per_frame_rows.extend(
            per_frame_metrics(
                model_name,
                rec,
            )
        )

        for (
            box,
            score,
            class_id,
        ) in zip(
            pred_boxes,
            pred_scores,
            pred_classes,
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "eval_id": sample["eval_id"],
                    "eval_image": sample["eval_image"],
                    "source_sequence": sample["source_sequence"],
                    "source_frame": sample["source_frame"],
                    "class": CLASS_NAMES.get(
                        int(class_id),
                        str(class_id),
                    ),
                    "confidence": float(score),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )

        if save_images:
            save_annotations(
                output_dir,
                model_name,
                rec,
                frame,
            )

        n_pred_operating = int(
            (
                pred_scores
                >= OPERATING_CONF
            ).sum()
        )

        print(
            f"[{idx:>3}/{len(samples)}] "
            f"{sample['eval_image']} "
            f"({sample['source_sequence']} / "
            f"frame {sample['source_frame']}) | "
            f"GT={len(gt['boxes']):>2} | "
            f"pred@{OPERATING_CONF:.2f}="
            f"{n_pred_operating:>2} | "
            f"{inference_ms:.1f} ms"
        )

    summary = aggregate_metrics(
        records,
        times,
    )

    summary.insert(
        0,
        "model",
        model_name,
    )

    return (
        summary,
        per_frame_rows,
        prediction_rows,
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ewaluacja YOLOv8n i Faster R-CNN "
            "na data/data_detection_evaluation"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "all",
        ],
        default="single",
        help=(
            "single = jedna klatka, "
            "all = cala zamrozona pula"
        ),
    )

    parser.add_argument(
        "--eval-id",
        type=int,
        default=1,
        help=(
            "Numer eval_XXX dla --mode single. "
            "Domyslnie 1."
        ),
    )

    parser.add_argument(
        "--models",
        choices=[
            "both",
            "yolo",
            "faster_rcnn",
        ],
        default="both",
    )

    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help=(
            "Nie zapisuj obrazow z ramkami. "
            "Metryki nadal zostana policzone."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ensure_exists(
        EVAL_ROOT,
        "katalogu data_detection_evaluation",
    )

    ensure_exists(
        YOLO_WEIGHTS,
        "wag YOLO",
    )

    if args.models in (
        "both",
        "faster_rcnn",
    ):
        ensure_exists(
            FRCNN_WEIGHTS,
            "wag Faster R-CNN",
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"PROJECT_ROOT: {PROJECT_ROOT}"
    )

    print(
        f"EVAL_ROOT:    {EVAL_ROOT}"
    )

    print(
        f"DEVICE:       {device}"
    )

    print(
        f"OPERATING_CONF={OPERATING_CONF}, "
        f"AP_MIN_CONF={AP_MIN_CONF}, "
        f"MATCH_IOU={MATCH_IOU}"
    )

    manifest, gt_by_eval_id = (
        load_evaluation_dataset()
    )

    print(
        f"Pula w manifest.csv: "
        f"{len(manifest)} klatek"
    )

    samples = select_samples(
        manifest=manifest,
        mode=args.mode,
        eval_id=args.eval_id,
    )

    print(
        f"Wybrano {len(samples)} "
        f"klatek do ewaluacji."
    )

    if args.mode == "single":
        print(
            "Tryb SINGLE: "
            f"{samples[0]['eval_image']} "
            f"<- "
            f"{samples[0]['source_sequence']} / "
            f"frame {samples[0]['source_frame']}"
        )
    else:
        if len(samples) != 100:
            print(
                "[UWAGA] Tryb ALL bierze "
                f"wszystkie {len(samples)} klatek "
                "z manifest.csv. "
                "Nie wymuszam sztucznie liczby 100."
            )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    output_dir = (
        RESULTS_ROOT
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Zapisujemy dokładnie użyte próbki.
    pd.DataFrame(
        [
            {
                "eval_id": s["eval_id"],
                "eval_image": s["eval_image"],
                "source_sequence": s["source_sequence"],
                "source_frame": s["source_frame"],
                "image": str(s["image_path"]),
            }
            for s in samples
        ]
    ).to_csv(
        output_dir
        / "sample_manifest.csv",
        index=False,
    )

    config = {
        "dataset": str(EVAL_ROOT),
        "mode": args.mode,
        "eval_id": (
            args.eval_id
            if args.mode == "single"
            else None
        ),
        "num_frames": len(samples),
        "models": args.models,
        "operating_conf": OPERATING_CONF,
        "ap_min_conf": AP_MIN_CONF,
        "match_iou": MATCH_IOU,
        "yolo_imgsz": YOLO_IMGSZ,
        "yolo_weights": str(
            YOLO_WEIGHTS
        ),
        "frcnn_weights": str(
            FRCNN_WEIGHTS
        ),
        "frcnn_min_size": (
            FRCNN_MIN_SIZE
        ),
        "frcnn_max_size": (
            FRCNN_MAX_SIZE
        ),
        "frcnn_anchor_sizes": (
            FRCNN_ANCHOR_SIZES
        ),
        "device": str(device),
    }

    (
        output_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    model_names = (
        [
            "yolo",
            "faster_rcnn",
        ]
        if args.models == "both"
        else [args.models]
    )

    summaries = []
    all_per_frame_rows = []
    all_prediction_rows = []

    for model_name in model_names:
        (
            summary,
            per_frame_rows,
            prediction_rows,
        ) = evaluate_model(
            model_name=model_name,
            samples=samples,
            gt_by_eval_id=gt_by_eval_id,
            output_dir=output_dir,
            device=device,
            save_images=(
                not args.no_save_images
            ),
        )

        summaries.append(
            summary
        )

        all_per_frame_rows.extend(
            per_frame_rows
        )

        all_prediction_rows.extend(
            prediction_rows
        )

    summary_df = pd.concat(
        summaries,
        ignore_index=True,
    )

    summary_df.to_csv(
        output_dir
        / "metrics_summary.csv",
        index=False,
    )

    pd.DataFrame(
        all_per_frame_rows
    ).to_csv(
        output_dir
        / "per_frame_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        all_prediction_rows
    ).to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )

    print(
        f"\n{'=' * 80}"
        f"\nPODSUMOWANIE"
        f"\n{'=' * 80}"
    )

    display_cols = [
        "model",
        "class",
        "precision",
        "recall",
        "ap50",
        "ap50_95",
        "mean_iou_tp",
        "median_iou_tp",
        "mean_inference_ms",
        "fps",
        "tp",
        "fp",
        "fn",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
    ):
        print(
            summary_df[
                display_cols
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

    print(
        f"\nWyniki zapisano w: "
        f"{output_dir}"
    )

    print(
        "Do pracy najwazniejszy "
        "jest wiersz class='player'."
    )


if __name__ == "__main__":
    main()
