"""
pitch_homography_diagnostics.py

Diagnostyka klasycznej homografii boiska pilkarskiego.

Ten modul nie zaklada, ze homografia musi zostac znaleziona dla kazdej klatki.
Jego celem jest pokazanie krok po kroku, czy algorytm naprawde widzi cechy boiska:
field mask, maske bialych linii, komponenty, odcinki, rodziny kierunkow,
przeciecia, kolo srodkowe oraz probe estymacji H.

Zasada pracy:
1. Najpierw diagnozujemy ekstrakcje cech.
2. Dopiero potem probujemy homografii.
3. Wynik H jest oznaczany jako accepted tylko po dodatkowej walidacji geometrycznej.

Wymagany plik obok notebooka:
    classical_homography_v2.py
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from classical_homography_v2 import ClassicalHomographyV2, PITCH_LENGTH_M, PITCH_WIDTH_M
    _HAS_CLASSICAL = True
except Exception:
    ClassicalHomographyV2 = None
    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0
    _HAS_CLASSICAL = False


@dataclass
class DiagnosticConfig:
    # Segmentacja murawy
    green_h_low: int = 30
    green_h_high: int = 95
    green_s_low: int = 25
    green_v_low: int = 25
    min_field_area_ratio: float = 0.15
    field_erode_iterations: int = 2

    # Biale linie
    white_s_max: int = 85
    white_v_min: int = 150
    line_component_min_area: int = 15
    line_component_max_area_ratio: float = 0.02
    line_component_min_aspect: float = 2.2
    line_component_max_thin_dim: int = 18

    # Odcinki
    hough_threshold: int = 18
    hough_min_line_length: int = 22
    hough_max_line_gap: int = 18
    segment_min_length: float = 25.0
    segment_support_radius: int = 4
    segment_min_support_ratio: float = 0.45

    # Scalanie i grupowanie
    merge_angle_tol: float = 6.0
    merge_dist_tol: float = 18.0
    merge_min_total_length: float = 45.0
    min_lines_per_direction: int = 2

    # Kosztowne kroki opcjonalne
    run_circle_detection: bool = False
    run_original_homography_attempt: bool = False

    # Walidacja H
    min_base_score: float = 0.25
    max_model_to_line_dist_px: float = 14.0
    max_line_to_model_dist_px: float = 28.0
    min_model_coverage: float = 0.20

    # Wizualizacje
    draw_thickness: int = 2


@dataclass
class HomographyQuality:
    base_score: float
    mean_model_to_line_px: float
    mean_line_to_model_px: float
    model_coverage: float
    visible_model_pixels: int
    accepted: bool
    reason: str


class PitchHomographyDiagnostics:
    def __init__(self, config: Optional[DiagnosticConfig] = None, verbose: bool = True):
        self.config = config or DiagnosticConfig()
        self.verbose = verbose

    def log(self, text: str) -> None:
        if self.verbose:
            print(f"[diagnostics] {text}")

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def read_image(image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Nie mozna wczytac obrazu: {image_path}")
        return image

    @staticmethod
    def save_image(path: str | Path, image: np.ndarray) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)

    @staticmethod
    def gray_to_bgr(mask: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
        vis = image.copy()
        color_img = np.zeros_like(vis)
        color_img[:, :] = color
        idx = mask > 0
        vis[idx] = cv2.addWeighted(vis, 1.0 - alpha, color_img, alpha, 0)[idx]
        return vis

    def make_base(self, width: int, height: int, min_score: float = 0.30) -> Optional[Any]:
        if not _HAS_CLASSICAL:
            return None
        return ClassicalHomographyV2(
            image_width=width,
            image_height=height,
            ransac_iterations=500,
            min_score=min_score,
            refine_steps=0,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # 1. Segmentacja murawy
    # ------------------------------------------------------------------
    def segment_field(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        cfg = self.config
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        green = cv2.inRange(
            hsv,
            (cfg.green_h_low, cfg.green_s_low, cfg.green_v_low),
            (cfg.green_h_high, 255, 255),
        )
        k7 = np.ones((7, 7), np.uint8)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k7, iterations=1)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k7, iterations=3)

        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((h, w), dtype=np.uint8), None, {
                "field_area_ratio": 0.0,
                "field_status": "no_green_contours",
            }

        largest = max(contours, key=cv2.contourArea)
        area_ratio = float(cv2.contourArea(largest)) / float(h * w)
        if area_ratio < cfg.min_field_area_ratio:
            return green, None, {
                "field_area_ratio": area_ratio,
                "field_status": "weak_green_mask_used_directly",
            }

        hull = cv2.convexHull(largest)
        field_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(field_mask, [hull], 255)
        k11 = np.ones((11, 11), np.uint8)
        field_mask = cv2.erode(field_mask, k11, iterations=cfg.field_erode_iterations)

        return field_mask, hull, {
            "field_area_ratio": area_ratio,
            "field_status": "ok_hull_eroded",
        }

    # ------------------------------------------------------------------
    # 2. Maska bialych pikseli i komponentow liniowych
    # ------------------------------------------------------------------
    def extract_white_masks(self, image: np.ndarray, field_mask: np.ndarray) -> Dict[str, np.ndarray]:
        cfg = self.config
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_raw = cv2.inRange(hsv, (0, 0, cfg.white_v_min), (180, cfg.white_s_max, 255))
        white_in_field = cv2.bitwise_and(white_raw, field_mask)

        k3 = np.ones((3, 3), np.uint8)
        white_clean = cv2.morphologyEx(white_in_field, cv2.MORPH_OPEN, k3, iterations=1)
        white_clean = cv2.morphologyEx(white_clean, cv2.MORPH_CLOSE, k3, iterations=1)

        return {
            "white_raw": white_raw,
            "white_in_field": white_in_field,
            "white_clean": white_clean,
        }

    def filter_line_components(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        cfg = self.config
        h, w = mask.shape[:2]
        max_area = int(cfg.line_component_max_area_ratio * h * w)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        line_mask = np.zeros_like(mask)
        component_vis = np.zeros((h, w, 3), dtype=np.uint8)
        rows: List[Dict[str, Any]] = []

        for i in range(1, n_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            aspect = float(max(bw, bh)) / max(float(min(bw, bh)), 1.0)
            thin_dim = min(bw, bh)
            fill_ratio = float(area) / max(float(bw * bh), 1.0)

            keep = True
            reason = "kept"
            if area < cfg.line_component_min_area:
                keep = False
                reason = "too_small"
            elif area > max_area:
                keep = False
                reason = "too_large"
            elif not (aspect >= cfg.line_component_min_aspect or thin_dim <= cfg.line_component_max_thin_dim):
                keep = False
                reason = "not_line_like"
            elif fill_ratio > 0.75 and area > 120:
                keep = False
                reason = "filled_blob"
            elif bh > (bw * 1.2) and thin_dim > 6 and area > 30:
                keep = False
                reason = "vertical_player"

            rows.append({
                "label": i,
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "area": area,
                "aspect": aspect,
                "thin_dim": int(thin_dim),
                "fill_ratio": fill_ratio,
                "keep": bool(keep),
                "reason": reason,
                "cx": float(centroids[i][0]),
                "cy": float(centroids[i][1]),
            })

            if keep:
                line_mask[labels == i] = 255
                component_vis[labels == i] = (0, 255, 0)
            else:
                component_vis[labels == i] = (0, 0, 255)

        return line_mask, component_vis, rows

    # ------------------------------------------------------------------
    # 3. Detekcja i filtrowanie odcinkow
    # ------------------------------------------------------------------
    @staticmethod
    def segment_angle(seg: np.ndarray) -> float:
        x1, y1, x2, y2 = [float(v) for v in seg]
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0

    @staticmethod
    def segment_length(seg: np.ndarray) -> float:
        x1, y1, x2, y2 = [float(v) for v in seg]
        return float(math.hypot(x2 - x1, y2 - y1))

    def segment_support_ratio(self, mask: np.ndarray, seg: np.ndarray, radius: int) -> float:
        x1, y1, x2, y2 = [float(v) for v in seg]
        length = max(self.segment_length(seg), 1.0)
        n = int(max(12, min(160, length / 3.0)))
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)
        h, w = mask.shape[:2]
        supported = 0
        for x, y in zip(xs, ys):
            xi = int(round(x))
            yi = int(round(y))
            x0 = max(0, xi - radius)
            x1b = min(w, xi + radius + 1)
            y0 = max(0, yi - radius)
            y1b = min(h, yi + radius + 1)
            if x0 >= x1b or y0 >= y1b:
                continue
            if np.any(mask[y0:y1b, x0:x1b] > 0):
                supported += 1
        return float(supported) / float(n)

    def detect_segments(self, line_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        cfg = self.config
        edges = cv2.Canny(line_mask, 50, 150)
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )
        if raw is None:
            return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.int32), []

        raw_segments = raw.reshape(-1, 4).astype(np.int32)
        kept = []
        rows: List[Dict[str, Any]] = []
        for idx, seg in enumerate(raw_segments):
            length = self.segment_length(seg)
            angle = self.segment_angle(seg)
            support = self.segment_support_ratio(line_mask, seg, cfg.segment_support_radius)
            keep = length >= cfg.segment_min_length and support >= cfg.segment_min_support_ratio
            rows.append({
                "idx": idx,
                "x1": int(seg[0]),
                "y1": int(seg[1]),
                "x2": int(seg[2]),
                "y2": int(seg[3]),
                "length": length,
                "angle": angle,
                "support": support,
                "keep": bool(keep),
            })
            if keep:
                kept.append(seg)

        filtered_segments = np.array(kept, dtype=np.int32).reshape(-1, 4) if kept else np.empty((0, 4), dtype=np.int32)
        return raw_segments, filtered_segments, rows

    # ------------------------------------------------------------------
    # 4. Grupowanie kierunkow i scalanie
    # ------------------------------------------------------------------
    @staticmethod
    def _cluster_orientations_two_groups(angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Lekki zamiennik KMeans dla orientacji modulo 180 stopni."""
        n = len(angles)
        if n == 0:
            return np.empty((0,), dtype=np.int32)
        if n < 4:
            return np.zeros((n,), dtype=np.int32)

        vecs = np.column_stack([
            np.cos(2.0 * np.radians(angles)),
            np.sin(2.0 * np.radians(angles)),
        ])

        first = int(np.argmax(weights))
        dots = vecs @ vecs[first]
        second = int(np.argmin(dots))
        centers = np.stack([vecs[first], vecs[second]], axis=0).astype(np.float64)

        labels = np.zeros((n,), dtype=np.int32)
        for _ in range(12):
            sim = vecs @ centers.T
            new_labels = np.argmax(sim, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for k in range(2):
                if not np.any(labels == k):
                    continue
                c = np.average(vecs[labels == k], axis=0, weights=weights[labels == k])
                norm = np.linalg.norm(c)
                if norm > 1e-9:
                    centers[k] = c / norm

        return labels

    def group_segments_by_angle(self, segments: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if len(segments) == 0:
            return np.empty((0,), dtype=np.int32), {"status": "no_segments"}
        if len(segments) < 4:
            labels = np.zeros((len(segments),), dtype=np.int32)
            return labels, {"status": "too_few_segments", "n_groups": 1}

        angles = np.array([self.segment_angle(s) for s in segments], dtype=np.float64)
        lengths = np.array([self.segment_length(s) for s in segments], dtype=np.float64)
        labels = self._cluster_orientations_two_groups(angles, lengths)

        stats = {
            "status": "ok",
            "n_groups": 2,
            "group_0_count": int(np.sum(labels == 0)),
            "group_1_count": int(np.sum(labels == 1)),
            "group_0_length": float(np.sum(lengths[labels == 0])),
            "group_1_length": float(np.sum(lengths[labels == 1])),
            "group_0_angle_mean": float(np.mean(angles[labels == 0])) if np.any(labels == 0) else None,
            "group_1_angle_mean": float(np.mean(angles[labels == 1])) if np.any(labels == 1) else None,
        }
        return labels.astype(np.int32), stats

    def merge_lines_with_classical(self, segments: np.ndarray, width: int, height: int) -> List[Dict[str, Any]]:
        base = self.make_base(width, height)
        if base is None or len(segments) == 0:
            return []
        return base.merge_collinear(
            segments,
            angle_tol=self.config.merge_angle_tol,
            dist_tol=self.config.merge_dist_tol,
            min_total_length=self.config.merge_min_total_length,
        )

    def group_merged_lines_with_classical(self, merged: List[Dict[str, Any]], width: int, height: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Nazwa zostaje dla kompatybilnosci notebooka, ale grupowanie robimy lokalnie,
        # bez sklearn.KMeans. Wczesniej KMeans potrafil zawieszac batch na kilku klatkach.
        if not merged:
            return [], []
        if len(merged) < 4:
            return merged, []
        angles = np.array([float(m.get("angle", 0.0)) for m in merged], dtype=np.float64)
        weights = np.array([float(m.get("total_length", 1.0)) for m in merged], dtype=np.float64)
        labels = self._cluster_orientations_two_groups(angles, weights)
        group_a = [m for m, lab in zip(merged, labels) if lab == 0]
        group_b = [m for m, lab in zip(merged, labels) if lab == 1]
        return group_a, group_b

    def detect_intersections(self, group_a: List[Dict[str, Any]], group_b: List[Dict[str, Any]], width: int, height: int) -> np.ndarray:
        base = self.make_base(width, height)
        if base is None or len(group_a) < 1 or len(group_b) < 1:
            return np.empty((0, 2), dtype=np.float64)
        return base.detect_intersections(group_a, group_b)

    def detect_center_circle(self, line_mask: np.ndarray, merged: List[Dict[str, Any]], width: int, height: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, ...]], str]:
        base = self.make_base(width, height)
        if base is None:
            return None, None, "classical_module_missing"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                C, params = base.detect_circle(line_mask, merged, max_trials=60)
        except Exception as exc:
            return None, None, f"circle_detection_error: {exc}"
        if C is None or params is None:
            return None, None, "not_found"
        return C, tuple(float(x) for x in params), "found"

    # ------------------------------------------------------------------
    # 5. Proba homografii i twarda walidacja
    # ------------------------------------------------------------------
    def project_pitch_model_mask(self, H: np.ndarray, width: int, height: int, thickness: int = 2) -> np.ndarray:
        base = self.make_base(width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        if base is None or H is None:
            return mask
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return mask

        for polyline in base.pitch_lines:
            pts = []
            for xm, ym in polyline:
                p = H_inv @ np.array([float(xm), float(ym), 1.0])
                if abs(p[2]) < 1e-10:
                    continue
                p = p / p[2]
                x = int(round(p[0]))
                y = int(round(p[1]))
                pts.append((x, y))
            for i in range(len(pts) - 1):
                cv2.line(mask, pts[i], pts[i + 1], 255, thickness)
        return mask

    def evaluate_homography_quality(self, H: Optional[np.ndarray], line_mask: np.ndarray) -> HomographyQuality:
        h, w = line_mask.shape[:2]
        if H is None:
            return HomographyQuality(0.0, 9999.0, 9999.0, 0.0, 0, False, "no_H")

        base = self.make_base(w, h)
        if base is None:
            return HomographyQuality(0.0, 9999.0, 9999.0, 0.0, 0, False, "classical_module_missing")

        try:
            base_score = float(base.score_homography(H, line_mask))
        except Exception:
            base_score = 0.0

        model_mask = self.project_pitch_model_mask(H, w, h, thickness=2)
        model_pixels = int(np.sum(model_mask > 0))
        if model_pixels == 0:
            return HomographyQuality(base_score, 9999.0, 9999.0, 0.0, 0, False, "no_visible_model")

        # Odleglosci Chamfera na pelnej rozdzielczosci.
        line_bin = (line_mask > 0).astype(np.uint8)
        model_bin = (model_mask > 0).astype(np.uint8)
        dist_to_line = cv2.distanceTransform((1 - line_bin).astype(np.uint8), cv2.DIST_L2, 3)
        dist_to_model = cv2.distanceTransform((1 - model_bin).astype(np.uint8), cv2.DIST_L2, 3)

        mean_model_to_line = float(np.mean(dist_to_line[model_bin > 0])) if np.any(model_bin > 0) else 9999.0
        mean_line_to_model = float(np.mean(dist_to_model[line_bin > 0])) if np.any(line_bin > 0) else 9999.0

        ys, xs = np.where(model_bin > 0)
        if len(xs) == 0:
            coverage = 0.0
        else:
            coverage = max((float(xs.max() - xs.min()) / max(w, 1)), (float(ys.max() - ys.min()) / max(h, 1)))

        reasons = []
        cfg = self.config
        if base_score < cfg.min_base_score:
            reasons.append(f"low_base_score={base_score:.3f}")
        if mean_model_to_line > cfg.max_model_to_line_dist_px:
            reasons.append(f"model_far_from_lines={mean_model_to_line:.1f}px")
        if mean_line_to_model > cfg.max_line_to_model_dist_px:
            reasons.append(f"lines_far_from_model={mean_line_to_model:.1f}px")
        if coverage < cfg.min_model_coverage:
            reasons.append(f"low_model_coverage={coverage:.3f}")

        accepted = len(reasons) == 0
        reason = "accepted" if accepted else "; ".join(reasons)
        return HomographyQuality(base_score, mean_model_to_line, mean_line_to_model, coverage, model_pixels, accepted, reason)

    def try_original_classical_homography(self, image: np.ndarray, line_mask: np.ndarray) -> Tuple[Optional[np.ndarray], HomographyQuality, str]:
        h, w = image.shape[:2]
        base = self.make_base(w, h, min_score=0.20)
        if base is None:
            return None, self.evaluate_homography_quality(None, line_mask), "classical_module_missing"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                H = base.estimate_from_image(image)
        except Exception as exc:
            return None, self.evaluate_homography_quality(None, line_mask), f"error: {exc}"
        quality = self.evaluate_homography_quality(H, line_mask)
        return H, quality, "original_classical_v2"

    # ------------------------------------------------------------------
    # 6. Wizualizacje
    # ------------------------------------------------------------------
    def draw_segments(self, image: np.ndarray, segments: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        vis = image.copy()
        for seg in segments:
            x1, y1, x2, y2 = [int(v) for v in seg]
            cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
        return vis

    def draw_grouped_segments(self, image: np.ndarray, segments: np.ndarray, labels: np.ndarray) -> np.ndarray:
        vis = image.copy()
        colors = [(255, 0, 0), (0, 255, 255), (0, 128, 255), (255, 255, 0)]
        for seg, lab in zip(segments, labels):
            x1, y1, x2, y2 = [int(v) for v in seg]
            cv2.line(vis, (x1, y1), (x2, y2), colors[int(lab) % len(colors)], self.config.draw_thickness)
        return vis

    def draw_merged_lines(self, image: np.ndarray, merged: List[Dict[str, Any]]) -> np.ndarray:
        vis = image.copy()
        for i, m in enumerate(merged):
            p1 = tuple(np.round(m["p1"]).astype(int))
            p2 = tuple(np.round(m["p2"]).astype(int))
            cv2.line(vis, p1, p2, (255, 0, 255), 3)
            mid = tuple(np.round((m["p1"] + m["p2"]) / 2.0).astype(int))
            cv2.putText(vis, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    def draw_intersections(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        vis = image.copy()
        h, w = image.shape[:2]
        for i, p in enumerate(pts):
            x, y = int(round(p[0])), int(round(p[1]))
            if -50 <= x <= w + 50 and -50 <= y <= h + 50:
                cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(vis, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        return vis

    def draw_circle(self, image: np.ndarray, params: Optional[Tuple[float, ...]]) -> np.ndarray:
        vis = image.copy()
        if params is None:
            cv2.putText(vis, "circle: not found", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            return vis
        xc, yc, a, b, theta = params
        center = (int(round(xc)), int(round(yc)))
        axes = (int(round(a)), int(round(b)))
        angle_deg = float(math.degrees(theta))
        cv2.ellipse(vis, center, axes, angle_deg, 0, 360, (0, 255, 255), 3)
        cv2.circle(vis, center, 5, (0, 0, 255), -1)
        cv2.putText(vis, "circle: found", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return vis

    def draw_homography_overlay(self, image: np.ndarray, H: Optional[np.ndarray], quality: HomographyQuality) -> np.ndarray:
        h, w = image.shape[:2]
        base = self.make_base(w, h)
        if base is not None and H is not None:
            vis = base.draw_overlay(image, H, color=(0, 255, 255), thickness=2)
        else:
            vis = image.copy()
        status = "H accepted" if quality.accepted else "H rejected"
        cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if quality.accepted else (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, quality.reason[:120], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    # ------------------------------------------------------------------
    # 7. Pelny przebieg diagnostyczny dla jednej klatki
    # ------------------------------------------------------------------
    def run_one(self, image_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
        image_path = Path(image_path)
        out = self.ensure_dir(output_dir)
        stem = image_path.stem
        image = self.read_image(image_path)
        h, w = image.shape[:2]
        self.log(f"{image_path.name}: start, size={w}x{h}")

        field_mask, hull, field_info = self.segment_field(image)
        white_masks = self.extract_white_masks(image, field_mask)
        line_mask, component_vis, component_rows = self.filter_line_components(white_masks["white_clean"])
        raw_segments, filtered_segments, segment_rows = self.detect_segments(line_mask)
        labels, group_stats = self.group_segments_by_angle(filtered_segments)
        merged = self.merge_lines_with_classical(filtered_segments, w, h)
        group_a, group_b = self.group_merged_lines_with_classical(merged, w, h)
        intersections = self.detect_intersections(group_a, group_b, w, h)
        if self.config.run_circle_detection:
            C, circle_params, circle_status = self.detect_center_circle(line_mask, merged, w, h)
        else:
            C, circle_params, circle_status = None, None, "skipped_by_config"
        if self.config.run_original_homography_attempt:
            H, quality, h_method = self.try_original_classical_homography(image, line_mask)
        else:
            H = None
            quality = self.evaluate_homography_quality(None, line_mask)
            h_method = "skipped_by_config"

        # Wizualizacje
        self.save_image(out / f"{stem}_00_original.jpg", image)
        self.save_image(out / f"{stem}_01_field_mask.png", field_mask)
        self.save_image(out / f"{stem}_02_field_overlay.jpg", self.overlay_mask(image, field_mask, (0, 160, 0), alpha=0.35))
        self.save_image(out / f"{stem}_03_white_raw.png", white_masks["white_raw"])
        self.save_image(out / f"{stem}_04_white_in_field.png", white_masks["white_in_field"])
        self.save_image(out / f"{stem}_05_white_clean.png", white_masks["white_clean"])
        self.save_image(out / f"{stem}_06_line_candidates.png", line_mask)
        self.save_image(out / f"{stem}_07_components_green_keep_red_reject.jpg", component_vis)
        self.save_image(out / f"{stem}_08_segments_raw.jpg", self.draw_segments(image, raw_segments, (0, 0, 255), thickness=1))
        self.save_image(out / f"{stem}_09_segments_filtered.jpg", self.draw_segments(image, filtered_segments, (0, 255, 0), thickness=2))
        self.save_image(out / f"{stem}_10_segments_grouped.jpg", self.draw_grouped_segments(image, filtered_segments, labels))
        self.save_image(out / f"{stem}_11_merged_lines.jpg", self.draw_merged_lines(image, merged))
        self.save_image(out / f"{stem}_12_intersections.jpg", self.draw_intersections(image, intersections))
        self.save_image(out / f"{stem}_13_center_circle.jpg", self.draw_circle(image, circle_params))
        self.save_image(out / f"{stem}_14_homography_overlay.jpg", self.draw_homography_overlay(image, H, quality))

        report: Dict[str, Any] = {
            "image": str(image_path),
            "image_name": image_path.name,
            "width": int(w),
            "height": int(h),
            **field_info,
            "white_raw_pixels": int(np.sum(white_masks["white_raw"] > 0)),
            "white_in_field_pixels": int(np.sum(white_masks["white_in_field"] > 0)),
            "line_candidate_pixels": int(np.sum(line_mask > 0)),
            "components_total": int(len(component_rows)),
            "components_kept": int(sum(1 for r in component_rows if r["keep"])),
            "segments_raw": int(len(raw_segments)),
            "segments_filtered": int(len(filtered_segments)),
            "segments_group_status": group_stats.get("status"),
            "segments_group_0_count": group_stats.get("group_0_count"),
            "segments_group_1_count": group_stats.get("group_1_count"),
            "merged_lines": int(len(merged)),
            "merged_group_a": int(len(group_a)),
            "merged_group_b": int(len(group_b)),
            "intersections": int(len(intersections)),
            "circle_status": circle_status,
            "circle_params": list(circle_params) if circle_params is not None else None,
            "homography_method": h_method,
            "homography_found": H is not None,
            "homography_accepted": bool(quality.accepted),
            "homography_reason": quality.reason,
            "homography_base_score": float(quality.base_score),
            "homography_mean_model_to_line_px": float(quality.mean_model_to_line_px),
            "homography_mean_line_to_model_px": float(quality.mean_line_to_model_px),
            "homography_model_coverage": float(quality.model_coverage),
            "outputs_dir": str(out),
        }

        # Zapis danych liczbowych do debugowania.
        serializable_components = component_rows[:500]
        serializable_segments = segment_rows[:500]
        with open(out / f"{stem}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(out / f"{stem}_components.json", "w", encoding="utf-8") as f:
            json.dump(serializable_components, f, ensure_ascii=False, indent=2)
        with open(out / f"{stem}_segments.json", "w", encoding="utf-8") as f:
            json.dump(serializable_segments, f, ensure_ascii=False, indent=2)

        if H is not None:
            np.savetxt(out / f"{stem}_H_original_classical.txt", H, fmt="%.10f")

        self.log(
            f"{image_path.name}: line_pixels={report['line_candidate_pixels']}, "
            f"segments={report['segments_filtered']}, merged={report['merged_lines']}, "
            f"circle={circle_status}, H={report['homography_accepted']}"
        )
        return report

    def run_batch(self, image_paths: List[str | Path], output_dir: str | Path) -> List[Dict[str, Any]]:
        out = self.ensure_dir(output_dir)
        reports = []
        for image_path in image_paths:
            stem = Path(image_path).stem
            reports.append(self.run_one(image_path, out / stem))
        with open(out / "batch_report.json", "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        return reports


def find_default_images(root: str | Path = ".") -> List[Path]:
    root = Path(root)
    names = ["000001.jpg", "000250.jpg", "000300.jpg", "000310.jpg", "000374.jpg"]
    return [root / name for name in names if (root / name).exists()]
