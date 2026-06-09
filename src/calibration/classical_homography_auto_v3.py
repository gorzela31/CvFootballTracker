"""
classical_homography_auto_v3.py

W pelni automatyczna wersja klasycznej homografii boiska.
Nie uzywa recznego klikania punktow. Homografia jest estymowana z cech obrazu:
1. maska murawy i bialych linii,
2. odcinki/proste boiska,
3. srodek boiska: linia srodkowa plus elipsa kola srodkowego,
4. fallback: VP-RANSAC z klasy ClassicalHomographyV2.

H mapuje piksele obrazu na metry boiska: image pixel -> pitch XY.
"""

import math
from typing import Optional

import cv2
import numpy as np

from classical_homography_v2 import ClassicalHomographyV2

try:
    from skimage.measure import EllipseModel, ransac as _sk_ransac
    from skimage.morphology import skeletonize as _sk_skeletonize
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    _sk_skeletonize = None


class ClassicalHomographyAutoV3(ClassicalHomographyV2):
    """
    Automatyczny kalibrator klasyczny.

    Najwazniejsza roznica wzgledem V2:
    - nie korzysta z recznych korespondencji,
    - ma mocniejsza maske boiska bez reklam,
    - ma automatyczny solver dla ujec z kolem srodkowym,
    - dopiero potem uruchamia VP-RANSAC.
    """

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        ransac_iterations: int = 2500,
        min_score: float = 0.08,
        refine_steps: int = 0,
        verbose: bool = True,
        circle_min_score: float = 0.12,
    ):
        super().__init__(
            image_width=image_width,
            image_height=image_height,
            ransac_iterations=ransac_iterations,
            min_score=min_score,
            refine_steps=refine_steps,
            verbose=verbose,
        )
        self.circle_min_score = circle_min_score

    # ------------------------------------------------------------------
    # Robustniejsza segmentacja pola
    # ------------------------------------------------------------------

    def segment_field(self, image: np.ndarray) -> tuple:
        """
        Wyznacza maske murawy bez wciagania reklam przez convex hull.

        Zamiast wypuklej otoczki z V2 wybieramy najwieksza zielona skladowa,
        ktora dochodzi do dolnej czesci kadru. Potem ja zamykamy i dylatujemy,
        zeby biale linie nadal byly wewnatrz maski pola.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        green = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
        if n_labels <= 1:
            return np.zeros((h, w), dtype=np.uint8), None

        candidates = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            reaches_bottom = y + bh > 0.55 * h
            wide_enough = bw > 0.35 * w
            big_enough = area > 0.08 * w * h
            if reaches_bottom and wide_enough and big_enough:
                candidates.append((area, i))

        if candidates:
            idx = max(candidates)[1]
        else:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        field_mask = (labels == idx).astype(np.uint8) * 255
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8), iterations=2)
        field_mask = cv2.dilate(field_mask, np.ones((17, 17), np.uint8), iterations=1)

        # Usuniecie drobnych wysp po dylatacji.
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean = np.zeros_like(field_mask)
            cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
            field_mask = clean
            hull = cv2.convexHull(largest)

        return field_mask, hull

    def extract_line_mask(self, image: np.ndarray, field_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Maska bialych linii oparta o jasnosc i niska saturacje.

        Parametry sa luzniejsze niz w V2, bo linie boiska bywaja szare,
        rozmyte i czesciowo zasloniete. Reklamy sa ograniczane przez maske pola.
        """
        if field_mask is None:
            field_mask, _ = self.segment_field(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Linie na boisku nie zawsze maja V > 175. Dajemy nizszy prog i silniej
        # polegamy na masce murawy oraz geometrii.
        white = cv2.inRange(hsv, (0, 0, 135), (180, 120, 255))
        line_mask = cv2.bitwise_and(white, field_mask)

        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        # Odrzucenie bardzo duzych zwartych plam, ale zostawienie dlugich linii i lukow.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
        filtered = np.zeros_like(line_mask)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area < 15:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            thin_dim = min(bw, bh)
            # Linie/luki: dlugie, cienkie, albo niewielkie fragmenty przerwanych linii.
            large_field_structure = (bw > 0.18 * self.image_width or bh > 0.18 * self.image_height)
            keep = (aspect >= 3.0) or (thin_dim <= 18 and area < 3000) or (area < 250) or large_field_structure
            # Grube prostokatne obiekty, typowo fragmenty zawodnikow, sa mniej przydatne.
            reject_blob = (aspect < 1.8 and area > 2500 and not large_field_structure)
            if keep and not reject_blob:
                filtered[labels == i] = 255

        return filtered

    # ------------------------------------------------------------------
    # Cechy srodka boiska
    # ------------------------------------------------------------------

    @staticmethod
    def _geom_angle(line_dict) -> float:
        p1, p2 = line_dict["p1"], line_dict["p2"]
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0

    def _find_center_line_candidate(self, merged_lines: list) -> Optional[dict]:
        """Wybiera najpewniejsza linie srodkowa: prawie pionowa i blisko srodka kadru."""
        if not merged_lines:
            return None

        best = None
        best_score = -1e18
        cx_img = self.image_width / 2.0

        for m in merged_lines:
            angle = self._geom_angle(m)
            verticalness = 1.0 - min(abs(angle - 90.0), 90.0) / 90.0
            if verticalness < 0.70:
                continue
            p1, p2 = m["p1"], m["p2"]
            midx = 0.5 * (p1[0] + p2[0])
            centrality = 1.0 - min(abs(midx - cx_img) / (0.45 * self.image_width), 1.0)
            length_score = math.log1p(float(m["total_length"]))
            span_y = abs(p2[1] - p1[1]) / max(self.image_height, 1)
            score = 3.0 * verticalness + 2.0 * centrality + length_score + span_y
            if score > best_score:
                best_score = score
                best = m

        return best

    def detect_center_circle(self, line_mask: np.ndarray, merged_lines: list):
        """
        Automatycznie dopasowuje elipse kola srodkowego.

        Walidacja jest zwiazana z wykryta linia srodkowa: srodek elipsy musi
        lezec blisko tej linii. To usuwa typowe falszywe elipsy z zawodnikow.
        """
        if not _HAS_SKIMAGE:
            return None, None, None

        center_line = self._find_center_line_candidate(merged_lines)
        if center_line is None:
            return None, None, None

        l_center = self._line_homogeneous(center_line["p1"], center_line["p2"])
        l_norm = math.hypot(float(l_center[0]), float(l_center[1]))
        if l_norm < 1e-9:
            return None, None, None

        # Szkielet maski jest duzo lepszy do RANSAC elipsy niz pelna gruba maska:
        # zawodnicy daja wtedy mniej punktow, a linie boiska zostaja czytelne.
        if _sk_skeletonize is not None:
            point_mask = (_sk_skeletonize(line_mask > 0).astype(np.uint8) * 255)
        else:
            point_mask = line_mask

        ys, xs = np.where(point_mask > 0)
        if len(xs) < 120:
            return None, None, None

        P = np.stack([xs, ys], axis=1).astype(np.float64)
        w, h = self.image_width, self.image_height

        # Centralny pas kadru, bez gornej linii reklam i bez samego dolu.
        P = P[(P[:, 0] > 0.08 * w) & (P[:, 0] < 0.92 * w) &
              (P[:, 1] > 0.18 * h) & (P[:, 1] < 0.78 * h)]
        if len(P) < 120:
            return None, None, None

        # Probkowanie ogranicza koszt RANSAC, ale zostawia duzo punktow linii.
        if len(P) > 1000:
            rng = np.random.RandomState(7)
            P_sample = P[rng.choice(len(P), 1000, replace=False)]
        else:
            P_sample = P

        def _valid(model, *args):
            try:
                xc, yc, ax1, ax2, th = model.params
            except Exception:
                return False
            lo, hi = sorted([float(ax1), float(ax2)])
            dist_to_center_line = abs(l_center[0] * xc + l_center[1] * yc + l_center[2]) / l_norm
            ratio = hi / max(lo, 1.0)
            return (
                dist_to_center_line < 0.045 * w and
                0.12 * w < xc < 0.88 * w and
                0.22 * h < yc < 0.68 * h and
                0.04 * w < lo < 0.16 * w and
                0.22 * w < hi < 0.40 * w and
                1.8 < ratio < 7.0
            )

        try:
            model, inliers = _sk_ransac(
                P_sample,
                EllipseModel,
                min_samples=6,
                residual_threshold=5.0,
                max_trials=600,
                is_model_valid=_valid,
                rng=12,
            )
        except Exception:
            return None, None, None

        if model is None or inliers is None:
            return None, None, None
        if int(inliers.sum()) < 80:
            return None, None, None

        params = model.params
        C = self._ellipse_to_conic(params)
        return C, params, center_line

    def _solve_with_center_circle_auto(self, line_mask: np.ndarray, merged_lines: list) -> tuple:
        """
        Estymuje H z kola srodkowego i linii srodkowej.

        Punkty automatyczne:
        - przeciecia linii srodkowej z elipsa: obrazy (0, -R) i (0, R),
        - przeciecia srednicy sprzezonej z elipsa: obrazy (-R, 0) i (R, 0).
        Potem wybierana jest orientacja o najlepszym score linii boiska.
        """
        C, params, center_line = self.detect_center_circle(line_mask, merged_lines)
        if C is None:
            return None, 0.0, None

        l_center = self._line_homogeneous(center_line["p1"], center_line["p2"])
        ns = self._intersect_line_conic(C, l_center)
        if len(ns) != 2:
            return None, 0.0, params

        direction = center_line["p2"] - center_line["p1"]
        norm = np.hypot(direction[0], direction[1])
        if norm < 1e-9:
            return None, 0.0, params
        direction = direction / norm

        # Aproksymacja punktu kierunkowego linii srodkowej w obrazie.
        # W ujeciach transmisyjnych wystarcza do odzyskania srednicy poprzecznej.
        vp_dir = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        polar = C @ vp_dir
        ew = self._intersect_line_conic(C, polar)
        if len(ew) != 2:
            return None, 0.0, params

        R = 9.15
        line_mask_small = cv2.resize(line_mask, (self._score_w, self._score_h), interpolation=cv2.INTER_NEAREST)

        best_H, best_score = None, 0.0
        for ns_sign in (1, -1):
            for ew_sign in (1, -1):
                img_pts = np.array([ns[0], ns[1], ew[0], ew[1]], dtype=np.float64)
                model_pts = np.array([
                    (0.0, -R * ns_sign),
                    (0.0, R * ns_sign),
                    (-R * ew_sign, 0.0),
                    (R * ew_sign, 0.0),
                ], dtype=np.float64)
                H, _ = cv2.findHomography(img_pts, model_pts, 0)
                if H is None:
                    continue
                if abs(H[2, 2]) > 1e-12:
                    H = H / H[2, 2]
                if not self._validate_homography(H):
                    continue
                score = self._score_homography(H, line_mask_small)
                if score > best_score:
                    best_H = H.copy()
                    best_score = float(score)

        if best_H is None or best_score < self.circle_min_score:
            return None, best_score, params
        return best_H, best_score, params

    # ------------------------------------------------------------------
    # Automatyczny pipeline
    # ------------------------------------------------------------------

    def estimate_from_image_debug(self, image: np.ndarray) -> dict:
        """Zwraca H oraz wszystkie elementy diagnostyczne do notebooka."""
        if image.shape[1] != self.image_width or image.shape[0] != self.image_height:
            image = cv2.resize(image, (self.image_width, self.image_height))

        # Sciezka 1: mocniejsza maska do automatycznego kola srodkowego.
        field_mask, _ = self.segment_field(image)
        line_mask = self.extract_line_mask(image, field_mask)
        segments = self.detect_segments(line_mask)
        merged = self.merge_collinear(segments, angle_tol=6.0, dist_tol=20.0, min_total_length=40.0)
        group_a, group_b = self.group_two_directions(merged) if len(merged) else ([], [])
        H_circle, score_circle, circle_params = self._solve_with_center_circle_auto(line_mask, merged)

        # Sciezka 2: oryginalny VP-RANSAC z V2 jako fallback.
        # Uzywa oryginalnej, bardziej konserwatywnej maski, bo dla ujec bez kola
        # daje mniej szumu i stabilniejsze grupowanie linii.
        field_mask_vp, _ = ClassicalHomographyV2.segment_field(self, image)
        line_mask_vp = ClassicalHomographyV2.extract_line_mask(self, image, field_mask_vp)
        segments_vp = self.detect_segments(line_mask_vp)
        merged_vp = ClassicalHomographyV2.merge_collinear(self, segments_vp)
        group_a_vp, group_b_vp = self.group_two_directions(merged_vp) if len(merged_vp) else ([], [])

        H_vp, score_vp = None, 0.0
        if len(group_a_vp) >= 2 and len(group_b_vp) >= 2:
            H_vp, score_vp = ClassicalHomographyV2.estimate_homography(self, line_mask_vp, group_a_vp, group_b_vp)

        method = None
        H, score = None, 0.0
        if H_circle is not None and score_circle >= score_vp:
            H, score, method = H_circle, score_circle, "CENTER_CIRCLE_AUTO"
        elif H_vp is not None:
            H, score, method = H_vp, score_vp, "VP_RANSAC_AUTO"

        if H is not None and self.refine_steps > 0:
            # Refine na masce zgodnej z wybrana metoda.
            refine_mask = line_mask if method == "CENTER_CIRCLE_AUTO" else line_mask_vp
            H, score = self.refine_homography(H, refine_mask)
            method = f"{method}+REFINE"

        if H is None or score < self.min_score:
            H = None
            method = "FAILED"

        return {
            "H": H,
            "score": float(score),
            "method": method,
            "field_mask": field_mask,
            "line_mask": line_mask,
            "segments": segments,
            "merged_lines": merged,
            "group_a": group_a,
            "group_b": group_b,
            "field_mask_vp": field_mask_vp,
            "line_mask_vp": line_mask_vp,
            "segments_vp": segments_vp,
            "merged_lines_vp": merged_vp,
            "group_a_vp": group_a_vp,
            "group_b_vp": group_b_vp,
            "circle_params": circle_params,
            "score_circle": float(score_circle),
            "score_vp": float(score_vp),
        }

    def estimate_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        debug = self.estimate_from_image_debug(image)
        self._log(
            f"method={debug['method']}, score={debug['score']:.3f}, "
            f"circle={debug['score_circle']:.3f}, vp={debug['score_vp']:.3f}, "
            f"segments={len(debug['segments'])}, lines={len(debug['merged_lines'])}, "
            f"groups={len(debug['group_a'])}/{len(debug['group_b'])}"
        )
        return debug["H"]
