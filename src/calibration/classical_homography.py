"""
Plik: src/calibration/classical_homography.py

Opis:
    Klasyczna estymacja homografii boiska pilkarskiego oparta na
    detekcji linii (Canny + Hough) i RANSAC z ocena jakosci
    przez nakladanie modelu linii boiska.

    Pipeline wewnetrzny:
        1. Segmentacja kolorystyczna pola (zielone -> maska boiska)
        2. Ekstrakcja bialych linii (HSV thresholding)
        3. Detekcja odcinkow linii (HoughLinesP)
        4. Wyznaczanie punktow przeciec linii + naroza (Shi-Tomasi)
        5. RANSAC: losowanie 4 par korespondencji (obraz<->boisko),
           obliczanie H, ocena przez nakladanie modelu linii na maske
        6. Zwrot najlepszej homografii

    Interfejs identyczny z TVCalibHomography (drop-in replacement).

Uzycie:
    from src.calibration.classical_homography import ClassicalHomography

    calibrator = ClassicalHomography(image_width=1280, image_height=720)
    H = calibrator.get_homography("sciezka/do/obrazu.jpg")
    pitch_coords = calibrator.project_point_to_pitch((x_px, y_px), H)
"""

import math
from typing import Optional

import cv2
import numpy as np

# Wymiary boiska FIFA w metrach
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


class ClassicalHomography:
    """
    Estymuje macierz homografii H (3x3) metoda klasyczna:
    detekcja linii + RANSAC z ocena jakosci przez nakladanie modelu boiska.

    H mapuje: wspolrzedne pikselowe obrazu -> metry na boisku.
    Uklad boiska: srodek = (0, 0), x: [-52.5, 52.5], y: [-34, 34].

    Parametry:
        image_width       : szerokosc obrazu wejsciowego (px)
        image_height      : wysokosc obrazu wejsciowego (px)
        ransac_iterations : liczba iteracji RANSAC
        min_score         : minimalny score F1 zeby zaakceptowac homografie
    """

    # Rozdzielczosc do scoringu (mniejsza = szybszy scoring)
    _SCORE_SCALE = 4

    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        ransac_iterations: int = 2000,
        min_score: float = 0.12,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.ransac_iterations = ransac_iterations
        self.min_score = min_score

        self._score_w = image_width // self._SCORE_SCALE
        self._score_h = image_height // self._SCORE_SCALE

        self._build_pitch_model()

        print(f"[ClassicalHomography] Inicjalizacja (obraz {image_width}x{image_height})")
        print(f"[ClassicalHomography] RANSAC: {ransac_iterations} iter, min_score: {min_score}")

    # ------------------------------------------------------------------
    # Model boiska (linie + keypoints)
    # ------------------------------------------------------------------

    def _build_pitch_model(self):
        """Definicja linii i keypointow boiska w metrach."""
        L = PITCH_LENGTH_M / 2   # 52.5
        W = PITCH_WIDTH_M / 2    # 34.0
        PA_D = 16.5              # glebokosc pola karnego
        PA_HW = 20.16            # polowa szerokosci pola karnego
        GA_D = 5.5               # glebokosc pola bramkowego
        GA_HW = 9.16             # polowa szerokosci pola bramkowego
        CC_R = 9.15              # promien kola srodkowego
        PS_D = 11.0              # odleglosc punktu karnego od linii bramkowej

        # Linie boiska jako polyline [(x,y), ...]
        self.pitch_lines = [
            # Granice
            [(-L, -W), (L, -W)],      # gorna
            [(-L, W), (L, W)],         # dolna
            [(-L, -W), (-L, W)],       # lewa
            [(L, -W), (L, W)],         # prawa
            # Linia srodkowa
            [(0, -W), (0, W)],
            # Lewe pole karne
            [(-L, -PA_HW), (-L + PA_D, -PA_HW)],
            [(-L, PA_HW), (-L + PA_D, PA_HW)],
            [(-L + PA_D, -PA_HW), (-L + PA_D, PA_HW)],
            # Prawe pole karne
            [(L, -PA_HW), (L - PA_D, -PA_HW)],
            [(L, PA_HW), (L - PA_D, PA_HW)],
            [(L - PA_D, -PA_HW), (L - PA_D, PA_HW)],
            # Lewe pole bramkowe
            [(-L, -GA_HW), (-L + GA_D, -GA_HW)],
            [(-L, GA_HW), (-L + GA_D, GA_HW)],
            [(-L + GA_D, -GA_HW), (-L + GA_D, GA_HW)],
            # Prawe pole bramkowe
            [(L, -GA_HW), (L - GA_D, -GA_HW)],
            [(L, GA_HW), (L - GA_D, GA_HW)],
            [(L - GA_D, -GA_HW), (L - GA_D, GA_HW)],
        ]

        # Kolo srodkowe (aproksymacja 36 segmentami)
        n_seg = 36
        circle = []
        for i in range(n_seg + 1):
            angle = 2 * math.pi * i / n_seg
            circle.append((CC_R * math.cos(angle), CC_R * math.sin(angle)))
        self.pitch_lines.append(circle)

        # Luki karne
        arc_half = math.acos((PA_D - PS_D) / CC_R)

        # Lewy luk karny (srodek: (-L+PS_D, 0), r=CC_R, poza polem karnym)
        left_arc = []
        for i in range(13):
            a = -arc_half + 2 * arc_half * i / 12
            x = (-L + PS_D) + CC_R * math.cos(a)
            y = CC_R * math.sin(a)
            left_arc.append((x, y))
        self.pitch_lines.append(left_arc)

        # Prawy luk karny
        right_arc = []
        for i in range(13):
            a = math.pi - arc_half + 2 * arc_half * i / 12
            x = (L - PS_D) + CC_R * math.cos(a)
            y = CC_R * math.sin(a)
            right_arc.append((x, y))
        self.pitch_lines.append(right_arc)

        # Keypoints boiska (przeciecia linii uzywane w RANSAC)
        self.pitch_keypoints = np.array([
            # Narozniki boiska
            (-L, -W), (L, -W), (-L, W), (L, W),
            # Linia srodkowa x granice
            (0, -W), (0, W),
            # Srodek
            (0, 0),
            # Narozniki lewego pola karnego
            (-L + PA_D, -PA_HW), (-L + PA_D, PA_HW),
            (-L, -PA_HW), (-L, PA_HW),
            # Narozniki prawego pola karnego
            (L - PA_D, -PA_HW), (L - PA_D, PA_HW),
            (L, -PA_HW), (L, PA_HW),
            # Narozniki lewego pola bramkowego
            (-L + GA_D, -GA_HW), (-L + GA_D, GA_HW),
            # Narozniki prawego pola bramkowego
            (L - GA_D, -GA_HW), (L - GA_D, GA_HW),
            # Kolo srodkowe x linia srodkowa
            (0, -CC_R), (0, CC_R),
        ], dtype=np.float64)

        # Przeskalowane linie do rozdzielczosci scoringu
        self._score_lines = []
        for polyline in self.pitch_lines:
            self._score_lines.append(np.array(polyline, dtype=np.float64))

    # ------------------------------------------------------------------
    # Ekstrakcja linii z obrazu
    # ------------------------------------------------------------------

    def _extract_line_mask(self, image: np.ndarray) -> np.ndarray:
        """Zwraca binarna maske bialych linii boiska."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Segmentacja zielonego pola
        green = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
        k7 = np.ones((7, 7), np.uint8)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k7, iterations=2)
        green = cv2.dilate(green, k7, iterations=3)

        # Biale linie (wysokie V, niskie S) w obrebie pola
        white = cv2.inRange(hsv, (0, 0, 185), (180, 55, 255))
        line_mask = cv2.bitwise_and(white, green)

        # Oczyszczenie morfologiczne
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, k3)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, k5)

        return line_mask

    # ------------------------------------------------------------------
    # Wyznaczanie kandydatow na keypoints
    # ------------------------------------------------------------------

    def _find_candidate_points(self, line_mask: np.ndarray) -> np.ndarray:
        """Znajduje kandydatow na keypoints z maski linii."""
        candidates = []

        # 1. Narozniki Shi-Tomasi na masce linii
        corners = cv2.goodFeaturesToTrack(
            line_mask, maxCorners=80, qualityLevel=0.01, minDistance=15
        )
        if corners is not None:
            for c in corners:
                candidates.append(c[0])

        # 2. Przeciecia odcinkow z HoughLinesP
        edges = cv2.Canny(line_mask, 50, 150)
        segments = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=40, minLineLength=40, maxLineGap=15,
        )

        if segments is not None:
            lines = segments.reshape(-1, 4)
            for i in range(len(lines)):
                for j in range(i + 1, min(len(lines), i + 60)):
                    pt = self._line_intersection(lines[i], lines[j])
                    if pt is None:
                        continue
                    px, py = pt
                    if 0 <= px < self.image_width and 0 <= py < self.image_height:
                        ix, iy = int(px), int(py)
                        if self._near_line(line_mask, ix, iy, radius=8):
                            candidates.append(pt)

        if not candidates:
            return np.empty((0, 2))

        candidates = np.array(candidates, dtype=np.float64)
        candidates = self._cluster_points(candidates, min_dist=10)
        return candidates

    @staticmethod
    def _line_intersection(seg1, seg2):
        """Przeciecie dwoch prostych (rozszerzonych odcinkow). Zwraca (x,y) lub None."""
        x1, y1, x2, y2 = seg1.astype(float)
        x3, y3, x4, y4 = seg2.astype(float)

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        # Kat miedzy liniami > 15 stopni
        a1 = math.atan2(y2 - y1, x2 - x1)
        a2 = math.atan2(y4 - y3, x4 - x3)
        diff = abs(a1 - a2) % math.pi
        if diff < math.radians(15) or diff > math.pi - math.radians(15):
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (px, py)

    @staticmethod
    def _near_line(mask, x, y, radius=5):
        """Sprawdza czy w sasiedztwie (x,y) jest bialy piksel na masce."""
        h, w = mask.shape
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        return np.any(mask[y0:y1, x0:x1] > 0)

    @staticmethod
    def _cluster_points(points, min_dist=10):
        """Usuwanie bliskich duplikatow (zachowaj pierwszy z klastra)."""
        if len(points) == 0:
            return points
        result = [points[0]]
        for p in points[1:]:
            dists = np.linalg.norm(np.array(result) - p, axis=1)
            if np.min(dists) > min_dist:
                result.append(p)
        return np.array(result)

    # ------------------------------------------------------------------
    # Scoring homografii
    # ------------------------------------------------------------------

    def _score_homography(self, H: np.ndarray, line_mask_small: np.ndarray) -> float:
        """
        Ocenia jakosc homografii H przez porownanie
        rzutowanego modelu linii z wykryta maska linii.

        Operuje na zmniejszonej rozdzielczosci (_SCORE_SCALE) dla szybkosci.

        Zwraca F1 score nakladania linii [0..1].
        """
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return 0.0

        sw, sh = self._score_w, self._score_h
        scale = self._SCORE_SCALE
        projected = np.zeros((sh, sw), dtype=np.uint8)
        n_visible = 0

        for polyline in self._score_lines:
            img_pts = []
            for xm, ym in polyline:
                p = H_inv @ np.array([xm, ym, 1.0])
                if abs(p[2]) < 1e-10:
                    continue
                p /= p[2]
                # Przeskalowanie do rozdzielczosci scoringu
                px = int(round(p[0] / scale))
                py = int(round(p[1] / scale))
                if -50 <= px < sw + 50 and -50 <= py < sh + 50:
                    img_pts.append((px, py))

            for k in range(len(img_pts) - 1):
                p1 = (max(-50, min(sw + 50, img_pts[k][0])),
                       max(-50, min(sh + 50, img_pts[k][1])))
                p2 = (max(-50, min(sw + 50, img_pts[k + 1][0])),
                       max(-50, min(sh + 50, img_pts[k + 1][1])))
                cv2.line(projected, p1, p2, 255, 1)
                n_visible += 1

        if n_visible == 0:
            return 0.0

        # Clip do rozmiaru
        projected = projected[:sh, :sw]

        # Dylatacja obu masek dla tolerancji
        k5 = np.ones((5, 5), np.uint8)
        proj_d = cv2.dilate(projected, k5)
        mask_d = cv2.dilate(line_mask_small, k5)

        inter = np.sum((proj_d > 0) & (mask_d > 0))
        proj_area = max(np.sum(proj_d > 0), 1)
        mask_area = max(np.sum(mask_d > 0), 1)

        precision = inter / proj_area
        recall = inter / mask_area
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # Estymacja homografii (RANSAC)
    # ------------------------------------------------------------------

    def get_homography(self, image_path: str) -> Optional[np.ndarray]:
        """
        Glowna metoda — estymuje homografie H (3x3) dla obrazu.

        Parametry:
            image_path : sciezka do obrazu .jpg / .png

        Zwraca:
            np.ndarray (3x3) lub None jesli estymacja sie nie powiodla
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ClassicalHomography] BLAD: nie mozna wczytac {image_path}")
            return None

        if image.shape[1] != self.image_width or image.shape[0] != self.image_height:
            image = cv2.resize(image, (self.image_width, self.image_height))

        line_mask = self._extract_line_mask(image)
        candidates = self._find_candidate_points(line_mask)

        if len(candidates) < 4:
            print(f"[ClassicalHomography] Za malo kandydatow: {len(candidates)}")
            return None

        # Zmniejszona maska do scoringu
        line_mask_small = cv2.resize(
            line_mask, (self._score_w, self._score_h),
            interpolation=cv2.INTER_NEAREST,
        )

        n_cands = len(candidates)
        n_kps = len(self.pitch_keypoints)

        best_H = None
        best_score = 0.0
        rng = np.random.RandomState(42)

        for _ in range(self.ransac_iterations):
            # Losuj 4 kandydatow (z minimalna odlegloscia)
            idx_src = rng.choice(n_cands, size=4, replace=False)
            src = candidates[idx_src]

            # Sprawdz minimalna odleglosc miedzy punktami
            dists = [np.linalg.norm(src[a] - src[b])
                     for a in range(4) for b in range(a + 1, 4)]
            if min(dists) < 20:
                continue

            # Sprawdz niekolinearnosc
            v1 = src[1] - src[0]
            v2 = src[2] - src[0]
            if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 100:
                continue

            # Losuj 4 keypoints boiska
            idx_dst = rng.choice(n_kps, size=4, replace=False)
            dst = self.pitch_keypoints[idx_dst]

            # Probuj 4 cykliczne permutacje dopasowania
            for shift in range(4):
                dst_shifted = np.roll(dst, shift, axis=0)

                H = cv2.getPerspectiveTransform(
                    src.astype(np.float32), dst_shifted.astype(np.float32)
                )
                if H is None:
                    continue

                # Szybkie sanity checks
                det = np.linalg.det(H[:2, :2])
                if det <= 0:
                    continue

                score = self._score_homography(H, line_mask_small)
                if score > best_score:
                    best_score = score
                    best_H = H.copy()

                    # Wczesne zakonczenie przy bardzo dobrym wyniku
                    if best_score > 0.50:
                        break

            if best_score > 0.50:
                break

        if best_H is not None and best_score >= self.min_score:
            print(f"[ClassicalHomography] Homografia znaleziona (F1={best_score:.3f})")
            return best_H
        else:
            print(f"[ClassicalHomography] Nie znaleziono dobrej H (best F1={best_score:.3f})")
            return None

    # ------------------------------------------------------------------
    # Projekcja punktow (identycznie jak TVCalibHomography)
    # ------------------------------------------------------------------

    def project_point_to_pitch(
        self, pixel_point: tuple, H: np.ndarray
    ) -> Optional[tuple]:
        """
        Rzutuje punkt pikselowy na plaszczyzne boiska.

        Parametry:
            pixel_point : (x, y) w pikselach
            H           : macierz homografii 3x3

        Zwraca:
            (x_m, y_m) w metrach lub None jesli punkt poza boiskiem
        """
        px, py = pixel_point
        p_h = np.array([px, py, 1.0], dtype=np.float64)
        world = H @ p_h
        world /= world[2]
        x_m, y_m = float(world[0]), float(world[1])

        if not (
            -PITCH_LENGTH_M / 2 <= x_m <= PITCH_LENGTH_M / 2
            and -PITCH_WIDTH_M / 2 <= y_m <= PITCH_WIDTH_M / 2
        ):
            return None
        return (x_m, y_m)

    def project_detections_to_pitch(
        self, detections: list, H: np.ndarray
    ) -> list:
        """
        Rzutuje liste detekcji na plaszczyzne boiska.

        Parametry:
            detections : lista dict {'bbox': [x1,y1,x2,y2], 'class': int, 'conf': float, ...}
            H          : macierz homografii 3x3

        Zwraca:
            lista dict z dodanym polem 'pitch_coords': (x_m, y_m) lub None
        """
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            pitch_coords = self.project_point_to_pitch((foot_x, foot_y), H)
            results.append({**det, "pitch_coords": pitch_coords})
        return results
