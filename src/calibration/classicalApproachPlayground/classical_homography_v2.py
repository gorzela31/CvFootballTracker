"""
Plik: src/calibration/classical_homography_v2.py

Opis:
    Klasyczna (bez uczenia) estymacja homografii boiska pilkarskiego.

    Potok E2E (jeden obraz -> macierz H 3x3):
        1. Segmentacja pola             segment_field()
        2. Maska bialych linii          extract_line_mask()
        3. Detekcja odcinkow            detect_segments()
        4. Scalanie kolinearnych        merge_collinear()
        5. Grupowanie w 2 kierunki      group_two_directions()
        6. Estymacja H (VP-RANSAC)      estimate_homography()
        7. Refinement (Nelder-Mead)     refine_homography()

    Solidna sciezka alternatywna (gdy auto-korespondencja zawodzi na ubogim ujeciu):
        estimate_homography_from_correspondences(img_pts, model_pts)
        -> cv2.findHomography(RANSAC) z recznie wskazanych / klikni\u0119tych punktow.
    Pomocnicze do debugowania: named_keypoints, detect_intersections(), draw_overlay().

    Metoda estymacji: VP-RANSAC.
        - Z kazdej grupy rownoleglych linii liczymy punkt zbiegu (vanishing point).
        - 2 punkty zbiegu definiuja kierunki osi boiska w obrazie.
        - Wystarcza wtedy 2 korespondencje punktowe (przeciecie linii w obrazie
          <-> znany keypoint boiska), aby wyznaczyc pelna homografie.
        - RANSAC losuje pary korespondencji, ocenia F1 dopasowania modelu linii
          do maski i wybiera najlepsza H.

    H mapuje: wspolrzedne pikselowe obrazu -> metry na boisku.
    Uklad boiska: srodek = (0, 0), x: [-52.5, 52.5], y: [-34, 34].

Uzycie:
    from src.calibration.classical_homography_v2 import ClassicalHomographyV2

    calibrator = ClassicalHomographyV2(image_width=1920, image_height=1080)
    H = calibrator.get_homography("sciezka/do/obrazu.jpg")
    pitch_xy = calibrator.project_point_to_pitch((x_px, y_px), H)
"""

import math
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

try:
    from skimage.measure import EllipseModel, ransac as _sk_ransac
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# Wymiary boiska FIFA w metrach
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


class ClassicalHomographyV2:
    """
    Estymuje macierz homografii H (3x3) metoda klasyczna.

    H mapuje: wspolrzedne pikselowe obrazu -> metry na boisku.

    Parametry:
        image_width       : szerokosc obrazu wejsciowego (px)
        image_height      : wysokosc obrazu wejsciowego (px)
        ransac_iterations : liczba iteracji VP-RANSAC
        min_score         : minimalny F1 zeby zaakceptowac homografie
        refine_steps      : liczba krokow Nelder-Mead refinement (0 = bez)
        verbose           : czy drukowac komunikaty diagnostyczne
    """

    # Skala podprobkowania przy liczeniu F1 (dla szybkosci)
    _SCORE_SCALE = 4

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        ransac_iterations: int = 5000,
        min_score: float = 0.30,
        refine_steps: int = 500,
        verbose: bool = True,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.ransac_iterations = ransac_iterations
        self.min_score = min_score
        self.refine_steps = refine_steps
        self.verbose = verbose

        self._score_w = image_width // self._SCORE_SCALE
        self._score_h = image_height // self._SCORE_SCALE

        self._build_pitch_model()
        self._build_pitch_keypoints()
        self._build_named_keypoints()

        self._log(f"Inicjalizacja (obraz {image_width}x{image_height}, "
                  f"RANSAC {ransac_iterations} iter, min_score {min_score})")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ClassicalHomographyV2] {msg}")

    # ==================================================================
    # Model boiska (metry)
    # ==================================================================

    def _build_pitch_model(self):
        """Definiuje linie boiska (do rysowania i scoringu) w metrach."""
        L = PITCH_LENGTH_M / 2
        W = PITCH_WIDTH_M / 2
        PA_D, PA_HW = 16.5, 20.16   # pole karne
        GA_D, GA_HW = 5.5, 9.16     # pole bramkowe
        CC_R = 9.15                 # kolo srodkowe
        PS_D = 11.0                 # punkt karny

        self.pitch_lines = [
            [(-L, -W), (L, -W)],
            [(-L, W), (L, W)],
            [(-L, -W), (-L, W)],
            [(L, -W), (L, W)],
            [(0, -W), (0, W)],
            [(-L, -PA_HW), (-L + PA_D, -PA_HW)],
            [(-L, PA_HW), (-L + PA_D, PA_HW)],
            [(-L + PA_D, -PA_HW), (-L + PA_D, PA_HW)],
            [(L, -PA_HW), (L - PA_D, -PA_HW)],
            [(L, PA_HW), (L - PA_D, PA_HW)],
            [(L - PA_D, -PA_HW), (L - PA_D, PA_HW)],
            [(-L, -GA_HW), (-L + GA_D, -GA_HW)],
            [(-L, GA_HW), (-L + GA_D, GA_HW)],
            [(-L + GA_D, -GA_HW), (-L + GA_D, GA_HW)],
            [(L, -GA_HW), (L - GA_D, -GA_HW)],
            [(L, GA_HW), (L - GA_D, GA_HW)],
            [(L - GA_D, -GA_HW), (L - GA_D, GA_HW)],
        ]

        # Kolo srodkowe
        n_seg = 36
        circle = [
            (CC_R * math.cos(2 * math.pi * i / n_seg),
             CC_R * math.sin(2 * math.pi * i / n_seg))
            for i in range(n_seg + 1)
        ]
        self.pitch_lines.append(circle)

        # Luki pol karnych
        arc_half = math.acos((PA_D - PS_D) / CC_R)
        left_arc = []
        for i in range(13):
            a = -arc_half + 2 * arc_half * i / 12
            left_arc.append(((-L + PS_D) + CC_R * math.cos(a), CC_R * math.sin(a)))
        self.pitch_lines.append(left_arc)

        right_arc = []
        for i in range(13):
            a = math.pi - arc_half + 2 * arc_half * i / 12
            right_arc.append(((L - PS_D) + CC_R * math.cos(a), CC_R * math.sin(a)))
        self.pitch_lines.append(right_arc)

        self._score_lines = [np.array(p, dtype=np.float64) for p in self.pitch_lines]

        # Linie podzielone na 2 kierunki (do liczenia punktow zbiegu modelu)
        self._pitch_along = [   # rownolegle do linii bocznych (os X)
            ("top_side",     [(-L, -W),     (L, -W)]),
            ("bot_side",     [(-L,  W),     (L,  W)]),
            ("pa_left_top",  [(-L, -PA_HW), (-L + PA_D, -PA_HW)]),
            ("pa_left_bot",  [(-L,  PA_HW), (-L + PA_D,  PA_HW)]),
            ("pa_right_top", [(L, -PA_HW),  (L - PA_D, -PA_HW)]),
            ("pa_right_bot", [(L,  PA_HW),  (L - PA_D,  PA_HW)]),
        ]
        self._pitch_across = [  # prostopadle (os Y)
            ("left_side",      [(-L, -W),          (-L,  W)]),
            ("right_side",     [(L,  -W),          (L,   W)]),
            ("center_line",    [(0,  -W),          (0,   W)]),
            ("pa_left_front",  [(-L + PA_D, -PA_HW), (-L + PA_D, PA_HW)]),
            ("pa_right_front", [(L - PA_D, -PA_HW),  (L - PA_D,  PA_HW)]),
        ]

    def _build_pitch_keypoints(self):
        """Charakterystyczne punkty boiska (przeciecia linii) w metrach."""
        L = PITCH_LENGTH_M / 2
        W = PITCH_WIDTH_M / 2
        PA_D, PA_HW = 16.5, 20.16
        GA_D, GA_HW = 5.5, 9.16
        CC_R = 9.15

        self.pitch_keypoints = np.array([
            (-L, -W), (L, -W), (-L, W), (L, W),          # naroza
            (0, -W), (0, W),                              # przeciecia linii srodkowej
            (-L + PA_D, -PA_HW), (-L + PA_D, PA_HW),      # rogi pola karnego (lewe)
            (-L, -PA_HW), (-L, PA_HW),
            (L - PA_D, -PA_HW), (L - PA_D, PA_HW),        # rogi pola karnego (prawe)
            (L, -PA_HW), (L, PA_HW),
            (-L + GA_D, -GA_HW), (-L + GA_D, GA_HW),      # pole bramkowe (lewe)
            (-L, -GA_HW), (-L, GA_HW),
            (L - GA_D, -GA_HW), (L - GA_D, GA_HW),        # pole bramkowe (prawe)
            (L, -GA_HW), (L, GA_HW),
            (0, -CC_R), (0, CC_R),                        # kolo srodkowe (gora/dol)
        ], dtype=np.float64)

    # ==================================================================
    # Geometria pomocnicza
    # ==================================================================

    @staticmethod
    def _line_homogeneous(p1, p2) -> np.ndarray:
        """Linia w postaci jednorodnej: l = p1 x p2."""
        return np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])

    @staticmethod
    def _segment_angle(seg) -> float:
        """Kat odcinka w stopniach [0, 180)."""
        x1, y1, x2, y2 = seg
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

    @staticmethod
    def _segment_length(seg) -> float:
        x1, y1, x2, y2 = seg
        return math.hypot(x2 - x1, y2 - y1)

    # ==================================================================
    # KROK 1: Segmentacja pola
    # ==================================================================

    def segment_field(self, image: np.ndarray) -> tuple:
        """
        Wyznacza maske pola gry (zielona murawa).

        Zwraca:
            field_mask : binarna maska obszaru boiska (po convex hull + erozji)
            hull       : wypukla otoczka najwiekszego zielonego konturu (lub None)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        green = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
        k7 = np.ones((7, 7), np.uint8)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k7, iterations=3)

        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((h, w), dtype=np.uint8), None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < h * w * 0.15:
            # Za malo zieleni — uzyj dylatowanej maski zamiast hull
            field_mask = cv2.dilate(green, k7, iterations=3)
            return field_mask, None

        hull = cv2.convexHull(largest)
        field_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(field_mask, [hull], 255)
        # Erozja odcina banery reklamowe tuz przy krawedzi boiska
        k11 = np.ones((11, 11), np.uint8)
        field_mask = cv2.erode(field_mask, k11, iterations=2)
        return field_mask, hull

    # ==================================================================
    # KROK 2: Maska bialych linii
    # ==================================================================

    def extract_line_mask(self, image: np.ndarray,
                          field_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Wyznacza binarna maske bialych linii boiska w obrebie pola.

        Filtruje komponenty po ksztalcie, by odrzucic napisy/numery/reklamy
        i zachowac tylko cienkie, wydluzone struktury (linie).
        """
        if field_mask is None:
            field_mask, _ = self.segment_field(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 175), (180, 65, 255))
        line_mask = cv2.bitwise_and(white, field_mask)

        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, k3)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, k5)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            line_mask, connectivity=8
        )
        filtered = np.zeros_like(line_mask)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area < 20:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            thin_dim = min(bw, bh)

            # Zawodnik to zazwyczaj obiekt wyższy niż szerszy, ale grubszy niż zwykła linia
            is_vertical_player = (bh > bw * 1.2) and (thin_dim > 6) and (area > 30)
            
            is_line = (aspect >= 4.0
                       or (thin_dim <= 12 and area < 800)
                       or area < 150)
            is_blob = ((aspect < 2.0 and area > 300)
                       or (area > 2000 and aspect < 3.0)
                       or is_vertical_player) # Dodajemy zawodników do blobów
            if is_line and not is_blob:
                filtered[labels == i] = 255
        return filtered

    # ==================================================================
    # KROK 3: Detekcja odcinkow
    # ==================================================================

    def detect_segments(self, line_mask: np.ndarray) -> np.ndarray:
        """Wykrywa odcinki linii (HoughLinesP). Zwraca tablice Nx4 (x1,y1,x2,y2)."""
        edges = cv2.Canny(line_mask, 50, 150)
        segments = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=20, minLineLength=20, maxLineGap=25,
        )
        if segments is None:
            return np.empty((0, 4), dtype=np.int32)
        return segments.reshape(-1, 4)

    # ==================================================================
    # KROK 4: Scalanie kolinearnych odcinkow
    # ==================================================================

    def merge_collinear(self, segments: np.ndarray,
                        angle_tol: float = 5.0,
                        dist_tol: float = 15.0,
                        min_total_length: float = 30.0) -> list:
        """
        Scala kolinearne odcinki w pelne linie.

        Zwraca liste slownikow z polami:
            p1, p2          : koncowe punkty linii (np.ndarray 2D)
            angle           : sredni kat [stopnie]
            mean, direction : srodek i kierunek (z PCA)
            total_length    : laczna dlugosc skladowych odcinkow
            n_segments      : liczba scalonych odcinkow
        """
        if len(segments) == 0:
            return []

        seg_data = []
        for s in segments:
            x1, y1, x2, y2 = s
            seg_data.append({
                'angle': self._segment_angle(s),
                'mid': ((x1 + x2) / 2, (y1 + y2) / 2),
                'length': self._segment_length(s),
                'x1': float(x1), 'y1': float(y1),
                'x2': float(x2), 'y2': float(y2),
            })
        seg_data.sort(key=lambda d: d['angle'])

        groups, used = [], [False] * len(seg_data)
        for i, si in enumerate(seg_data):
            if used[i]:
                continue
            group = [si]
            used[i] = True
            for j in range(i + 1, len(seg_data)):
                if used[j]:
                    continue
                sj = seg_data[j]
                adiff = abs(si['angle'] - sj['angle'])
                adiff = min(adiff, 180 - adiff)
                if adiff > angle_tol:
                    continue
                dx, dy = si['x2'] - si['x1'], si['y2'] - si['y1']
                L = math.hypot(dx, dy)
                if L < 1e-6:
                    continue
                dist = abs(dy * sj['mid'][0] - dx * sj['mid'][1] +
                           si['x2'] * si['y1'] - si['y2'] * si['x1']) / L
                if dist < dist_tol:
                    group.append(sj)
                    used[j] = True
            groups.append(group)

        merged = []
        for group in groups:
            pts = np.array([(s['x1'], s['y1']) for s in group] +
                           [(s['x2'], s['y2']) for s in group], dtype=np.float64)
            mean = pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts - mean)
            direction = Vt[0]
            proj = (pts - mean) @ direction
            p1 = mean + proj.min() * direction
            p2 = mean + proj.max() * direction
            total_length = sum(s['length'] for s in group)
            if total_length >= min_total_length:
                merged.append({
                    'p1': p1, 'p2': p2,
                    'angle': float(np.mean([s['angle'] for s in group])),
                    'mean': mean, 'direction': direction,
                    'n_segments': len(group), 'total_length': total_length,
                })
        merged.sort(key=lambda m: m['total_length'], reverse=True)
        return merged

    # ==================================================================
    # KROK 5: Grupowanie w 2 kierunki
    # ==================================================================

    def group_two_directions(self, merged_lines: list) -> tuple:
        """Dzieli linie na 2 grupy kierunkowe (k-means na katach 2*theta)."""
        if len(merged_lines) < 4:
            if len(merged_lines) < 2:
                return merged_lines, []
            median_a = np.median([m['angle'] for m in merged_lines])
            ga = [m for m in merged_lines if abs(m['angle'] - median_a) < 30]
            gb = [m for m in merged_lines if abs(m['angle'] - median_a) >= 30]
            return ga, gb

        angles = np.array([m['angle'] for m in merged_lines])
        weights = np.array([m['total_length'] for m in merged_lines])
        vecs = np.column_stack([np.cos(2 * np.radians(angles)),
                                np.sin(2 * np.radians(angles))])
        labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(
            vecs, sample_weight=weights)
        ga = [m for m, l in zip(merged_lines, labels) if l == 0]
        gb = [m for m, l in zip(merged_lines, labels) if l == 1]
        return ga, gb

    # ==================================================================
    # Punkty zbiegu (vanishing points)
    # ==================================================================

    def compute_vanishing_point(self, lines: list) -> Optional[np.ndarray]:
        """Punkt zbiegu grupy rownoleglych linii (wsp. jednorodne)."""
        if len(lines) < 2:
            return None
        homo = [self._line_homogeneous(m['p1'], m['p2']) for m in lines]
        if len(homo) == 2:
            return np.cross(homo[0], homo[1])
        weights = np.array([m['total_length'] for m in lines])
        A = np.array(homo) * weights[:, np.newaxis]
        _, _, Vt = np.linalg.svd(A)
        return Vt[-1]

    def _line_intersections(self, group_a: list, group_b: list) -> np.ndarray:
        """Przeciecia kazdej linii grupy A z kazda linia grupy B (w obrazie)."""
        homo_a = [self._line_homogeneous(m['p1'], m['p2']) for m in group_a]
        homo_b = [self._line_homogeneous(m['p1'], m['p2']) for m in group_b]
        pts = []
        for la in homo_a:
            for lb in homo_b:
                p = np.cross(la, lb)
                if abs(p[2]) < 1e-10:
                    continue
                p = p / p[2]
                if (-self.image_width <= p[0] <= 2 * self.image_width and
                        -self.image_height <= p[1] <= 2 * self.image_height):
                    pts.append(p[:2])
        if not pts:
            return np.empty((0, 2))
        return np.array(pts, dtype=np.float64)

    def _solve_h_from_vps(self, vp_x: np.ndarray, vp_y: np.ndarray,
                          img_pts: np.ndarray, model_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Wyznacza H z 2 punktow zbiegu + 2 korespondencji punktowych.

        Buduje H^{-1} = [alpha*vp_x | beta*vp_y | t], t3 = 1.
        4 rownania (2 punkty x 2 wspolrzedne), 4 niewiadome (alpha,beta,t1,t2).
        """
        vx, vy = vp_x.astype(np.float64), vp_y.astype(np.float64)
        A = np.zeros((4, 4)); b = np.zeros(4)
        for i in range(2):
            u, v = float(img_pts[i][0]), float(img_pts[i][1])
            X, Y = float(model_pts[i][0]), float(model_pts[i][1])
            A[2 * i] = [X * (vx[0] - u * vx[2]), Y * (vy[0] - u * vy[2]), 1.0, 0.0]
            b[2 * i] = u
            A[2 * i + 1] = [X * (vx[1] - v * vx[2]), Y * (vy[1] - v * vy[2]), 0.0, 1.0]
            b[2 * i + 1] = v
        try:
            alpha, beta, t1, t2 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        if abs(alpha) < 1e-12 or abs(beta) < 1e-12:
            return None
        H_inv = np.array([
            [alpha * vx[0], beta * vy[0], t1],
            [alpha * vx[1], beta * vy[1], t2],
            [alpha * vx[2], beta * vy[2], 1.0],
        ])
        try:
            H = np.linalg.inv(H_inv)
        except np.linalg.LinAlgError:
            return None
        if abs(H[2, 2]) > 1e-10:
            H /= H[2, 2]
        return H

    # ==================================================================
    # KROK 6: Estymacja homografii (VP-RANSAC)
    # ==================================================================

    def estimate_homography(self, line_mask: np.ndarray,
                            group_a: list, group_b: list) -> tuple:
        """
        Pelny automat. Najpierw probuje metody z kolem srodkowym (gdy widoczne),
        potem VP-RANSAC. Zwraca (H, F1) najlepszej metody lub (None, 0.0).
        """
        merged = list(group_a) + list(group_b)
        H_c, s_c = self._solve_with_circle(line_mask, group_a, group_b, merged)
        if H_c is not None:
            self._log(f"Metoda kola: F1={s_c:.3f}")
        H_v, s_v = self._estimate_vp_ransac(line_mask, group_a, group_b)
        if H_v is not None:
            self._log(f"Metoda VP-RANSAC: F1={s_v:.3f}")
        if s_c >= s_v and H_c is not None:
            return H_c, s_c
        return H_v, s_v

    def _estimate_vp_ransac(self, line_mask: np.ndarray,
                            group_a: list, group_b: list) -> tuple:
        """
        VP-RANSAC: estymuje H wykorzystujac 2 punkty zbiegu + 2 korespondencje.

        Zwraca (H, F1) lub (None, 0.0).
        """
        vp_a = self.compute_vanishing_point(group_a)
        vp_b = self.compute_vanishing_point(group_b)
        if vp_a is None or vp_b is None:
            return None, 0.0

        # Kandydaci na punkty obrazu: przeciecia linii (najpewniejsze),
        # uzupelnione narozami Shi-Tomasi.
        candidates = self._line_intersections(group_a, group_b)
        corners = self._find_candidate_points(line_mask)
        if len(candidates) and len(corners):
            candidates = np.vstack([candidates, corners])
        elif len(corners):
            candidates = corners
        n_cands = len(candidates)
        if n_cands < 2:
            return None, 0.0

        line_mask_small = cv2.resize(
            line_mask, (self._score_w, self._score_h),
            interpolation=cv2.INTER_NEAREST)
        n_kps = len(self.pitch_keypoints)

        best_H, best_score = None, 0.0
        rng = np.random.RandomState(42)

        for _ in range(self.ransac_iterations):
            src = candidates[rng.choice(n_cands, size=2, replace=False)]
            if np.linalg.norm(src[0] - src[1]) < 30:
                continue
            dst = self.pitch_keypoints[rng.choice(n_kps, size=2, replace=False)]

            for vp_x, vp_y in ((vp_a, vp_b), (vp_b, vp_a)):
                for perm in ((0, 1), (1, 0)):
                    H = self._solve_h_from_vps(vp_x, vp_y, src, dst[list(perm)])
                    if H is None:
                        continue
                    if np.linalg.det(H[:2, :2]) <= 0:
                        continue
                    if not self._validate_homography(H):
                        continue
                    score = self._score_homography(H, line_mask_small)
                    if score > best_score:
                        best_score, best_H = score, H.copy()

        return best_H, best_score

    def _find_candidate_points(self, line_mask: np.ndarray) -> np.ndarray:
        """Naroza (Shi-Tomasi) na masce linii — uzupelnienie korespondencji."""
        corners = cv2.goodFeaturesToTrack(
            line_mask, maxCorners=80, qualityLevel=0.01, minDistance=15)
        if corners is None:
            return np.empty((0, 2))
        return corners.reshape(-1, 2).astype(np.float64)

    # ==================================================================
    # Scoring + walidacja
    # ==================================================================

    def _score_homography(self, H: np.ndarray, line_mask_small: np.ndarray) -> float:
        """F1 dopasowania rzutu modelu linii do maski (na podprobce)."""
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
                px, py = int(round(p[0] / scale)), int(round(p[1] / scale))
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
        projected = projected[:sh, :sw]

        # Pokrycie obrazu rzutem (kara za model w malym rogu)
        ys, xs = np.where(projected > 0)
        coverage = max((xs.max() - xs.min()) / sw,
                       (ys.max() - ys.min()) / sh) if len(xs) else 0.0

        k3 = np.ones((3, 3), np.uint8)
        proj_d = cv2.dilate(projected, k3)
        mask_d = cv2.dilate(line_mask_small, k3)
        inter = np.sum((proj_d > 0) & (mask_d > 0))
        precision = inter / max(np.sum(proj_d > 0), 1)
        recall = inter / max(np.sum(mask_d > 0), 1)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)

        if coverage < 0.25:
            f1 *= 0.3
        elif coverage < 0.40:
            f1 *= 0.6
        return f1

    def score_homography(self, H: np.ndarray, line_mask: np.ndarray) -> float:
        """Publiczny scoring (pelna maska -> podprobka -> F1)."""
        lm_small = cv2.resize(line_mask, (self._score_w, self._score_h),
                              interpolation=cv2.INTER_NEAREST)
        return self._score_homography(H, lm_small)

    def _validate_homography(self, H: np.ndarray) -> bool:
        """Odrzuca H mapujace obraz na nierealistyczny obszar boiska."""
        iw, ih = self.image_width, self.image_height
        L, W = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2

        center = H @ np.array([iw / 2.0, ih / 2.0, 1.0])
        if abs(center[2]) < 1e-10:
            return False
        center /= center[2]
        if abs(center[0]) > L * 2.5 or abs(center[1]) > W * 2.5:
            return False

        corners = [(0, 0), (iw, 0), (iw, ih), (0, ih)]
        pitch_pts = []
        for cx, cy in corners:
            p = H @ np.array([float(cx), float(cy), 1.0])
            if abs(p[2]) < 1e-10:
                return False
            p /= p[2]
            if abs(p[0]) > L * 4 or abs(p[1]) > W * 4:
                return False
            pitch_pts.append(p[:2])

        pts = np.array(pitch_pts)
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
        area = abs(area) / 2.0
        return 200 <= area <= 25000

    # ==================================================================
    # KROK 7: Refinement (Nelder-Mead)
    # ==================================================================

    def refine_homography(self, H_init: np.ndarray, line_mask: np.ndarray) -> tuple:
        """Lokalna optymalizacja H (Nelder-Mead) maksymalizujaca F1."""
        lm_small = cv2.resize(line_mask, (self._score_w, self._score_h),
                              interpolation=cv2.INTER_NEAREST)

        def neg_score(h_flat):
            return -self._score_homography(h_flat.reshape(3, 3), lm_small)

        h_flat = H_init.flatten()
        initial = -neg_score(h_flat)
        result = minimize(neg_score, h_flat, method='Nelder-Mead',
                          options={'maxiter': self.refine_steps,
                                   'xatol': 1e-10, 'fatol': 1e-6})
        final = -result.fun
        if final > initial:
            return result.x.reshape(3, 3), final
        return H_init, initial

    # ==================================================================
    # Potok E2E
    # ==================================================================

    def get_homography(self, image_path: str) -> Optional[np.ndarray]:
        """Estymuje H (3x3) dla obrazu. Zwraca macierz lub None."""
        image = cv2.imread(image_path)
        if image is None:
            self._log(f"BLAD: nie mozna wczytac {image_path}")
            return None
        if image.shape[1] != self.image_width or image.shape[0] != self.image_height:
            image = cv2.resize(image, (self.image_width, self.image_height))
        return self.estimate_from_image(image)

    def estimate_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Potok E2E na wczytanym obrazie BGR (uzywany tez przez get_homography)."""
        field_mask, _ = self.segment_field(image)
        line_mask = self.extract_line_mask(image, field_mask)

        segments = self.detect_segments(line_mask)
        if len(segments) < 4:
            self._log(f"Za malo odcinkow: {len(segments)}")
            return None

        merged = self.merge_collinear(segments)
        group_a, group_b = self.group_two_directions(merged)
        self._log(f"Odcinki: {len(segments)}, linie: {len(merged)}, "
                  f"grupy: {len(group_a)}/{len(group_b)}")

        if len(group_a) < 2 or len(group_b) < 2:
            self._log("Za malo linii w grupach kierunkowych")
            return None

        H, score = self.estimate_homography(line_mask, group_a, group_b)
        self._log(f"VP-RANSAC F1: {score:.3f}")

        if H is None or score < self.min_score:
            self._log(f"Nie znaleziono H (F1={score:.3f})")
            return None

        if self.refine_steps > 0:
            H, score = self.refine_homography(H, line_mask)

        self._log(f"Homografia znaleziona (F1={score:.3f})")
        return H

    # ==================================================================
    # Nazwane keypointy boiska (do wspomaganej / recznej korespondencji)
    # ==================================================================

    def _build_named_keypoints(self):
        """Slownik {nazwa: (x_m, y_m)} charakterystycznych punktow boiska.

        Uklad: srodek (0,0); lewa bramka x=-L, prawa x=+L; gora obrazu zwykle -y.
        Ulatwia reczne klikanie: wybierasz nazwe i klikasz odpowiadajacy punkt.
        """
        L, W = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2
        PA_D, PA_HW = 16.5, 20.16
        GA_D, GA_HW = 5.5, 9.16
        PS_D = 11.0
        GOAL_HW = 3.66

        kp = {}
        for side, sx in (("L", -L), ("R", L)):
            s = 1 if side == "L" else -1   # kierunek "w glab pola" od linii bramkowej
            kp[f"{side}_corner_top"]          = (sx, -W)
            kp[f"{side}_corner_bot"]          = (sx,  W)
            kp[f"{side}_box_gl_top"]          = (sx, -PA_HW)
            kp[f"{side}_box_gl_bot"]          = (sx,  PA_HW)
            kp[f"{side}_box_front_top"]       = (sx + s * PA_D, -PA_HW)
            kp[f"{side}_box_front_bot"]       = (sx + s * PA_D,  PA_HW)
            kp[f"{side}_goalarea_gl_top"]     = (sx, -GA_HW)
            kp[f"{side}_goalarea_gl_bot"]     = (sx,  GA_HW)
            kp[f"{side}_goalarea_front_top"]  = (sx + s * GA_D, -GA_HW)
            kp[f"{side}_goalarea_front_bot"]  = (sx + s * GA_D,  GA_HW)
            kp[f"{side}_post_top"]            = (sx, -GOAL_HW)
            kp[f"{side}_post_bot"]            = (sx,  GOAL_HW)
            kp[f"{side}_penalty_spot"]        = (sx + s * PS_D, 0.0)
        kp["C_top"]    = (0.0, -W)
        kp["C_bot"]    = (0.0,  W)
        kp["C_center"] = (0.0, 0.0)
        self.named_keypoints = kp

    # ==================================================================
    # Pomocnicze do debugowania / wizualizacji
    # ==================================================================

    def detect_intersections(self, group_a: list, group_b: list) -> np.ndarray:
        """Publiczny wrapper na przeciecia linii grup kierunkowych (w obrazie)."""
        return self._line_intersections(group_a, group_b)

    def draw_overlay(self, image: np.ndarray, H: Optional[np.ndarray],
                     color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
        """Rysuje model linii boiska na obrazie wg H (pixel -> metry)."""
        vis = image.copy()
        if H is None:
            return vis
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return vis
        for polyline in self.pitch_lines:
            pts = []
            for xm, ym in polyline:
                p = H_inv @ np.array([xm, ym, 1.0])
                if abs(p[2]) < 1e-10:
                    continue
                p /= p[2]
                pts.append((int(round(p[0])), int(round(p[1]))))
            for k in range(len(pts) - 1):
                cv2.line(vis, pts[k], pts[k + 1], color, thickness)
        return vis

    # ==================================================================
    # SOLIDNA estymacja H z recznych / klikni\u0119tych korespondencji
    # ==================================================================

    def estimate_homography_from_correspondences(
        self,
        img_pts,
        model_pts,
        line_mask: Optional[np.ndarray] = None,
        ransac_thresh: float = 8.0,
    ) -> tuple:
        """
        Najpewniejsza droga: H z >=4 korespondencji (piksel obrazu -> metry boiska).

        Uzywa cv2.findHomography z RANSAC (gdy >4 punktow), wiec toleruje
        pojedyncze bledne wskazania. Gdy podasz line_mask, zwraca tez F1.

        Parametry:
            img_pts       : lista/tablica (N,2) punktow w pikselach obrazu
            model_pts     : lista/tablica (N,2) odpowiadajacych punktow w metrach
            line_mask     : maska linii do oceny F1 (opcjonalnie)
            ransac_thresh : prog RANSAC w metrach (reprojekcja na boisko)

        Zwraca (H, score) lub (None, 0.0).
        """
        img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        model_pts = np.asarray(model_pts, dtype=np.float64).reshape(-1, 2)
        if len(img_pts) != len(model_pts) or len(img_pts) < 4:
            self._log(f"Za malo/niespojne korespondencje: {len(img_pts)} vs {len(model_pts)}")
            return None, 0.0

        method = cv2.RANSAC if len(img_pts) > 4 else 0
        H, mask = cv2.findHomography(img_pts, model_pts, method, ransac_thresh)
        if H is None:
            return None, 0.0
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        n_in = int(mask.sum()) if mask is not None else len(img_pts)
        score = self.score_homography(H, line_mask) if line_mask is not None else float("nan")
        self._log(f"findHomography: {n_in}/{len(img_pts)} inlierow, F1={score:.3f}")
        return H, score

    # ==================================================================
    # Wykrycie kola srodkowego (elipsa) — pelny automat
    # ==================================================================

    def detect_circle(self, line_mask: np.ndarray, merged_lines: list,
                      max_points: int = 300, max_trials: int = 120):
        """
        Dopasowuje elipse do pikseli kola srodkowego (RANSAC).

        Usuwa piksele lezace na prostych liniach (zostaje luk kola), filtruje
        do srodkowego pasa kadru i dopasowuje elipse z walidacja rozmiaru/pozycji.

        Zwraca (conic_3x3, params) lub (None, None).
        Wymaga scikit-image; bez niego zwraca (None, None).
        """
        if not _HAS_SKIMAGE:
            return None, None
        ys, xs = np.where(line_mask > 0)
        if len(xs) < 60:
            return None, None
        P = np.stack([xs, ys], axis=1).astype(np.float64)

        keep = np.ones(len(P), dtype=bool)
        for m in merged_lines:
            if m["total_length"] < 120:
                continue
            a, b = m["p1"], m["p2"]
            d = b - a
            L = np.hypot(d[0], d[1])
            if L < 1:
                continue
            n = np.array([-d[1], d[0]]) / L
            along = (P - a) @ (d / L)
            perp = np.abs((P - a) @ n)
            keep &= ~((perp < 10) & (along > -30) & (along < L + 30))
        Q = P[keep]
        Q = Q[(Q[:, 1] > 0.12 * self.image_height) & (Q[:, 1] < 0.9 * self.image_height)]
        if len(Q) < 40:
            return None, None
        if len(Q) > max_points:
            idx = np.random.RandomState(0).choice(len(Q), max_points, replace=False)
            Q = Q[idx]

        iw, ih = self.image_width, self.image_height

        def _valid(model, *args):
            try:
                xc, yc, ax1, ax2, th = model.params
            except Exception:
                return False
            lo, hi = sorted([ax1, ax2])
            return (lo > 0.04 * iw and hi < 0.45 * iw and hi / max(lo, 1) < 5.0
                    and 0.03 * iw < xc < 0.97 * iw and 0.1 * ih < yc < 0.95 * ih)

        try:
            model, inliers = _sk_ransac(
                Q, EllipseModel, min_samples=6, residual_threshold=3.0,
                max_trials=max_trials, is_model_valid=_valid, rng=0,
            )
        except Exception:
            return None, None
        if model is None or inliers is None or int(inliers.sum()) < 40:
            return None, None
        params = model.params  # (xc, yc, a, b, theta)
        return self._ellipse_to_conic(params), params

    @staticmethod
    def _ellipse_to_conic(params) -> np.ndarray:
        """Macierz koniki 3x3 (x^T C x = 0) z parametrow elipsy (xc,yc,a,b,theta)."""
        xc, yc, a, b, th = params
        ct, st = math.cos(th), math.sin(th)
        # macierz formy w ukladzie osi elipsy: diag(1/a^2, 1/b^2)
        R = np.array([[ct, -st], [st, ct]])
        D = np.diag([1.0 / (a * a), 1.0 / (b * b)])
        M = R @ D @ R.T          # 2x2 forma kwadratowa
        cxy = np.array([xc, yc])
        C = np.zeros((3, 3))
        C[:2, :2] = M
        C[:2, 2] = C[2, :2] = -M @ cxy
        C[2, 2] = cxy @ M @ cxy - 1.0
        return C

    @staticmethod
    def _intersect_line_conic(C: np.ndarray, line: np.ndarray) -> list:
        """Punkty przeciecia prostej (a,b,c) z konika C (max 2, wsp. kartezjanskie)."""
        a, b, c = line
        if abs(a) >= abs(b):
            p0 = np.array([-c / a, 0.0]) if abs(a) > 1e-9 else np.array([0.0, 0.0])
        else:
            p0 = np.array([0.0, -c / b])
        dir_ = np.array([b, -a], dtype=np.float64)
        nd = np.hypot(dir_[0], dir_[1])
        if nd < 1e-9:
            return []
        dir_ /= nd

        def hom(t):
            return np.array([p0[0] + t * dir_[0], p0[1] + t * dir_[1], 1.0])

        d3 = np.array([dir_[0], dir_[1], 0.0])
        p3 = np.array([p0[0], p0[1], 1.0])
        A = d3 @ C @ d3
        B = 2.0 * (d3 @ C @ p3)
        Cc = p3 @ C @ p3
        if abs(A) < 1e-12:
            return []
        disc = B * B - 4 * A * Cc
        if disc < 0:
            return []
        sq = math.sqrt(disc)
        pts = []
        for t in ((-B + sq) / (2 * A), (-B - sq) / (2 * A)):
            p = hom(t)
            pts.append(p[:2])
        return pts

    def _solve_with_circle(self, line_mask: np.ndarray, group_a: list, group_b: list,
                           merged_lines: list) -> tuple:
        """
        Pelny automat z kola srodkowego:
          - dopasuj elipse (obraz kola R=9.15 m wokol (0,0)),
          - linia srodkowa (najdluzsza linia przez srodek elipsy) ∩ elipsa -> obrazy (0, ±R),
          - biegun kierunku linii srodkowej wzgl. koniki ∩ elipsa -> obrazy (±R, 0),
          - z 4 punktow kola: findHomography; orientacje rozstrzyga F1.
        Zwraca (H, F1) lub (None, 0.0).
        """
        C, params = self.detect_circle(line_mask, merged_lines)
        if C is None:
            return None, 0.0
        xc, yc = params[0], params[1]

        # linia srodkowa = najdluzsza linia przechodzaca najblizej srodka elipsy
        best_line, best_d = None, 1e9
        for m in merged_lines:
            if m["total_length"] < 150:
                continue
            l = self._line_homogeneous(m["p1"], m["p2"])
            nrm = math.hypot(l[0], l[1])
            if nrm < 1e-9:
                continue
            d = abs(l[0] * xc + l[1] * yc + l[2]) / nrm
            if d < best_d:
                best_d, best_line, best_m = d, l, m
        if best_line is None or best_d > 0.2 * self.image_width:
            return None, 0.0

        ns = self._intersect_line_conic(C, best_line)         # (0, ±R)
        vp_center = self.compute_vanishing_point([best_m])    # tylko 1 linia -> brak; uzyj kierunku
        # kierunek linii srodkowej jako punkt w nieskonczonosci
        d_dir = best_m["direction"]
        vp_dir = np.array([d_dir[0], d_dir[1], 0.0])
        polar = C @ vp_dir
        ew = self._intersect_line_conic(C, polar)             # (±R, 0)
        if len(ns) != 2 or len(ew) != 2:
            return None, 0.0

        R = 9.15
        lm_small = cv2.resize(line_mask, (self._score_w, self._score_h),
                              interpolation=cv2.INTER_NEAREST)
        best_H, best_score = None, 0.0
        # orientacje: ktory z NS to -R/+R, ktory z EW to -R/+R, oraz globalny znak x
        for ns_sign in (1, -1):
            for ew_sign in (1, -1):
                img_pts = np.array([ns[0], ns[1], ew[0], ew[1]], dtype=np.float64)
                model_pts = np.array([
                    (0.0, -R * ns_sign), (0.0, R * ns_sign),
                    (-R * ew_sign, 0.0), (R * ew_sign, 0.0),
                ], dtype=np.float64)
                H, _ = cv2.findHomography(img_pts, model_pts, 0)
                if H is None:
                    continue
                if abs(H[2, 2]) > 1e-12:
                    H = H / H[2, 2]
                if not self._validate_homography(H):
                    continue
                score = self._score_homography(H, lm_small)
                if score > best_score:
                    best_score, best_H = score, H.copy()
        return best_H, best_score

    # ==================================================================
    # Projekcja
    # ==================================================================

    def project_point_to_pitch(self, pixel_point: tuple,
                               H: np.ndarray) -> Optional[tuple]:
        """Rzutuje punkt pikselowy na wspolrzedne boiska (metry) lub None."""
        px, py = pixel_point
        world = H @ np.array([px, py, 1.0], dtype=np.float64)
        world /= world[2]
        x_m, y_m = float(world[0]), float(world[1])
        if (-PITCH_LENGTH_M / 2 <= x_m <= PITCH_LENGTH_M / 2 and
                -PITCH_WIDTH_M / 2 <= y_m <= PITCH_WIDTH_M / 2):
            return (x_m, y_m)
        return None

    def project_detections_to_pitch(self, detections: list, H: np.ndarray) -> list:
        """Rzutuje srodek dolnej krawedzi bbox kazdej detekcji na boisko."""
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            foot = ((x1 + x2) / 2.0, y2)
            results.append({**det, "pitch_coords": self.project_point_to_pitch(foot, H)})
        return results
