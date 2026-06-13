"""
Plik: src/calibration/keypoints_homography.py

Opis:
    Wrapper nad modelem YOLOv8x-pose (dataset: roboflow-jvuqo/football-field-detection-f07vi)
    do detekcji 32 keypoints boiska piłkarskiego i obliczenia macierzy homografii H (3x3).

Pipeline wewnętrzny (dwa etapy):
    1. Detekcja keypoints boiska
       - Model YOLOv8-pose wykrywa 32 charakterystyczne punkty boiska
         (narożniki, pola karne, łuki, środek) na obrazie klatki
       - Wynikiem są pary (x_px, y_px, confidence) dla każdego keypointa
    2. Obliczenie homografii (RANSAC)
       - Wykryte keypoints (>= KEYPOINT_CONFIDENCE_THRESHOLD) są
         dopasowywane do ich znanych pozycji metrycznych na szablonie boiska
       - cv2.findHomography z RANSAC oblicza H minimalizując błąd
         reprojekcji i odrzucając outliery

Układ współrzędnych (zgodny z TVCalib i PitchRenderer):
    Wycentrowany: X ∈ [-52.5, +52.5] m, Y ∈ [-34, +34] m
    Środek boiska = (0, 0)

Użycie:
    from src.calibration.keypoints_homography import KeypointsHomography

    calibrator = KeypointsHomography(
        model_weights="models/pitch_keypoints/trained_keypoints.pt"
    )
    H = calibrator.get_homography("sciezka/do/obrazu.jpg")
    pitch_coords = calibrator.project_point_to_pitch((x_px, y_px), H)
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Wymiary boiska FIFA w metrach (half-values dla układu wycentrowanego)
PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0

# Minimalna liczba keypoints do obliczenia homografii (RANSAC wymaga min. 4)
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4

# Próg pewności keypointa poniżej którego punkt jest odrzucany
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# Próg błędu reprojekcji RANSAC (w pikselach)
RANSAC_REPROJ_THRESHOLD = 10.0

# ---------------------------------------------------------------------------
# Szablon boiska — pozycje 32 keypoints w metrach (układ wycentrowany)
#
# Dataset: roboflow-jvuqo/football-field-detection-f07vi v16
# kpt_shape: [32, 3]  (x, y, visibility)
#
# Kolejność wynika z analizy flip_idx (data.yaml) i geometrii boiska FIFA:
#   Grupa A (0-5 ↔ 24-29): 6 punktów na lewej/prawej linii bramkowej (x=±52.5)
#   Grupa B (6-7 ↔ 22-23): narożniki dalekich krawędzi pól karnych (x=±36)
#   Grupa C (8 ↔ 21):      środek dalszej krawędzi pola karnego (x=±36, y=0)
#   Grupa D (9-12 ↔ 17-20): narożniki pola bramkowego + łuk karny ∩ linia pola karnego
#   Grupa E (13-16):        4 punkty na linii środkowej (x=0), self-symmetric
#   Grupa F (30 ↔ 31):      punkty karne (x=±41.5, y=0)
#
# Standardowe wymiary FIFA (metry):
#   Długość: 105m  → x ∈ [-52.5, 52.5]
#   Szerokość: 68m → y ∈ [-34, 34]
#   Pole karne: głębokość 16.5m, szerokość 40.32m (±20.16 od osi)
#   Pole bramkowe: głębokość 5.5m, szerokość 18.32m (±9.16 od osi)
#   Punkt karny: 11m od linii bramkowej → x = ±41.5
#   Promień łuku karnego = promień koła środkowego = 9.15m
#   Łuk karny ∩ linia pola karnego: y_offset = sqrt(9.15² - 5.5²) ≈ 7.31m
# ---------------------------------------------------------------------------
PITCH_KEYPOINTS_TEMPLATE_M: np.ndarray = np.array([
    # --- Grupa A: lewa linia bramkowa (x=-52.5) ---
    [-52.5, -34.0],   # 0: lewy górny narożnik boiska
    [-52.5, -20.16],  # 1: górna krawędź lewego pola karnego
    [-52.5,  -9.16],  # 2: górna krawędź lewego pola bramkowego
    [-52.5,   9.16],  # 3: dolna krawędź lewego pola bramkowego
    [-52.5,  20.16],  # 4: dolna krawędź lewego pola karnego
    [-52.5,  34.0],   # 5: lewy dolny narożnik boiska

    # --- Grupa B: daleka krawędź lewego pola karnego (x=-36) ---
    [-36.0, -20.16],  # 6: lewy górny narożnik dalszej krawędzi pola karnego
    [-36.0,  20.16],  # 7: lewy dolny narożnik dalszej krawędzi pola karnego

    # --- Grupa C: środek dalszej krawędzi lewego pola karnego ---
    [-36.0,   0.0],   # 8: środek dalszej krawędzi lewego pola karnego

    # --- Grupa D: lewe pole bramkowe + łuk karny (x=-47 i x=-36) ---
    [-47.0,  -9.16],  # 9:  górny narożnik dalszej krawędzi lewego pola bramkowego
    [-47.0,   9.16],  # 10: dolny narożnik dalszej krawędzi lewego pola bramkowego
    [-36.0,  -7.31],  # 11: górne przecięcie łuku karnego z krawędzią pola karnego
    [-36.0,   7.31],  # 12: dolne przecięcie łuku karnego z krawędzią pola karnego

    # --- Grupa E: linia środkowa (x=0), self-symmetric ---
    [  0.0, -34.0],   # 13: górna krawędź linii środkowej (przy bocznej)
    [  0.0,  -9.15],  # 14: górne przecięcie koła środkowego z linią środkową
    [  0.0,   9.15],  # 15: dolne przecięcie koła środkowego z linią środkową
    [  0.0,  34.0],   # 16: dolna krawędź linii środkowej (przy bocznej)

    # --- Grupa D: prawe pole bramkowe + łuk karny (mirror 9-12) ---
    [ 47.0,  -9.16],  # 17: górny narożnik dalszej krawędzi prawego pola bramkowego
    [ 47.0,   9.16],  # 18: dolny narożnik dalszej krawędzi prawego pola bramkowego
    [ 36.0,  -7.31],  # 19: górne przecięcie łuku karnego (prawe pole)
    [ 36.0,   7.31],  # 20: dolne przecięcie łuku karnego (prawe pole)

    # --- Grupa C: środek dalszej krawędzi prawego pola karnego (mirror 8) ---
    [ 36.0,   0.0],   # 21: środek dalszej krawędzi prawego pola karnego

    # --- Grupa B: daleka krawędź prawego pola karnego (mirror 6-7) ---
    [ 36.0, -20.16],  # 22: prawy górny narożnik dalszej krawędzi pola karnego
    [ 36.0,  20.16],  # 23: prawy dolny narożnik dalszej krawędzi pola karnego

    # --- Grupa A: prawa linia bramkowa (x=+52.5), mirror 0-5 ---
    [ 52.5, -34.0],   # 24: prawy górny narożnik boiska
    [ 52.5, -20.16],  # 25: górna krawędź prawego pola karnego
    [ 52.5,  -9.16],  # 26: górna krawędź prawego pola bramkowego
    [ 52.5,   9.16],  # 27: dolna krawędź prawego pola bramkowego
    [ 52.5,  20.16],  # 28: dolna krawędź prawego pola karnego
    [ 52.5,  34.0],   # 29: prawy dolny narożnik boiska

    # --- Grupa F: punkty karne (mirror pair) ---
    [-41.5,   0.0],   # 30: lewy punkt karny (11m od linii bramkowej)
    [ 41.5,   0.0],   # 31: prawy punkt karny
], dtype=np.float32)


class KeypointsHomography:
    """
    Estymuje macierz homografii H dla obrazu transmisji piłkarskiej
    przy użyciu modelu YOLOv8-pose wykrywającego 32 keypoints boiska.

    H mapuje: współrzędne pikselowe obrazu → metry na boisku (układ wycentrowany).
    Układ boiska: środek = (0, 0), X ∈ [-52.5, 52.5], Y ∈ [-34, 34].

    Parametry:
        model_weights  : ścieżka do pliku wag modelu YOLOv8-pose (.pt)
        conf_threshold : minimalny poziom pewności keypointa (domyślnie 0.5)
    """

    def __init__(
        self,
        model_weights: str,
        conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
    ):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if self._cuda_available() else "cpu"

        print(f"[KeypointsHomography] Urządzenie: {self.device}")
        print(f"[KeypointsHomography] Wczytywanie modelu: {model_weights}")

        self.model = YOLO(str(model_weights))

        print("[KeypointsHomography] Model gotowy ✓")

    def get_homography(self, image_path: str) -> Optional[np.ndarray]:
        """
        Główna metoda — dla podanego obrazu zwraca macierz homografii H (3x3).

        Parametry:
            image_path : ścieżka do obrazu .jpg / .png

        Zwraca:
            np.ndarray (3x3) lub None jeśli kalibracja się nie powiodła
        """
        keypoints = self._detect_keypoints(image_path)
        if keypoints is None:
            return None

        # Wybierz keypoints z wystarczającą pewnością
        confident_mask = keypoints[:, 2] >= self.conf_threshold
        n_confident = confident_mask.sum()

        if n_confident < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            print(
                f"[KeypointsHomography] Za mało pewnych keypoints: {n_confident} "
                f"(wymagane >= {MIN_KEYPOINTS_FOR_HOMOGRAPHY})"
            )
            return None

        src_pts = keypoints[confident_mask, :2]           # (N, 2) — piksele
        dst_pts = PITCH_KEYPOINTS_TEMPLATE_M[confident_mask]  # (N, 2) — metry

        return self._compute_homography(src_pts, dst_pts)

    def project_point_to_pitch(
        self, pixel_point: tuple, H: np.ndarray
    ) -> Optional[tuple]:
        """
        Rzutuje punkt pikselowy (np. stopy zawodnika) na płaszczyznę boiska.

        Parametry:
            pixel_point : (x, y) w pikselach
            H           : macierz homografii 3x3

        Zwraca:
            (x_m, y_m) w metrach (układ wycentrowany) lub None jeśli poza boiskiem
        """
        px, py = pixel_point
        p_h = np.array([px, py, 1.0], dtype=np.float64)
        world = H @ p_h
        world /= world[2]
        x_m, y_m = float(world[0]), float(world[1])

        if not (
            -PITCH_HALF_LENGTH_M <= x_m <= PITCH_HALF_LENGTH_M
            and -PITCH_HALF_WIDTH_M <= y_m <= PITCH_HALF_WIDTH_M
        ):
            return None
        return (x_m, y_m)

    def project_detections_to_pitch(
        self, detections: list, H: np.ndarray
    ) -> list:
        """
        Rzutuje listę detekcji na płaszczyznę boiska.

        Parametry:
            detections : lista dict {'bbox': [x1,y1,x2,y2], 'class': int, 'conf': float}
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

    # ------------------------------------------------------------------
    # Metody wewnętrzne
    # ------------------------------------------------------------------

    def _detect_keypoints(self, image_path: str) -> Optional[np.ndarray]:
        """
        Uruchamia YOLOv8-pose na obrazie i zwraca 32 keypoints.

        Zwraca:
            np.ndarray kształtu (32, 3) — (x_px, y_px, confidence) per keypoint,
            lub None jeśli model nie wykrył żadnego boiska
        """
        results = self.model(image_path, verbose=False, device=self.device)

        if not results or len(results[0].keypoints) == 0:
            print(f"[KeypointsHomography] Brak detekcji boiska w: {image_path}")
            return None

        # Weź detekcję z najwyższym confidence (powinno być max 1 boisko na klatce)
        kp_data = results[0].keypoints
        if kp_data.conf is None or len(kp_data.conf) == 0:
            return None

        best_idx = int(kp_data.conf.mean(dim=1).argmax())

        xy = kp_data.xy[best_idx].cpu().numpy()     # (32, 2)
        conf = kp_data.conf[best_idx].cpu().numpy()  # (32,)

        return np.column_stack([xy, conf])            # (32, 3)

    def _compute_homography(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Oblicza macierz homografii H metodą RANSAC.

        Parametry:
            src_pts : (N, 2) — wykryte pozycje keypoints w pikselach
            dst_pts : (N, 2) — odpowiadające pozycje na szablonie boiska (metry)

        Zwraca:
            np.ndarray (3x3) lub None jeśli RANSAC się nie powiódł / za mało inlierów
        """
        H, mask = cv2.findHomography(
            src_pts.astype(np.float32),
            dst_pts.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        )
        if H is None:
            print("[KeypointsHomography] RANSAC nie znalazł homografii")
            return None

        n_inliers = int(mask.sum()) if mask is not None else 0
        if n_inliers < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            print(f"[KeypointsHomography] Za mało inlierów RANSAC: {n_inliers}")
            return None

        return H

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
