"""
Plik: src/calibration/keypoints_homography.py

Opis:
    Wrapper nad modelem YOLOv8x-pose do detekcji 32 keypointów boiska
    piłkarskiego i obliczenia macierzy homografii H (3x3).

Pipeline wewnętrzny:
    1. Detekcja keypointów boiska
       - obraz jest domyślnie fizycznie skalowany metodą STRETCH do 640x640,
         zgodnie z preprocessingiem zbioru użytego do treningu modelu,
       - YOLOv8-pose wykrywa 32 charakterystyczne punkty boiska,
       - współrzędne wykrytych punktów są skalowane z powrotem do
         oryginalnej rozdzielczości klatki,
       - wynikiem są pary (x_px, y_px, confidence).

    2. Obliczenie homografii
       - keypointy z confidence >= KEYPOINT_CONFIDENCE_THRESHOLD są
         dopasowywane do ich znanych pozycji metrycznych na boisku,
       - cv2.findHomography z RANSAC wyznacza H w kierunku:
             IMAGE [px] -> PITCH [m]

Układ współrzędnych:
    X ∈ [-52.5, +52.5] m
    Y ∈ [-34.0, +34.0] m
    środek boiska = (0, 0)

Użycie:
    from src.calibration.keypoints_homography import KeypointsHomography

    calibrator = KeypointsHomography(
        model_weights="models/pitch_keypoints/trained_keypoints.pt"
    )

    H = calibrator.get_homography("sciezka/do/obrazu.jpg")

Opcjonalnie można odtworzyć stare zachowanie bez wymuszonego stretchu:
    calibrator = KeypointsHomography(
        model_weights="models/pitch_keypoints/trained_keypoints.pt",
        use_stretch=False,
    )
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# KONFIGURACJA
# =============================================================================

# Wymiary boiska FIFA w metrach (half-values dla układu wycentrowanego)
PITCH_HALF_LENGTH_M = 52.5
PITCH_HALF_WIDTH_M = 34.0

# Minimalna liczba keypointów do obliczenia homografii.
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4

# Finalny próg pewności używany w metodzie keypointowej.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.60

# Dataset football-field-detection v16 był preprocessingowany przez
# Resize to 640x640 (Stretch), dlatego ten sam preprocessing stosujemy
# domyślnie podczas inferencji.
DEFAULT_STRETCH_SIZE = 640

# Homografia jest estymowana IMAGE [px] -> PITCH [m].
# Z tego powodu próg RANSAC ma jednostkę METRÓW.
RANSAC_REPROJ_THRESHOLD = 10.0


# =============================================================================
# SZABLON 32 KEYPOINTÓW BOISKA
# =============================================================================

# Kolejność indeksów jest zgodna z finalną, poprawioną reprezentacją datasetu:
#
#   K00-K05  : lewa linia bramkowa
#   K06-K07  : lewe pole bramkowe
#   K08      : lewy punkt karny
#   K09-K12  : lewe pole karne + przecięcia łuku karnego
#   K13-K16  : linia środkowa + przecięcia koła środkowego
#   K17-K20  : prawe pole karne + przecięcia łuku karnego
#   K21      : prawy punkt karny
#   K22-K23  : prawe pole bramkowe
#   K24-K29  : prawa linia bramkowa
#   K30-K31  : lewy/prawy punkt koła środkowego
#
# Standardowe wartości:
#   długość boiska: 105 m
#   szerokość boiska: 68 m
#   pole karne: 16.5 m głębokości, ±20.16 m od osi
#   pole bramkowe: 5.5 m głębokości, ±9.16 m od osi
#   punkt karny: 11 m od linii bramkowej
#   promień koła środkowego: 9.15 m
#   przecięcie łuku karnego z linią pola karnego: około ±7.31 m

PITCH_KEYPOINTS_TEMPLATE_M: np.ndarray = np.array([
    [-52.5, -34.0],   # K00 top-left pitch corner
    [-52.5, -20.16],  # K01 goal line / top penalty-area edge
    [-52.5,  -9.16],  # K02 goal line / top goal-area edge
    [-52.5,   9.16],  # K03 goal line / bottom goal-area edge
    [-52.5,  20.16],  # K04 goal line / bottom penalty-area edge
    [-52.5,  34.0],   # K05 bottom-left pitch corner

    [-47.0,  -9.16],  # K06 far/top goal-area corner
    [-47.0,   9.16],  # K07 far/bottom goal-area corner

    [-41.5,   0.0],   # K08 left penalty spot

    [-36.0, -20.16],  # K09 far/top penalty-area corner
    [-36.0,  -7.31],  # K10 upper penalty-arc intersection
    [-36.0,   7.31],  # K11 lower penalty-arc intersection
    [-36.0,  20.16],  # K12 far/bottom penalty-area corner

    [  0.0, -34.0],   # K13 top halfway line / touchline
    [  0.0,  -9.15],  # K14 upper center-circle / halfway intersection
    [  0.0,   9.15],  # K15 lower center-circle / halfway intersection
    [  0.0,  34.0],   # K16 bottom halfway line / touchline

    [ 36.0, -20.16],  # K17 far/top right penalty-area corner
    [ 36.0,  -7.31],  # K18 upper right penalty-arc intersection
    [ 36.0,   7.31],  # K19 lower right penalty-arc intersection
    [ 36.0,  20.16],  # K20 far/bottom right penalty-area corner

    [ 41.5,   0.0],   # K21 right penalty spot

    [ 47.0,  -9.16],  # K22 far/top right goal-area corner
    [ 47.0,   9.16],  # K23 far/bottom right goal-area corner

    [ 52.5, -34.0],   # K24 top-right pitch corner
    [ 52.5, -20.16],  # K25 right goal line / top penalty-area edge
    [ 52.5,  -9.16],  # K26 right goal line / top goal-area edge
    [ 52.5,   9.16],  # K27 right goal line / bottom goal-area edge
    [ 52.5,  20.16],  # K28 right goal line / bottom penalty-area edge
    [ 52.5,  34.0],   # K29 bottom-right pitch corner

    [ -9.15,   0.0],  # K30 left center-circle point
    [  9.15,   0.0],  # K31 right center-circle point
], dtype=np.float32)


# =============================================================================
# KLASA
# =============================================================================

class KeypointsHomography:
    """
    Estymuje macierz homografii H dla obrazu transmisji piłkarskiej
    przy użyciu YOLOv8-pose wykrywającego 32 keypointy boiska.

    H mapuje:
        IMAGE [px] -> PITCH [m]

    Parametry:
        model_weights:
            ścieżka do wag YOLOv8-pose

        conf_threshold:
            minimalny confidence punktu używanego do homografii

        use_stretch:
            jeśli True, przed inferencją obraz jest fizycznie skalowany
            do stretch_size x stretch_size

        stretch_size:
            rozmiar kwadratowego wejścia modelu, domyślnie 640
    """

    def __init__(
        self,
        model_weights: str,
        conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
        use_stretch: bool = True,
        stretch_size: int = DEFAULT_STRETCH_SIZE,
    ):
        self.conf_threshold = float(conf_threshold)
        self.use_stretch = bool(use_stretch)
        self.stretch_size = int(stretch_size)

        if not (0.0 <= self.conf_threshold <= 1.0):
            raise ValueError("conf_threshold musi należeć do [0, 1].")

        if self.stretch_size <= 0:
            raise ValueError("stretch_size musi być > 0.")

        self.device = ("cuda" if self._cuda_available() else "cpu")

        print(f"[KeypointsHomography] Urządzenie: {self.device}")

        print(f"[KeypointsHomography] Wczytywanie modelu: {model_weights}")

        print(
            f"[KeypointsHomography] Stretch resize: {self.use_stretch}"
            + (f" ({self.stretch_size}x{self.stretch_size})" if self.use_stretch else "")
        )

        print(f"[KeypointsHomography] Confidence threshold: {self.conf_threshold:.2f}")

        self.model = YOLO(str(model_weights))

        print("[KeypointsHomography] Model gotowy")

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def get_homography(self, image_path: str) -> Optional[np.ndarray]:
        """
        Dla podanego obrazu zwraca macierz homografii H (3x3).

        H:
            IMAGE [px] -> PITCH [m]

        Zwraca:
            np.ndarray (3x3) lub None
        """

        keypoints = self._detect_keypoints(image_path)

        if keypoints is None:
            return None

        confident_mask = keypoints[:, 2] >= self.conf_threshold

        n_confident = int(confident_mask.sum())

        if n_confident < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            print(
                f"[KeypointsHomography] Za mało pewnych keypointów: {n_confident} (wymagane >= {MIN_KEYPOINTS_FOR_HOMOGRAPHY})"
            )
            return None

        # Współrzędne obrazu są już po skalowaniu z powrotem
        # do oryginalnej rozdzielczości.
        src_pts = keypoints[confident_mask, :2]

        dst_pts = PITCH_KEYPOINTS_TEMPLATE_M[confident_mask]

        return self._compute_homography(src_pts, dst_pts)

    def project_point_to_pitch(self, pixel_point: tuple, H: np.ndarray) -> Optional[tuple]:
        """
        Rzutuje punkt obrazu na płaszczyznę boiska.

        Parametry:
            pixel_point:
                (x_px, y_px)

            H:
                IMAGE [px] -> PITCH [m]

        Zwraca:
            (x_m, y_m), jeśli punkt leży w granicach boiska,
            w przeciwnym razie None.
        """
        px, py = pixel_point
        p_h = np.array([px, py, 1.0], dtype=np.float64)
        world = H @ p_h
        if abs(world[2]) < 1e-12:
            return None

        world /= world[2]
        x_m = float(world[0])
        y_m = float(world[1])

        if not (-PITCH_HALF_LENGTH_M <= x_m <= PITCH_HALF_LENGTH_M and -PITCH_HALF_WIDTH_M <= y_m <= PITCH_HALF_WIDTH_M):
            return None

        return (x_m, y_m)

    def project_detections_to_pitch(self, detections: list, H: np.ndarray) -> list:
        """
        Rzutuje detekcje obiektów na płaszczyznę boiska.

        Punkt reprezentujący zawodnika:
            środek dolnej krawędzi bboxa.

        Do każdego słownika dodawane jest:
            pitch_coords = (x_m, y_m) lub None
        """
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            pitch_coords = self.project_point_to_pitch((foot_x, foot_y), H)
            results.append({**det, "pitch_coords": pitch_coords})
        return results

    # -------------------------------------------------------------------------
    # DETEKCJA KEYPOINTÓW
    # -------------------------------------------------------------------------

    def _detect_keypoints(self, image_path: str) -> Optional[np.ndarray]:
        """
        Uruchamia YOLOv8-pose i zwraca 32 keypointy.

        Przy use_stretch=True:
            original W x H
            -> cv2.resize(..., stretch_size x stretch_size)
            -> YOLO
            -> x/y przeskalowane z powrotem do original W x H

        Zwraca:
            np.ndarray (32,3):
                [x_px, y_px, confidence]

        Współrzędne x/y są zawsze w układzie ORYGINALNEGO obrazu.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[KeypointsHomography] Nie można odczytać obrazu: {image_path}")
            return None

        orig_h, orig_w = image.shape[:2]

        # ---------------------------------------------------------------------
        # Inferencja
        # ---------------------------------------------------------------------

        if self.use_stretch:
            model_input = cv2.resize(image, (self.stretch_size, self.stretch_size), interpolation=cv2.INTER_LINEAR)

            results = self.model(model_input, verbose=False, device=self.device, imgsz=self.stretch_size)

        else:
            # Stare zachowanie bez jawnego stretch resize.
            # Zostawione wyłącznie dla reprodukcji wcześniejszych eksperymentów.
            results = self.model(str(image_path), verbose=False, device=self.device)

        # ---------------------------------------------------------------------
        # Odczyt wyniku
        # ---------------------------------------------------------------------

        if not results:
            print(f"[KeypointsHomography] Brak detekcji boiska w: {image_path}")
            return None

        result = results[0]

        if result.keypoints is None or result.keypoints.conf is None or len(result.keypoints.xy) == 0:
            print(f"[KeypointsHomography] Brak detekcji boiska w: {image_path}")
            return None

        kp_data = result.keypoints

        # Jeśli model zwróci kilka detekcji obiektu pitch,
        # wybieramy tę o najwyższym średnim confidence 32 keypointów.
        best_idx = int(kp_data.conf.mean(dim=1).argmax().item())

        xy = kp_data.xy[best_idx].detach().cpu().numpy().astype(np.float64)

        conf = kp_data.conf[best_idx].detach().cpu().numpy().astype(np.float64)

        if xy.shape != (32, 2) or conf.shape != (32,):
            print(f"[KeypointsHomography] Nieprawidłowy shape: xy={xy.shape}, conf={conf.shape}")
            return None

        # ---------------------------------------------------------------------
        # Powrót z przestrzeni stretch NxN do oryginalnej rozdzielczości
        # ---------------------------------------------------------------------

        if self.use_stretch:
            scale_x = orig_w / float(self.stretch_size)

            scale_y = orig_h / float(self.stretch_size)

            xy[:, 0] *= scale_x
            xy[:, 1] *= scale_y

        return np.column_stack([xy, conf])

    # -------------------------------------------------------------------------
    # HOMOGRAFIA
    # -------------------------------------------------------------------------

    def _compute_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Oblicza homografię metodą RANSAC.

        src_pts:
            IMAGE [px]

        dst_pts:
            PITCH [m]

        Zatem:
            H: IMAGE [px] -> PITCH [m]

        W konsekwencji ransacReprojThreshold jest wyrażony w metrach.
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

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
