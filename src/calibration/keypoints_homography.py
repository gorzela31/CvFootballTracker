"""
Plik: src/calibration/keypoints_homography.py

Opis:
    Wrapper nad modelem YOLOv8x-pose (roboflow/sports) do detekcji
    keypoints boiska piłkarskiego i obliczenia macierzy homografii H (3x3).

Pipeline wewnętrzny (dwa etapy):
    1. Detekcja keypoints boiska
       - Model YOLOv8x-pose wykrywa charakterystyczne punkty boiska
         (narożniki, pola karne, środek, łuki) na obrazie klatki
       - Wynikiem są pary (x_px, y_px, confidence) dla każdego keypointa
    2. Obliczenie homografii (RANSAC)
       - Wykryte keypoints są dopasowywane do ich znanych pozycji na
         szablonie boiska (w metrach, układ FIFA: 105m x 68m)
       - cv2.findHomography z RANSAC oblicza macierz H minimalizując
         błąd reprojekcji i odrzucając outliery

Użycie:
    from src.calibration.keypoints_homography import KeypointsHomography

    calibrator = KeypointsHomography(
        model_weights="models/pitch_keypoints/pitch_keypoints_yolov8x_pose.pt"
    )
    H = calibrator.get_homography("sciezka/do/obrazu.jpg")

    pitch_coords = calibrator.project_point_to_pitch((x_px, y_px), H)
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Wymiary boiska FIFA w metrach
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Minimalna liczba keypoints z wystarczającą pewnością do obliczenia H.
# findHomography wymaga min. 4 par; większa liczba = stabilniejszy RANSAC.
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4

# Próg pewności keypointa — punkty poniżej progu są odrzucane
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# Próg błędu reprojekcji RANSAC (w pikselach)
RANSAC_REPROJ_THRESHOLD = 10.0

# ---------------------------------------------------------------------------
# Szablon boiska — pozycje keypoints w metrach (układ: (0,0) = lewy dolny róg)
#
# TODO: Zweryfikuj kolejność keypoints z datasetu roboflow/sports.
#       Kolejność MUSI być identyczna z etykietami w modelu YOLOv8-pose.
#       Sprawdź plik konfiguracyjny sports/configs/soccer.py lub
#       annotacje datasetu na Roboflow Universe ("Football Field Keypoints").
# ---------------------------------------------------------------------------
PITCH_KEYPOINTS_TEMPLATE_M: np.ndarray = np.array([
    # [x_m, y_m] — TODO: uzupelnij na podstawie roboflow/sports
], dtype=np.float32)


class KeypointsHomography:
    """
    Estymuje macierz homografii H dla obrazu transmisji piłkarskiej
    przy użyciu modelu YOLOv8-pose wykrywającego keypoints boiska.

    H mapuje: współrzędne pikselowe obrazu → metry na boisku.
    Układ boiska: lewy dolny róg = (0, 0), prawy górny = (105, 68).

    Parametry:
        model_weights : ścieżka do pliku wag modelu YOLOv8-pose
        conf_threshold: minimalny poziom pewności keypointa (domyślnie 0.5)
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
        # TODO: Implement
        raise NotImplementedError(
            "KeypointsHomography.get_homography() nie jest jeszcze zaimplementowane. "
            "Zaimplementuj zgodnie z pipeline'em: detekcja keypoints → RANSAC homografia."
        )

    def project_point_to_pitch(
        self, pixel_point: tuple, H: np.ndarray
    ) -> Optional[tuple]:
        """
        Rzutuje punkt pikselowy (np. stopy zawodnika) na płaszczyznę boiska.

        Parametry:
            pixel_point : (x, y) w pikselach
            H           : macierz homografii 3x3

        Zwraca:
            (x_m, y_m) w metrach lub None jeśli punkt poza boiskiem
        """
        px, py = pixel_point
        p_h = np.array([px, py, 1.0], dtype=np.float64)
        world = H @ p_h
        world /= world[2]
        x_m, y_m = float(world[0]), float(world[1])

        if not (0.0 <= x_m <= PITCH_LENGTH_M and 0.0 <= y_m <= PITCH_WIDTH_M):
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
        Uruchamia YOLOv8-pose na obrazie i zwraca wykryte keypoints.

        Zwraca:
            np.ndarray kształtu (N, 3) — (x_px, y_px, confidence) per keypoint,
            lub None jeśli model nie wykrył boiska
        """
        # TODO: Implement
        raise NotImplementedError

    def _compute_homography(
        self,
        keypoints_px: np.ndarray,
        keypoints_template_m: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Oblicza macierz homografii H metodą RANSAC.

        Parametry:
            keypoints_px         : (N, 2) — wykryte pozycje w pikselach
            keypoints_template_m : (N, 2) — odpowiadające pozycje na szablonie (metry)

        Zwraca:
            np.ndarray (3x3) lub None jeśli RANSAC się nie powiódł
        """
        if len(keypoints_px) < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            return None

        H, mask = cv2.findHomography(
            keypoints_px,
            keypoints_template_m,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        )
        if H is None:
            return None

        n_inliers = int(mask.sum()) if mask is not None else 0
        if n_inliers < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            return None

        return H

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
