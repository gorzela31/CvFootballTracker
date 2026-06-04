"""
Plik: src/classification/team_classifier.py

Opis:
    Klasyfikacja zawodnikow do druzyn na podstawie koloru koszulki.
    Metoda: K-means (k=2) na srednim kolorze HSV gornej czesci bbox.

    Algorytm:
        1. Dla kazdego gracza (class_id==1) wycinamy gornych 40% bbox (koszulka)
        2. Obliczamy sredni kolor w przestrzeni HSV (kanaly H, S)
        3. K-means (k=2) dopasowywany na pierwszej klatce z >=4 graczami
        4. Kolejne klatki uzywaja predict() na ustalonych centrach klastrow

    Ograniczenia:
        - Bramkarze moga zostac blednie przypisani (inny kolor koszulki)
        - Zaklada sie stabilne oswietlenie w obrebie jednego klipu

Uzycie:
    from src.classification.team_classifier import TeamClassifier

    classifier = TeamClassifier()
    team_ids = classifier.classify(frame_bgr, detections)
    # team_ids[i]: 0 lub 1 dla graczy, -1 dla ball/referee
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


# Kolory druzyn (BGR) — uzywane zarowno w bboxach jak i na minimapie
TEAM_COLORS_BGR = {
    0: (255, 80, 50),   # Team A — niebieski
    1: (80, 50, 255),   # Team B — czerwony
}

# Kolory domyslne per klasa (gdy brak team_id)
DEFAULT_COLORS_BGR = {
    0: (255, 255, 255),  # ball — bialy
    1: (255, 200, 0),    # player (brak druzyny) — cyjan
    2: (0, 140, 255),    # referee — pomaranczowy
}


class TeamClassifier:
    """
    Klasyfikator druzyn na podstawie koloru koszulki (K-means HSV).

    Parametry:
        n_clusters     : liczba druzyn (domyslnie 2)
        jersey_ratio   : ile gornej czesci bbox wycinac jako koszulke (0-1)
        min_players    : minimalna liczba graczy do dopasowania K-means
    """

    def __init__(self, n_clusters=2, jersey_ratio=0.4, min_players=4):
        self.n_clusters = n_clusters
        self.jersey_ratio = jersey_ratio
        self.min_players = min_players
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.fitted = False

    def _extract_jersey_color(self, frame_bgr, bbox):
        """Wyodrebnia sredni kolor H, S z gornej czesci bbox (koszulka)."""
        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        crop_h = max(1, int((y2 - y1) * self.jersey_ratio))
        crop = frame_bgr[y1:y1 + crop_h, x1:x2]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hs = hsv[:, :, :2].reshape(-1, 2).astype(np.float32)
        return hs.mean(axis=0)

    def classify(self, frame_bgr, detections):
        """
        Klasyfikuje graczy do druzyn.

        Parametry:
            frame_bgr   : klatka BGR (np.ndarray)
            detections  : sv.Detections z polem class_id

        Zwraca:
            np.ndarray of int — team_id per detekcja:
                0 lub 1 = druzyna
                -1      = nie-gracz (ball, referee) lub brak danych
        """
        n = len(detections)
        team_ids = np.full(n, -1, dtype=int)

        if n == 0:
            return team_ids

        player_indices = []
        features = []

        for i in range(n):
            if int(detections.class_id[i]) != 1:  # 1 = player
                continue

            feat = self._extract_jersey_color(frame_bgr, detections.xyxy[i])
            if feat is not None:
                player_indices.append(i)
                features.append(feat)

        if len(features) < self.n_clusters:
            return team_ids

        features_arr = np.array(features, dtype=np.float32)

        if not self.fitted and len(features) >= self.min_players:
            labels = self.kmeans.fit_predict(features_arr)
            self.fitted = True
        elif self.fitted:
            labels = self.kmeans.predict(features_arr)
        else:
            return team_ids

        for idx, player_i in enumerate(player_indices):
            team_ids[player_i] = int(labels[idx])

        return team_ids


def get_detection_color(class_id, team_id=-1):
    """Zwraca kolor BGR dla detekcji na podstawie klasy i druzyny."""
    if class_id == 1 and team_id in TEAM_COLORS_BGR:
        return TEAM_COLORS_BGR[team_id]
    return DEFAULT_COLORS_BGR.get(class_id, (180, 180, 180))


def annotate_frame_with_teams(frame, detections, team_ids, class_names):
    """
    Rysuje bounding boxy kolorowane wg druzyny.

    Parametry:
        frame       : klatka BGR
        detections  : sv.Detections
        team_ids    : np.ndarray z team_id per detekcja (-1 / 0 / 1)
        class_names : dict {class_id: name}

    Zwraca:
        annotated frame (kopia)
    """
    annotated = frame.copy()

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        cls_id = int(detections.class_id[i])
        tid = detections.tracker_id[i] if detections.tracker_id is not None else None
        team_id = int(team_ids[i]) if team_ids is not None else -1

        color = get_detection_color(cls_id, team_id)

        # Bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        cls_name = class_names.get(cls_id, "?")
        if tid is not None and int(tid) != -1:
            label = f"#{int(tid)} {cls_name}"
        else:
            label = cls_name

        # Tlo etykiety + tekst
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

        # Tekst czarny na kolorowym tle
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated
