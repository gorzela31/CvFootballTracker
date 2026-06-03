"""
Plik: src/utils/visualization.py
 
Opis:
    Moduł wizualizacji bird's eye view (BEV) — rzut z góry na boisko.
    Przyjmuje detekcje zawodników (bbox z YOLO) oraz macierz homografii H
    i rysuje ich pozycje na standardowej mapie 2D boiska.
 
Układ współrzędnych boiska (TVCalib):
    Środek boiska = (0, 0)
    Oś X: od -52.5m (lewa linia bramkowa) do +52.5m (prawa linia bramkowa)
    Oś Y: od -34.0m (górna linia boczna) do +34.0m (dolna linia boczna)
 
Użycie:
    from src.utils.visualization import BirdEyeView
 
    bev = BirdEyeView()
    img = bev.draw(detections_with_pitch_coords, original_frame)
    cv2.imshow("BEV", img)
"""
 
from typing import Optional
import cv2
import numpy as np
 
# --- Wymiary boiska FIFA (metry, układ środkowy) ---
PITCH_LENGTH_M = 105.0   # oś X: -52.5 do +52.5
PITCH_WIDTH_M = 68.0     # oś Y: -34.0 do +34.0
 
# Wymiary pola karnego, pola bramkowego i łuku
PENALTY_LENGTH_M = 16.5
PENALTY_WIDTH_M = 40.32
GOAL_LENGTH_M = 5.5
GOAL_WIDTH_M = 18.32
CENTER_CIRCLE_R_M = 9.15
PENALTY_SPOT_M = 11.0
 
# Kolory zawodników wg klasy YOLO (BGR)
CLASS_COLORS = {
    0: (0, 200, 0),      # zawodnik drużyna A — zielony
    1: (0, 80, 220),     # zawodnik drużyna B — czerwony
    2: (0, 200, 200),    # bramkarz — żółty
    3: (200, 200, 200),  # sędzia — biały
}
DEFAULT_COLOR = (180, 180, 180)
 
 
class BirdEyeView:
    """
    Rysuje mapę bird's eye view boiska z zawodnikami.
 
    Parametry:
        scale       : piksele na metr (domyślnie 10 → mapa 1050x680 px)
        padding_px  : margines wokół boiska w pikselach
        bg_color    : kolor tła (BGR)
        pitch_color : kolor trawy (BGR)
        line_color  : kolor linii boiska (BGR)
    """
 
    def __init__(
        self,
        scale: int = 10,
        padding_px: int = 30,
        bg_color=(30, 30, 30),
        pitch_color=(34, 100, 34),
        line_color=(255, 255, 255),
    ):
        self.scale = scale
        self.padding = padding_px
        self.bg_color = bg_color
        self.pitch_color = pitch_color
        self.line_color = line_color
 
        # Rozmiar obrazu BEV
        self.pitch_w_px = int(PITCH_LENGTH_M * scale)   # 1050
        self.pitch_h_px = int(PITCH_WIDTH_M * scale)    # 680
        self.img_w = self.pitch_w_px + 2 * padding_px
        self.img_h = self.pitch_h_px + 2 * padding_px
 
    def draw(
        self,
        detections: list,
        original_frame: Optional[np.ndarray] = None,
        show_frame_thumbnail: bool = True,
    ) -> np.ndarray:
        """
        Główna metoda — rysuje BEV z zawodnikami.
 
        Parametry:
            detections          : lista dict z polami:
                                  'bbox': [x1,y1,x2,y2]
                                  'class': int
                                  'conf': float
                                  'pitch_coords': (x_m, y_m) lub None
            original_frame      : oryginalny obraz (opcjonalnie — miniatura w rogu)
            show_frame_thumbnail: czy pokazać miniaturę oryginalnego obrazu
 
        Zwraca:
            np.ndarray — obraz BEV (BGR)
        """
        img = self._draw_pitch()
        img = self._draw_players(img, detections)
 
        if show_frame_thumbnail and original_frame is not None:
            img = self._draw_thumbnail(img, original_frame)
 
        img = self._draw_legend(img)
        return img
 
    def pitch_to_pixel(self, x_m: float, y_m: float) -> tuple:
        """
        Przelicza współrzędne metryczne na pikselowe w obrazie BEV.
 
        Układ TVCalib (środek=0,0) → układ obrazu (lewy górny=0,0)
        """
        px = int((x_m + PITCH_LENGTH_M / 2) * self.scale) + self.padding
        py = int((y_m + PITCH_WIDTH_M / 2) * self.scale) + self.padding
        return (px, py)
 
    # ------------------------------------------------------------------
    # Metody wewnętrzne
    # ------------------------------------------------------------------
 
    def _draw_pitch(self) -> np.ndarray:
        """Rysuje puste boisko z liniami."""
        img = np.full((self.img_h, self.img_w, 3), self.bg_color, dtype=np.uint8)
 
        # Trawa
        x0, y0 = self.padding, self.padding
        x1, y1 = self.padding + self.pitch_w_px, self.padding + self.pitch_h_px
        cv2.rectangle(img, (x0, y0), (x1, y1), self.pitch_color, -1)
 
        s = self.scale
        p = self.padding
        lc = self.line_color
        t = 1  # grubość linii
 
        # Obwód boiska
        cv2.rectangle(img, (x0, y0), (x1, y1), lc, t)
 
        # Linia środkowa
        mid_x = p + self.pitch_w_px // 2
        cv2.line(img, (mid_x, y0), (mid_x, y1), lc, t)
 
        # Koło środkowe
        center = (mid_x, p + self.pitch_h_px // 2)
        cv2.circle(img, center, int(CENTER_CIRCLE_R_M * s), lc, t)
        cv2.circle(img, center, 3, lc, -1)
 
        # Punkt środkowy
        cv2.circle(img, center, 3, lc, -1)
 
        # Pole karne lewe
        pk_w = int(PENALTY_WIDTH_M * s)
        pk_l = int(PENALTY_LENGTH_M * s)
        pk_y0 = p + (self.pitch_h_px - pk_w) // 2
        cv2.rectangle(img, (x0, pk_y0), (x0 + pk_l, pk_y0 + pk_w), lc, t)
 
        # Pole karne prawe
        cv2.rectangle(img, (x1 - pk_l, pk_y0), (x1, pk_y0 + pk_w), lc, t)
 
        # Pole bramkowe lewe
        gb_w = int(GOAL_WIDTH_M * s)
        gb_l = int(GOAL_LENGTH_M * s)
        gb_y0 = p + (self.pitch_h_px - gb_w) // 2
        cv2.rectangle(img, (x0, gb_y0), (x0 + gb_l, gb_y0 + gb_w), lc, t)
 
        # Pole bramkowe prawe
        cv2.rectangle(img, (x1 - gb_l, gb_y0), (x1, gb_y0 + gb_w), lc, t)
 
        # Punkt karny lewy
        ps_x = x0 + int(PENALTY_SPOT_M * s)
        cv2.circle(img, (ps_x, center[1]), 3, lc, -1)
 
        # Punkt karny prawy
        ps_x2 = x1 - int(PENALTY_SPOT_M * s)
        cv2.circle(img, (ps_x2, center[1]), 3, lc, -1)
 
        return img
 
    def _draw_players(self, img: np.ndarray, detections: list) -> np.ndarray:
        """Rysuje zawodników na mapie BEV."""
        for det in detections:
            coords = det.get("pitch_coords")
            if coords is None:
                continue
 
            x_m, y_m = coords
            px, py = self.pitch_to_pixel(x_m, y_m)
 
            # Sprawdź czy piksel mieści się w obrazie
            if not (0 <= px < self.img_w and 0 <= py < self.img_h):
                continue
 
            cls = det.get("class", 0)
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
            conf = det.get("conf", 1.0)
 
            # Koło zawodnika
            cv2.circle(img, (px, py), 7, color, -1)
            cv2.circle(img, (px, py), 8, (255, 255, 255), 1)
 
            # Confidence score
            cv2.putText(
                img,
                f"{conf:.2f}",
                (px + 9, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )
 
        return img
 
    def _draw_thumbnail(self, img: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Wstawia miniaturę oryginalnego obrazu w prawym górnym rogu."""
        thumb_w = self.img_w // 3
        thumb_h = int(thumb_w * frame.shape[0] / frame.shape[1])
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
 
        x_off = self.img_w - thumb_w - 5
        y_off = 5
        img[y_off:y_off + thumb_h, x_off:x_off + thumb_w] = thumb
        cv2.rectangle(img, (x_off, y_off), (x_off + thumb_w, y_off + thumb_h), (255, 255, 255), 1)
        return img
 
    def _draw_legend(self, img: np.ndarray) -> np.ndarray:
        """Rysuje legendę klas w lewym dolnym rogu."""
        labels = {
            0: "Ball",
            1: "Team 1",
            2: "Team 2",
            3: "Referee",
        }
        x, y = 5, self.img_h - 5 - len(labels) * 18
        for cls, label in labels.items():
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
            cv2.circle(img, (x + 6, y + 6), 5, color, -1)
            cv2.putText(img, label, (x + 15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            y += 18
        return img