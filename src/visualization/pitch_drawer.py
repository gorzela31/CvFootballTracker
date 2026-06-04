"""
Plik: src/visualization/pitch_drawer.py

Opis:
    Rysowanie widoku z gory boiska pilkarskiego w wymiarach FIFA
    (105m x 68m). Generuje wszystkie standardowe oznaczenia: obwod,
    linia srodkowa, kola srodkowe, pola karne i bramkowe, luki pol
    karnych ("D"), luki narozne, punkty karne.

    Uklad wspolrzednych: wycentrowany (zgodny z konwencja TVCalib).
        X w [-52.5, +52.5] m  (dlugosc)
        Y w [-34.0, +34.0] m  (szerokosc)
    Y dodatnie rysowane u DOLU obrazu (strona bliska kamery transmisji).

    Klasa zaprojektowana jako reusable - moze byc uzywana zarowno do
    minimapy (ten plik dalej w module), jak i pozniej do heatmap pozycji,
    wizualizacji bledu reprojekcji itp.
"""

import math
import cv2
import numpy as np


# Wymiary boiska FIFA (m)
PITCH_LENGTH_M    = 105.0
PITCH_WIDTH_M     = 68.0
CENTER_CIRCLE_R   = 9.15
PENALTY_AREA_LEN  = 16.5
PENALTY_AREA_W    = 40.32
GOAL_AREA_LEN     = 5.5
GOAL_AREA_W       = 18.32
PENALTY_SPOT_DIST = 11.0
PENALTY_ARC_R     = 9.15
CORNER_ARC_R      = 1.0


class PitchRenderer:
    """
    Renderuje boisko w ciemnym motywie. Wszystkie kolory w formacie BGR (OpenCV).
    """

    # Paleta - dark theme
    COLOR_BG       = (32, 20, 13)     # tlo poza boiskiem  (#0d1421)
    COLOR_PITCH    = (50, 38, 28)     # powierzchnia boiska (#1c2632)
    COLOR_LINES    = (230, 230, 230)  # linie (off-white, mniej harsh niz pure)
    COLOR_BALL     = (255, 255, 255)  # bialy
    COLOR_PLAYER   = (255, 200, 0)    # cyjan        (#00c8ff)
    COLOR_TEAM_A   = (255, 80, 50)    # niebieski    (team 0)
    COLOR_TEAM_B   = (80, 50, 255)    # czerwony     (team 1)
    COLOR_REFEREE  = (0, 140, 255)    # pomaranczowy (#ff8c00)
    COLOR_TEXT     = (200, 200, 200)  # meta info (frame counter, legenda)

    def __init__(
        self,
        width_px: int = 500,
        margin_px: int = 25,
        line_thickness: int = 2,
    ):
        self.width_px = width_px
        self.margin_px = margin_px
        self.line_thickness = line_thickness

        self.pitch_w_px = width_px - 2 * margin_px
        self.pitch_h_px = int(self.pitch_w_px * PITCH_WIDTH_M / PITCH_LENGTH_M)
        self.canvas_h_px = self.pitch_h_px + 2 * margin_px

        self.scale_x = self.pitch_w_px / PITCH_LENGTH_M
        self.scale_y = self.pitch_h_px / PITCH_WIDTH_M

    @property
    def canvas_size(self) -> tuple:
        """Zwraca (height_px, width_px) calego canvas."""
        return (self.canvas_h_px, self.width_px)

    def m_to_px(self, x_m: float, y_m: float) -> tuple:
        """
        Konwertuje wspolrzedne boiska (m) -> piksele canvasu.
        Y dodatnie na boisku = gora obrazu.
        """
        x_px = int(self.margin_px + (x_m + PITCH_LENGTH_M / 2) * self.scale_x)
        y_px = int(self.margin_px + (y_m + PITCH_WIDTH_M / 2) * self.scale_y)
        return x_px, y_px

    def render_empty(self) -> np.ndarray:
        """Generuje canvas z naniesionymi liniami boiska, bez obiektow."""
        canvas = np.full(
            (self.canvas_h_px, self.width_px, 3),
            self.COLOR_BG,
            dtype=np.uint8,
        )

        # Powierzchnia boiska (lekko jasniejszy prostokat wewnetrzny)
        tl = self.m_to_px(-PITCH_LENGTH_M / 2,  PITCH_WIDTH_M / 2)
        br = self.m_to_px( PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2)
        cv2.rectangle(canvas, tl, br, self.COLOR_PITCH, thickness=-1)

        # Obwod
        cv2.rectangle(canvas, tl, br, self.COLOR_LINES, thickness=self.line_thickness)

        # Linia srodkowa
        cv2.line(
            canvas,
            self.m_to_px(0,  PITCH_WIDTH_M / 2),
            self.m_to_px(0, -PITCH_WIDTH_M / 2),
            self.COLOR_LINES, self.line_thickness,
        )

        # Kolo srodkowe + punkt
        center = self.m_to_px(0, 0)
        r_center = int(CENTER_CIRCLE_R * self.scale_x)
        cv2.circle(canvas, center, r_center, self.COLOR_LINES, self.line_thickness)
        cv2.circle(canvas, center, 3, self.COLOR_LINES, -1)

        # Pola karne, bramkowe, punkty karne, luki "D" - per strona boiska
        for sign in [-1, +1]:
            self._draw_box_side(canvas, sign)

        # Luki narozne
        self._draw_corner_arcs(canvas)

        return canvas

    def _draw_box_side(self, canvas: np.ndarray, sign: int) -> None:
        """Rysuje pole karne, bramkowe, punkt karny i luk dla jednej strony boiska."""
        x_outer = sign * PITCH_LENGTH_M / 2

        # Pole karne
        x_inner_p = sign * (PITCH_LENGTH_M / 2 - PENALTY_AREA_LEN)
        tl = self.m_to_px(min(x_outer, x_inner_p),  PENALTY_AREA_W / 2)
        br = self.m_to_px(max(x_outer, x_inner_p), -PENALTY_AREA_W / 2)
        cv2.rectangle(canvas, tl, br, self.COLOR_LINES, self.line_thickness)

        # Pole bramkowe
        x_inner_g = sign * (PITCH_LENGTH_M / 2 - GOAL_AREA_LEN)
        tl = self.m_to_px(min(x_outer, x_inner_g),  GOAL_AREA_W / 2)
        br = self.m_to_px(max(x_outer, x_inner_g), -GOAL_AREA_W / 2)
        cv2.rectangle(canvas, tl, br, self.COLOR_LINES, self.line_thickness)

        # Punkt karny
        x_spot = sign * (PITCH_LENGTH_M / 2 - PENALTY_SPOT_DIST)
        spot = self.m_to_px(x_spot, 0)
        cv2.circle(canvas, spot, 3, self.COLOR_LINES, -1)

        # Luk pola karnego ("D") - czesc kola wystajaca poza pole karne
        # Liczymy kat: cos(α) = odleglosc od punktu karnego do brzegu pola / promien luku
        d_to_box = abs(PENALTY_AREA_LEN - PENALTY_SPOT_DIST)  # 5.5 m
        angle_offset = math.degrees(math.acos(d_to_box / PENALTY_ARC_R))
        r_arc = int(PENALTY_ARC_R * self.scale_x)

        if sign == -1:
            start_a, end_a = -angle_offset, +angle_offset           # otwarty na prawo
        else:
            start_a, end_a = 180 - angle_offset, 180 + angle_offset # otwarty na lewo

        cv2.ellipse(
            canvas, spot, (r_arc, r_arc), 0,
            start_a, end_a, self.COLOR_LINES, self.line_thickness,
        )

    def _draw_corner_arcs(self, canvas: np.ndarray) -> None:
        """Rysuje 4 male luki naroznikow boiska (do wnetrza boiska)."""
        r = max(3, int(CORNER_ARC_R * self.scale_x))
        # m_to_px: X rośnie w prawo, Y rośnie w dół (Y_m=+34 -> dol obrazu)
        # Luki musza isc do wnetrza boiska:
        corners = [
            (-PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2,   0,  90),  # gorny lewy (px) -> w prawo+dol
            (+PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2,  90, 180),  # gorny prawy (px) -> w lewo+dol
            (+PITCH_LENGTH_M / 2, +PITCH_WIDTH_M / 2, 180, 270),  # dolny prawy (px) -> w lewo+gore
            (-PITCH_LENGTH_M / 2, +PITCH_WIDTH_M / 2, 270, 360),  # dolny lewy (px) -> w prawo+gore
        ]
        for x_m, y_m, s, e in corners:
            cv2.ellipse(
                canvas, self.m_to_px(x_m, y_m), (r, r), 0,
                s, e, self.COLOR_LINES, self.line_thickness,
            )

    def get_class_color(self, class_name: str, team_id: int = -1) -> tuple:
        if class_name == "player" and team_id == 0:
            return self.COLOR_TEAM_A
        if class_name == "player" and team_id == 1:
            return self.COLOR_TEAM_B
        return {
            "ball":    self.COLOR_BALL,
            "player":  self.COLOR_PLAYER,
            "referee": self.COLOR_REFEREE,
        }.get(class_name, (128, 128, 128))

    def draw_object(
        self,
        canvas: np.ndarray,
        x_m: float,
        y_m: float,
        class_name: str,
        radius_px: int = 5,
        team_id: int = -1,
    ) -> None:
        """Rysuje pojedynczy obiekt na minimapie."""
        color = self.get_class_color(class_name, team_id)
        c = self.m_to_px(x_m, y_m)

        if class_name == "ball":
            # Pilka wieksza, z ciemnym pierscieniem dla widocznosci
            cv2.circle(canvas, c, radius_px + 2, (40, 40, 40), -1)
            cv2.circle(canvas, c, radius_px + 1, color, -1)
        else:
            cv2.circle(canvas, c, radius_px + 1, (20, 20, 20), -1)
            cv2.circle(canvas, c, radius_px, color, -1)