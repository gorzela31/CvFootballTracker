"""
Plik: src/visualization/minimap.py

Opis:
    Renderowanie minimapy boiska z naniesionymi pozycjami obiektow
    (ball, player, referee). Opakowuje PitchRenderer w wygodny interfejs
    przyjmujacy detekcje z TVCalibHomography.project_detections_to_pitch().

    Klasa wspiera padding pionowy do zadanej wysokosci, co ulatwia
    komponowanie minimapy obok klatki wideo o roznych wymiarach (np.hstack).
"""

import cv2
import numpy as np

from src.visualization.pitch_drawer import PitchRenderer


DEFAULT_CLASS_NAMES = {0: "ball", 1: "player", 2: "referee"}


class MinimapRenderer:
    """
    Renderuje minimape boiska wraz z elementami informacyjnymi
    (licznik klatek, legenda kolorow).

    Parametry:
        width_px         : szerokosc minimapy
        margin_px        : margines wokol boiska
        line_thickness   : grubosc linii boiska
        class_names      : mapowanie {class_id: nazwa}
        show_frame_info  : licznik klatek u gory
        show_legend      : legenda kolorow u dolu
    """

    def __init__(
        self,
        width_px: int = 500,
        margin_px: int = 25,
        line_thickness: int = 2,
        class_names: dict = None,
        show_frame_info: bool = True,
        show_legend: bool = True,
    ):
        self.pitch = PitchRenderer(width_px, margin_px, line_thickness)
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.show_frame_info = show_frame_info
        self.show_legend = show_legend

    @property
    def canvas_size(self) -> tuple:
        """(height, width) bazowego canvas (bez paddingu)."""
        return self.pitch.canvas_size

    def render(
        self,
        detections: list,
        frame_idx: int = None,
        total_frames: int = None,
        target_height: int = None,
    ) -> np.ndarray:
        """
        Renderuje minimape.

        Parametry:
            detections    : lista dictow z polami 'class' (int) oraz
                            'pitch_coords' (tuple x_m, y_m) lub None
            frame_idx     : numer biezacej klatki (do wyswietlenia)
            total_frames  : laczna liczba klatek
            target_height : pad pionowy do tej wysokosci (px) - uzyteczne
                            gdy komponujemy obok klatki wideo
        """
        canvas = self.pitch.render_empty()
        pitch_canvas_h = canvas.shape[0]   # wysokosc samego boiska, bez paddingu

        for d in detections:
            pitch_coords = d.get("pitch_coords")
            if pitch_coords is None:
                continue
            class_name = self.class_names.get(d.get("class"), "unknown")
            x_m, y_m = pitch_coords
            self.pitch.draw_object(canvas, x_m, y_m, class_name, radius_px=5)

        if self.show_frame_info and frame_idx is not None:
            self._draw_frame_info(canvas, frame_idx, total_frames)

        # Padding ZANIM rysujemy legende - zeby legenda miala miejsce ponizej boiska
        if target_height is not None and target_height > canvas.shape[0]:
            pad_h = target_height - canvas.shape[0]
            padding = np.full(
                (pad_h, canvas.shape[1], 3),
                self.pitch.COLOR_BG,
                dtype=np.uint8,
            )
            canvas = np.vstack([canvas, padding])

        if self.show_legend:
            # Legenda zaczyna sie 20px ponizej dolnej krawedzi boiska.
            # Jezeli nie ma paddingu (brak target_height), spada do dolnej czesci canvas.
            legend_y_top = pitch_canvas_h + 20
            max_y = canvas.shape[0] - 16 * 3 - 5
            if legend_y_top > max_y:
                legend_y_top = max_y
            self._draw_legend(canvas, legend_y_top)

        return canvas

    def _draw_frame_info(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        total_frames: int = None,
    ) -> None:
        """Tekst z numerem klatki u gory canvas."""
        text = f"Frame: {frame_idx + 1}"
        if total_frames is not None:
            text += f" / {total_frames}"
        cv2.putText(
            canvas, text, (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.pitch.COLOR_TEXT, 1, cv2.LINE_AA,
        )

    def _draw_legend(self, canvas: np.ndarray, y_top: int) -> None:
        """Legenda kolorow umieszczona pod boiskiem (lub na koncu canvas)."""
        w = canvas.shape[1]
        items = [
            ("ball",    self.pitch.COLOR_BALL),
            ("player",  self.pitch.COLOR_PLAYER),
            ("referee", self.pitch.COLOR_REFEREE),
        ]
        x0 = w - 90
        for i, (name, color) in enumerate(items):
            cy = y_top + i * 16
            cv2.circle(canvas, (x0, cy), 4, color, -1)
            cv2.putText(
                canvas, name, (x0 + 10, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                self.pitch.COLOR_TEXT, 1, cv2.LINE_AA,
            )