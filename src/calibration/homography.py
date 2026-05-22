"""
Plik: src/calibration/homography.py

Opis:
    Wrapper nad modelem TVCalib (Theiner et al., WACV 2023).
    Odpowiada za estymację macierzy homografii H (3x3) która
    mapuje współrzędne pikselowe obrazu na rzeczywiste współrzędne
    płaszczyzny boiska (w metrach).

Pipeline wewnętrzny (dwa etapy TVCalib):
    1. Segmentacja semantyczna boiska
       - Sieć neuronowa (HRNet) wykrywa linie i łuki boiska na obrazie
       - Wynikiem są maski klas dla każdego elementu boiska
    2. Optymalizacja homografii
       - Z masek wyodrębniane są szkielety i ekstrema linii (keypoints)
       - Algorytm iteracyjnie minimalizuje błąd reprojekcji modelu
         3D boiska na obraz (różniczkowalny renderer)
       - Wynikiem jest macierz H oraz parametry kamery

Użycie:
    from src.calibration.homography import TVCalibHomography

    calibrator = TVCalibHomography(
        model_weights="src/calibration/tvcalib/data/segment_localization/train_59.pt"
    )
    H = calibrator.get_homography("ścieżka/do/obrazu.jpg")

    # Rzutowanie punktu (stopy zawodnika) na boisko
    pitch_coords = calibrator.project_point_to_pitch((x_px, y_px), H)
"""

import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# --- Dodaj TVCalib i sn_segmentation do ścieżki Pythona ---
TVCALIB_DIR = Path(__file__).parent / "tvcalib"
SN_SEG_DIR = TVCALIB_DIR / "sn_segmentation"
sys.path.insert(0, str(TVCALIB_DIR))
sys.path.insert(0, str(SN_SEG_DIR))

from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from tvcalib.inference import (
    InferenceDatasetCalibration,
    InferenceDatasetSegmentation,
    InferenceSegmentationModel,
    get_camera_from_per_sample_output,
)
from tvcalib.module import TVCalibModule
from tvcalib.sncalib_dataset import custom_list_collate
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import (
    SoccerPitchLineCircleSegments,
    SoccerPitchSNCircleCentralSplit,
)
from sn_segmentation.src.custom_extremities import (
    generate_class_synthesis,
    get_line_extremities,
)

# Wymiary boiska FIFA w metrach
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


class TVCalibHomography:
    """
    Estymuje macierz homografii H dla obrazu transmisji piłkarskiej
    przy użyciu modelu TVCalib.

    H mapuje: współrzędne pikselowe obrazu → metry na boisku.
    Układ boiska: lewy dolny róg = (0, 0), prawy górny = (105, 68).

    Parametry:
        model_weights : ścieżka do pliku wag modelu segmentacji (train_59.pt)
        image_width   : szerokość obrazu wejściowego w pikselach (domyślnie 1280)
        image_height  : wysokość obrazu wejściowego w pikselach (domyślnie 720)
        optim_steps   : liczba kroków optymalizacji homografii (domyślnie 2000)
        lens_dist     : czy używać korekcji dystorsji soczewki (domyślnie False)
    """

    # Rozmiar wejścia modelu segmentacji — stały, wymagany przez TVCalib
    SEG_WIDTH = 455
    SEG_HEIGHT = 256

    def __init__(
        self,
        model_weights: str,
        image_width: int = 1280,
        image_height: int = 720,
        optim_steps: int = 2000,
        lens_dist: bool = False,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.optim_steps = optim_steps
        self.lens_dist = lens_dist
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[TVCalibHomography] Urządzenie: {self.device}")
        print(f"[TVCalibHomography] Wczytywanie modelu segmentacji: {model_weights}")

        # Model segmentacji semantycznej boiska
        self.model_seg = InferenceSegmentationModel(model_weights, self.device)

        # Model 3D boiska (wzorzec linii i łuków wg FIFA) — dwie wersje: GPU i CPU
        base_field = SoccerPitchSNCircleCentralSplit()
        self.object3d = SoccerPitchLineCircleSegments(device=self.device, base_field=base_field)
        self.object3d_cpu = SoccerPitchLineCircleSegments(device="cpu", base_field=base_field)

        # Funkcje przetwarzania masek segmentacji na keypoints
        # generate_class_synthesis: szkieletyzacja masek (radius=4px)
        # get_line_extremities: sampling punktów wzdłuż linii (4 punkty/linię, 8/łuk)
        self.fn_generate_class_synthesis = partial(generate_class_synthesis, radius=4)
        self.fn_get_line_extremities = partial(
            get_line_extremities,
            maxdist=30,
            width=self.SEG_WIDTH,
            height=self.SEG_HEIGHT,
            num_points_lines=4,
            num_points_circles=8,
        )

        print("[TVCalibHomography] Model gotowy ✓")

    def get_homography(self, image_path: str) -> Optional[np.ndarray]:
        """
        Główna metoda — dla podanego obrazu zwraca macierz homografii H (3x3).

        Parametry:
            image_path : ścieżka do obrazu .jpg / .png

        Zwraca:
            np.ndarray (3x3) lub None jeśli kalibracja się nie powiodła
        """
        # Etap 1: segmentacja boiska → keypoints
        image_ids, keypoints_raw = self._run_segmentation([image_path])
        if not keypoints_raw:
            print(f"[TVCalibHomography] BŁĄD: segmentacja nie zwróciła keypoints dla {image_path}")
            return None

        # Etap 2: optymalizacja homografii → parametry kamery
        df = self._run_calibration(image_ids, keypoints_raw)
        if df is None:
            print(f"[TVCalibHomography] BŁĄD: kalibracja nie powiodła się dla {image_path}")
            return None

        # Wyciągnij H dla pierwszego (jedynego) obrazu
        sample = df.iloc[0]
        cam = get_camera_from_per_sample_output(sample, self.lens_dist)
        H = cam.get_homography_raster().detach().cpu().numpy().squeeze()
        return H

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

        if not (-PITCH_LENGTH_M / 2 <= x_m <= PITCH_LENGTH_M / 2 and
                -PITCH_WIDTH_M / 2 <= y_m <= PITCH_WIDTH_M / 2):
            return None
        return (x_m, y_m)

    def project_detections_to_pitch(
        self, detections: list, H: np.ndarray
    ) -> list:
        """
        Rzutuje listę detekcji YOLO na płaszczyznę boiska.

        Parametry:
            detections : lista dict {'bbox': [x1,y1,x2,y2], 'class': int, 'conf': float}
            H          : macierz homografii 3x3

        Zwraca:
            lista dict z dodanym polem 'pitch_coords': (x_m, y_m) lub None
        """
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            foot_x = (x1 + x2) / 2.0  # środek dolnej krawędzi bbox = punkt kontaktu z boiskiem
            foot_y = y2
            pitch_coords = self.project_point_to_pitch((foot_x, foot_y), H)
            results.append({**det, "pitch_coords": pitch_coords})
        return results

    # ------------------------------------------------------------------
    # Metody wewnętrzne
    # ------------------------------------------------------------------

    def _run_segmentation(self, image_paths: list) -> tuple:
        """
        Uruchamia segmentację semantyczną na liście obrazów.

        Zwraca:
            (image_ids, keypoints_raw) — listy tej samej długości
        """
        image_path = Path(image_paths[0])
        dataset_seg = InferenceDatasetSegmentation(
            image_path.parent,
            self.image_width,
            self.image_height,
        )
        # Filtruj tylko żądane obrazy
        target_names = {Path(p).name for p in image_paths}
        dataset_seg.image_files = [img for img in dataset_seg.image_files if img.name in target_names]

        dataloader_seg = torch.utils.data.DataLoader(
            dataset_seg,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_list_collate,
        )

        image_ids = []
        keypoints_raw = []

        for batch_dict in dataloader_seg:
            with torch.no_grad():
                sem_lines = self.model_seg.inference(batch_dict["image"].to(self.device))
            sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

            for mask in sem_lines:
                skeleton = self.fn_generate_class_synthesis(mask)
                kp = self.fn_get_line_extremities(skeleton)
                keypoints_raw.append(kp)

            image_ids.extend(batch_dict["image_id"])

        return image_ids, keypoints_raw

    def _run_calibration(self, image_ids: list, keypoints_raw: list):
        """
        Uruchamia optymalizację homografii TVCalib.

        Zwraca:
            pandas DataFrame z parametrami kamery per obraz, lub None przy błędzie
        """
        try:
            import pandas as pd

            batch_size_calib = len(keypoints_raw)

            # Model kalibracji — optymalizator parametrów kamery
            # get_cam_distr(1.96, batch_size, 1): rozkład a priori parametrów kamery
            # (1.96 = współczynnik skali FoV typowy dla transmisji TV)
            model_calib = TVCalibModule(
                self.object3d,
                get_cam_distr(1.96, batch_size_calib, 1),
                get_dist_distr(batch_size_calib, 1) if self.lens_dist else None,
                (self.image_height, self.image_width),
                self.optim_steps,
                self.device,
                log_per_step=False,
                tqdm_kwqargs=None,
            )

            dataset_calib = InferenceDatasetCalibration(
                keypoints_raw,
                self.image_width,
                self.image_height,
                self.object3d,
            )
            dataloader_calib = torch.utils.data.DataLoader(
                dataset_calib,
                batch_size=batch_size_calib,
                collate_fn=custom_list_collate,
            )

            per_sample_output = defaultdict(list)
            per_sample_output["image_id"] = [[x] for x in image_ids]

            for x_dict in dataloader_calib:
                _batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]
                points_line = x_dict["lines__px_projected_selection_shuffled"]
                points_circle = x_dict["circles__px_projected_selection_shuffled"]

                per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)
                output_dict = tensor2list(
                    detach_dict({**cam.get_parameters(_batch_size), **per_sample_loss})
                )
                output_dict["points_line"] = points_line
                output_dict["points_circle"] = points_circle

                for k in output_dict.keys():
                    per_sample_output[k].extend(output_dict[k])

            df = pd.DataFrame.from_dict(per_sample_output)
            df = df.explode(
                column=[k for k, v in per_sample_output.items() if isinstance(v, list)]
            )
            df.set_index("image_id", inplace=True, drop=False)
            return df

        except Exception as e:
            import traceback
            print(f"[TVCalibHomography] Błąd kalibracji: {e}")
            traceback.print_exc()
            return None