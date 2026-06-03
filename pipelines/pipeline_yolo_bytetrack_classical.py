"""
Plik: pipelines/pipeline_yolo_bytetrack_classical.py

Opis:
    Pipeline detekcja -> tracking -> kalibracja -> projekcja na boisko.

    Komponenty:
        Detekcja:    YOLOv8n (wagi wytrenowane na SoccerNet, 3 klasy)
        Tracking:    ByteTrack (supervision)
        Kalibracja:  Klasyczna homografia (detekcja linii + RANSAC)
        Projekcja:   pozycje stop (srodek dolnej krawedzi bbox) -> metry na boisku
        Wizualizacja: minimapa boiska w ciemnym motywie obok klatki glownej

    Wejscie:
        Katalog z klatkami SoccerNet: SNMOT-XXX/img1/*.jpg

    Wyjscie (w results/<RUN_NAME>/):
        - output.mp4   : wideo z bboxami, klasami, ID trackow oraz minimapa
        - tracks.csv   : pozycje per (frame, track_id) w pikselach i metrach
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from src.calibration.classical_homography import ClassicalHomography
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.visualization.minimap import MinimapRenderer


# ==========================================================================
# KONFIGURACJA
# ==========================================================================
RUN_NAME = "pipeline_yolo_bt_classical"

# Sciezki wejsciowe
FRAMES_DIR = PROJECT_ROOT / "data" / "tracking_dataset" / "tracking" / "test" / "SNMOT-123" / "img1"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n" / "trained_detection_yolov8n.pt"

# Sciezki wyjsciowe
OUTPUT_DIR = PROJECT_ROOT / "results" / RUN_NAME
OUTPUT_VIDEO = OUTPUT_DIR / "output.mp4"
OUTPUT_CSV = OUTPUT_DIR / "tracks.csv"

# Parametry przetwarzania
CONF_THRESHOLD = 0.10
FPS = 25
CALIB_STRIDE = 5

# Parametry klasycznej homografii
RANSAC_ITERATIONS = 2000
RANSAC_MIN_SCORE = 0.12

# Mapowanie klas
CLASS_NAMES = {0: "ball", 1: "player", 2: "referee"}

# Wizualizacja
MINIMAP_WIDTH_PX = 500


# ==========================================================================
# HELPERS
# ==========================================================================

def sv_detections_to_dict_list(detections: sv.Detections) -> list:
    """Konwertuje sv.Detections do listy dictow."""
    result = []
    for i in range(len(detections)):
        tid = (
            int(detections.tracker_id[i])
            if detections.tracker_id is not None and detections.tracker_id[i] != -1
            else None
        )
        result.append({
            "bbox": detections.xyxy[i].tolist(),
            "class": int(detections.class_id[i]),
            "conf": float(detections.confidence[i]),
            "track_id": tid,
        })
    return result


def build_annotation_labels(detections: sv.Detections) -> list:
    """Etykiety '#ID class' do narysowania nad bboxami."""
    labels = []
    for i in range(len(detections)):
        cls = CLASS_NAMES.get(int(detections.class_id[i]), "?")
        tid = detections.tracker_id[i] if detections.tracker_id is not None else None
        if tid is not None and tid != -1:
            labels.append(f"#{tid} {cls}")
        else:
            labels.append(cls)
    return labels


# ==========================================================================
# GLOWNA FUNKCJA PIPELINE
# ==========================================================================

def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Wczytanie listy klatek ----
    frame_files = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"Brak klatek .jpg w {FRAMES_DIR}")
    print(f"[1/6] Znaleziono {len(frame_files)} klatek w {FRAMES_DIR.name}")

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise RuntimeError(f"Nie udalo sie wczytac {frame_files[0]}")
    h, w = first_frame.shape[:2]

    # ---- 2. Inicjalizacja detektora YOLO ----
    print(f"[2/6] Ladowanie YOLO z: {YOLO_WEIGHTS}")
    detector = YOLO(str(YOLO_WEIGHTS))

    # ---- 3. Inicjalizacja kalibratora (klasyczna homografia) ----
    print(f"[3/6] Inicjalizacja klasycznej homografii (obrazy {w}x{h})")
    calibrator = ClassicalHomography(
        image_width=w,
        image_height=h,
        ransac_iterations=RANSAC_ITERATIONS,
        min_score=RANSAC_MIN_SCORE,
    )

    # ---- 4. Plan kalibracji ----
    n_planned_calibs = len(frame_files) // CALIB_STRIDE + 1
    print(f"[4/6] Plan kalibracji: co {CALIB_STRIDE} klatek (~{n_planned_calibs} kalibracji)")
    H = None

    # ---- 5. Inicjalizacja trackera, anotatorow i minimapy ----
    tracker = ByteTrackTracker(frame_rate=FPS)
    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)

    minimap_renderer = MinimapRenderer(width_px=MINIMAP_WIDTH_PX)

    output_w = w + minimap_renderer.canvas_size[1]
    output_h = h
    output_w = output_w - (output_w % 2)

    # ---- 6. Petla po klatkach ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (output_w, output_h))

    print(f"[5/6] Przetwarzanie {len(frame_files)} klatek (output {output_w}x{output_h})...")
    csv_rows = []
    n_calibrations_ok = 0
    n_calibrations_failed = 0

    for frame_idx, frame_path in enumerate(frame_files):
        # Rekalibracja H
        if frame_idx % CALIB_STRIDE == 0:
            new_H = calibrator.get_homography(str(frame_path))
            if new_H is not None:
                H = new_H
                n_calibrations_ok += 1
            elif H is None:
                print(
                    f"  UWAGA: Pierwsza kalibracja (klatka {frame_idx}) nie powiodla sie. "
                    "Brak projekcji do momentu pierwszej udanej kalibracji."
                )
                n_calibrations_failed += 1
            else:
                n_calibrations_failed += 1

        frame = cv2.imread(str(frame_path))

        # Detekcja YOLO
        yolo_result = detector.predict(
            source=frame, conf=CONF_THRESHOLD, verbose=False, imgsz=960
        )[0]
        detections = sv.Detections.from_ultralytics(yolo_result)

        # Tracking
        detections = tracker.update(detections)

        # Projekcja na boisko (tylko jesli H dostepna)
        dets_dict = sv_detections_to_dict_list(detections)
        if H is not None:
            dets_projected = calibrator.project_detections_to_pitch(dets_dict, H)
        else:
            dets_projected = [{**d, "pitch_coords": None} for d in dets_dict]

        # CSV
        for d in dets_projected:
            pitch = d["pitch_coords"]
            csv_rows.append({
                "frame": frame_idx,
                "track_id": d["track_id"],
                "class": CLASS_NAMES.get(d["class"], "?"),
                "conf": d["conf"],
                "bbox_x1": d["bbox"][0],
                "bbox_y1": d["bbox"][1],
                "bbox_x2": d["bbox"][2],
                "bbox_y2": d["bbox"][3],
                "pitch_x_m": pitch[0] if pitch else None,
                "pitch_y_m": pitch[1] if pitch else None,
            })

        # Anotacja klatki
        labels = build_annotation_labels(detections)
        annotated = box_annotator.annotate(frame.copy(), detections=detections)
        annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)

        # Minimapa
        minimap = minimap_renderer.render(
            dets_projected,
            frame_idx=frame_idx,
            total_frames=len(frame_files),
            target_height=h,
        )

        composite = np.hstack([annotated, minimap])
        if composite.shape[1] != output_w:
            composite = composite[:, :output_w]

        writer.write(composite)

        if (frame_idx + 1) % 50 == 0:
            print(f"      {frame_idx + 1}/{len(frame_files)}")

    writer.release()

    # ---- 7. Zapis CSV ----
    print(f"[6/6] Zapis wynikow...")
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    n_tracks = df.dropna(subset=["track_id"])["track_id"].nunique()
    n_with_pitch = df["pitch_x_m"].notna().sum()
    print(f"\n  Wideo:    {OUTPUT_VIDEO}")
    print(f"  CSV:      {OUTPUT_CSV}")
    print(f"  Unikalne tracki: {n_tracks}")
    print(f"  Detekcje z pozycja na boisku: {n_with_pitch}/{len(df)}")
    print(f"  Kalibracji: {n_calibrations_ok} OK, {n_calibrations_failed} nieudanych")


if __name__ == "__main__":
    run_pipeline()
