"""
Plik: pipelines/pipeline_fasterrcnn_bytetrack_tvcalib.py

Opis:
    Pipeline detekcja -> tracking -> kalibracja -> projekcja na boisko.

    Komponenty:
        Detekcja:    Faster R-CNN ResNet50-FPN v2 (fine-tuned na SoccerNet, 3 klasy)
        Tracking:    ByteTrack (supervision)
        Kalibracja:  TVCalib z dynamiczna rekalibracja co CALIB_STRIDE klatek
        Projekcja:   pozycje stop (srodek dolnej krawedzi bbox) -> metry na boisku
        Wizualizacja: minimapa boiska w ciemnym motywie obok klatki glownej

    Wejscie:
        Katalog z klatkami SoccerNet: SNMOT-XXX/img1/*.jpg

    Wyjscie (w results/<RUN_NAME>/):
        - output.mp4   : wideo z bboxami, klasami, ID trackow oraz minimapa
        - tracks.csv   : pozycje per (frame, track_id) w pikselach i metrach

Uzycie:
    # Domyslne sciezki
    python pipelines/pipeline_fasterrcnn_bytetrack_tvcalib.py

    # Wlasne sciezki
    python pipelines/pipeline_fasterrcnn_bytetrack_tvcalib.py \
        --frames data/tracking_dataset/tracking/test/SNMOT-116/img1 \
        --frcnn-weights models/faster_rcnn/trained_fasterrcnn_resnet50.pt \
        --conf 0.15 --calib-stride 10
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from src.detection.faster_rcnn import FasterRCNNDetector
from src.calibration.homography import TVCalibHomography
from src.tracking.bytetrack_tracker import ByteTrackTracker
from src.visualization.minimap import MinimapRenderer
from src.classification.team_classifier import TeamClassifier, annotate_frame_with_teams


# ==========================================================================
# DOMYSLNA KONFIGURACJA
# ==========================================================================
RUN_NAME = "pipeline_frcnn_bt_tvcalib"

FRAMES_DIR = PROJECT_ROOT / "data" / "tracking_dataset" / "tracking" / "test" / "SNMOT-123" / "img1"
FRCNN_WEIGHTS = PROJECT_ROOT / "models" / "faster_rcnn" / "trained_fasterrcnn_resnet50.pt"
TVCALIB_WEIGHTS = PROJECT_ROOT / "src" / "calibration" / "tvcalib" / "data" / "segment_localization" / "train_59.pt"

CONF_THRESHOLD = 0.10
FPS = 25
CALIB_STRIDE = 25 # co ile klatek rekalibrowac homografie TVCalib (im mniejszy, tym bardziej odporny na dryft ale wolniejszy pipeline)
OPTIM_STEPS = 500 # liczba krokow optymalizacji TVCalib (im wiecej, tym dokladniejsza ale wolniejsza kalibracja)

CLASS_NAMES = {0: "ball", 1: "player", 2: "referee"}
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
            if detections.tracker_id is not None
            and detections.tracker_id[i] is not None
            and int(detections.tracker_id[i]) != -1
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
        if tid is not None and int(tid) != -1:
            labels.append(f"#{int(tid)} {cls}")
        else:
            labels.append(cls)
    return labels


# ==========================================================================
# GLOWNA FUNKCJA PIPELINE
# ==========================================================================

def run_pipeline(
    frames_dir: Path = FRAMES_DIR,
    frcnn_weights: Path = FRCNN_WEIGHTS,
    tvcalib_weights: Path = TVCALIB_WEIGHTS,
    conf: float = CONF_THRESHOLD,
    calib_stride: int = CALIB_STRIDE,
    run_name: str = RUN_NAME,
):
    output_dir = PROJECT_ROOT / "results" / run_name
    _ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_video = output_dir / f"{_ts}_output.mp4"
    output_csv = output_dir / f"{_ts}_tracks.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.time()

    # ---- 1. Wczytanie listy klatek ----
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"Brak klatek .jpg w {frames_dir}")
    print(f"[1/6] Znaleziono {len(frame_files)} klatek w {frames_dir.name}")

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise RuntimeError(f"Nie udalo sie wczytac {frame_files[0]}")
    h, w = first_frame.shape[:2]
    print(f"      Rozdzielczosc: {w}x{h}")

    # ---- 2. Inicjalizacja detektora Faster R-CNN ----
    if frcnn_weights.exists():
        print(f"[2/6] Ladowanie Faster R-CNN (fine-tuned): {frcnn_weights.name}")
        detector = FasterRCNNDetector(weights_path=str(frcnn_weights), conf_threshold=conf)
    else:
        print(f"[2/6] Brak fine-tuned wag ({frcnn_weights.name}), uzywam COCO pretrained")
        print("      UWAGA: brak rozroznienia player/referee w trybie COCO")
        detector = FasterRCNNDetector(weights_path=None, conf_threshold=conf)

    # ---- 3. Inicjalizacja kalibratora TVCalib ----
    print(f"[3/6] Inicjalizacja TVCalib")
    calibrator = TVCalibHomography(
        model_weights=str(tvcalib_weights),
        image_width=w,
        image_height=h,
        optim_steps=OPTIM_STEPS,
    )

    # ---- 4. Plan kalibracji ----
    n_planned_calibs = len(frame_files) // calib_stride + 1
    print(f"[4/6] Plan kalibracji: co {calib_stride} klatek (~{n_planned_calibs} kalibracji)")
    H = None

    # ---- 5. Inicjalizacja trackera, anotatorow i minimapy ----
    tracker = ByteTrackTracker(frame_rate=FPS)

    team_classifier = TeamClassifier()

    minimap_renderer = MinimapRenderer(width_px=MINIMAP_WIDTH_PX)

    output_w = w + minimap_renderer.canvas_size[1]
    output_h = h
    output_w = output_w - (output_w % 2)

    # ---- 6. Petla po klatkach ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, FPS, (output_w, output_h))

    print(f"[5/6] Przetwarzanie {len(frame_files)} klatek (output {output_w}x{output_h})...")
    print(f"      Detekcja: Faster R-CNN | Kalibracja: TVCalib | Tracking: ByteTrack")
    print()

    csv_rows = []
    n_calibrations_ok = 0
    n_calibrations_failed = 0
    t_det_total = 0.0
    t_calib_total = 0.0

    for frame_idx, frame_path in enumerate(frame_files):
        # --- Rekalibracja H ---
        if frame_idx % calib_stride == 0:
            t0 = time.time()
            try:
                new_H = calibrator.get_homography(str(frame_path))
            except Exception as e:
                print(f"      [WARN] Kalibracja klatki {frame_idx} rzucila wyjatek: {e}")
                new_H = None
            t_calib_total += time.time() - t0

            if new_H is not None:
                H = new_H
                n_calibrations_ok += 1
            else:
                n_calibrations_failed += 1
                if H is None and frame_idx == 0:
                    print(f"      [WARN] Pierwsza kalibracja nie powiodla sie, "
                          f"projekcja niedostepna do pierwszej udanej kalibracji")

        frame = cv2.imread(str(frame_path))

        # --- Detekcja Faster R-CNN ---
        t0 = time.time()
        detections = detector.predict(frame, conf=conf)
        t_det_total += time.time() - t0

        # --- Tracking ---
        if len(detections) > 0:
            detections = tracker.update(detections)

        # --- Klasyfikacja druzyn ---
        team_ids = team_classifier.classify(frame, detections)

        # --- Projekcja na boisko ---
        dets_dict = sv_detections_to_dict_list(detections)
        for i, d in enumerate(dets_dict):
            d["team_id"] = int(team_ids[i])
        if H is not None:
            dets_projected = calibrator.project_detections_to_pitch(dets_dict, H)
        else:
            dets_projected = [{**d, "pitch_coords": None} for d in dets_dict]

        # --- CSV ---
        for d in dets_projected:
            pitch = d["pitch_coords"]
            csv_rows.append({
                "frame": frame_idx,
                "track_id": d["track_id"],
                "class": CLASS_NAMES.get(d["class"], "?"),
                "conf": round(d["conf"], 4),
                "bbox_x1": round(d["bbox"][0], 1),
                "bbox_y1": round(d["bbox"][1], 1),
                "bbox_x2": round(d["bbox"][2], 1),
                "bbox_y2": round(d["bbox"][3], 1),
                "pitch_x_m": round(pitch[0], 3) if pitch else None,
                "pitch_y_m": round(pitch[1], 3) if pitch else None,
                "team_id": d.get("team_id", -1),
            })

        # --- Anotacja klatki ---
        if len(detections) > 0:
            annotated = annotate_frame_with_teams(frame, detections, team_ids, CLASS_NAMES)
        else:
            annotated = frame.copy()

        # --- Minimapa ---
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

        # Progress co 50 klatek
        if (frame_idx + 1) % 50 == 0 or frame_idx == len(frame_files) - 1:
            elapsed = time.time() - t_pipeline_start
            fps_proc = (frame_idx + 1) / elapsed
            print(f"      {frame_idx + 1:>4d}/{len(frame_files)} "
                  f"({fps_proc:.1f} fps, {elapsed:.0f}s elapsed)")

    writer.release()

    # ---- 7. Zapis CSV ----
    print(f"\n[6/6] Zapis wynikow...")
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_csv, index=False)

    # Podsumowanie
    t_total = time.time() - t_pipeline_start
    n_tracks = df.dropna(subset=["track_id"])["track_id"].nunique() if len(df) > 0 else 0
    n_with_pitch = df["pitch_x_m"].notna().sum() if len(df) > 0 else 0
    n_frames = len(frame_files)

    print(f"\n{'='*60}")
    print(f"  Pipeline:  {run_name}")
    print(f"  Wideo:     {output_video}")
    print(f"  CSV:       {output_csv}")
    print(f"  Klatki:    {n_frames}")
    print(f"  Unikalne tracki: {n_tracks}")
    print(f"  Detekcje z pozycja na boisku: {n_with_pitch}/{len(df)}")
    print(f"  Kalibracje: {n_calibrations_ok} OK, {n_calibrations_failed} nieudanych")
    print(f"  --- Czasy ---")
    print(f"  Detekcja (lacznie):    {t_det_total:.1f}s  ({t_det_total/n_frames*1000:.0f} ms/klatka)")
    print(f"  Kalibracja (lacznie):  {t_calib_total:.1f}s  ({t_calib_total/max(n_calibrations_ok+n_calibrations_failed,1)*1000:.0f} ms/kalibracja)")
    print(f"  Calosc:                {t_total:.1f}s  ({n_frames/t_total:.1f} fps)")
    print(f"{'='*60}")


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline: Faster R-CNN + ByteTrack + TVCalib"
    )
    parser.add_argument(
        "--frames", type=str, default=str(FRAMES_DIR),
        help="Katalog z klatkami .jpg (domyslnie SNMOT-123/img1)"
    )
    parser.add_argument(
        "--frcnn-weights", type=str, default=str(FRCNN_WEIGHTS),
        help="Sciezka do wag Faster R-CNN (None = COCO pretrained)"
    )
    parser.add_argument(
        "--tvcalib-weights", type=str, default=str(TVCALIB_WEIGHTS),
        help="Sciezka do wag TVCalib (segmentacja)"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"Prog confidence detekcji (domyslnie {CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--calib-stride", type=int, default=CALIB_STRIDE,
        help=f"Co ile klatek rekalibracja homografii (domyslnie {CALIB_STRIDE})"
    )
    parser.add_argument(
        "--run-name", type=str, default=RUN_NAME,
        help=f"Nazwa uruchomienia / podkatalog wynikow (domyslnie {RUN_NAME})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        frames_dir=Path(args.frames),
        frcnn_weights=Path(args.frcnn_weights),
        tvcalib_weights=Path(args.tvcalib_weights),
        conf=args.conf,
        calib_stride=args.calib_stride,
        run_name=args.run_name,
    )
