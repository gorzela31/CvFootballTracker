"""
Plik: mot_to_yolo.py
Opis: Skrypt konwertujacy zbior danych z formatu MOT (SoccerNet) do formatu YOLO.
Obsluguje podzialy: train, valid, test. Generuje strukture katalogow:
yoloformat/[split]/[clip_name]/images oraz labels.
Automatycznie mapuje klasy na podstawie plikow gameinfo.ini.

Klasy wyjsciowe (3):
  0: ball
  1: player    (player + goalkeeper, obie druzyny)
  2: referee   (main + side referee)
Tracklety nie pasujace do zadnej kategorii (np. "other") sa pomijane.

Subsampling klatek:
  Kazdy klip ma 750 klatek (25 fps, 30 sekund). Sasiednie klatki sa mocno
  skorelowane, wiec dla traina bierzemy co N-ta klatke. Walidacja i test
  pozostaja w pelnej rozdzielczosci dla rzetelnej oceny.
"""

import os
import cv2
import shutil
from pathlib import Path

# Co ile klatek pobierac probki dla kazdego podzialu.
# 1 = wszystkie klatki, 3 = co trzecia (250 z 750), 5 = co piata (150 z 750).
FRAME_STRIDE = {
    "train": 3,
    "valid": 1,
    "test":  1,
}

def get_class_mapping(gameinfo_path):
    """
    Parsuje plik gameinfo.ini i zwraca mapowanie {tracklet_id: class_id}.
    Wartosc None oznacza tracklet do pominiecia (np. "other").
    """
    mapping = {}
    if not os.path.exists(gameinfo_path):
        return mapping

    with open(gameinfo_path, 'r') as f:
        for line in f:
            if "trackletID_" not in line:
                continue
            parts = line.strip().split('=')
            tid = int(parts[0].split('_')[1])
            desc = parts[1].lower()

            if 'ball' in desc:
                cid = 0
            elif 'player' in desc or 'goalkeeper' in desc:
                # obejmuje "player team left/right" oraz
                # "goalkeeper team left/right" i niespojna forme "goalkeepers"
                cid = 1
            elif 'referee' in desc:
                # obejmuje "main", "side top", "side bottom"
                cid = 2
            else:
                cid = None
            mapping[tid] = cid
    return mapping

def convert_clip(clip_source_path, output_base_dir, split_name):
    clip_name = os.path.basename(clip_source_path)
    target_dir = Path(output_base_dir) / split_name / clip_name
    img_target = target_dir / "images"
    lbl_target = target_dir / "labels"

    img_target.mkdir(parents=True, exist_ok=True)
    lbl_target.mkdir(parents=True, exist_ok=True)

    img_src_dir = os.path.join(clip_source_path, "img1")
    gt_path = os.path.join(clip_source_path, "gt", "gt.txt")
    gameinfo_path = os.path.join(clip_source_path, "gameinfo.ini")

    if not os.path.exists(gt_path):
        return

    class_map = get_class_mapping(gameinfo_path)
    stride = FRAME_STRIDE.get(split_name, 1)

    # Pobranie wymiarow z pierwszego dostepnego zdjecia
    images_list = sorted(os.listdir(img_src_dir))
    if not images_list:
        return

    sample = cv2.imread(os.path.join(img_src_dir, images_list[0]))
    if sample is None:
        print(f"OSTRZEZENIE: nie udalo sie wczytac {images_list[0]} w {clip_name}")
        return
    h_img, w_img, _ = sample.shape

    with open(gt_path, 'r') as f:
        lines = f.readlines()

    frame_data = {}
    skipped_unknown = 0
    for line in lines:
        p = line.strip().split(',')
        if len(p) < 6:
            continue
        fid, tid = int(p[0]), int(p[1])

        # Subsampling: bierzemy tylko klatki spelniajace warunek stride
        # (fid w SoccerNet zaczyna sie od 1, wiec uzywamy ((fid-1) % stride))
        if (fid - 1) % stride != 0:
            continue

        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])

        cid = class_map.get(tid)
        if cid is None:
            skipped_unknown += 1
            continue

        xc = (x + w/2) / w_img
        yc = (y + h/2) / h_img
        wn = w / w_img
        hn = h / h_img

        yolo_line = f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"
        if fid not in frame_data:
            frame_data[fid] = []
        frame_data[fid].append(yolo_line)

    print(f"Konwersja klipu: {split_name}/{clip_name} "
          f"(stride={stride}, klatek={len(frame_data)}, pominiete tracklety={skipped_unknown})")

    for fid, annots in frame_data.items():
        img_name = f"{fid:06d}.jpg"
        txt_name = f"{fid:06d}.txt"

        src_img_path = os.path.join(img_src_dir, img_name)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, img_target / img_name)
            with open(lbl_target / txt_name, 'w') as f:
                f.writelines(annots)

def process_all_splits(base_path, output_path, config):
    # Przetwarza wskazane podzialy i foldery zgodnie z konfiguracja
    for split, folders in config.items():
        print(f"\n=== Sekcja: {split} (stride={FRAME_STRIDE.get(split, 1)}) ===")
        for folder in folders:
            clip_path = os.path.join(base_path, split, folder)
            if os.path.exists(clip_path):
                convert_clip(clip_path, output_path, split)
            else:
                print(f"Pominiecie - brak folderu: {clip_path}")

if __name__ == "__main__":
    SOURCE_DATA = "data/tracking_dataset/tracking"
    YOLO_DATA = "data/tracking_dataset/yoloformat"

    # Konfiguracja: wybierz foldery dla kazdego podzialu
    processing_config = {
        "train": [f for f in os.listdir(os.path.join(SOURCE_DATA, "train")) if os.path.isdir(os.path.join(SOURCE_DATA, "train", f))],
        "valid": [f for f in os.listdir(os.path.join(SOURCE_DATA, "valid")) if os.path.isdir(os.path.join(SOURCE_DATA, "valid", f))],
        "test":  []
    }

    # Konwersja adnotacji i kopiowanie obrazow
    process_all_splits(SOURCE_DATA, YOLO_DATA, processing_config)

    print("\nProces zakonczony sukcesem.")