"""
Plik: mot_to_yolo.py
Opis: Skrypt konwertujacy zbior danych z formatu MOT (SoccerNet) do formatu YOLO.
Obsluguje podzialy: train, valid, test. Generuje strukture katalogow:
yoloformat/[split]/[clip_name]/images oraz labels.
Automatycznie mapuje klasy na podstawie plikow gameinfo.ini.
"""

import os
import cv2
import shutil
from pathlib import Path

def get_class_mapping(gameinfo_path):
    # Parsuje plik gameinfo.ini i zwraca mapowanie {tracklet_id: class_id}
    mapping = {}
    if not os.path.exists(gameinfo_path):
        return mapping
    
    with open(gameinfo_path, 'r') as f:
        for line in f:
            if "trackletID_" in line:
                parts = line.strip().split('=')
                tid = int(parts[0].split('_')[1])
                desc = parts[1].lower()
                
                if 'ball' in desc:
                    cid = 0
                elif 'team left' in desc:
                    cid = 1
                elif 'team right' in desc:
                    cid = 2
                elif 'referee' in desc:
                    cid = 3
                else:
                    cid = 0
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
    
    # Pobranie wymiarow z pierwszego dostepnego zdjecia
    images_list = sorted(os.listdir(img_src_dir))
    if not images_list:
        return
        
    sample = cv2.imread(os.path.join(img_src_dir, images_list[0]))
    h_img, w_img, _ = sample.shape

    with open(gt_path, 'r') as f:
        lines = f.readlines()

    frame_data = {}
    for line in lines:
        p = line.strip().split(',')
        if len(p) < 6: continue
        fid, tid = int(p[0]), int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        
        cid = class_map.get(tid, 0)
        
        xc = (x + w/2) / w_img
        yc = (y + h/2) / h_img
        wn = w / w_img
        hn = h / h_img
        
        yolo_line = f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"
        if fid not in frame_data: 
            frame_data[fid] = []
        frame_data[fid].append(yolo_line)

    print(f"Konwersja klipu: {split_name}/{clip_name}")
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
        print(f"Rozpoczynanie przetwarzania sekcji: {split}")
        for folder in folders:
            clip_path = os.path.join(base_path, split, folder)
            if os.path.exists(clip_path):
                convert_clip(clip_path, output_path, split)
            else:
                print(f"Pominiecie - brak folderu: {clip_path}")

def generate_yolo_lists(output_path):
    """
    Generuje pliki .txt z listami absolutnych sciezek do obrazow dla kazdego podzialu.
    Umozliwia to YOLO poprawne odczytanie danych przy strukturze wielofolderowej.
    """
    splits = ['train', 'valid', 'test']
    for split in splits:
        split_dir = Path(output_path) / split
        if not split_dir.exists():
            continue
            
        image_paths = []
        for path in split_dir.rglob('*.jpg'):
            if 'images' in path.parts:
                image_paths.append(str(path.absolute()))
        
        if image_paths:
            list_file = Path(output_path) / f"{split}.txt"
            with open(list_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(image_paths))
            print(f"Wygenerowano liste: {list_file}")

if __name__ == "__main__":
    SOURCE_DATA = "data/tracking_dataset/tracking"
    YOLO_DATA = "data/tracking_dataset/yoloformat"
    
    # Konfiguracja: wybierz foldery dla kazdego podzialu
    # Mozesz wpisac nazwy recznie lub uzyc listdir dla wszystkich
    processing_config = {
        "train": ["SNMOT-060", "SNMOT-061"],
        "valid": ["SNMOT-160", "SNMOT-161"],
        "test":  ["SNMOT-116", "SNMOT-117"]
    }

    # 1. Konwersja adnotacji i kopiowanie obrazow
    process_all_splits(SOURCE_DATA, YOLO_DATA, processing_config)
    
    # 2. Generowanie list sciezrek dla frameworka YOLO
    generate_yolo_lists(YOLO_DATA)

    print("Proces zakonczony sukcesem.")