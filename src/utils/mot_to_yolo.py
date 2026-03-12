import os
import cv2
import shutil

def convert_soccernet_to_yolo(sn_clip_path, output_base_dir):
    img_dir = os.path.join(sn_clip_path, "img1")
    gt_path = os.path.join(sn_clip_path, "gt", "gt.txt")
    
    if os.path.exists(output_base_dir):
        print("🧹 Usuwanie starego folderu i plików cache...")
        shutil.rmtree(output_base_dir)
        
    yolo_images_dir = os.path.join(output_base_dir, "images", "train")
    yolo_labels_dir = os.path.join(output_base_dir, "labels", "train")
    
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    first_img_name = os.listdir(img_dir)[0]
    sample_img = cv2.imread(os.path.join(img_dir, first_img_name))
    img_height, img_width, _ = sample_img.shape

    with open(gt_path, 'r') as f:
        lines = f.readlines()

    frame_annotations = {}
    skipped_boxes = 0 
    
    for line in lines:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        
        # Prawdziwe współrzędne 2 punktów ramki
        x1 = float(parts[2])
        y1 = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        
        x2 = x1 + w
        y2 = y1 + h
        
        # Bezpieczne przycinanie ramek, jeśli wystają poza obraz
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            skipped_boxes += 1
            continue

        # ROZWIĄZANIE PROBLEMU: Wymuszamy klasę 0 ("obiekt"), bo SoccerNet trzyma
        # prawdziwe etykiety w gameinfo.ini, a w gt.txt wpisuje bezużyteczne "-1".
        class_id = 0

        # Normalizacja YOLO
        x_center = (x1 + w / 2) / img_width
        y_center = (y1 + h / 2) / img_height
        norm_w = w / img_width
        norm_h = h / img_height
        
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
        
        if frame_id not in frame_annotations:
            frame_annotations[frame_id] = []
        frame_annotations[frame_id].append(yolo_line)

    print(f"Rozpoczynam konwersję (odrzucono {skipped_boxes} ramek poza krawędzią ekranu)...")
    
    for frame_id, annots in frame_annotations.items():
        base_name = f"{frame_id:06d}"
        src_img = os.path.join(img_dir, f"{base_name}.jpg")
        dst_img = os.path.join(yolo_images_dir, f"{sn_clip_path[-9:]}_{base_name}.jpg") 
        dst_txt = os.path.join(yolo_labels_dir, f"{sn_clip_path[-9:]}_{base_name}.txt")
        
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            with open(dst_txt, 'w') as f_out:
                f_out.writelines(annots)

    print(f"✅ Konwersja zakończona! Skopiowano i przetworzono {len(frame_annotations)} klatek.")

if __name__ == "__main__":
    SOURCE_CLIP = "data/tracking_dataset/tracking/train/SNMOT-060"
    OUTPUT_YOLO_DATASET = "data/yolo_dataset"
    convert_soccernet_to_yolo(SOURCE_CLIP, OUTPUT_YOLO_DATASET)