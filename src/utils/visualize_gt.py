import cv2
import os

def draw_soccernet_gt(image_path, gt_file_path, target_frame_id, output_path):
    """
    Funkcja wczytuje klatkę, parsuje plik gt.txt w formacie MOT i rysuje bounding boxy.
    """
    # 1. Wczytanie obrazu
    img = cv2.imread(image_path)
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu z {image_path}")
        return

    # 2. Otwarcie pliku z adnotacjami (gt.txt)
    if not os.path.exists(gt_file_path):
        print(f"Błąd: Nie znaleziono pliku {gt_file_path}")
        return

    with open(gt_file_path, 'r') as f:
        lines = f.readlines()

    # 3. Parsowanie linijek i rysowanie
    boxes_drawn = 0
    for line in lines:
        parts = line.strip().split(',')
        
        # Pobieramy ID klatki z pierwszej kolumny
        frame_id = int(parts[0])
        
        # Interesuje nas tylko klatka, którą właśnie wczytaliśmy
        if frame_id == target_frame_id:
            # Format MOT: frame_id, track_id, x, y, w, h, active, label, visibility
            x = int(float(parts[2]))
            y = int(float(parts[3]))
            w = int(float(parts[4]))
            h = int(float(parts[5]))
            
            # Rysowanie prostokąta (Obraz, Lewy_Górny, Prawy_Dolny, Kolor BGR(Zielony), Grubość)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes_drawn += 1

    # 4. Zapisanie wyniku
    # Upewniamy się, że folder docelowy istnieje
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"✅ Sukces! Narysowano {boxes_drawn} obiektów na klatce {target_frame_id}.")
    print(f"Wynik zapisano w: {output_path}")

if __name__ == "__main__":
    # Ścieżki dopasowane do Twojego screena (folder SNMOT-060)
    # Zwróć uwagę, że klatki są numerowane od 000001 do 00000... coś tam
    base_dir = "data/tracking_dataset/tracking/train/SNMOT-060"
    
    SAMPLE_IMAGE = os.path.join(base_dir, "img1/000001.jpg") 
    SAMPLE_GT = os.path.join(base_dir, "gt/gt.txt")
    
    # Zapiszemy wynik w folderze results (który jest ignorowany przez Gita)
    OUTPUT_FILE = "results/test_bbox_SNMOT_060.jpg"
    
    draw_soccernet_gt(
        image_path=SAMPLE_IMAGE, 
        gt_file_path=SAMPLE_GT, 
        target_frame_id=1, 
        output_path=OUTPUT_FILE
    )