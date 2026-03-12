import cv2
import os
from ultralytics import YOLO

def generate_test_video():
    # 1. Ścieżki
    model_path = "models/best.pt"  # Upewnij się, że plik tu jest!
    frames_dir = "data/tracking_dataset/tracking/train/SNMOT-060/img1"
    output_video = "results/pierwszy_test_yolo.mp4"
    
    # Upewniamy się, że folder docelowy istnieje
    os.makedirs("results", exist_ok=True)
    
    # 2. Wczytanie Twojego modelu
    print(f"⏳ Ładowanie modelu z {model_path}...")
    model = YOLO(model_path)
    
    # 3. Pobranie listy klatek wideo
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    frames.sort() # Sortujemy, żeby klatki szły po kolei (000001, 000002...)
    
    # Bierzemy WSZYSTKIE klatki z folderu (pełne 30 sekund)
    test_frames = frames 
    
    # Pobieramy wymiary z pierwszej klatki
    first_frame = cv2.imread(os.path.join(frames_dir, test_frames[0]))
    h, w, _ = first_frame.shape
    
    # 4. Inicjalizacja "sklejacza" wideo (OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Kodek MP4
    fps = 25.0
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    print(f"🎬 Rozpoczynam generowanie wideo ({len(test_frames)} klatek)...")
    
    # 5. Główna pętla detekcji
    for i, frame_name in enumerate(test_frames):
        img_path = os.path.join(frames_dir, frame_name)
        
        # YOLO robi detekcję (conf=0.25 oznacza, że ignoruje domysły poniżej 25% pewności)
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        # Wynik .plot() automatycznie rysuje ramki i procenty na zdjęciu!
        annotated_frame = results[0].plot()
        
        # Wrzucamy gotową klatkę do wideo
        out.write(annotated_frame)
        
        if (i+1) % 30 == 0:
            print(f"Przetworzono {i+1}/{len(test_frames)} klatek...")
            
    out.release()
    print(f"✅ GOTOWE! Otwórz plik: {output_video}")

if __name__ == "__main__":
    generate_test_video()