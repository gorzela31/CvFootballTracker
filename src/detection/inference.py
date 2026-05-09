"""
Plik: inference.py
Opis: Skrypt realizuje proces inferencji przy uzyciu wytrenowanego modelu YOLO 
na sekwencji obrazow (klatek wideo). Program wczytuje klatki z okreslonego 
katalogu, wykonuje detekcje obiektow dla kazdej klatki, nanosi wizualizacje 
wynikow (bounding boxy, etykiety klas) i zapisuje koncowy wynik w formie 
pliku wideo MP4.
"""

import cv2
import os
from ultralytics import YOLO

def generate_test_video():
    # 1. Konfiguracja sciezek projektowych
    project_root = os.getcwd()
    model_path = os.path.join(project_root, "models", "yolov8n", "trained_detection_yolov8n.pt")
    frames_dir = os.path.join(project_root, "data", "tracking_dataset", "tracking", "test", "SNMOT-123", "img1")
    output_video = os.path.join(project_root, "results", "tracking_test_output_1.mp4")
    
    # Tworzenie katalogu wynikowego, jesli nie istnieje
    os.makedirs("results", exist_ok=True)
    
    # 2. Inicjalizacja modelu wagami w formacie PyTorch
    print(f"Ladowanie modelu z lokalizacji: {model_path}")
    model = YOLO(model_path)
    
    # 3. Pobieranie i sortowanie listy klatek wejsciowych
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    frames.sort()
    
    # Weryfikacja wymiarow obrazu na podstawie pierwszej klatki
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    h, w, _ = first_frame.shape
    
    # 4. Inicjalizacja obiektu VideoWriter do zapisu pliku wynikowego
    # Wykorzystano kodek mp4v oraz standardowa predkosc 25 klatek na sekunde
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25.0
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    print(f"Rozpoczecie przetwarzania sekwencji: {len(frames)} klatek")
    
    # 5. Petla przetwarzania sekwencyjnego i detekcji
    for i, frame_name in enumerate(frames):
        img_path = os.path.join(frames_dir, frame_name)
        
        # Wykonanie predykcji z ustalonym progiem ufnosci (confidence threshold)
        # verbose=False ogranicza nadmiarowe logowanie w konsoli
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        # Generowanie obrazu z naniesionymi wynikami detekcji (bounding boxy)
        annotated_frame = results[0].plot(
        line_width=1,      # Minimalizacja grubosci ramki obiektow
        font_size=0.5,     # Skalowanie rozmiaru etykiet klasyfikacyjnych
        labels=True,
        conf=True
    )
        # Zapis przetworzonej klatki do bufora wideo
        out.write(annotated_frame)
        
        # Monitorowanie postepu prac co 30 klatek
        if (i + 1) % 30 == 0:
            print(f"Przetworzono: {i + 1} / {len(frames)} klatek")
            
    # Zwolnienie zasobow i zamkniecie pliku wideo
    out.release()
    print(f"Proces zakonczony. Plik wynikowy zapisano w: {output_video}")

if __name__ == "__main__":
    generate_test_video()