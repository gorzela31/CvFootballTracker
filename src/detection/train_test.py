import os
from ultralytics import YOLO

def main():
    # 1. Ścieżki absolutne (żeby YOLO się nie zgubiło)
    project_root = os.getcwd()
    dataset_dir = os.path.join(project_root, "data", "yolo_dataset")
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    # 2. Dynamiczne tworzenie pliku data.yaml
    # Na razie jako zbiór treningowy i walidacyjny podajemy to samo (bo mamy tylko 1 klip testowy)
    yaml_content = f"""
path: {dataset_dir}
train: images/train
val: images/train

# Tymczasowo uczymy model wykrywać wszystkie obiekty jako jedną klasę
nc: 1
names: ['object']
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Utworzono plik konfiguracyjny: {yaml_path}")

    # 3. Inicjalizacja modelu YOLO
    # Pobieramy najmniejszy i najszybszy model (Nano), idealny do testów na CPU
    print("⏳ Pobieranie wag modelu YOLOv8n...")
    model = YOLO("yolov8n.pt") 

    # 4. Uruchomienie treningu
    print("🔥 Rozpoczynamy testowy trening na CPU (1 epoka)...")
    results = model.train(
        data=yaml_path,
        epochs=1,          # Tylko 1 epoka, żeby zobaczyć czy działa
        imgsz=640,         # Rozdzielczość, do której YOLO przeskaluje zdjęcia
        device='cpu',      # Wymuszamy CPU, skoro lokalnie nie masz GPU
        batch=4,           # Mały batch size, żeby nie zapchać RAMu
        project="results", # Gdzie zapisać wyniki
        name="yolo_test_run" # Nazwa folderu z wynikami
    )
    
    print("✅ Trening zakończony! Sprawdź folder results/yolo_test_run/")

if __name__ == "__main__":
    main()