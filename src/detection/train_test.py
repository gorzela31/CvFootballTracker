"""
Plik: src/detection/train_test.py

Opis:
    Sprawdza konfigurację zbioru YOLO przez minimalny trening testowy.

Generuje tymczasowy `data.yaml` i uruchamia krótki trening CPU, aby wykryć
problemy z etykietami lub strukturą danych przed właściwym treningiem.
"""

import os
from ultralytics import YOLO

def create_dataset_config(dataset_dir, yaml_path):
    """Generuje plik data.yaml niezbędny do zainicjowania procesu uczenia."""
    content = f"""
path: {dataset_dir}
train: train
val: valid
test: test

names:
  0: ball
  1: team_1
  2: team_2
  3: referee
"""
    os.makedirs(dataset_dir, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content.strip())

def main():
    project_root = os.getcwd()
    dataset_dir = os.path.join(project_root, "data", "tracking_dataset", "yoloformat")
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    create_dataset_config(dataset_dir, yaml_path)

    # Inicjalizacja bazowego modelu YOLO (wersja Nano dla optymalizacji zasobow)
    base_model_path = os.path.join(project_root, "models", "yolov8n", "yolov8n.pt")
    model = YOLO(base_model_path)

    # Uruchomienie testowej iteracji uczenia
    try:
        model.train(
            data=yaml_path,
            epochs=1,
            imgsz=640,
            device='cpu',
            batch=2,
            project="results",
            name="environment_verification"
        )
    except Exception as e:
        print(f"Wystapil blad krytyczny podczas testu srodowiska: {e}")

if __name__ == "__main__":
    main()