"""
Plik: download_calibration.py
Opis: Skrypt do pobierania danych 'calibration' z bazy SoccerNet
      dla treningu i ewaluacji modeli estymacji homografii.
Funkcjonalność:
  - Pobiera wybrane splity danych (train, valid, test) z SoccerNet
  - Zapisuje dane do folderu data/calibration_dataset/calibration/
  - Każdy split zawiera: obrazy klatek (.jpg), anotacje kalibracji (.json)
    oraz metadane kamery (match_info_cam_gt.json)
Bezpieczeństwo:
  - Domyślnie nic nie pobiera - trzeba odkomentować wybrane splity w kodzie
  - Wymaga zainstalowanego pakietu SoccerNet (pip install SoccerNet)
Uwagi dot. rozmiarów:
  - train  ~12 GB  (używany do treningu modelu segmentacji boiska)
  - valid  ~1.5 GB (używany do ewaluacji i doboru hiperparametrów)
  - test   ~1.5 GB (bez ground truth - do submisji challenge)
Użycie:
  1. Ustaw zmienną SOCCERNET_PASSWORD (patrz niżej)
  2. Odkomentuj żądane splity w liście splits_to_download
  3. Uruchom: python scripts/download_calibration.py
"""

import os
from SoccerNet.Downloader import SoccerNetDownloader


def main():
    # Pobieramy absolutną ścieżkę do folderu głównego projektu
    project_root = os.getcwd()

    # Ustawiamy docelową ścieżkę: data/calibration_dataset
    data_dir = os.path.join(project_root, "data", "calibration_dataset")

    # Tworzymy strukturę katalogów, jeśli nie istnieje
    os.makedirs(data_dir, exist_ok=True)

    print(f"Dane zostaną pobrane do: {data_dir}")

    # Inicjalizacja oficjalnego downloadera ze wskazanym katalogiem
    myDownloader = SoccerNetDownloader(LocalDirectory=data_dir)

    # ==========================================================
    # USTAW BEZPIECZNIK: Odkomentuj (usuń '#') linijki poniżej,
    # aby wybrać, które zbiory chcesz teraz pobrać.
    # ==========================================================
    splits_to_download = [
         "valid",  # Zbiór walidacyjny (~1.5 GB) - zalecany na start
         "train",  # Zbiór treningowy  (~12 GB)  - do treningu segmentacji
         "test",   # Zbiór testowy     (~1.5 GB) - bez GT, do challenge
    ]

    # Zabezpieczenie przed odpaleniem "pustego" skryptu
    if not splits_to_download:
        print("\n UWAGA: Nic nie wybrano!")
        print("Musisz odkomentować przynajmniej jeden zbiór (valid, train lub test) w kodzie skryptu.")
        return

    # Właściwe pobieranie
    print(f"\nRozpoczynam pobieranie paczki calibration dla zbiorów: {splits_to_download}...")
    myDownloader.downloadDataTask(task="calibration", split=splits_to_download)

    print(f"\n Gotowe! Sprawdź folder: {data_dir}/calibration/")
    print("\nStruktura pobranych danych:")
    print(f"  {data_dir}/calibration/")
    print("  ├── <split>/")
    print("  │   ├── <match_id>/")
    print("  │   │   ├── <frame_id>.jpg         # klatka wideo")
    print("  │   │   └── <frame_id>.json        # anotacje kalibracji (GT homografia)")
    print("  │   └── match_info_cam_gt.json     # typy kamer dla całego splitu")


if __name__ == "__main__":
    main()
