"""
Plik: download_tracking.py
Opis: Skrypt do pobierania danych 'tracking' z bazy SoccerNet dla treningu modeli detekcji i śledzenia.
Funkcjonalność:
  - Pobiera wybrane splity danych (train, test) z SoccerNet
  - Zapisuje dane do folderu data/tracking_dataset/tracking/
  - Każdy klip zawiera: klatki wideo (img1/), adnotacje MOT (gt/), metadane (gameinfo.ini)
Bezpieczeństwo:
  - Domyślnie nic nie pobiera - trzeba odkomentować wybrane splity w kodzie
  - Wymaga zainstalowanego pakietu SoccerNet (pip install SoccerNet)
Użycie:
  1. Odkomentuj żądane splity w liście splits_to_download
  2. Uruchom: python scripts/download_tracking.py
"""

import os
from SoccerNet.Downloader import SoccerNetDownloader

def main():
    # Pobieramy absolutną ścieżkę do folderu głównego projektu
    project_root = os.getcwd()
    
    # Ustawiamy docelową ścieżkę: data/tracking_dataset
    data_dir = os.path.join(project_root, "data", "tracking_dataset")
    
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
        # "train",  # Zbiór treningowy
        # "challenge",  # Zbiór walidacyjny
        # "test",   # Zbiór testowy
    ]
    
    # Zabezpieczenie przed odpaleniem "pustego" skryptu
    if not splits_to_download:
        print("\n UWAGA: Nic nie wybrano!")
        print("Musisz odkomentować przynajmniej jeden zbiór (train, valid lub test) w kodzie skryptu.")
        return

    # Właściwe pobieranie
    print(f"\nRozpoczynam pobieranie paczki tracking dla zbiorów: {splits_to_download}...")
    myDownloader.downloadDataTask(task="tracking", split=splits_to_download)
    
    print(f"\n✅ Gotowe! Sprawdź folder {data_dir}/tracking/")

if __name__ == "__main__":
    main()