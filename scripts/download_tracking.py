"""
Plik: scripts/download_tracking.py

Opis:
    Pobiera wybrane splity zbioru SoccerNet-Tracking.

Klipy z klatkami, adnotacjami MOT i metadanymi trafiają do
`data/tracking_dataset/tracking/`. Lista splitów jest domyślnie pusta,
aby przypadkowe uruchomienie nie rozpoczęło dużego pobierania.
"""

import os
from SoccerNet.Downloader import SoccerNetDownloader

def main():
    project_root = os.getcwd()
    
    data_dir = os.path.join(project_root, "data", "tracking_dataset")
    
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Dane zostaną pobrane do: {data_dir}")
    
    myDownloader = SoccerNetDownloader(LocalDirectory=data_dir)
    
    # Wybierz splity do pobrania. Pusta lista oznacza brak pobierania.
    splits_to_download = [
        # "train",  # Zbiór treningowy
        # "challenge",  # Zbiór walidacyjny
        # "test",   # Zbiór testowy
    ]
    
    # Nie pobieraj niczego bez jawnie wybranych splitów.
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