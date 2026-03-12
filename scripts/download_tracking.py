import os
from SoccerNet.Downloader import SoccerNetDownloader

def main():
    # Pobieramy absolutną ścieżkę do folderu głównego projektu
    project_root = os.getcwd()
    
    # Ustawiamy nową, docelową ścieżkę: data/tracking_dataset
    data_dir = os.path.join(project_root, "data", "tracking_dataset")
    
    # Tworzymy strukturę katalogów, jeśli nie istnieje
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Dane zostaną pobrane do: {data_dir}")
    
    # Inicjalizacja oficjalnego downloadera ze wskazanym nowym katalogiem
    myDownloader = SoccerNetDownloader(LocalDirectory=data_dir)
    
    # Pobieramy zadanie "tracking", TYLKO zbiór "train"
    print("Rozpoczynam pobieranie paczki tracking (zbiór train)...")
    myDownloader.downloadDataTask(task="tracking", split=["train"])
    
    print(f"\n✅ Gotowe! Sprawdź folder {data_dir}/tracking/train/")

if __name__ == "__main__":
    main()