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
        # "train",  # Zbiór treningowy (ok. 50-60 folderów)
        # "challenge",  # Zbiór walidacyjny (do testowania w trakcie nauki w Colabie) <- nie istnieje w tym zadaniu.
        # "test",   # Zbiór testowy (do ostatecznej oceny modelu na koniec)
    ]
    
    # Zabezpieczenie przed odpaleniem "pustego" skryptu
    if not splits_to_download:
        print("\n⚠️ UWAGA: Nic nie wybrano!")
        print("Musisz odkomentować przynajmniej jeden zbiór (train, valid lub test) w kodzie skryptu.")
        return

    # Właściwe pobieranie
    print(f"\nRozpoczynam pobieranie paczki tracking dla zbiorów: {splits_to_download}...")
    myDownloader.downloadDataTask(task="tracking", split=splits_to_download)
    
    print(f"\n✅ Gotowe! Sprawdź folder {data_dir}/tracking/")

if __name__ == "__main__":
    main()