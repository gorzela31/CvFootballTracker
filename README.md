# CvFootballTracker

Repozytorium projektu zrealizowanego w ramach pracy magisterskiej:

**„Porównanie metod detekcji i lokalizacji zawodników piłki nożnej na płaszczyźnie boiska w oparciu o obraz transmisji telewizyjnej”**

## Cel projektu

Celem projektu jest detekcja obiektów na obrazie transmisji meczu piłkarskiego oraz odwzorowanie ich położenia z układu współrzędnych obrazu na dwuwymiarowy model boiska.

Główna część badawcza obejmuje porównanie:

- dwóch metod detekcji obiektów: **YOLOv8n** oraz **Faster R-CNN ResNet50-FPN**, 
- dwóch metod estymacji homografii: **detekcji 32 punktów charakterystycznych boiska za pomocą YOLOv8x-pose i RANSAC** oraz **TVCalib**.

Połączenie tych metod daje cztery warianty systemu:

| Wariant | Detekcja | Estymacja homografii |
| --- | --- | --- |
| I | YOLOv8n | Punkty charakterystyczne boiska |
| II | YOLOv8n | TVCalib |
| III | Faster R-CNN | Punkty charakterystyczne boiska |
| IV | Faster R-CNN | TVCalib |

Śledzenie obiektów z wykorzystaniem ByteTrack oraz klasyfikacja zawodników do dwóch drużyn na podstawie koloru strojów pełnią w projekcie funkcję pomocniczą. Nie stanowią osobnego przedmiotu porównania metod.

## Ogólny przepływ danych

Dla kolejnych klatek transmisji wykonywane są dwa główne tory przetwarzania:

1. Detekcja obiektów, śledzenie ich pomiędzy klatkami oraz pomocnicza klasyfikacja zawodników do drużyn.
2. Estymacja macierzy homografii opisującej odwzorowanie obrazu transmisji na płaszczyznę boiska.

Po wyznaczeniu obu wyników jako położenie obiektu na obrazie przyjmowany jest środek dolnej krawędzi jego ramki ograniczającej. Punkt ten jest rzutowany za pomocą macierzy homografii na model boiska o wymiarach **105 m × 68 m**, którego środek stanowi początek układu współrzędnych.

Wyniki mogą zostać zapisane do pliku CSV oraz przedstawione w formie wideo zawierającego detekcje i minimapę boiska.

## Wykorzystane dane

W projekcie wykorzystano trzy główne źródła danych.

### SoccerNet-Tracking

Zbiór został wykorzystany do przygotowania danych treningowych i testowych dla modeli detekcji YOLOv8n oraz Faster R-CNN. Zawiera sekwencje transmisji piłkarskich oraz adnotacje obiektów w formacie MOT.

W projekcie rozpoznawane są trzy klasy:

- `ball`,
- `player`,
- `referee`.

### SoccerNet-Calibration

Zbiór zawiera klatki transmisji wraz z adnotacjami elementów geometrii boiska. Został wykorzystany przede wszystkim do przygotowania referencyjnych homografii na potrzeby ewaluacji metod kalibracji oraz jako dodatkowe źródło obrazów podczas rozszerzania zbioru treningowego detektora punktów charakterystycznych.

### Roboflow football-field-detection

Zbiór z platformy Roboflow zawierający adnotacje 32 punktów charakterystycznych boiska stanowił bazę do treningu modelu YOLOv8x-pose wykorzystywanego w jednej z dwóch metod estymacji homografii.

Surowe materiały SoccerNet nie są przechowywane w tym repozytorium. Dostęp do nich wymaga uzyskania odpowiednich uprawnień zgodnie z zasadami SoccerNet.

## Struktura repozytorium

```text
CvFootballTracker/
├── data/                   # dane wejściowe; katalog lokalny, ignorowany przez Git
├── models/                 # wagi wykorzystywanych modeli
├── notebooks/              # notebooki treningowe i pomocnicze
├── pipelines/              # cztery kompletne warianty systemu
├── results/                # wyniki działania i ewaluacji; katalog lokalny, ignorowany przez Git
├── scripts/                # pobieranie danych, przygotowanie zbiorów i ewaluacja
├── src/
│   ├── calibration/        # estymacja homografii i projekcja współrzędnych
│   │   ├── classicalApproachPlayground/
│   │   └── tvcalib/       # submoduł Git z forkiem TVCalib
│   ├── classification/     # pomocnicza klasyfikacja zawodników do drużyn
│   ├── detection/          # implementacja i obsługa detektorów
│   ├── tracking/           # ByteTrack
│   ├── utils/              # narzędzia pomocnicze
│   └── visualization/      # minimapa i rysowanie modelu boiska
├── opis_pracy.md           # założenia pracy magisterskiej
├── requirements.txt        # zależności środowiska Python
├── .gitmodules             # konfiguracja submodułu TVCalib
└── README.md
```

Katalogi `data`, `results` oraz większość dużych plików wag modeli nie są wersjonowane. Jest to zamierzone i wynika z rozmiaru danych oraz ograniczeń licencyjnych.

Szczegóły zawartości poszczególnych części repozytorium znajdują się w plikach README umieszczonych w odpowiednich katalogach.

## Instalacja na Windows

Projekt był rozwijany i testowany lokalnie na Windows 11. W pracy wykorzystano Python 3.13.11.

### 1. Pobranie repozytorium

Repozytorium zawiera TVCalib jako submoduł Git, dlatego zalecane jest pobranie projektu za pomocą Git, a nie przez opcję pobrania ZIP z GitHub.

```powershell
git clone https://github.com/gorzela31/CvFootballTracker.git
cd CvFootballTracker
git submodule update --init --recursive
```

Jeżeli repozytorium zostało wcześniej sklonowane bez submodułów, wystarczy wykonać ostatnią komendę z poziomu głównego katalogu projektu.

### 2. Utworzenie środowiska wirtualnego

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalacja zależności

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Warianty wykorzystujące TVCalib wymagają poprawnie zainicjalizowanego submodułu `src/calibration/tvcalib`.

## Wagi modeli

Po sklonowaniu repozytorium należy sprawdzić obecność wymaganych wag.

```text
models/
├── yolov8n/
│   └── trained_detection_yolov8n.pt
├── faster_rcnn/
│   └── trained_fasterrcnn_resnet50.pt
└── pitch_keypoints/
    └── trained_keypoints.pt

src/calibration/tvcalib/data/segment_localization/
└── train_59.pt
```

Wagi detektora YOLOv8n są wersjonowane w głównym repozytorium. Wagi Faster R-CNN i modelu punktów charakterystycznych są wyłączone z wersjonowania i należy umieścić je ręcznie w podanych katalogach albo odtworzyć na podstawie notebooków treningowych.

TVCalib korzysta z gotowego modelu segmentacyjnego `train_59.pt`. Wrapper projektu oczekuje tego pliku pod ścieżką podaną wyżej. Jeżeli plik nie znajduje się w submodule po jego inicjalizacji, należy przygotować go zgodnie z repozytorium TVCalib i umieścić w oczekiwanym katalogu.

## Przygotowanie danych

Wszystkie poniższe komendy należy uruchamiać z głównego katalogu repozytorium.

### SoccerNet-Tracking

Po uzyskaniu dostępu do danych SoccerNet można użyć skryptu:

```powershell
python scripts/download_tracking.py
```

Przed uruchomieniem należy sprawdzić listę `splits_to_download` w skrypcie i pozostawić wyłącznie potrzebne podziały danych.

Domyślna struktura oczekiwana przez pipeline'y wygląda następująco:

```text
data/tracking_dataset/tracking/
└── test/
    └── SNMOT-123/
        ├── img1/
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ...
        ├── gt/
        └── gameinfo.ini
```

Do samego uruchomienia pipeline'u potrzebny jest katalog z kolejnymi klatkami `.jpg`. Adnotacje GT są wymagane dopiero podczas przygotowywania zbiorów ewaluacyjnych.

### SoccerNet-Calibration

Dane można pobrać przy pomocy:

```powershell
python scripts/download_calibration.py
```

Przed uruchomieniem również należy sprawdzić `splits_to_download`, ponieważ poszczególne części zbioru zajmują dużo miejsca.

Skrypty projektu oczekują danych pod:

```text
data/calibration_dataset/calibration/
```

### Własna sekwencja obrazów

Pipeline'y mogą pracować również na innym katalogu zawierającym kolejno nazwane klatki `.jpg`. Dla modeli wytrenowanych w ramach pracy najlepsze rezultaty należy oczekiwać dla materiału o charakterze zbliżonym do standardowej transmisji telewizyjnej meczu piłkarskiego.

Domyślne pipeline'y YOLO korzystają ze ścieżek zapisanych w sekcji konfiguracyjnej na początku pliku. W wariantach Faster R-CNN katalog klatek można dodatkowo przekazać przez parametr `--frames`.

## Uruchamianie pipeline'ów

Każdy z czterech wariantów jest osobnym skryptem w katalogu `pipelines`.

### Wariant I: YOLOv8n + punkty charakterystyczne

```powershell
python pipelines/pipeline_yolo_bytetrack_keypoints.py
```

### Wariant II: YOLOv8n + TVCalib

```powershell
python pipelines/pipeline_yolo_bytetrack_tvcalib.py
```

### Wariant III: Faster R-CNN + punkty charakterystyczne

```powershell
python pipelines/pipeline_fasterrcnn_bytetrack_keypoints.py
```

Przykład z własnym katalogiem klatek:

```powershell
python pipelines/pipeline_fasterrcnn_bytetrack_keypoints.py `
    --frames "D:\dane\mecz\img1" `
    --frcnn-weights "models\faster_rcnn\trained_fasterrcnn_resnet50.pt" `
    --keypoints-weights "models\pitch_keypoints\trained_keypoints.pt"
```

### Wariant IV: Faster R-CNN + TVCalib

```powershell
python pipelines/pipeline_fasterrcnn_bytetrack_tvcalib.py
```

Przykład z własnym katalogiem klatek:

```powershell
python pipelines/pipeline_fasterrcnn_bytetrack_tvcalib.py `
    --frames "D:\dane\mecz\img1" `
    --frcnn-weights "models\faster_rcnn\trained_fasterrcnn_resnet50.pt" `
    --tvcalib-weights "src\calibration\tvcalib\data\segment_localization\train_59.pt"
```

Domyślne parametry wejściowe, progi detekcji oraz częstotliwość aktualizacji homografii są zapisane w sekcji konfiguracyjnej każdego pipeline'u.

## Wyniki działania

Pipeline'y tworzą katalogi w `results/` i zapisują przede wszystkim:

- plik wideo `.mp4` z naniesionymi detekcjami oraz minimapą,
- plik `.csv` zawierający informacje o detekcjach, identyfikatorach śledzenia oraz wyznaczonych współrzędnych na boisku.

Przykładowa struktura:

```text
results/
└── pipeline_yolo_bt_keypoints/
    ├── <timestamp>_output.mp4
    └── <timestamp>_tracks.csv
```

Katalog `results` jest ignorowany przez Git.

## Ewaluacja

Skrypty wykorzystywane podczas badań znajdują się w katalogu `scripts`. Szczegółowy opis ich przeznaczenia znajduje się w `scripts/README.md`.

Główny przebieg ewaluacji detekcji:

```powershell
python scripts/prepare_detection_evaluation_dataset.py --overwrite
python scripts/evaluate_detection.py --mode all
```

Główny przebieg ewaluacji homografii:

```powershell
python scripts/prepare_homography_evaluation_dataset.py --overwrite
python scripts/generate_homography_evaluation_gt.py --overwrite
python scripts/evaluate_homography_keypoints.py --mode all
python scripts/evaluate_homography_tvcalib.py --mode all
```

Ewaluacja modeli detekcji i metod homografii została przeprowadzona osobno. Wynika to z braku jednego zbioru danych zawierającego jednocześnie kompletne adnotacje detekcji zawodników oraz ich rzeczywiste współrzędne na płaszczyźnie boiska.

## Trening modeli

Notebooki wykorzystane podczas treningu znajdują się w katalogu `notebooks`. Trening modeli wymagających większych zasobów obliczeniowych był wykonywany w Google Colab z wykorzystaniem GPU.

Notebooki obejmują między innymi:

- trening YOLOv8n na danych przygotowanych z SoccerNet-Tracking,
- kolejne etapy treningu Faster R-CNN,
- trening detektora 32 punktów charakterystycznych boiska,
- notebooki pomocnicze do przygotowania wizualizacji.

Szczegóły znajdują się w `notebooks/README.md`.

## Uwagi

Repozytorium zawiera zarówno finalne elementy rozwiązania, jak i część kodu dokumentującego wcześniejsze eksperymenty. Dotyczy to przede wszystkim katalogu `src/calibration/classicalApproachPlayground`, w którym pozostawiono próby klasycznego wyznaczania homografii. Podejście to nie zostało wykorzystane jako jedna z dwóch finalnych metod porównywanych w pracy.
