# calibration

Moduł odpowiedzialny za estymację homografii oraz przeliczanie współrzędnych punktów z obrazu transmisji na metryczny układ boiska.

W projekcie finalnie porównywane są dwie metody.

## `keypoints_homography.py`

Metoda oparta na detekcji 32 zdefiniowanych punktów charakterystycznych boiska za pomocą modelu YOLOv8x-pose.

Przebieg:

1. obraz jest przygotowywany zgodnie z preprocessingiem modelu,
2. wykrywane są punkty charakterystyczne,
3. punkty o wystarczającym confidence są łączone ze znanymi współrzędnymi na modelu boiska,
4. `cv2.findHomography` z RANSAC wyznacza macierz w kierunku obraz `[px]` -> boisko `[m]`.

Domyślna ścieżka wag wykorzystywana przez pipeline'y:

```text
models/pitch_keypoints/trained_keypoints.pt
```

## `homography.py`

Warstwa integracyjna dla TVCalib. Klasa `TVCalibHomography` uruchamia segmentację oznaczeń boiska, przygotowuje dane wejściowe dla procesu kalibracji, a następnie przelicza uzyskane parametry kamery do postaci macierzy homografii wykorzystywanej przez pozostałą część projektu.

Domyślna ścieżka modelu segmentacyjnego:

```text
src/calibration/tvcalib/data/segment_localization/train_59.pt
```

## `tvcalib`

Submoduł Git wskazujący na fork zewnętrznego projektu TVCalib. Jego zawartość nie jest kopiowana bezpośrednio do głównego repozytorium.

Po sklonowaniu projektu należy wykonać:

```powershell
git submodule update --init --recursive
```

## `classicalApproachPlayground`

Katalog dokumentujący wcześniejsze eksperymenty z klasycznym podejściem do wykrywania geometrii boiska na podstawie przetwarzania obrazu. Podejście to zostało odrzucone ze względu na niewystarczającą stabilność i nie stanowi jednej z dwóch finalnych metod porównywanych w pracy.

## Układ współrzędnych

Finalna projekcja korzysta z modelu boiska 105 m × 68 m ze środkiem w punkcie `(0, 0)`:

```text
X: -52.5 ... +52.5 m
Y: -34.0 ... +34.0 m
```

Do rzutowania pozycji wykrytego obiektu wykorzystywany jest środek dolnej krawędzi jego ramki ograniczającej.
