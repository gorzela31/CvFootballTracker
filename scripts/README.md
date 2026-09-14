# scripts

Katalog zawiera skrypty pomocnicze wykorzystywane do pobierania danych, przygotowywania zbiorów treningowych i testowych oraz przeprowadzania ewaluacji.

Do zwykłego uruchomienia gotowego pipeline'u nie trzeba wykonywać wszystkich skryptów z tego katalogu. Część z nich służy wyłącznie do odtworzenia procedury badawczej opisanej w pracy magisterskiej.

## Pobieranie danych

### `download_tracking.py`

Pobiera podzbiory zadania SoccerNet-Tracking do:

```text
data/tracking_dataset/
```

Przed uruchomieniem należy ustawić wymagane podziały w `splits_to_download` i posiadać dostęp do danych SoccerNet.

### `download_calibration.py`

Pobiera dane SoccerNet-Calibration do:

```text
data/calibration_dataset/
```

Przed uruchomieniem należy sprawdzić listę `splits_to_download`, ponieważ pełny zbiór zajmuje dużo miejsca.

## Przygotowanie danych do ewaluacji

### `prepare_detection_evaluation_dataset.py`

Tworzy stałą pulę klatek przeznaczonych do porównania YOLOv8n i Faster R-CNN na podstawie części testowej SoccerNet-Tracking.

Domyślny wynik:

```text
data/data_detection_evaluation/
```

### `prepare_homography_evaluation_dataset.py`

Przygotowuje stałą pulę par obraz + adnotacja ze zbioru SoccerNet-Calibration do późniejszej ewaluacji homografii.

Domyślny wynik:

```text
data/data_homography_evaluation/
```

### `generate_homography_evaluation_gt.py`

Na podstawie adnotacji SoccerNet-Calibration wyznacza referencyjne macierze homografii `H_GT` dla przygotowanej puli ewaluacyjnej. Generuje również dane i wizualizacje pomocnicze używane podczas porównania obu metod.

### `build_combined_keypoints_evaluation_dataset.py`

Buduje połączony zbiór testowy do dodatkowej oceny detektora 32 punktów charakterystycznych. Łączy przykłady pochodzące z przygotowanej puli SoccerNet-Calibration z testową częścią `football-field-detection`.

## Przygotowanie danych do treningu keypointów

### `build_football_field_detection_16_extended.py`

Tworzy rozszerzoną wersję zbioru `football-field-detection` poprzez dołączenie dodatkowych klatek z SoccerNet-Calibration i wygenerowanie dla nich etykiet 32 punktów charakterystycznych.

Skrypt służy do przygotowania danych przed treningiem modelu YOLOv8x-pose.

## Ewaluacja detekcji

### `evaluate_detection.py`

Uruchamia porównanie YOLOv8n i Faster R-CNN na zamrożonej puli testowej. Oblicza między innymi:

- Precision,
- Recall,
- AP@0.5,
- AP@0.5:0.95,
- średnią i medianę IoU,
- TP, FP i FN,
- czas przetwarzania oraz FPS.

Przykład:

```powershell
python scripts/evaluate_detection.py --mode all
```

### `detection_confidence_sweep.py`

Analizuje wpływ progu confidence na wyniki detekcji na podstawie zapisanych predykcji. Nie uruchamia modeli ponownie.

## Ewaluacja homografii

### `evaluate_homography_keypoints.py`

Ocenia homografię wyznaczoną na podstawie 32 punktów YOLOv8x-pose i RANSAC względem `H_GT`.

Każda klatka jest oceniana niezależnie. W przeciwieństwie do kompletnego pipeline'u evaluator nie korzysta z ostatniej poprawnej homografii przy błędzie bieżącej estymacji.

### `evaluate_homography_tvcalib.py`

Ocenia homografię wyznaczaną przez TVCalib względem tego samego `H_GT` i na tych samych punktach testowych co metoda keypointowa.

Dzięki temu wyniki obu metod mogą być bezpośrednio zestawione.

### `evaluate_pitch_keypoints_combined.py`

Dodatkowa ewaluacja samego modelu punktów charakterystycznych na połączonym zbiorze testowym.

## Skrypty pomocnicze i diagnostyczne

### `generate_homography_gt.py`

Narzędzie do wyznaczania i wizualizacji referencyjnej homografii dla pojedynczego obrazu i odpowiadającej mu adnotacji.

### `visualize_pitch_keypoints_resized.py`

Skrypt pomocniczy do analizy zachowania modelu punktów charakterystycznych przy skalowaniu obrazu i do generowania wizualizacji wyników.

## Typowy przebieg badań

Detekcja:

```powershell
python scripts/prepare_detection_evaluation_dataset.py --overwrite
python scripts/evaluate_detection.py --mode all
```

Homografia:

```powershell
python scripts/prepare_homography_evaluation_dataset.py --overwrite
python scripts/generate_homography_evaluation_gt.py --overwrite
python scripts/evaluate_homography_keypoints.py --mode all
python scripts/evaluate_homography_tvcalib.py --mode all
```
