# src

Katalog `src` zawiera główną implementację modułów wykorzystywanych przez cztery pipeline'y projektu.

Kod został podzielony według funkcji realizowanych w systemie, tak aby metody detekcji, kalibracji, śledzenia i wizualizacji mogły być rozwijane i testowane niezależnie.

## Moduły

### `detection`

Obsługa modeli detekcji obiektów. Zawiera przede wszystkim wrapper Faster R-CNN oraz starsze skrypty treningowe i inferencyjne.

YOLOv8n jest w kompletnych pipeline'ach obsługiwany bezpośrednio przez bibliotekę Ultralytics.

### `calibration`

Implementacja obu finalnych metod estymacji homografii:

- detekcja 32 punktów charakterystycznych boiska i RANSAC,
- integracja z TVCalib.

Znajduje się tutaj również katalog `classicalApproachPlayground` dokumentujący wcześniejsze eksperymenty z podejściem klasycznym oraz submoduł `tvcalib`.

### `tracking`

Wrapper ByteTrack wykorzystywany do zachowania identyfikatorów obiektów pomiędzy kolejnymi klatkami.

### `classification`

Pomocnicze przypisanie zawodników do dwóch drużyn na podstawie koloru górnej części stroju.

### `visualization`

Rysowanie modelu boiska, minimapy oraz nanoszenie na nią pozycji wyznaczonych przez pipeline.

### `utils`

Narzędzia pomocnicze używane podczas przygotowywania danych i wcześniejszych etapów projektu.

## Powiązanie z pipeline'ami

Skrypty znajdujące się w `pipelines` są warstwą wysokiego poziomu. Importują moduły z `src`, łączą je w odpowiedniej kolejności i odpowiadają za pełne przetwarzanie sekwencji klatek.
