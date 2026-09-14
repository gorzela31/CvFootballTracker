# utils

Katalog zawiera narzędzia pomocnicze wykorzystywane na różnych etapach przygotowania projektu.

## `mot_to_yolo.py`

Konwertuje adnotacje SoccerNet-Tracking z formatu MOT do formatu wykorzystywanego podczas treningu detektorów.

Podczas konwersji klasy źródłowe są sprowadzane do trzech klas używanych w projekcie:

```text
0 - ball
1 - player
2 - referee
```

Zawodnicy i bramkarze obu drużyn są łączeni do wspólnej klasy `player`.

## `visualization.py`

Starszy moduł pomocniczy do tworzenia widoku bird's-eye view i nanoszenia pozycji obiektów na model boiska.

## `visualize_gt.py`

Narzędzie pomocnicze do wizualnej kontroli adnotacji ground truth.

Główna wizualizacja używana obecnie przez kompletne pipeline'y znajduje się w `src/visualization`.
