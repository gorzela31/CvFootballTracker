# pipelines

Katalog zawiera cztery kompletne warianty systemu detekcji i lokalizacji zawodników. Każdy pipeline łączy jedną z dwóch metod detekcji z jedną z dwóch metod estymacji homografii.

## Warianty

| Plik | Detekcja | Homografia |
| --- | --- | --- |
| `pipeline_yolo_bytetrack_keypoints.py` | YOLOv8n | YOLOv8x-pose + RANSAC |
| `pipeline_yolo_bytetrack_tvcalib.py` | YOLOv8n | TVCalib |
| `pipeline_fasterrcnn_bytetrack_keypoints.py` | Faster R-CNN | YOLOv8x-pose + RANSAC |
| `pipeline_fasterrcnn_bytetrack_tvcalib.py` | Faster R-CNN | TVCalib |

## Wspólny przebieg

Niezależnie od wybranego wariantu pipeline realizuje ten sam ogólny schemat:

1. wczytanie kolejnej klatki,
2. detekcja obiektów,
3. przypisanie identyfikatorów ByteTrack,
4. pomocnicza klasyfikacja zawodników do dwóch drużyn,
5. estymacja lub aktualizacja homografii,
6. rzutowanie środka dolnej krawędzi bbox na płaszczyznę boiska,
7. zapis wyników i przygotowanie wizualizacji.

Tracking i klasyfikacja drużyn są wspólne dla wszystkich wariantów i pełnią funkcję pomocniczą.

## Wejście

Pipeline'y pracują na katalogu zawierającym kolejne klatki `.jpg`, przykładowo:

```text
data/tracking_dataset/tracking/test/SNMOT-123/img1/
```

Domyślne ścieżki do danych i wag znajdują się w sekcji konfiguracyjnej każdego skryptu.

Warianty Faster R-CNN udostępniają dodatkowo argumenty CLI, między innymi `--frames`, `--frcnn-weights`, `--calib-stride` i `--run-name`.

Warianty YOLO korzystają obecnie z wartości ustawionych bezpośrednio na początku skryptu.

## Wyjście

Wyniki są zapisywane w `results/<RUN_NAME>/` i obejmują:

- wideo z detekcjami oraz minimapą,
- plik CSV z detekcjami, ID śledzenia, klasą, confidence, bbox oraz współrzędnymi na boisku.

Jeżeli estymacja homografii dla kolejnej klatki się nie powiedzie, pipeline może korzystać z ostatniej poprawnie wyznaczonej macierzy. Mechanizm ten dotyczy działania kompletnego systemu i nie jest wykorzystywany podczas niezależnej ewaluacji metod homografii.
