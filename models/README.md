# models

Katalog przeznaczony na wagi modeli wykorzystywanych przez projekt.

## Struktura

```text
models/
├── yolov8n/
│   ├── yolov8n.pt
│   └── trained_detection_yolov8n.pt
├── faster_rcnn/
│   └── trained_fasterrcnn_resnet50.pt
└── pitch_keypoints/
    └── trained_keypoints.pt
```

### `yolov8n`

Zawiera bazowe wagi YOLOv8n oraz model wytrenowany do detekcji klas `ball`, `player` i `referee` na danych przygotowanych z SoccerNet-Tracking.

### `faster_rcnn`

Docelowe miejsce na wagi modelu Faster R-CNN ResNet50-FPN wykorzystanego w drugim wariancie detekcji.

### `pitch_keypoints`

Docelowe miejsce na wagi modelu YOLOv8x-pose wykrywającego 32 punkty charakterystyczne boiska. Wynik modelu jest następnie wykorzystywany do estymacji homografii metodą RANSAC.

## Wersjonowanie

Pliki `.pt` są domyślnie ignorowane przez Git ze względu na ich rozmiar. Wyjątek stanowią pliki znajdujące się w `models/yolov8n`, które zostały zachowane w głównym repozytorium.

Wagi Faster R-CNN i modelu punktów charakterystycznych należy umieścić lokalnie w ścieżkach oczekiwanych przez pipeline'y albo odtworzyć na podstawie notebooków z katalogu `notebooks`.

Wagi TVCalib nie są przechowywane w tym katalogu. Wrapper TVCalib oczekuje pliku:

```text
src/calibration/tvcalib/data/segment_localization/train_59.pt
```
