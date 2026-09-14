# detection

Moduł zawiera kod związany z detekcją obiektów na obrazie transmisji piłkarskiej.

Finalne porównanie w pracy obejmuje dwa modele:

- YOLOv8n,
- Faster R-CNN ResNet50-FPN.

Oba modele zostały przygotowane do rozpoznawania klas `ball`, `player` i `referee`.

## `faster_rcnn.py`

Wrapper modelu Faster R-CNN wykorzystywany przez kompletne pipeline'y. Zwraca wyniki w formacie `supervision.Detections`, dzięki czemu mogą być bezpośrednio przekazane do ByteTrack i dalszych etapów systemu.

Model może pracować z własnymi wagami wytrenowanymi na SoccerNet albo w trybie bazującym na wagach COCO.

## `inference.py`

Starszy skrypt pomocniczy do uruchamiania inferencji YOLO na sekwencji obrazów i zapisywania wyniku do pliku wideo.

## `train_faster_rcnn.py` i `train_test.py`

Skrypty związane z wcześniejszym lokalnym treningiem i sprawdzaniem poprawności danych. Finalny trening modeli opisany w pracy był realizowany głównie w notebookach z katalogu `notebooks` i w środowisku Google Colab.

W kompletnych pipeline'ach YOLOv8n jest ładowany bezpośrednio przez bibliotekę Ultralytics, dlatego w tym katalogu nie ma osobnego wrappera YOLO odpowiadającego `faster_rcnn.py`.
