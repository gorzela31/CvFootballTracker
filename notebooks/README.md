# notebooks

Katalog zawiera notebooki wykorzystywane podczas treningu modeli oraz przygotowywania materiałów pomocniczych do pracy magisterskiej. Trening był wykonywany głównie w Google Colab z wykorzystaniem GPU.

Notebooki nie są wymagane do zwykłego uruchomienia gotowych pipeline'ów, jeżeli odpowiednie wagi modeli zostały już umieszczone w katalogu `models`.

## Zawartość

### `01_yolov8n_detection_training.ipynb`

Trening modelu YOLOv8n do detekcji klas `ball`, `player` i `referee` na danych przygotowanych ze zbioru SoccerNet-Tracking.

### `02_faster_rcnn_detection_training*.ipynb`

Kolejne iteracje treningu modelu Faster R-CNN ResNet50-FPN. Pliki dokumentują rozwój konfiguracji modelu, w tym pracę z natywną rozdzielczością obrazu, mniejszymi kotwicami RPN i kolejnymi seriami treningowymi.

Wersje oznaczone `v3` i `v4` są kolejnymi etapami eksperymentów, a nie osobnymi metodami porównywanymi w pracy.

### `03_train_pitch_keypoint_detector_v1.ipynb`

Początkowy notebook treningu detektora punktów charakterystycznych boiska, oparty na materiale Roboflow dotyczącym zbioru `football-field-detection`.

### `03_train_keypoints_detection_extended.ipynb`

Trening modelu YOLOv8x-pose na rozszerzonym zbiorze punktów charakterystycznych boiska. Model ten stanowi podstawę finalnej metody homografii opartej na 32 punktach charakterystycznych.

### `04_keypoints_visualization_for_thesis.ipynb`

Notebook pomocniczy służący do przygotowania wizualizacji detekcji punktów charakterystycznych oraz nałożenia modelu boiska na obraz transmisji.

## Dane i wyniki

Notebooki treningowe korzystają z danych przygotowanych wcześniej lokalnie, a następnie przeniesionych do środowiska Google Colab, najczęściej za pośrednictwem Google Drive.

Ścieżki zapisane w notebookach odpowiadają środowisku wykorzystanemu podczas realizacji pracy i przed ponownym treningiem mogą wymagać dostosowania do własnej struktury Google Drive.
