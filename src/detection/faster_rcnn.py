"""
Plik: src/detection/faster_rcnn.py

Opis:
    Wrapper nad modelem Faster R-CNN (torchvision) do detekcji
    zawodnikow, pilki i sedziow na obrazach transmisji pilkarskiej.

    Obslugiuje dwa tryby pracy:
        1. Fine-tuned na SoccerNet (3 klasy: ball, player, referee)
        2. COCO pretrained z mapowaniem klas (person->player, sports ball->ball)

    Zwraca detekcje w formacie sv.Detections (kompatybilne z ByteTrack).

Uzycie:
    from src.detection.faster_rcnn import FasterRCNNDetector

    # Fine-tuned
    detector = FasterRCNNDetector(weights_path="models/faster_rcnn/trained.pt")

    # COCO pretrained (fallback)
    detector = FasterRCNNDetector(weights_path=None)

    detections = detector.predict(frame_bgr, conf=0.3)
"""

import cv2
import numpy as np
import torch
import supervision as sv
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


# Klasy zgodne z modelem YOLO (dla spojnosci miedzy pipeline'ami)
CLASS_NAMES = {0: "ball", 1: "player", 2: "referee"}

# Mapowanie COCO -> SoccerNet (dla trybu pretrained)
COCO_TO_SOCCERNET = {
    1: 1,    # COCO 'person' -> SoccerNet 'player'
    37: 0,   # COCO 'sports ball' -> SoccerNet 'ball'
}

NUM_CLASSES = 4  # background + ball + player + referee

# Konfiguracja kotwic zgodna z treningiem v3/v4 (przesun kotwice pod mala pilke)
# Jesli trenujesz nowy model z innymi kotwicami, zmien ponizej:
#_ANCHOR_SIZES  = ((16,), (32,), (64,), (128,), (256,))   # v3/v4
_ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))  # domyslne torchvision (v2 i starsze)
_ANCHOR_RATIOS = (0.5, 1.0, 2.0)


class FasterRCNNDetector:
    """
    Detektor Faster R-CNN ResNet50-FPN v2.

    Parametry:
        weights_path   : sciezka do wag fine-tuned (.pt), None = COCO pretrained
        conf_threshold : domyslny prog confidence
        device         : 'cuda' / 'cpu' / None (auto)
        min_size       : krotszy bok obrazu po wewn. resizingu FPN
        max_size       : dluzszy bok obrazu po wewn. resizingu FPN
    """

    def __init__(self, weights_path=None, conf_threshold=0.5, device=None,
                 min_size=800, max_size=1333):   # v3/v4: 1080/1920 | v2 i starsze: 800/1333
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_coco_mapping = weights_path is None

        if weights_path is not None:
            # Model fine-tuned na SoccerNet — architektura musi byc identyczna z treningiem
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                min_size=min_size,
                max_size=max_size,
                rpn_pre_nms_top_n_test=2000,
                rpn_post_nms_top_n_test=1000,
            )
            # Kotwice przesuniete w dol pod mala pilke (v3/v4); nie sa zapisywane w state_dict
            model.rpn.anchor_generator = AnchorGenerator(
                _ANCHOR_SIZES, (_ANCHOR_RATIOS,) * len(_ANCHOR_SIZES)
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            print(f"[FasterRCNN] Zaladowano wagi fine-tuned: {weights_path}")
            print(f"[FasterRCNN] Rozdzielczosc: min={min_size}, max={max_size} | kotwice: {_ANCHOR_SIZES}")
        else:
            # COCO pretrained (91 klas) — domyslna konfiguracja torchvision
            model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
            print("[FasterRCNN] COCO pretrained (mapowanie: person->player, ball->ball)")
            print("[FasterRCNN] UWAGA: brak rozroznienia player/referee w trybie COCO")

        model.to(self.device)
        model.eval()
        self.model = model
        print(f"[FasterRCNN] Urzadzenie: {self.device}")

    def predict(self, source, conf=None, verbose=False, imgsz=None):
        """
        Uruchamia detekcje na pojedynczym obrazie.

        Parametry:
            source  : np.ndarray (BGR) lub sciezka do pliku
            conf    : prog confidence (nadpisuje domyslny)
            verbose : nieuzywane (kompatybilnosc API)
            imgsz   : nieuzywane (Faster R-CNN ma wbudowany resizer)

        Zwraca:
            sv.Detections z polami xyxy, confidence, class_id
        """
        conf = conf if conf is not None else self.conf_threshold

        if isinstance(source, (str, bytes)):
            source = cv2.imread(str(source))

        # BGR -> RGB, HWC -> CHW, [0,1]
        image_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        # Filtrowanie po confidence
        mask = scores >= conf
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if self.use_coco_mapping:
            mapped_labels = []
            keep = []
            for i, label in enumerate(labels):
                if int(label) in COCO_TO_SOCCERNET:
                    mapped_labels.append(COCO_TO_SOCCERNET[int(label)])
                    keep.append(i)
            if keep:
                boxes = boxes[keep]
                scores = scores[keep]
                labels = np.array(mapped_labels, dtype=int)
            else:
                return self._empty_detections()
        else:
            # Fine-tuned: etykiety 1-indexed (1=ball, 2=player, 3=referee)
            labels = labels - 1

        if len(boxes) == 0:
            return self._empty_detections()

        return sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=labels.astype(int),
        )

    @staticmethod
    def _empty_detections() -> sv.Detections:
        """Zwraca puste detekcje z poprawnymi polami (kompatybilne z ByteTrack)."""
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )
