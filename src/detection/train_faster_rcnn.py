"""
Plik: src/detection/train_faster_rcnn.py

Opis:
    Skrypt treningowy Faster R-CNN ResNet50-FPN v2 na danych SoccerNet
    w formacie YOLO (wygenerowanym przez src/utils/mot_to_yolo.py).

    Przygotowany do uruchomienia zarowno lokalnie (szybki test CPU)
    jak i na Google Colab (pelny trening GPU).

    Klasy:
        0: ball         (w Faster R-CNN label=1, bo 0=background)
        1: player       (label=2)
        2: referee      (label=3)

Uzycie:
    # Pelny trening (GPU/Colab)
    python src/detection/train_faster_rcnn.py

    # Szybki test (CPU)
    python src/detection/train_faster_rcnn.py --test
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F


# ======================================================================
# KONFIGURACJA
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Sciezki do danych YOLO
TRAIN_IMAGES_DIR = PROJECT_ROOT / "data" / "yoloformat" / "train"
VALID_IMAGES_DIR = PROJECT_ROOT / "data" / "yoloformat" / "valid"

# Wyjscie
OUTPUT_DIR = PROJECT_ROOT / "models" / "faster_rcnn"
OUTPUT_WEIGHTS = OUTPUT_DIR / "trained_fasterrcnn_resnet50.pt"

# Hiperparametry
NUM_CLASSES = 4   # background + ball + player + referee
IMG_SIZE = 960
BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 15
LR_GAMMA = 0.1
NUM_WORKERS = 2
PATIENCE = 15     # early stopping: ile epok bez poprawy


# ======================================================================
# DATASET
# ======================================================================

class YOLOFormatDataset(Dataset):
    """
    Wczytuje obrazy i adnotacje w formacie YOLO i konwertuje
    na format wymagany przez torchvision Faster R-CNN.

    Struktura wejsciowa:
        base_dir/
            CLIP_NAME/
                images/  *.jpg
                labels/  *.txt  (class cx cy w h — znormalizowane 0-1)

    Format wyjsciowy:
        image: Tensor [3, H, W] float32 [0,1]
        target: dict {
            'boxes': Tensor [N, 4] float32 (x1, y1, x2, y2 w pikselach)
            'labels': Tensor [N] int64 (1=ball, 2=player, 3=referee)
        }
    """

    def __init__(self, base_dir, img_size=960):
        self.img_size = img_size
        self.samples = []  # lista (image_path, label_path)

        base = Path(base_dir)
        for clip_dir in sorted(base.iterdir()):
            images_dir = clip_dir / "images"
            labels_dir = clip_dir / "labels"
            if not images_dir.is_dir():
                continue
            for img_path in sorted(images_dir.glob("*.jpg")):
                lbl_path = labels_dir / img_path.with_suffix(".txt").name
                self.samples.append((img_path, lbl_path))

        print(f"  Dataset: {len(self.samples)} obrazow z {base_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # Wczytanie i resize obrazu
        image = cv2.imread(str(img_path))
        orig_h, orig_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = F.to_tensor(image)  # [3, H, W] float32 [0,1]

        # Wczytanie adnotacji
        boxes = []
        labels = []

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_yolo = int(parts[0])  # 0=ball, 1=player, 2=referee
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # YOLO normalized -> piksele na zresizowanym obrazie
                    x1 = (cx - bw / 2) * self.img_size
                    y1 = (cy - bh / 2) * self.img_size
                    x2 = (cx + bw / 2) * self.img_size
                    y2 = (cy + bh / 2) * self.img_size

                    x1 = max(0.0, x1)
                    y1 = max(0.0, y1)
                    x2 = min(float(self.img_size), x2)
                    y2 = min(float(self.img_size), y2)

                    if x2 > x1 + 1 and y2 > y1 + 1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_yolo + 1)  # +1 bo 0=background

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        return image, target


def collate_fn(batch):
    """Collate dla zmiennej liczby obiektow per obraz."""
    return tuple(zip(*batch))


# ======================================================================
# TRENING
# ======================================================================

def create_model(num_classes, pretrained=True):
    """Tworzy model Faster R-CNN z nowa glowica klasyfikacyjna."""
    if pretrained:
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Jeden epoch treningowy. Zwraca sredni loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Filtruj puste targety (Faster R-CNN wymaga >= 1 box)
        valid = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].shape[0] > 0]
        if not valid:
            continue
        images, targets = zip(*valid)
        images, targets = list(images), list(targets)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {losses.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Walidacja — sredni loss na zbiorze walidacyjnym."""
    model.train()  # Faster R-CNN zwraca loss tylko w trybie train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        valid = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].shape[0] > 0]
        if not valid:
            continue
        images, targets = zip(*valid)
        images, targets = list(images), list(targets)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(test_mode=False):
    """Glowna funkcja treningowa."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Urzadzenie: {device}")

    if test_mode:
        print("=== TRYB TESTOWY (1 epoch, batch_size=2, CPU) ===")
        epochs = 1
        batch_size = 2
    else:
        epochs = NUM_EPOCHS
        batch_size = BATCH_SIZE

    # Dataset
    print("Ladowanie danych...")
    train_dataset = YOLOFormatDataset(TRAIN_IMAGES_DIR, img_size=IMG_SIZE)
    valid_dataset = YOLOFormatDataset(VALID_IMAGES_DIR, img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS if not test_mode else 0,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS if not test_mode else 0,
    )

    # Model
    print("Tworzenie modelu Faster R-CNN ResNet50-FPN v2...")
    model = create_model(NUM_CLASSES, pretrained=True)
    model.to(device)

    # Optimizer i scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # Trening
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\nRozpoczecie treningu ({epochs} epok, batch={batch_size})")
    print(f"Wyjscie: {OUTPUT_WEIGHTS}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, valid_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"lr: {lr_now:.6f} | czas: {elapsed:.0f}s"
        )

        # Zapis najlepszego modelu
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), str(OUTPUT_WEIGHTS))
            print(f"  -> Zapisano najlepszy model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= PATIENCE and not test_mode:
            print(f"\nEarly stopping po {epoch} epokach (brak poprawy przez {PATIENCE} epok)")
            break

    print(f"\nTrening zakonczony. Najlepszy val_loss: {best_val_loss:.4f}")
    print(f"Wagi zapisane w: {OUTPUT_WEIGHTS}")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN na SoccerNet")
    parser.add_argument("--test", action="store_true", help="Tryb testowy (1 epoch CPU)")
    args = parser.parse_args()

    train(test_mode=args.test)
