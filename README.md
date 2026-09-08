# CvFootballTracker
**Master's Thesis:** Comparison of football player detection and 2D pitch localization methods based on TV broadcast using the SoccerNet database.

**Temat pracy magisterskiej:** Porównanie metod detekcji i lokalizacji zawodników piłki nożnej na płaszczyźnie boiska w oparciu o obraz transmisji telewizyjnej.

---

## Overview

This repository contains the source code for my Master's thesis. The project implements a computer vision pipeline that processes standard football TV broadcasts to detect players, track them across frames, and project their real-world coordinates onto a 2D tactical map (bird's-eye view).

The system is evaluated using the [SoccerNet](https://www.soccer-net.org/) database.

## Pipeline Architecture

The processing pipeline consists of three main stages:

1. **Detection:** Locating players, referees, and the ball. The project compares two object detection architectures: YOLOv8n and Faster R-CNN (ResNet50-FPN v2).
2. **Multi-Object Tracking (MOT):** Maintaining consistent player IDs across video frames using ByteTrack.
3. **Team Classification:** Assigning players to teams based on jersey color using K-means clustering in HSV color space.
4. **Pitch Localization:** Detecting pitch keypoints and lines to calculate a homography matrix. The project compares two approaches: TVCalib (neural segmentation-based) and YOLO Keypoints (YOLOv8x-pose + RANSAC, based on roboflow/sports). This transformation maps the bottom-center of player bounding boxes (feet contact points) to a standard 2D pitch coordinate system.

## Dataset

Data is sourced from the SoccerNet dataset. Two specific subsets are used:
* `tracking`: Used for training and evaluating detection and MOT modules.
* `camera-calibration`: Used for pitch line extraction and homography evaluation (contains broadcast frames from 500+ matches with ground truth camera parameters).

## Repository Structure

The project is designed for a hybrid workflow: local development for source code and Google Colab for GPU-intensive training.

```text
├── data/                   # Git-ignored: Raw datasets and annotations
├── models/
│   ├── yolov8n/            # Wagi detektora YOLOv8n
│   ├── faster_rcnn/        # Wagi detektora Faster R-CNN
│   └── pitch_keypoints/    # Wagi modelu YOLOv8x-pose (keypoints boiska, roboflow/sports)
├── notebooks/              # Notebooki treningowe (Colab) – YOLOv8n, Faster R-CNN
├── pipelines/              # Gotowe pipeline'y E2E (4 kombinacje: detektor × kalibracja)
├── scripts/                # Skrypty pomocnicze (pobieranie danych, modeli)
├── src/
│   ├── calibration/
│   │   ├── homography.py               # Wrapper TVCalib
│   │   ├── keypoints_homography.py     # Wrapper YOLO Keypoints + RANSAC
│   │   ├── classicalApproachPlayground/# Eksperymenty z klasycznym podejściem (archiwum)
│   │   └── tvcalib/                    # Submoduł TVCalib (Theiner et al., WACV 2023)
│   ├── classification/     # Klasyfikacja drużyn (K-means na kolorze koszulek)
│   ├── detection/          # Detektory: YOLOv8n, Faster R-CNN – inferencja i trening
│   ├── tracking/           # Śledzenie obiektów (ByteTrack)
│   ├── utils/              # Konwersja danych, wizualizacja GT
│   └── visualization/      # Rysowanie minimapki i boiska 2D
├── results/                # Git-ignored: wyjściowe wideo i CSV
├── opis_pracy.md           # Thesis requirements
├── requirements.txt        # Python dependencies
└── .gitignore