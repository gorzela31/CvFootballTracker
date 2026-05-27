# CvFootballTracker
**Master's Thesis:** Comparison of football player detection and 2D pitch localization methods based on TV broadcast using the SoccerNet database.

**Temat pracy magisterskiej:** Porównanie metod detekcji i lokalizacji zawodników piłki nożnej na płaszczyźnie boiska w oparciu o obraz transmisji telewizyjnej.

---

## Overview

This repository contains the source code for my Master's thesis. The project implements a computer vision pipeline that processes standard football TV broadcasts to detect players, track them across frames, and project their real-world coordinates onto a 2D tactical map (bird's-eye view).

The system is evaluated using the [SoccerNet](https://www.soccer-net.org/) database.

## Pipeline Architecture

The processing pipeline consists of three main stages:

1. **Detection:** Locating players, referees, and the ball. The project compares different object detection architectures, primarily focusing on YOLO (v8/v10) and RT-DETR.
2. **Multi-Object Tracking (MOT):** Maintaining consistent player IDs across video frames. Methods evaluated include ByteTrack and BoT-SORT, with a focus on handling severe occlusions.
3. **Pitch Localization:** Detecting pitch keypoints and lines to calculate a homography matrix. This transformation maps the bottom-center of player bounding boxes (feet contact points) to a standard 2D pitch coordinate system.

## Dataset

Data is sourced from the SoccerNet dataset. Two specific subsets are used:
* `tracking`: Used for training and evaluating detection and MOT modules.
* `camera-calibration`: Used for pitch line extraction and homography evaluation (contains broadcast frames from 500+ matches with ground truth camera parameters).

## Repository Structure

The project is designed for a hybrid workflow: local development for source code and Google Colab for GPU-intensive training.

```text
├── configs/                # Model hyperparameters (.yaml)
├── data/                   # Git-ignored: Raw datasets and annotations
├── models/                 # trained weights of models (.pt)
├── notebooks/              # Jupyter notebooks for Google Colab execution
├── pipelines/              # Ready E2E run scripts
├── scripts/                # CLI entry points for pipeline execution
├── src/  
│   ├── calibration/        # Homography and pitch projection math                  
│   ├── detection/          # Object detection models and inference logic
│   ├── tracking/           # tracking algorithms integration
│   └── utils/              # Data parsers
│   └── visualization/      # Drawing of pitch and minimap
├── results/                # Git-ignored: output media
├── opis_pracy.md           # Thesis requirements
├── requirements.txt        # Python dependencies
└── .gitignore