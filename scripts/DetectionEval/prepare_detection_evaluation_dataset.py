"""
Przygotowanie zróżnicowanej puli ewaluacyjnej detekcji z SoccerNet-Tracking.

Umieść plik jako:
    CvFootballTracker/scripts/prepare_detection_evaluation_dataset.py

Domyślne uruchomienie:
    python scripts/prepare_detection_evaluation_dataset.py

Założenia domyślne:
    - przejście po wszystkich katalogach SNMOT-* w zbiorze testowym,
    - wybór 2 klatek z każdej sekwencji, rozłożonych w czasie,
    - przy 49 sekwencjach daje to 98 klatek,
    - do docelowych 100 dobierane są 2 dodatkowe klatki z dwóch różnych
      sekwencji (o ile jest to możliwe),
    - wybór jest deterministyczny dla ustalonego SEED,
    - eksportowane obrazy dostają nazwy eval_001.jpg ... eval_100.jpg,
    - razem z obrazami eksportowany jest ground truth oraz manifest źródeł.

Wyjście:
    data/data_detection_evaluation/
        images/
            eval_001.jpg
            ...
        labels/
            eval_001.txt
            ...
        ground_truth.csv
        manifest.csv
        selection_info.json

Format labels/*.txt:
    YOLO:
        class_id x_center y_center width height
    gdzie:
        0 = ball
        1 = player (w tym goalkeeper)
        2 = referee

ground_truth.csv zachowuje dodatkowo oryginalne track_id oraz bbox w pikselach.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TEST_ROOT = (
    PROJECT_ROOT / "data" / "tracking_dataset" / "tracking" / "test"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "data_detection_evaluation"
)

SEED = 17

CLASS_NAMES = {
    0: "ball",
    1: "player",
    2: "referee",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Przygotuj zróżnicowany zbiór klatek do ewaluacji detekcji."
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
        help="Katalog testowy SoccerNet-Tracking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Katalog docelowy eksportu.",
    )
    parser.add_argument(
        "--per-sequence",
        type=int,
        default=2,
        help="Minimalna liczba klatek wybierana z każdej sekwencji.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help=(
            "Docelowa liczba klatek. Po pobraniu --per-sequence z każdego klipu "
            "brakujące klatki są dobierane z różnych sekwencji."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed gwarantujący powtarzalny wybór.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Usuń istniejący katalog wyjściowy przed eksportem.",
    )
    return parser.parse_args()


def parse_gameinfo_class_map(gameinfo_path: Path) -> Dict[int, int | None]:
    """
    Mapowanie SoccerNet track_id -> class_id.

    0 = ball
    1 = player + goalkeeper
    2 = referee
    """
    mapping: Dict[int, int | None] = {}

    with gameinfo_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("trackletID_") or "=" not in line:
                continue

            left, description = line.split("=", 1)
            track_id = int(left.split("_")[-1])
            description = description.lower()

            if "ball" in description:
                class_id = 0
            elif "player" in description or "goalkeeper" in description:
                class_id = 1
            elif "referee" in description:
                class_id = 2
            else:
                class_id = None

            mapping[track_id] = class_id

    return mapping


def read_image_size(seqinfo_path: Path) -> Tuple[int, int]:
    """Zwraca (width, height) z seqinfo.ini."""
    config = configparser.ConfigParser()
    config.read(seqinfo_path, encoding="utf-8")

    width = int(config["Sequence"]["imWidth"])
    height = int(config["Sequence"]["imHeight"])
    return width, height


def load_ground_truth(
    clip_dir: Path,
) -> Tuple[Dict[int, List[dict]], Tuple[int, int]]:
    """
    Wczytuje GT do:
        frame_id -> lista obiektów.

    Oryginalny MOT:
        frame, track_id, x, y, width, height, mark, ...
    """
    gt_path = clip_dir / "gt" / "gt.txt"
    gameinfo_path = clip_dir / "gameinfo.ini"
    seqinfo_path = clip_dir / "seqinfo.ini"

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not gameinfo_path.exists():
        raise FileNotFoundError(gameinfo_path)
    if not seqinfo_path.exists():
        raise FileNotFoundError(seqinfo_path)

    class_map = parse_gameinfo_class_map(gameinfo_path)
    width, height = read_image_size(seqinfo_path)

    frames: Dict[int, List[dict]] = defaultdict(list)

    with gt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split(",")
            if len(parts) < 6:
                continue

            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])

            # MOT column 7: mark/confidence. Pomijamy nieaktywne adnotacje.
            if len(parts) >= 7 and float(parts[6]) <= 0:
                continue

            class_id = class_map.get(track_id)
            if class_id is None:
                continue

            frames[frame_id].append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                }
            )

    return frames, (width, height)


def valid_clip_dirs(test_root: Path) -> List[Path]:
    clips: List[Path] = []

    for clip_dir in sorted(test_root.glob("SNMOT-*")):
        if not clip_dir.is_dir():
            continue

        required = [
            clip_dir / "img1",
            clip_dir / "gt" / "gt.txt",
            clip_dir / "gameinfo.ini",
            clip_dir / "seqinfo.ini",
        ]

        if all(path.exists() for path in required):
            clips.append(clip_dir)

    return clips


def eligible_frames(
    clip_dir: Path,
    gt_by_frame: Dict[int, List[dict]],
) -> List[int]:
    """
    Klatka kwalifikuje się, jeśli:
      - istnieje plik obrazu,
      - istnieje co najmniej jedna rozpoznawalna adnotacja GT,
      - zawiera co najmniej jednego zawodnika/bramkarza.

    W praktyce zabezpiecza to przed wyborem klatek nieużytecznych
    z punktu widzenia głównego celu pracy.
    """
    frames: List[int] = []

    for frame_id in sorted(gt_by_frame):
        objects = gt_by_frame[frame_id]

        if not any(obj["class_id"] == 1 for obj in objects):
            continue

        image_path = clip_dir / "img1" / f"{frame_id:06d}.jpg"
        if image_path.exists():
            frames.append(frame_id)

    return frames


def stratified_temporal_sample(
    frame_ids: List[int],
    n: int,
    rng: random.Random,
) -> List[int]:
    """
    Dzieli dostępne klatki na n przedziałów czasowych i losuje po jednej
    klatce z każdego przedziału.

    Dla n=2:
      - jedna klatka z wcześniejszej części klipu,
      - jedna z późniejszej części klipu.

    Dzięki temu nie wybieramy dwóch prawie sąsiednich obrazów.
    """
    if n <= 0:
        return []

    if len(frame_ids) < n:
        raise ValueError(
            f"Za mało klatek ({len(frame_ids)}) do pobrania {n} próbek."
        )

    selected: List[int] = []
    total = len(frame_ids)

    for bin_idx in range(n):
        start = math.floor(bin_idx * total / n)
        end = math.floor((bin_idx + 1) * total / n)

        bucket = frame_ids[start:end]
        if not bucket:
            continue

        selected.append(rng.choice(bucket))

    return selected


def choose_samples(
    clips: List[Path],
    per_sequence: int,
    target: int,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], Dict[str, Dict[int, List[dict]]]]:
    rng = random.Random(seed)

    if target < len(clips) * per_sequence:
        raise ValueError(
            f"--target={target} jest mniejszy niż liczba bazowych próbek "
            f"{len(clips)} x {per_sequence} = {len(clips) * per_sequence}. "
            "Zwiększ target albo zmniejsz --per-sequence."
        )

    gt_cache: Dict[str, Dict[int, List[dict]]] = {}
    eligible_cache: Dict[str, List[int]] = {}
    selected_by_clip: Dict[str, set[int]] = defaultdict(set)

    samples: List[Tuple[Path, int]] = []

    # 1) Równa reprezentacja każdego klipu.
    for clip_dir in clips:
        gt_by_frame, _ = load_ground_truth(clip_dir)
        gt_cache[clip_dir.name] = gt_by_frame

        frames = eligible_frames(clip_dir, gt_by_frame)
        eligible_cache[clip_dir.name] = frames

        chosen = stratified_temporal_sample(
            frames,
            per_sequence,
            rng,
        )

        for frame_id in chosen:
            samples.append((clip_dir, frame_id))
            selected_by_clip[clip_dir.name].add(frame_id)

    # 2) Jeśli potrzebujemy np. 100 zamiast 98, dobieramy brakujące
    #    klatki z różnych sekwencji w round-robin.
    remaining = target - len(samples)

    if remaining > 0:
        clip_order = clips.copy()
        rng.shuffle(clip_order)

        while remaining > 0:
            progress = False

            for clip_dir in clip_order:
                if remaining <= 0:
                    break

                available = [
                    f
                    for f in eligible_cache[clip_dir.name]
                    if f not in selected_by_clip[clip_dir.name]
                ]

                if not available:
                    continue

                # Preferuj klatkę jak najdalej od już wybranych w tej sekwencji.
                already = selected_by_clip[clip_dir.name]

                if already:
                    distances = {
                        f: min(abs(f - old) for old in already)
                        for f in available
                    }
                    max_distance = max(distances.values())
                    farthest = [
                        f for f, d in distances.items()
                        if d == max_distance
                    ]
                    frame_id = rng.choice(farthest)
                else:
                    frame_id = rng.choice(available)

                samples.append((clip_dir, frame_id))
                selected_by_clip[clip_dir.name].add(frame_id)

                remaining -= 1
                progress = True

            if not progress:
                raise RuntimeError(
                    "Nie udało się dobrać wymaganej liczby unikalnych klatek."
                )

    # Kolejność eksportu losujemy deterministycznie, aby eval_001.. nie tworzyły
    # bloków pochodzących z jednej sekwencji.
    rng.shuffle(samples)

    return samples, gt_cache


def prepare_output(output_root: Path, overwrite: bool) -> Tuple[Path, Path]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Katalog już istnieje: {output_root}\n"
                "Jeśli chcesz wygenerować pulę od nowa, użyj --overwrite."
            )
        shutil.rmtree(output_root)

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    return images_dir, labels_dir


def write_yolo_label(
    label_path: Path,
    objects: List[dict],
    image_width: int,
    image_height: int,
) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for obj in objects:
            x_center = (obj["x"] + obj["w"] / 2.0) / image_width
            y_center = (obj["y"] + obj["h"] / 2.0) / image_height
            w_norm = obj["w"] / image_width
            h_norm = obj["h"] / image_height

            f.write(
                f'{obj["class_id"]} '
                f'{x_center:.8f} '
                f'{y_center:.8f} '
                f'{w_norm:.8f} '
                f'{h_norm:.8f}\n'
            )


def main() -> None:
    args = parse_args()

    test_root = args.test_root.resolve()
    output_root = args.output.resolve()

    if not test_root.exists():
        raise FileNotFoundError(
            f"Nie znaleziono zbioru testowego: {test_root}"
        )

    clips = valid_clip_dirs(test_root)

    if not clips:
        raise RuntimeError(
            f"Nie znaleziono poprawnych katalogów SNMOT-* w {test_root}"
        )

    print("=" * 72)
    print("PRZYGOTOWANIE PULI EWALUACYJNEJ DETEKCJI")
    print("=" * 72)
    print(f"Test root:              {test_root}")
    print(f"Liczba sekwencji:       {len(clips)}")
    print(f"Klatek / sekwencję:     {args.per_sequence}")
    print(f"Bazowa liczba klatek:   {len(clips) * args.per_sequence}")
    print(f"Docelowa liczba klatek: {args.target}")
    print(f"Seed:                   {args.seed}")

    if len(clips) * args.per_sequence != args.target:
        difference = args.target - len(clips) * args.per_sequence
        print(
            f"UWAGA: {len(clips)} x {args.per_sequence} = "
            f"{len(clips) * args.per_sequence}, więc do target={args.target} "
            f"zostanie dobranych dodatkowo {difference} klatek."
        )

    samples, gt_cache = choose_samples(
        clips=clips,
        per_sequence=args.per_sequence,
        target=args.target,
        seed=args.seed,
    )

    images_dir, labels_dir = prepare_output(
        output_root,
        overwrite=args.overwrite,
    )

    manifest_rows: List[dict] = []
    gt_rows: List[dict] = []

    sequence_counts = Counter()

    for export_idx, (clip_dir, frame_id) in enumerate(samples, start=1):
        export_name = f"eval_{export_idx:03d}.jpg"
        label_name = f"eval_{export_idx:03d}.txt"

        source_image = clip_dir / "img1" / f"{frame_id:06d}.jpg"
        destination_image = images_dir / export_name
        destination_label = labels_dir / label_name

        if not source_image.exists():
            raise FileNotFoundError(source_image)

        shutil.copy2(source_image, destination_image)

        objects = gt_cache[clip_dir.name][frame_id]
        width, height = read_image_size(clip_dir / "seqinfo.ini")

        write_yolo_label(
            destination_label,
            objects,
            image_width=width,
            image_height=height,
        )

        class_counts = Counter(obj["class_name"] for obj in objects)
        sequence_counts[clip_dir.name] += 1

        manifest_rows.append(
            {
                "eval_id": export_idx,
                "eval_image": export_name,
                "source_sequence": clip_dir.name,
                "source_frame": frame_id,
                "source_image": str(source_image.relative_to(PROJECT_ROOT)),
                "image_width": width,
                "image_height": height,
                "num_gt_total": len(objects),
                "num_players": class_counts.get("player", 0),
                "num_referees": class_counts.get("referee", 0),
                "num_balls": class_counts.get("ball", 0),
            }
        )

        for obj in objects:
            gt_rows.append(
                {
                    "eval_id": export_idx,
                    "eval_image": export_name,
                    "source_sequence": clip_dir.name,
                    "source_frame": frame_id,
                    "track_id": obj["track_id"],
                    "class_id": obj["class_id"],
                    "class_name": obj["class_name"],
                    "x": obj["x"],
                    "y": obj["y"],
                    "width": obj["w"],
                    "height": obj["h"],
                    "x1": obj["x1"],
                    "y1": obj["y1"],
                    "x2": obj["x2"],
                    "y2": obj["y2"],
                }
            )

    manifest_path = output_root / "manifest.csv"
    gt_path = output_root / "ground_truth.csv"
    info_path = output_root / "selection_info.json"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(manifest_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with gt_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(gt_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(gt_rows)

    selection_info = {
        "source": "SoccerNet-Tracking test",
        "test_root": str(test_root),
        "output_root": str(output_root),
        "number_of_sequences_found": len(clips),
        "per_sequence": args.per_sequence,
        "target": args.target,
        "exported_images": len(samples),
        "seed": args.seed,
        "selection_strategy": (
            "stratified temporal sampling within every sequence; "
            "remaining frames selected from different sequences with "
            "preference for maximum temporal distance from already selected frames"
        ),
        "class_mapping": CLASS_NAMES,
        "sequence_sample_counts": dict(sorted(sequence_counts.items())),
    }

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(
            selection_info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("GOTOWE")
    print("=" * 72)
    print(f"Wyeksportowano: {len(samples)} obrazów")
    print(f"Obrazy:         {images_dir}")
    print(f"Etykiety YOLO:  {labels_dir}")
    print(f"Manifest:       {manifest_path}")
    print(f"Ground truth:   {gt_path}")
    print(f"Info:           {info_path}")
    print()
    print("Liczba próbek z poszczególnych sekwencji:")
    for sequence, count in sorted(sequence_counts.items()):
        print(f"  {sequence}: {count}")


if __name__ == "__main__":
    main()
