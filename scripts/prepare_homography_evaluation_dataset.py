#!/usr/bin/env python3
"""
Przygotowanie zamrożonej puli ewaluacyjnej homografii z SoccerNet-Calibration.

Umieść plik jako:
    CvFootballTracker/scripts/prepare_homography_evaluation_dataset.py

Domyślne źródło:
    data/calibration_dataset/test/

Zakładany format źródła:
    00000.jpg
    00000.json
    00001.jpg
    00001.json
    ...

Domyślne wyjście:
    data/data_homography_evaluation/
        images/
            eval_homo_001.jpg
            ...
        annotations/
            eval_homo_001.json
            ...
        manifest.csv
        selection_info.json

Działanie:
    - znajduje pary obraz + JSON o tym samym stemie,
    - sortuje je naturalnie/numerowo,
    - wybiera pierwsze 100 kompletnych par,
    - kopiuje je do osobnego katalogu i zmienia nazwy na eval_homo_001...
    - zapisuje manifest pozwalający odtworzyć pochodzenie każdej próbki.

UWAGA:
    Domyślnie skrypt KOPIUJE pliki, aby nie niszczyć źródłowego zbioru testowego.
    Jeśli naprawdę chcesz je usunąć ze źródła, użyj --move.

Przykłady:
    python scripts/prepare_homography_evaluation_dataset.py

    python scripts/prepare_homography_evaluation_dataset.py --overwrite

    python scripts/prepare_homography_evaluation_dataset.py --target 100 --move --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "calibration_dataset" / "calibration" / "test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "data_homography_evaluation"


def natural_key(path_or_stem: str | Path):
    """Naturalne sortowanie: 2 przed 10, 00002 przed 00010."""
    text = Path(path_or_stem).stem if isinstance(path_or_stem, Path) else str(path_or_stem)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Przygotuj 100 pierwszych kompletnych par JPG+JSON do ewaluacji homografii."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Katalog źródłowy. Domyślnie: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Katalog docelowy. Domyślnie: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help="Liczba par do wybrania. Domyślnie: 100.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Przenieś pliki zamiast je kopiować. Domyślnie źródło pozostaje nietknięte.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Usuń istniejący katalog wyjściowy przed przygotowaniem nowej puli.",
    )
    return parser.parse_args()


def collect_pairs(source_root: Path) -> List[Tuple[Path, Path]]:
    """
    Zwraca kompletne pary (obraz, json) o tym samym stemie.

    Obsługiwane rozszerzenia obrazów: .jpg, .jpeg, .png
    Preferencja przy kilku obrazach o tym samym stemie: jpg -> jpeg -> png.
    """
    if not source_root.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu źródłowego: {source_root}")

    image_by_stem = {}

    for ext in (".jpg", ".jpeg", ".png"):
        for image_path in source_root.glob(f"*{ext}"):
            image_by_stem.setdefault(image_path.stem, image_path)

    json_by_stem = {
        p.stem: p
        for p in source_root.glob("*.json")
    }

    common_stems = sorted(
        set(image_by_stem) & set(json_by_stem),
        key=natural_key,
    )

    return [
        (image_by_stem[stem], json_by_stem[stem])
        for stem in common_stems
    ]


def main() -> None:
    args = parse_args()

    if args.target <= 0:
        raise ValueError("--target musi być > 0.")

    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if source_root == output_root:
        raise ValueError("Katalog źródłowy i docelowy nie mogą być takie same.")

    pairs = collect_pairs(source_root)

    if len(pairs) < args.target:
        raise RuntimeError(
            f"Znaleziono tylko {len(pairs)} kompletnych par obraz+JSON, "
            f"a wymagane jest {args.target}."
        )

    selected = pairs[: args.target]

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Katalog wyjściowy już istnieje: {output_root}\n"
                f"Użyj --overwrite, jeśli chcesz przygotować pulę od nowa."
            )

    images_dir = output_root / "images"
    annotations_dir = output_root / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    transfer = shutil.move if args.move else shutil.copy2
    mode_name = "move" if args.move else "copy"

    print(f"Źródło: {source_root}")
    print(f"Znaleziono kompletnych par: {len(pairs)}")
    print(f"Wybrano pierwszych par: {len(selected)}")
    print(f"Tryb: {mode_name}")
    print(f"Wyjście: {output_root}\n")

    for eval_id, (source_image, source_json) in enumerate(selected, start=1):
        eval_stem = f"eval_homo_{eval_id:03d}"

        target_image = images_dir / f"{eval_stem}{source_image.suffix.lower()}"
        target_json = annotations_dir / f"{eval_stem}.json"

        original_image = str(source_image.relative_to(PROJECT_ROOT)) if source_image.is_relative_to(PROJECT_ROOT) else str(source_image)
        original_json = str(source_json.relative_to(PROJECT_ROOT)) if source_json.is_relative_to(PROJECT_ROOT) else str(source_json)

        transfer(str(source_image), str(target_image))
        transfer(str(source_json), str(target_json))

        manifest_rows.append(
            {
                "eval_id": eval_id,
                "eval_stem": eval_stem,
                "eval_image": target_image.name,
                "eval_annotation": target_json.name,
                "source_stem": source_image.stem,
                "source_image": original_image,
                "source_annotation": original_json,
            }
        )

        print(
            f"[{eval_id:03d}/{len(selected):03d}] "
            f"{source_image.name} + {source_json.name} -> {eval_stem}"
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_id",
                "eval_stem",
                "eval_image",
                "eval_annotation",
                "source_stem",
                "source_image",
                "source_annotation",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    selection_info = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "available_complete_pairs": len(pairs),
        "selected_pairs": len(selected),
        "selection_rule": "first complete image+json pairs after natural/numeric sorting",
        "transfer_mode": mode_name,
        "naming": "eval_homo_001 ... eval_homo_NNN",
    }

    with (output_root / "selection_info.json").open("w", encoding="utf-8") as f:
        json.dump(selection_info, f, indent=2, ensure_ascii=False)

    print("\nGotowe.")
    print(f"Manifest: {manifest_path}")
    print(f"Obrazy:   {images_dir}")
    print(f"JSON-y:   {annotations_dir}")


if __name__ == "__main__":
    main()
