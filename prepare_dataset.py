"""
Sorts your two downloaded datasets into the final training folder
structure: data/person/, data/hazard/, data/neither/
"""

import argparse
import os
import shutil
from pathlib import Path


def copy_file(src: Path, dst_dir: Path, prefix: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{prefix}_{src.name}"
    shutil.copy2(src, dst)


def sort_human_dataset(root: Path, out_dir: Path):
    counts = {"person": 0, "neither": 0}
    dir_1 = root / "1"
    dir_0 = root / "0"

    if dir_1.exists():
        for img in dir_1.glob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                copy_file(img, out_dir / "person", "human1")
                counts["person"] += 1
    else:
        print(f"WARNING: folder '1' not found under {root}")

    if dir_0.exists():
        for img in dir_0.glob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                copy_file(img, out_dir / "neither", "human0")
                counts["neither"] += 1
    else:
        print(f"WARNING: folder '0' not found under {root}")

    return counts


def sort_fire_dataset(root: Path, out_dir: Path):
    counts = {"hazard": 0, "neither": 0, "missing_label": 0}
    splits = ["train", "valid", "test"]

    for split in splits:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"

        if not images_dir.exists():
            print(f"WARNING: {images_dir} not found, skipping split '{split}'")
            continue

        for img in images_dir.glob("*"):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            label_file = labels_dir / (img.stem + ".txt")

            if not label_file.exists():
                counts["missing_label"] += 1
                continue

            content = label_file.read_text().strip()
            if content:
                copy_file(img, out_dir / "hazard", f"fire_{split}")
                counts["hazard"] += 1
            else:
                copy_file(img, out_dir / "neither", f"fire_{split}")
                counts["neither"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-dataset", type=str, required=True)
    parser.add_argument("--fire-dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    human_counts = sort_human_dataset(Path(args.human_dataset), out_dir)
    fire_counts = sort_fire_dataset(Path(args.fire_dataset), out_dir)

    print("\n=== Human Detection Dataset ===")
    print(human_counts)
    print("\n=== Indoor Fire Smoke Dataset ===")
    print(fire_counts)

    print("\n=== FINAL COUNTS in data/ ===")
    for label in ["person", "hazard", "neither"]:
        folder = out_dir / label
        n = len(list(folder.glob("*"))) if folder.exists() else 0
        print(f"  {label}: {n} images")


if __name__ == "__main__":
    main()