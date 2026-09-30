"""
Rover Computer Vision System - Dataset Preprocessing & Bounding Box Validation

Utility module for inspecting annotation correctness, checking coordinate boundaries,
and verifying YOLO format integrity before fine-tuning.
"""

from pathlib import Path
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = SCRIPT_DIR / "dataset"

NUM_CLASSES = 12

def validate_dataset(dataset_path: Path):
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[ERROR] Dataset paths do not exist in {dataset_path}")
        return False

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in image_extensions]

    valid_count = 0
    invalid_count = 0

    print(f"Validating {len(images)} images in {dataset_path}...")

    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARNING] Missing label for {img_path.name}")
            invalid_count += 1
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        valid = True
        for line_no, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                valid = False
                break
            try:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                valid = False
                break

            if not (0 <= cls_id < NUM_CLASSES) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                valid = False
                break

        if valid:
            valid_count += 1
        else:
            invalid_count += 1

    print(f"Validation finished: {valid_count} valid, {invalid_count} invalid.")
    return invalid_count == 0

if __name__ == "__main__":
    validate_dataset(DATASET_DIR)
