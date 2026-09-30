"""
Rover Computer Vision System - Stage 1: Fine-Tuning Script

This script splits the compiled pseudo-annotated dataset into 85% train / 15% validation sets,
generates the Ultralytics data.yaml configuration, and fine-tunes YOLO26n on custom rover classes.
"""

from pathlib import Path
import random
import shutil
from ultralytics import YOLO

# ============================================================
# PATH CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Compiled dataset source from annotation step
COMPILED_DATASET_DIR = PROJECT_ROOT / "annotation" / "compiled_dataset"

# Training output directory
TRAINING_DIR = SCRIPT_DIR / "dataset"
TRAIN_IMAGES = TRAINING_DIR / "images" / "train"
VAL_IMAGES = TRAINING_DIR / "images" / "val"
TRAIN_LABELS = TRAINING_DIR / "labels" / "train"
VAL_LABELS = TRAINING_DIR / "labels" / "val"

# Base model to fine-tune
MODEL_PATH = PROJECT_ROOT / "models" / "pretrained" / "yolo26n.pt"

# Output fine-tuned models
FINE_TUNED_MODEL_DIR = PROJECT_ROOT / "models" / "fine_tuned"

# ============================================================
# HYPERPARAMETERS & SETTINGS
# ============================================================

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
RANDOM_SEED = 42

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4
DEVICE = 0  # 0 for GPU, 'cpu' for CPU

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "bench",
    "dog",
    "backpack",
    "handbag",
    "sports_ball",
    "bottle",
    "tv",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("\n" + "=" * 60)
    print("ROVER CV SYSTEM: YOLO26n FINE-TUNING")
    print("=" * 60)
    print(f"Project Root     : {PROJECT_ROOT}")
    print(f"Source Dataset   : {COMPILED_DATASET_DIR}")
    print(f"Target Dataset   : {TRAINING_DIR}")
    print(f"Pretrained Model : {MODEL_PATH}")

    if not COMPILED_DATASET_DIR.exists():
        print(f"\n[ERROR] Compiled dataset not found at {COMPILED_DATASET_DIR}")
        print("Please run annotation/annotate.py first.")
        raise SystemExit(1)

    source_images_dir = COMPILED_DATASET_DIR / "images"
    source_labels_dir = COMPILED_DATASET_DIR / "labels"

    # Create destination directories
    for directory in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        directory.mkdir(parents=True, exist_ok=True)

    # Find valid image-label pairs
    raw_images = [
        p for p in source_images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    valid_pairs = []
    for img_path in raw_images:
        lbl_path = source_labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            valid_pairs.append(img_path)

    print(f"\nTotal Source Images Found : {len(raw_images)}")
    print(f"Valid Image-Label Pairs   : {len(valid_pairs)}")

    if not valid_pairs:
        print("\n[ERROR] No valid image-label pairs found for training.")
        raise SystemExit(1)

    # Shuffle with reproducible seed
    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * TRAIN_RATIO)
    train_split = valid_pairs[:split_idx]
    val_split = valid_pairs[split_idx:]

    print("\n" + "-" * 60)
    print(f"Dataset Split (85/15): Total = {len(valid_pairs)}")
    print(f"  Train Set : {len(train_split)} ({len(train_split)/len(valid_pairs)*100:.1f}%)")
    print(f"  Val Set   : {len(val_split)} ({len(val_split)/len(valid_pairs)*100:.1f}%)")
    print("-" * 60)

    # Copy files to train/val subdirectories
    def copy_files(image_list, dest_img_dir, dest_lbl_dir):
        for img_p in image_list:
            lbl_p = source_labels_dir / f"{img_p.stem}.txt"
            shutil.copy2(img_p, dest_img_dir / img_p.name)
            shutil.copy2(lbl_p, dest_lbl_dir / lbl_p.name)

    print("\nPreparing training split...")
    copy_files(train_split, TRAIN_IMAGES, TRAIN_LABELS)
    print("Preparing validation split...")
    copy_files(val_split, VAL_IMAGES, VAL_LABELS)

    # Write data.yaml
    yaml_path = TRAINING_DIR / "data.yaml"
    yaml_content = f"path: {TRAINING_DIR.resolve()}\n\ntrain: images/train\nval: images/val\n\nnc: {len(CLASS_NAMES)}\n\nnames:\n"
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\ndata.yaml generated at: {yaml_path}")

    # Load & Train Model
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Pretrained model weights missing: {MODEL_PATH}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("STARTING YOLO26n FINE-TUNING")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    runs_dir = SCRIPT_DIR / "runs"
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        patience=20,
        cos_lr=True,
        amp=True,
        cache=False,
        project=str(runs_dir),
        name="yolo26n_finetuned",
        exist_ok=True,
    )

    best_weights = runs_dir / "yolo26n_finetuned" / "weights" / "best.pt"
    if best_weights.exists():
        FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, FINE_TUNED_MODEL_DIR / "best.pt")
        print(f"\nUpdated centralized fine-tuned model weight: {FINE_TUNED_MODEL_DIR / 'best.pt'}")

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
