"""
Data Preparation Module for AML-Based Room Identifier.
Filters, validates, maps, and copies room images from the raw dataset to data/processed/,
and configures the data/custom/ directory for user phone photos.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2

from src.utils import setup_logger, load_config, ensure_dirs, compute_class_weights, resolve_path

logger = setup_logger("prepare_data")


def validate_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image file is readable, not corrupt, and can be converted to RGB.
    Returns (is_valid, error_message).
    """
    try:
        # Check with PIL
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            if img_rgb.size[0] == 0 or img_rgb.size[1] == 0:
                return False, "Zero dimension image"
        
        # Check with OpenCV
        cv_img = cv2.imread(str(image_path))
        if cv_img is None:
            return False, "OpenCV failed to decode image"
        
        return True, None
    except Exception as e:
        return False, str(e)


def inspect_raw_dataset(raw_dir: Path) -> Dict[str, Dict[str, any]]:
    """
    Inspect raw dataset directory structure, subfolders, file types, counts, and split files.
    """
    logger.info(f"Inspecting raw dataset directory at: {raw_dir.resolve()}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found at: {raw_dir}")

    structure: Dict[str, Dict[str, any]] = {}
    total_raw_images = 0
    split_files = []

    for item in sorted(raw_dir.iterdir()):
        if item.is_dir():
            files = [f for f in item.iterdir() if f.is_file()]
            img_files = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
            exts = list(set(f.suffix.lower() for f in files))
            structure[item.name] = {
                "total_files": len(files),
                "image_files": len(img_files),
                "extensions": exts,
            }
            total_raw_images += len(img_files)
        elif item.is_file():
            if "train" in item.name.lower() or "test" in item.name.lower() or "split" in item.name.lower():
                split_files.append(item.name)

    logger.info(f"Found {len(structure)} subdirectories with {total_raw_images} total images.")
    logger.info(f"Split text files found: {split_files if split_files else 'None'}")
    for name, info in structure.items():
        logger.info(f"  - {name}: {info['image_files']} images (Extensions: {info['extensions']})")

    return structure


def setup_custom_directory(custom_dir: Path, target_classes: List[str]) -> None:
    """
    Create data/custom/<class_name>/ subdirectories for user-supplied phone photos.
    """
    ensure_dirs(str(custom_dir))
    for c in target_classes:
        class_folder = custom_dir / c
        ensure_dirs(str(class_folder))
        # Add a .gitkeep so empty directories are preserved
        gitkeep = class_folder / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    logger.info(f"Created custom dataset directories at: {custom_dir.resolve()} for classes: {target_classes}")


def prepare_dataset(config_path: str = "config.yaml") -> Dict[str, any]:
    """
    Main data preparation function:
    1. Inspects raw dataset.
    2. Maps MIT category names to standardized target room classes.
    3. Validates and copies images to data/processed/<class_name>/.
    4. Sets up data/custom/ for phone photos.
    5. Scans custom photos and reports distribution and imbalance.
    """
    config = load_config(config_path)
    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    processed_dir = resolve_path(config["dataset"]["processed_dir"])
    custom_dir = resolve_path(config["dataset"]["custom_dir"])
    folder_mapping: Dict[str, str] = config["dataset"]["folder_mapping"]

    # All unique target classes
    target_classes = sorted(list(set(folder_mapping.values())))

    # 1. Inspect raw dataset
    raw_structure = inspect_raw_dataset(raw_dir)

    # 2. Setup processed and custom directories
    ensure_dirs(str(processed_dir))
    for c in target_classes:
        ensure_dirs(str(processed_dir / c))
    setup_custom_directory(custom_dir, target_classes)

    # 3. Copy and validate images from raw to processed
    logger.info("Copying and validating images into data/processed/ ...")
    skipped_corrupt: List[Tuple[str, str]] = []
    processed_counts: Dict[str, int] = {c: 0 for c in target_classes}

    # Map raw folders
    for raw_folder_name, target_class in folder_mapping.items():
        src_folder = raw_dir / raw_folder_name
        if not src_folder.exists() or not src_folder.is_dir():
            logger.info(f"Notice: Source folder '{raw_folder_name}' not found in raw dataset (skipped).")
            continue

        images = [f for f in src_folder.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        dest_folder = processed_dir / target_class

        logger.info(f"Mapping '{raw_folder_name}' -> '{target_class}' ({len(images)} candidate images)...")
        for img_path in images:
            is_valid, err = validate_image(img_path)
            if not is_valid:
                skipped_corrupt.append((str(img_path), err or "Unknown validation error"))
                logger.warning(f"Skipping corrupt/unreadable image: {img_path} ({err})")
                continue

            # Target filename includes source prefix to prevent name collisions
            dest_file = dest_folder / f"{raw_folder_name}_{img_path.name}"
            shutil.copy2(img_path, dest_file)
            processed_counts[target_class] += 1

    # Check for custom photos
    custom_counts: Dict[str, int] = {c: 0 for c in target_classes}
    for c in target_classes:
        c_folder = custom_dir / c
        if c_folder.exists():
            c_images = [f for f in c_folder.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
            custom_counts[c] = len(c_images)

    total_custom = sum(custom_counts.values())
    if total_custom > 0:
        logger.info(f"Found {total_custom} user-supplied custom photos in {custom_dir}: {custom_counts}")

    # 4. Summary & Imbalance Analysis
    total_counts = {c: processed_counts[c] + custom_counts[c] for c in target_classes}
    total_active_images = sum(total_counts.values())

    logger.info("=" * 60)
    logger.info("DATASET PREPARATION SUMMARY & CLASS DISTRIBUTION")
    logger.info("=" * 60)
    for c in target_classes:
        logger.info(f"  Class '{c:<12}': Processed = {processed_counts[c]:>4} | Custom = {custom_counts[c]:>3} | Total = {total_counts[c]:>4}")
    logger.info(f"Total Images Available for Pipeline: {total_active_images}")
    logger.info(f"Corrupt/Skipped Images: {len(skipped_corrupt)}")
    logger.info("=" * 60)

    # Imbalance Check
    counts_list = [total_counts[c] for c in target_classes if total_counts[c] > 0]
    if counts_list:
        max_c = max(counts_list)
        min_c = min(counts_list)
        ratio = max_c / min_c if min_c > 0 else float("inf")
        logger.info(f"Class Imbalance Ratio (Max / Min): {ratio:.2f} (Max: {max_c}, Min: {min_c})")
        if ratio > 2.0:
            logger.warning(f"Dataset is moderately/severely imbalanced (ratio {ratio:.2f} > 2.0). Class weighting in CrossEntropyLoss will be used.")
        else:
            logger.info("Dataset is well-balanced across active classes.")

    # Calculate class weights for active classes
    active_classes = [c for c in target_classes if total_counts[c] > 0]
    weights = compute_class_weights(total_counts, active_classes)
    weights_dict = {c: float(weights[i]) for i, c in enumerate(active_classes)}
    logger.info(f"Computed Balanced Class Weights: {weights_dict}")

    return {
        "target_classes": target_classes,
        "active_classes": active_classes,
        "processed_counts": processed_counts,
        "custom_counts": custom_counts,
        "total_counts": total_counts,
        "class_weights": weights_dict,
        "skipped_corrupt": skipped_corrupt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and validate dataset for Room Identifier.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    prepare_dataset(args.config)
