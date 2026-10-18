"""
Dataset and DataLoader creation for AML-Based Room Identifier.
Implements custom PyTorch RoomDataset and stratified train/val/test splitting (70/15/15).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transforms import get_train_transforms, get_val_test_transforms
from src.utils import setup_logger, load_config, compute_class_weights, resolve_path

logger = setup_logger("dataset")


class RoomDataset(Dataset):
    """
    Custom PyTorch Dataset for room image classification.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Any] = None,
    ) -> None:
        """
        Args:
            image_paths: List of absolute or relative paths to image files.
            labels: List of integer class labels corresponding to image_paths.
            transform: PyTorch transform pipeline to apply to images.
        """
        assert len(image_paths) == len(labels), "image_paths and labels must have the same length"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image at {img_path}: {e}")
            # Fallback black image if corrupted
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(image)

        return image_tensor, label, img_path


def discover_all_images(
    processed_dir: Path,
    custom_dir: Path,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Discover all image filepaths and their class labels from data/processed and data/custom.
    Returns (image_paths, string_labels, active_class_names).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths: List[str] = []
    string_labels: List[str] = []

    # Check processed directory
    if processed_dir.exists():
        for class_dir in sorted(processed_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                for f in class_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in valid_exts:
                        image_paths.append(str(f.resolve()))
                        string_labels.append(class_name)

    # Check custom directory
    if custom_dir.exists():
        for class_dir in sorted(custom_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                for f in class_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in valid_exts:
                        image_paths.append(str(f.resolve()))
                        string_labels.append(class_name)

    # Determine unique classes with at least one sample
    class_counts: Dict[str, int] = {}
    for lbl in string_labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    active_classes = sorted([cls for cls, count in class_counts.items() if count > 0])
    logger.info(f"Discovered {len(image_paths)} total images across {len(active_classes)} active classes.")
    for cls in active_classes:
        logger.info(f"  Class '{cls}': {class_counts[cls]} samples")

    return image_paths, string_labels, active_classes


def get_stratified_splits(
    image_paths: List[str],
    string_labels: List[str],
    active_classes: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Dict[str, int],
    Dict[int, str],
]:
    """
    Perform a stratified 70/15/15 train/val/test split.
    """
    class_to_idx = {cls_name: i for i, cls_name in enumerate(active_classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(active_classes)}
    numeric_labels = [class_to_idx[lbl] for lbl in string_labels]

    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-5, f"Split ratios must sum to 1.0, got {total_ratio}"

    # First split: Train vs (Val + Test)
    temp_ratio = val_ratio + test_ratio  # 0.30
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        numeric_labels,
        test_size=temp_ratio,
        stratify=numeric_labels,
        random_state=seed,
    )

    # Second split: Val vs Test (50/50 split of the remaining 30% = 15% val, 15% test)
    val_relative_ratio = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1.0 - val_relative_ratio),
        stratify=temp_labels,
        random_state=seed,
    )

    logger.info(
        f"Stratified Split Completed (Seed={seed}): "
        f"Train={len(train_paths)} ({len(train_paths)/len(image_paths):.1%}), "
        f"Val={len(val_paths)} ({len(val_paths)/len(image_paths):.1%}), "
        f"Test={len(test_paths)} ({len(test_paths)/len(image_paths):.1%})"
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
        class_to_idx,
        idx_to_class,
    )


def create_dataloaders(
    config_path: str = "config.yaml",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int], torch.Tensor]:
    """
    Build and return train, validation, and test PyTorch DataLoaders along with
    class mappings and class weights for balanced training.
    """
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    seed = config["project"].get("seed", 42)

    processed_dir = resolve_path(dataset_cfg["processed_dir"])
    custom_dir = resolve_path(dataset_cfg["custom_dir"])
    image_size = dataset_cfg.get("image_size", 224)
    batch_size = dataset_cfg.get("batch_size", 32)
    num_workers = dataset_cfg.get("num_workers", 2)
    train_split = dataset_cfg.get("train_split", 0.70)
    val_split = dataset_cfg.get("val_split", 0.15)
    test_split = dataset_cfg.get("test_split", 0.15)

    # Discover images
    image_paths, string_labels, active_classes = discover_all_images(processed_dir, custom_dir)
    if not image_paths:
        raise RuntimeError("No images found in processed or custom datasets. Run prepare_data.py first.")

    # Split dataset
    (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
        class_to_idx,
        idx_to_class,
    ) = get_stratified_splits(
        image_paths,
        string_labels,
        active_classes,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        seed=seed,
    )

    # Transforms
    train_transform = get_train_transforms(image_size=image_size)
    val_test_transform = get_val_test_transforms(image_size=image_size)

    # Datasets
    train_dataset = RoomDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = RoomDataset(val_paths, val_labels, transform=val_test_transform)
    test_dataset = RoomDataset(test_paths, test_labels, transform=val_test_transform)

    # Dataloaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Compute class weights for training set
    train_class_counts: Dict[str, int] = {}
    for idx in train_labels:
        c_name = idx_to_class[idx]
        train_class_counts[c_name] = train_class_counts.get(c_name, 0) + 1

    class_weights = compute_class_weights(train_class_counts, active_classes)

    return train_loader, val_loader, test_loader, active_classes, class_to_idx, class_weights


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes, c2i, weights = create_dataloaders()
    print("Classes:", classes)
    print("Class-to-Index:", c2i)
    print("Class Weights:", weights)
    print(f"Batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
