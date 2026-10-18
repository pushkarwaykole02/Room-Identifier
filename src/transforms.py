"""
Data transformations and augmentation pipeline for Room Identifier.
Implements ImageNet normalization, training augmentations, and validation/test preprocessing.
"""

import sys
from pathlib import Path
from typing import Tuple

import torchvision.transforms as T

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Construct training data transformations with augmentations:
    - RandomResizedCrop (scale 0.8 to 1.0)
    - RandomHorizontalFlip (50% probability)
    - RandomRotation (up to 15 degrees)
    - ColorJitter (brightness and contrast)
    - Normalization with ImageNet mean and std
    """
    return T.Compose([
        T.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms(image_size: int = 224) -> T.Compose:
    """
    Construct validation and testing transformations:
    - Resize to slightly larger scale (256 for 224)
    - CenterCrop to exact image_size
    - Normalization with ImageNet mean and std
    """
    resize_dim = int(image_size * 1.143)  # 256 for 224
    return T.Compose([
        T.Resize(size=resize_dim),
        T.CenterCrop(size=(image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> T.Compose:
    """
    Construct inference transform for single images / webcam frames.
    """
    return get_val_test_transforms(image_size=image_size)
