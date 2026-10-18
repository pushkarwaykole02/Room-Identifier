import os
import sys
import random
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# Ensure UTF-8 stdout/stderr on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logger(name: str = "room_id", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a standardized logger with console and optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set deterministic random seeds across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(p: str | Path) -> Path:
    """
    Resolve a path. If it does not exist relative to CWD, resolve it relative to PROJECT_ROOT.
    """
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file into a dictionary.
    """
    path = resolve_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_name: str = "auto") -> torch.device:
    """
    Determine and return torch.device based on availability and config.
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif device_name == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def ensure_dirs(*dirs: str) -> None:
    """
    Create directories if they do not exist.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file with indentation.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file into a dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_class_weights(
    class_counts: Dict[str, int],
    classes: list[str]
) -> torch.Tensor:
    """
    Compute balanced inverse-frequency class weights:
    weight_c = total_samples / (num_classes * count_c)
    """
    total_samples = sum(class_counts[c] for c in classes if class_counts.get(c, 0) > 0)
    num_classes = len(classes)
    weights = []
    for c in classes:
        cnt = class_counts.get(c, 0)
        if cnt > 0:
            w = total_samples / (num_classes * cnt)
        else:
            w = 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)
