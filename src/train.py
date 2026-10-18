"""
Training Pipeline for AML-Based Room Identifier.
Executes two-phase transfer learning (Phase 1: Feature Extraction, Phase 2: Fine-Tuning)
with CrossEntropyLoss, Adam optimizer, ReduceLROnPlateau scheduler, and early stopping.
Generates learning curves saved to outputs/training_curves.png and checkpoints to models/best_model.pt.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloaders
from src.model import build_model, RoomClassifier
from src.utils import setup_logger, load_config, set_seed, get_device, ensure_dirs, resolve_path

logger = setup_logger("train")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    phase_name: str = "Train",
) -> Tuple[float, float]:
    """
    Train model for one epoch. Returns (avg_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"[{phase_name} Epoch {epoch:02d}] Train", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    phase_name: str = "Val",
) -> Tuple[float, float]:
    """
    Evaluate model on validation/test set. Returns (avg_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"[{phase_name} Epoch {epoch:02d}] Eval", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_training_curves(
    history: Dict[str, List[float]],
    phase1_epochs: int,
    save_path: str,
) -> None:
    """
    Plot and save training and validation loss and accuracy curves.
    Highlights Phase 1 (feature extraction) and Phase 2 (fine-tuning).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e", linewidth=2)
    if phase1_epochs < len(epochs):
        ax1.axvline(x=phase1_epochs + 0.5, color="#d62728", linestyle="--", alpha=0.7, label="Phase 2 Start")
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val Accuracy", color="#9467bd", linewidth=2)
    if phase1_epochs < len(epochs):
        ax2.axvline(x=phase1_epochs + 0.5, color="#d62728", linestyle="--", alpha=0.7, label="Phase 2 Start")
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("AML Room Identifier - 2-Phase Transfer Learning Curves", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved learning curves to: {save_path}")


def train_room_classifier(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Main training workflow for Room Identifier.
    """
    config = load_config(config_path)
    seed = config["project"].get("seed", 42)
    set_seed(seed)

    device_pref = config["project"].get("device", "auto")
    device = get_device(device_pref)
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Output directories
    checkpoint_path = str(resolve_path(config["model"].get("checkpoint_path", "models/best_model.pt")))
    curves_path = str(resolve_path(config["outputs"].get("training_curves", "outputs/training_curves.png")))
    ensure_dirs(os.path.dirname(checkpoint_path), os.path.dirname(curves_path))

    # Data Loaders
    train_loader, val_loader, test_loader, classes, class_to_idx, class_weights = create_dataloaders(config_path)
    num_classes = len(classes)
    logger.info(f"Active training classes ({num_classes}): {classes}")

    # Model Setup
    model = build_model(num_classes=num_classes, config=config).to(device)

    # Loss Criterion with balanced class weights
    if config["training"].get("use_class_weights", True):
        weights_tensor = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        logger.info(f"Using Weighted CrossEntropyLoss with weights: {weights_tensor.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Training History Tracker
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    p1_cfg = config["training"]["phase1"]
    p2_cfg = config["training"]["phase2"]
    sched_cfg = config["training"]["scheduler"]
    es_cfg = config["training"]["early_stopping"]

    # =========================================================================
    # PHASE 1: Feature Extraction (Freeze Backbone, Train Classifier Head)
    # =========================================================================
    p1_epochs = p1_cfg.get("epochs", 5)
    p1_lr = p1_cfg.get("lr", 0.001)
    p1_wd = p1_cfg.get("weight_decay", 1e-4)

    logger.info("=" * 65)
    logger.info(f"STARTING PHASE 1: Classifier Head Training ({p1_epochs} Epochs, LR={p1_lr})")
    logger.info("=" * 65)

    model.freeze_backbone()
    optimizer_p1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=p1_lr,
        weight_decay=p1_wd,
    )

    current_epoch = 0
    for epoch in range(1, p1_epochs + 1):
        current_epoch += 1
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer_p1, device, current_epoch, "Phase 1")
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device, current_epoch, "Phase 1")

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["learning_rates"].append(p1_lr)

        logger.info(
            f"[Phase 1] Epoch {epoch:02d}/{p1_epochs:02d} -> "
            f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | "
            f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_epoch = current_epoch
            # Save checkpoint
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_names": classes,
                "class_to_idx": class_to_idx,
                "backbone": config["model"].get("backbone", "mobilenet_v2"),
                "dropout_rate": config["model"].get("dropout_rate", 0.3),
                "image_size": config["dataset"].get("image_size", 224),
            }, checkpoint_path)
            logger.info(f"[*] Checkpoint saved at Epoch {current_epoch} (Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%)")

    # =========================================================================
    # PHASE 2: Fine-Tuning (Unfreeze Last Blocks, Lower LR)
    # =========================================================================
    p2_epochs = p2_cfg.get("epochs", 12)
    p2_lr = p2_cfg.get("lr", 0.0001)
    p2_wd = p2_cfg.get("weight_decay", 1e-4)
    unfreeze_layers = p2_cfg.get("unfreeze_layers", 4)

    logger.info("=" * 65)
    logger.info(f"STARTING PHASE 2: Fine-Tuning ({p2_epochs} Epochs, LR={p2_lr}, Unfreeze Last {unfreeze_layers} Blocks)")
    logger.info("=" * 65)

    model.unfreeze_backbone(unfreeze_layers=unfreeze_layers)
    optimizer_p2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=p2_lr,
        weight_decay=p2_wd,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2,
        mode="min",
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 2),
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )

    es_patience = es_cfg.get("patience", 5)
    min_delta = es_cfg.get("min_delta", 0.001)

    for epoch in range(1, p2_epochs + 1):
        current_epoch += 1
        curr_lr = optimizer_p2.param_groups[0]["lr"]
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer_p2, device, current_epoch, "Phase 2")
        v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, device, current_epoch, "Phase 2")

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["learning_rates"].append(curr_lr)

        scheduler.step(v_loss)

        logger.info(
            f"[Phase 2] Epoch {epoch:02d}/{p2_epochs:02d} (LR: {curr_lr:.2e}) -> "
            f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | "
            f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%"
        )

        # Check for improvement
        if v_loss < (best_val_loss - min_delta):
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_epoch = current_epoch
            patience_counter = 0
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "class_names": classes,
                "class_to_idx": class_to_idx,
                "backbone": config["model"].get("backbone", "mobilenet_v2"),
                "dropout_rate": config["model"].get("dropout_rate", 0.3),
                "image_size": config["dataset"].get("image_size", 224),
            }, checkpoint_path)
            logger.info(f"[*] Checkpoint updated at Epoch {current_epoch} (Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"EarlyStopping patience counter: {patience_counter}/{es_patience}")
            if patience_counter >= es_patience:
                logger.info(f"Early stopping triggered at epoch {current_epoch} (No validation improvement for {es_patience} epochs).")
                break

    # Plot learning curves
    plot_training_curves(history, phase1_epochs=p1_epochs, save_path=curves_path)

    logger.info("=" * 65)
    logger.info(f"TRAINING COMPLETED! Best Val Acc: {best_val_acc*100:.2f}% (Loss: {best_val_loss:.4f}) at Epoch {best_epoch}")
    logger.info(f"Best checkpoint saved to: {checkpoint_path}")
    logger.info("=" * 65)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history,
        "classes": classes,
        "checkpoint_path": checkpoint_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Room Classifier.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    train_room_classifier(args.config)
