"""
Evaluation Module for AML-Based Room Identifier.
Evaluates best_model.pt on the held-out TEST set only.
Calculates Accuracy, Precision, Recall, F1, Macro Averages, Confusion Matrix,
identifies top confused room pairs with mitigation suggestions, and saves metrics.json.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloaders
from src.model import load_checkpoint
from src.utils import setup_logger, load_config, get_device, save_json, ensure_dirs, resolve_path

logger = setup_logger("evaluate")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """
    Plot and save a high-resolution, publication-ready Confusion Matrix heatmap.
    Displays raw counts and recall percentages per cell.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="white", font_scale=1.05)

    # Normalize by row (true class)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    # Combined annotation: Count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    plt.figure(figsize=(9, 7.5), dpi=300)
    cmap = sns.light_palette("#1f77b4", as_cmap=True)
    ax = sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall"},
        linewidths=1.2,
        linecolor="white",
        square=True,
    )

    plt.title("Held-Out Test Set Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Class", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Confusion Matrix to: {save_path}")


def find_top_confused_pairs(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Identify off-diagonal cells with the highest misclassification counts.
    """
    confusions = []
    num_classes = len(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                true_name = class_names[i]
                pred_name = class_names[j]
                confusions.append({
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                    "true_total": int(cm[i].sum()),
                    "error_rate": float(cm[i, j] / (cm[i].sum() + 1e-7)),
                })

    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:top_k]


@torch.no_grad()
def evaluate_model(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Run full evaluation on held-out test set using best_model.pt.
    """
    config = load_config(config_path)
    device_pref = config["project"].get("device", "auto")
    device = get_device(device_pref)

    checkpoint_path = str(resolve_path(config["model"].get("checkpoint_path", "models/best_model.pt")))
    cm_path = str(resolve_path(config["outputs"].get("confusion_matrix", "outputs/confusion_matrix.png")))
    metrics_json_path = str(resolve_path(config["outputs"].get("metrics_json", "outputs/metrics.json")))
    ensure_dirs(os.path.dirname(cm_path), os.path.dirname(metrics_json_path))

    # Load Model
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    # Load test dataloader
    _, _, test_loader, classes, class_to_idx, _ = create_dataloaders(config_path)
    model_classes = checkpoint.get("class_names", classes)

    logger.info("=" * 65)
    logger.info("EVALUATING MODEL ON HELD-OUT TEST SET ONLY")
    logger.info(f"Test Samples: {len(test_loader.dataset)} | Target Classes: {model_classes}")
    logger.info("=" * 65)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    pbar = tqdm(test_loader, desc="Testing", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Overall Accuracy
    accuracy = float(accuracy_score(y_true, y_pred))

    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        target_names=model_classes,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=model_classes,
        digits=4,
        zero_division=0,
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(model_classes))))
    plot_confusion_matrix(cm, model_classes, cm_path)

    # Top confused room pairs
    top_confused = find_top_confused_pairs(cm, model_classes, top_k=3)

    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})

    logger.info("\n" + "=" * 65)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 65)
    logger.info(f"\n{report_text}")
    logger.info(f"Overall Test Accuracy : {accuracy * 100:.2f}%")
    logger.info(f"Macro Precision       : {macro_avg.get('precision', 0.0):.4f}")
    logger.info(f"Macro Recall          : {macro_avg.get('recall', 0.0):.4f}")
    logger.info(f"Macro F1-Score        : {macro_avg.get('f1-score', 0.0):.4f}")
    logger.info("=" * 65)

    if top_confused:
        logger.info("Top Misclassified Room Pairs:")
        for idx, conf in enumerate(top_confused, 1):
            logger.info(
                f"  {idx}. True: '{conf['true_class']}' -> Predicted: '{conf['predicted_class']}' "
                f"({conf['count']}/{conf['true_total']} errors, {conf['error_rate']*100:.1f}%)"
            )

    # Compile final metrics dictionary
    metrics_summary = {
        "dataset": {
            "test_samples": len(y_true),
            "num_classes": len(model_classes),
            "classes": model_classes,
        },
        "overall_accuracy": accuracy,
        "macro_avg": {
            "precision": macro_avg.get("precision", 0.0),
            "recall": macro_avg.get("recall", 0.0),
            "f1_score": macro_avg.get("f1-score", 0.0),
        },
        "weighted_avg": {
            "precision": weighted_avg.get("precision", 0.0),
            "recall": weighted_avg.get("recall", 0.0),
            "f1_score": weighted_avg.get("f1-score", 0.0),
        },
        "per_class": {
            cls_name: {
                "precision": report[cls_name]["precision"],
                "recall": report[cls_name]["recall"],
                "f1_score": report[cls_name]["f1-score"],
                "support": int(report[cls_name]["support"]),
            }
            for cls_name in model_classes
        },
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": top_confused,
    }

    # Save to metrics.json
    save_json(metrics_summary, metrics_json_path)
    logger.info(f"Saved test metrics to: {metrics_json_path}")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Room Classifier on test set.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    evaluate_model(args.config)
