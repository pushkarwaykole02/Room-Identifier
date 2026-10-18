"""
Model Architecture Module for AML-Based Room Identifier.
Implements transfer learning CNN with ImageNet-pretrained MobileNetV2 (or ResNet18)
with custom Global Average Pooling, Dropout, and Dense classification head.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, ResNet18_Weights

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger, resolve_path

logger = setup_logger("model")


class RoomClassifier(nn.Module):
    """
    Room Classification Neural Network using ImageNet-pretrained backbone
    and a custom classification head (Global Avg Pool -> Dropout -> Linear).
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "mobilenet_v2",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        """
        Args:
            num_classes: Number of room categories to predict.
            backbone_name: CNN architecture ('mobilenet_v2' or 'resnet18').
            pretrained: Whether to load ImageNet pre-trained weights.
            dropout_rate: Dropout probability in the classification head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name.lower()
        self.dropout_rate = dropout_rate

        if self.backbone_name == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            # Feature extractor
            self.features = base_model.features
            # Global Average Pooling
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # Classifier Head
            in_features = base_model.classifier[1].in_features  # 1280
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        elif self.backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            # Remove original fc and avgpool
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = base_model.fc.in_features  # 512
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Use 'mobilenet_v2' or 'resnet18'.")

        logger.info(
            f"Initialized RoomClassifier: Backbone={self.backbone_name}, "
            f"Pretrained={pretrained}, Classes={num_classes}, Dropout={dropout_rate}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor and classification head.
        Returns raw class logits.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone layers (Phase 1). Only classification head parameters are trainable.
        """
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        logger.info("Backbone frozen: Only classifier head is trainable (Phase 1).")

    def unfreeze_backbone(self, unfreeze_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone layers for fine-tuning (Phase 2).
        Args:
            unfreeze_layers: If specified, only unfreeze the last N blocks of features.
                             If None, unfreeze all backbone layers.
        """
        if unfreeze_layers is None or unfreeze_layers <= 0:
            for param in self.features.parameters():
                param.requires_grad = True
            logger.info("Full backbone unfrozen for fine-tuning.")
        else:
            # Unfreeze only the last N blocks
            if self.backbone_name == "mobilenet_v2":
                total_blocks = len(self.features)
                unfreeze_start = max(0, total_blocks - unfreeze_layers)
                for idx, block in enumerate(self.features):
                    req_grad = idx >= unfreeze_start
                    for param in block.parameters():
                        param.requires_grad = req_grad
                logger.info(f"Unfrozen last {unfreeze_layers} blocks of MobileNetV2 features (blocks {unfreeze_start} to {total_blocks-1}).")
            else:
                for param in self.features.parameters():
                    param.requires_grad = True
                logger.info("Unfrozen ResNet features for fine-tuning.")

        for param in self.classifier.parameters():
            param.requires_grad = True

    def count_trainable_parameters(self) -> Dict[str, int]:
        """
        Count total and trainable parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_model(
    num_classes: int = 5,
    config: Optional[Dict[str, Any]] = None,
) -> RoomClassifier:
    """
    Factory function to build RoomClassifier according to configuration.
    """
    if config is None:
        config = {}
    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "mobilenet_v2")
    pretrained = model_cfg.get("pretrained", True)
    dropout = model_cfg.get("dropout_rate", 0.3)

    return RoomClassifier(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=pretrained,
        dropout_rate=dropout,
    )


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[RoomClassifier, Dict[str, Any]]:
    """
    Load a saved model checkpoint and reconstruct the RoomClassifier.
    Returns (model, metadata_dict).
    """
    chk_file = resolve_path(checkpoint_path)
    if not chk_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(str(chk_file), map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names)
    backbone = checkpoint.get("backbone", "mobilenet_v2")
    dropout = checkpoint.get("dropout_rate", 0.3)

    model = RoomClassifier(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
        dropout_rate=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Successfully loaded checkpoint from {checkpoint_path} (Classes: {class_names})")
    return model, checkpoint


if __name__ == "__main__":
    net = build_model(num_classes=5)
    print("Initial parameter counts:", net.count_trainable_parameters())
    net.freeze_backbone()
    print("After freeze_backbone:", net.count_trainable_parameters())
    net.unfreeze_backbone(unfreeze_layers=4)
    print("After unfreeze_backbone(4):", net.count_trainable_parameters())
    sample_input = torch.randn(2, 3, 224, 224)
    out = net(sample_input)
    print("Output shape:", out.shape)
