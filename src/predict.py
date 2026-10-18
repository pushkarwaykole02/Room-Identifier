"""
Inference Module and CLI for AML-Based Room Identifier.
Supports predicting room type from image path, PIL Image, or NumPy array (webcam).
CLI usage: python src/predict.py <path_to_image> [--config config.yaml] [--top_k 3]
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import torch
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transforms import get_inference_transforms
from src.model import load_checkpoint, RoomClassifier
from src.utils import setup_logger, load_config, get_device

logger = setup_logger("predict")


class RoomPredictor:
    """
    Predictor helper for loading model once and serving single image inference requests.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/best_model.pt",
        device_name: str = "auto",
        image_size: int = 224,
    ) -> None:
        self.device = get_device(device_name)
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.class_names: List[str] = self.checkpoint.get("class_names", [])
        self.image_size: int = self.checkpoint.get("image_size", image_size)
        self.transform = get_inference_transforms(image_size=self.image_size)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Run inference on an input image (filepath, PIL Image, or NumPy array).
        Returns:
            Dictionary with predicted class, confidence, all class probabilities, and top-k predictions.
        """
        # 1. Convert input to PIL Image RGB
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at: {image_input}")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 2. Transform and Batch
        start_time = time.perf_counter()
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 3. Model Forward Pass
        outputs = self.model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Format Results
        all_probs: Dict[str, float] = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }

        # Sorted ranking
        sorted_indices = np.argsort(probabilities)[::-1]
        top_idx = sorted_indices[0]
        top_class = self.class_names[top_idx]
        top_confidence = float(probabilities[top_idx])

        top_k_list = [
            {
                "rank": rank + 1,
                "class": self.class_names[idx],
                "confidence": float(probabilities[idx]),
                "percentage": f"{probabilities[idx] * 100:.2f}%",
            }
            for rank, idx in enumerate(sorted_indices[:top_k])
        ]

        return {
            "predicted_class": top_class,
            "confidence": top_confidence,
            "confidence_percentage": f"{top_confidence * 100:.2f}%",
            "all_probabilities": all_probs,
            "top_k": top_k_list,
            "inference_time_ms": round(inference_time_ms, 2),
        }


def predict_cli(
    image_path: str,
    config_path: str = "config.yaml",
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    CLI interface for running single image prediction.
    """
    config = load_config(config_path)
    checkpoint_path = config["model"].get("checkpoint_path", "models/best_model.pt")
    device_name = config["project"].get("device", "auto")
    image_size = config["dataset"].get("image_size", 224)

    predictor = RoomPredictor(
        checkpoint_path=checkpoint_path,
        device_name=device_name,
        image_size=image_size,
    )

    result = predictor.predict(image_path, top_k=top_k)

    print("\n" + "=" * 55)
    print("AML ROOM IDENTIFIER - PREDICTION RESULT")
    print("=" * 55)
    print(f"Image Path        : {image_path}")
    print(f"Predicted Room    : {result['predicted_class'].upper()}")
    print(f"Confidence Score  : {result['confidence_percentage']}")
    print(f"Inference Latency : {result['inference_time_ms']:.2f} ms")
    print("-" * 55)
    print(f"Top-{top_k} Candidates:")
    for item in result["top_k"]:
        bar = "#" * int(item["confidence"] * 25)
        print(f"  {item['rank']}. {item['class']:<14}: {item['percentage']:>7}  |{bar:<25}|")
    print("=" * 55 + "\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict room category from a photo.")
    parser.add_argument("image_path", type=str, help="Path to room image file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top classes to display")
    args = parser.parse_args()

    predict_cli(args.image_path, args.config, args.top_k)
