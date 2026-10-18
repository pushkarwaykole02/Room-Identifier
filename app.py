"""
Gradio Web Application for AML-Based Room Identifier.
Provides interactive image upload and live webcam capture, real-time prediction,
confidence distributions across all room categories, and latency benchmarks.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure UTF-8 stdout/stderr on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gradio as gr
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import RoomPredictor
from src.utils import load_config, setup_logger

logger = setup_logger("app")

# Initialize global predictor
config_path = "config.yaml"
if os.path.exists(config_path):
    config = load_config(config_path)
    checkpoint_path = config["model"].get("checkpoint_path", "models/best_model.pt")
    device_pref = config["project"].get("device", "auto")
    image_size = config["dataset"].get("image_size", 224)
else:
    checkpoint_path = "models/best_model.pt"
    device_pref = "auto"
    image_size = 224

# Predictor instance (lazy initialized or loaded on launch)
predictor: Optional[RoomPredictor] = None
predictor_mtime: float = 0.0

def get_predictor() -> Optional[RoomPredictor]:
    global predictor, predictor_mtime
    if os.path.exists(checkpoint_path):
        current_mtime = os.path.getmtime(checkpoint_path)
        if predictor is None or current_mtime > predictor_mtime:
            logger.info("Loading / updating RoomPredictor from checkpoint...")
            predictor = RoomPredictor(
                checkpoint_path=checkpoint_path,
                device_name=device_pref,
                image_size=image_size,
            )
            predictor_mtime = current_mtime
            logger.info("RoomPredictor ready.")
    return predictor


def predict_room_ui(
    input_image: Optional[Image.Image]
) -> Tuple[Dict[str, float], str, str]:
    """
    Gradio inference callback.
    Args:
        input_image: PIL Image from file upload or webcam.
    Returns:
        (confidence_dict, markdown_status, latency_text)
    """
    if input_image is None:
        return {}, "### ⚠️ Please upload an image or capture a webcam photo.", "Latency: 0 ms"

    pred_inst = get_predictor()
    if pred_inst is None:
        return {}, "### ❌ Model checkpoint not found! Please run `python src/train.py` first.", "Latency: N/A"

    try:
        res = pred_inst.predict(input_image, top_k=8)
        top_cls = res["predicted_class"].replace("_", " ").title()
        conf_pct = res["confidence_percentage"]
        time_ms = res["inference_time_ms"]
        all_probs = res["all_probabilities"]

        # Format label dictionary for Gradio gr.Label
        formatted_probs = {
            k.replace("_", " ").title(): v
            for k, v in all_probs.items()
        }

        # Room badge color styling
        room_icons = {
            "Bedroom": "🛏️",
            "Kitchen": "🍳",
            "Living Room": "🛋️",
            "Bathroom": "🚿",
            "Hallway": "🚪",
            "Dining Room": "🍽️",
            "Children Room": "🧸",
            "Closet": "👔",
            "Office": "💻",
        }
        icon = room_icons.get(top_cls, "🏠")

        status_md = f"""
<div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; padding: 18px 24px; color: #f8fafc; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
  <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; margin-bottom: 6px;">Predicted Indoor Scene</div>
  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
    <div style="font-size: 2rem; font-weight: 800; color: #38bdf8; display: flex; align-items: center; gap: 10px;">
      <span>{icon}</span> <span>{top_cls}</span>
    </div>
    <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; font-size: 1.25rem; font-weight: 700; padding: 4px 16px; border-radius: 9999px;">
      {conf_pct} Confidence
    </div>
  </div>
</div>
"""
        latency_str = f"⚡ Inference Latency: **{time_ms:.2f} ms** | Device: **{pred_inst.device.type.upper()}**"
        return formatted_probs, status_md, latency_str

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {}, f"### ❌ Error during inference: {str(e)}", "Latency: Error"


CUSTOM_CSS = """
#title-box {
    text-align: center;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    padding: 24px;
    border-radius: 14px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}
#title-box h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#title-box p {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin: 0;
}
"""


def create_demo() -> gr.Blocks:
    """
    Build the Gradio interface with modern layout, webcam support, and confidence charts.
    """
    with gr.Blocks(title="AML Room Identifier") as demo:
        with gr.Column(elem_id="title-box"):
            gr.Markdown("""
            # 🏠 AML-Based Room Identifier
            **Real-Time Indoor Scene Classification using CNN & Transfer Learning (MobileNetV2)**
            """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Input Camera / Photo")
                image_input = gr.Image(
                    type="pil",
                    label="Upload Room Photo or Use Webcam",
                    sources=["upload", "webcam", "clipboard"],
                    height=320,
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    submit_btn = gr.Button("🔍 Identify Room", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Classification Output")
                result_banner = gr.Markdown(
                    """
<div style="background: #1e293b; border-radius: 12px; padding: 20px; color: #94a3b8; text-align: center; border: 1px dashed #475569;">
  📸 Upload an indoor photo or capture with webcam to see the predicted room category.
</div>
                    """
                )

                confidence_chart = gr.Label(
                    num_top_classes=8,
                    label="Confidence Distribution Across Room Categories",
                )

                latency_info = gr.Markdown("⚡ Latency: -- ms")

        # Interaction wiring
        submit_btn.click(
            fn=predict_room_ui,
            inputs=[image_input],
            outputs=[confidence_chart, result_banner, latency_info],
        )
        image_input.change(
            fn=predict_room_ui,
            inputs=[image_input],
            outputs=[confidence_chart, result_banner, latency_info],
        )
        clear_btn.click(
            fn=lambda: (None, {}, """
<div style="background: #1e293b; border-radius: 12px; padding: 20px; color: #94a3b8; text-align: center; border: 1px dashed #475569;">
  📸 Upload an indoor photo or capture with webcam to see the predicted room category.
</div>
""", "⚡ Latency: -- ms"),
            inputs=[],
            outputs=[image_input, confidence_chart, result_banner, latency_info],
        )

        gr.Markdown("""
        ---
        <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
          AML-Based Room Identifier • Deep Learning Computer Vision Mini-Project • PyTorch & Gradio
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    # Launch on localhost with fallback if port is occupied
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            theme=gr.themes.Soft(),
            css=CUSTOM_CSS,
        )
    except OSError:
        logger.warning(f"Port {port} is occupied. Attempting to launch on an available port...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=None,
            share=False,
            theme=gr.themes.Soft(),
            css=CUSTOM_CSS,
        )
