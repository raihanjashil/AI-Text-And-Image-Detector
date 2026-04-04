"""
image_detector.py
-----------------
Core logic for AI vs Real image classification.
Loads pre-trained ResNet18 fine-tuned on CIFAKE dataset.

Architecture:
    - ResNet18 pretrained on ImageNet
    - Final FC layer replaced: fc(512 → 2)
    - Class 0 = FAKE (AI-generated), Class 1 = REAL

All ML logic lives here — app.py just calls predict().
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "resnet18_cifake.pth")

# ── ImageNet normalisation (required — model was pretrained on ImageNet) ──────
_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std= [0.229, 0.224, 0.225],   # ImageNet std
    ),
])

# ── Model cache ───────────────────────────────────────────────────────────────
_model = None


def _load_model():
    """Lazy-load model once and cache in module global."""
    global _model

    if _model is not None:
        return _model

    # Rebuild the exact architecture used during training
    model = models.resnet18(weights=None)           # no pretrained weights
    model.fc = nn.Linear(512, 2)                    # 2-class output head

    state_dict = torch.load(_MODEL_PATH, map_location=DEVICE)

    # Handle checkpoint dicts that wrap the state_dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image: Image.Image) -> dict:
    """
    Run inference on a PIL Image.

    Parameters
    ----------
    image : PIL.Image.Image
        RGB image (any size — will be resized to 224x224 internally).

    Returns
    -------
    dict with keys:
        label     : "AI" or "Real"
        ai_prob   : float (0–1)
        real_prob : float (0–1)
    """
    model = _load_model()

    # Ensure RGB (handles RGBA PNGs etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = _TRANSFORM(image).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                            # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]          # (2,)

    # Class mapping: 0 = FAKE (AI), 1 = REAL
    ai_prob   = float(probs[0])
    real_prob = float(probs[1])
    label     = "Real" if real_prob > ai_prob else "AI"

    return {
        "label":     label,
        "ai_prob":   ai_prob,
        "real_prob": real_prob,
    }


# ── Grad-CAM explainability ───────────────────────────────────────────────────

def get_gradcam(image: Image.Image, pred_class: int) -> Image.Image:
    """
    Generate a Grad-CAM heatmap overlay on the original image.

    Parameters
    ----------
    image      : PIL.Image.Image  — original RGB image (any size)
    pred_class : int              — 0 = FAKE (AI), 1 = REAL

    Returns
    -------
    PIL.Image.Image — original image blended with the CAM heatmap (RGB, 224×224)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        return None

    import numpy as np

    model = _load_model()

    # layer2 gives the best spatial resolution (4×4) for 28×28 inputs
    target_layers = [model.layer2[-1]]

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = _TRANSFORM(image).unsqueeze(0).to(DEVICE)

    targets = [ClassifierOutputTarget(pred_class)]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # (H, W)

    # Build display image at 224×224 for a clear overlay
    DISPLAY = 224
    img_display = np.array(image.resize((DISPLAY, DISPLAY))) / 255.0

    # Upscale CAM to display size
    from PIL import Image as _PIL
    cam_up = np.array(
        _PIL.fromarray((grayscale_cam * 255).astype("uint8")).resize(
            (DISPLAY, DISPLAY), resample=_PIL.BILINEAR
        )
    ) / 255.0

    overlay = show_cam_on_image(img_display.astype("float32"), cam_up, use_rgb=True)
    return Image.fromarray(overlay)


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python image_detector.py path/to/image.jpg")
        raise SystemExit(1)

    img    = Image.open(path).convert("RGB")
    result = predict(img)

    print(f"\nImage   : {path}")
    print(f"Label   : {result['label']}")
    print(f"AI prob : {result['ai_prob']:.1%}")
    print(f"Real %  : {result['real_prob']:.1%}")