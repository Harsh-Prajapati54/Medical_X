"""
predict.py — Run inference on a chest X-ray using the trained DenseNet121 model.
             Supports multilabel classification across 14 diseases (NIH ChestX-ray14).
             Optionally generates a Grad-CAM heatmap for explainability.

Usage:
    python predict.py --image path/to/xray.png
    python predict.py --image path/to/xray.png --model model_weights.pth
    python predict.py --image path/to/xray.png --gradcam
    python predict.py --image path/to/xray.png --threshold 0.4 --gradcam
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES        = len(ALL_LABELS)   # 14
IMAGE_SIZE         = 320              # 320x320 input size used during training
DEFAULT_MODEL_PATH = "best_model.pth"
DEFAULT_THRESHOLD  = 0.5
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> nn.Module:
    """Load DenseNet121 with custom classifier head and saved weights."""
    model = torchvision.models.densenet121(weights=None)

    # Recreate the same classifier used during training
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1024, out_features=NUM_CLASSES, bias=True)
    )

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Handle both raw state_dict and full checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    print(f"[✓] Model loaded from '{model_path}' | device: {DEVICE}")
    return model


# ── Transform ──────────────────────────────────────────────────────────────────
def get_transform() -> transforms.Compose:
    """Standard ImageNet preprocessing used during training."""
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ── Predict ────────────────────────────────────────────────────────────────────
def predict(image_path: str, model: nn.Module, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Run multilabel inference on a single image.
    Returns predicted diseases and per-class probabilities.
    """
    transform = get_transform()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 320, 320)

    with torch.inference_mode():
        logits = model(input_tensor)             # (1, 14)
        probs  = torch.sigmoid(logits).squeeze() # (14,) — multilabel uses sigmoid, NOT softmax

    probs_np = probs.cpu().numpy()
    predicted_labels = [ALL_LABELS[i] for i, p in enumerate(probs_np) if p >= threshold]

    if not predicted_labels:
        predicted_labels = ["No Finding"]

    return {
        "image_path":    image_path,
        "predictions":   predicted_labels,
        "probabilities": {ALL_LABELS[i]: float(probs_np[i]) for i in range(NUM_CLASSES)},
        "threshold":     threshold,
        "input_tensor":  transform(image).unsqueeze(0).to(DEVICE),
        "pil_image":     image,
    }


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM for DenseNet121. Target layer: model.features.norm5"""

    def __init__(self, model: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer = model.features.denseblock4.denselayer16.conv2
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.clone()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad(set_to_none=True)
        output[0, class_idx].backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam     = torch.sum(weights * self.activations, dim=1).squeeze()
        cam     = F.relu(cam)
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.detach().cpu().numpy()


def overlay_gradcam(image_path: str, heatmap: np.ndarray, save_path: str = "gradcam_result.jpg"):
    """Overlay Grad-CAM heatmap on original X-ray and save."""
    img     = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result  = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, result)
    print(f"[✓] Grad-CAM saved to '{save_path}'")
    return result


# ── Visualize ──────────────────────────────────────────────────────────────────
def show_result(result: dict, gradcam_img: np.ndarray = None):
    """Display X-ray with top predictions and optional Grad-CAM overlay."""
    top_preds = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)[:5]
    colors    = ["red" if k in result["predictions"] else "gray" for k, _ in top_preds]

    n_cols = 2 if gradcam_img is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(result["pil_image"], cmap="gray")
    axes[0].set_title("Input X-Ray", fontsize=12)
    axes[0].axis("off")

    if gradcam_img is not None:
        axes[1].imshow(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis("off")

    pred_text = "Detected: " + (", ".join(result["predictions"]))
    fig.suptitle(pred_text, fontsize=13,
                 color="red" if result["predictions"] != ["No Finding"] else "green", y=1.02)

    # Bar chart of top-5 probabilities
    fig2, ax = plt.subplots(figsize=(7, 3))
    labels = [f"{k} ({v*100:.1f}%)" for k, v in top_preds]
    ax.barh(labels[::-1], [v for _, v in top_preds[::-1]], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.axvline(result["threshold"], color="black", linestyle="--", linewidth=1,
               label=f"Threshold ({result['threshold']})")
    ax.set_title("Top-5 Disease Probabilities", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 — DenseNet121 Inference")
    parser.add_argument("--image",     type=str,   required=True,
                        help="Path to chest X-ray image (.png / .jpg)")
    parser.add_argument("--model",     type=str,   default=DEFAULT_MODEL_PATH,
                        help=f"Path to model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Sigmoid threshold for positive label (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--gradcam",   action="store_true",
                        help="Generate Grad-CAM heatmap for the top predicted disease")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip displaying plots")
    args = parser.parse_args()

    model  = load_model(args.model)
    result = predict(args.image, model, threshold=args.threshold)

    print("\n── Result ──────────────────────────────────────")
    print(f"  Image      : {result['image_path']}")
    print(f"  Threshold  : {result['threshold']}")
    print(f"  Detected   : {', '.join(result['predictions'])}")
    print("\n  All probabilities:")
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        marker = " ◄" if label in result["predictions"] else ""
        print(f"    {label:<22}: {prob*100:5.2f}%{marker}")
    print("─────────────────────────────────────────────────\n")

    gradcam_img = None
    if args.gradcam:
        top_label = max(result["probabilities"], key=result["probabilities"].get)
        top_idx   = ALL_LABELS.index(top_label)
        print(f"[→] Generating Grad-CAM for: {top_label} (class {top_idx})")
        gcam        = GradCAM(model)
        heatmap     = gcam.generate(result["input_tensor"], class_idx=top_idx)
        gradcam_img = overlay_gradcam(args.image, heatmap)

    if not args.no_plot:
        show_result(result, gradcam_img)


if __name__ == "__main__":
    main()
