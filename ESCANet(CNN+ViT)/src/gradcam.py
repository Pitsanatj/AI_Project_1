"""
src/gradcam.py
━━━━━━━━━━━━━━
Grad-CAM visualisation — works with all models in this project.

Target layers per model
────────────────────────
  ecsamet       → model.features[6]   (last EfficientNetV2-S block before CBAM)
  hybrid_ecsanet→ model.cnn_branch.features[-1]
  lictnet       → model.lstl.saa
  swin_t        → model.model.layers[-1]
  deit_s        → model.model.blocks[-1]
  mobilevit_s   → model.model.stages[-1]
  fastvit_sa12  → model.model.stages[-1]
  mobilenet_v3  → model.model.features[-1]
  shufflenet_v2 → model.model.conv5

Usage
-----
  from src.gradcam import GradCAM, get_default_target_layer, save_gradcam_grid

  target_layer = get_default_target_layer(model, model_name)
  gcam = GradCAM(model, target_layer)
  cam  = gcam(input_tensor)            # (1, H, W) float in [0,1]
  gcam.remove_hooks()

  save_gradcam_grid(
      images, labels, preds, cams,
      class_labels, save_path,
      n=16,
  )
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Standard Grad-CAM (Selvaraju et al., 2017).

    Parameters
    ----------
    model        : nn.Module  — full model (eval mode recommended)
    target_layer : nn.Module  — layer whose activation maps to visualise
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output could be tuple (some timm layers)
            self._activations = output[0] if isinstance(output, tuple) else output

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0] if isinstance(grad_out, tuple) else grad_out

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,   # (1, C, H, W) normalised float32
        class_idx:    int | None = None,
    ) -> np.ndarray:
        """
        Returns Grad-CAM heatmap (H, W) float32 in [0, 1].
        class_idx=None → use predicted class.
        """
        self.model.zero_grad()
        self._activations = None
        self._gradients   = None

        # Forward
        output = self.model(input_tensor)   # (1, num_classes)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        # Backward for target class
        score = output[0, class_idx]
        score.backward()

        # Pool gradients over spatial dims
        acts = self._activations.detach()   # (1, C, H, W)  or (1, T, C) for ViT
        grads = self._gradients.detach()

        # Handle ViT / Swin token outputs (B, T, C) → treat as (B, C, T, 1)
        if acts.dim() == 3:
            acts  = acts.permute(0, 2, 1).unsqueeze(-1)
            grads = grads.permute(0, 2, 1).unsqueeze(-1)

        weights = grads.mean(dim=(2, 3), keepdim=True)       # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)       # (1, 1, H', W')
        cam = F.relu(cam)

        # Resize to input spatial size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()                     # (H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


# ── Default target layers per model ──────────────────────────────────────────
def get_default_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the recommended Grad-CAM target layer for each architecture."""
    key = model_name.lower()

    try:
        if key == "efficientnet_gru":
            # ดึงจากชั้น features ตัวสุดท้ายของ EfficientNetV2-S
            return model.features[-1]
        
        if key == "ecsamet":
            # Last EfficientNetV2-S block BEFORE CBAM — rich spatial features
            return model.features[6]

        elif key == "hybrid_ecsanet":
            # Last block of the CNN branch
            return model.cnn_branch.features[-1]

        elif key == "lictnet":
            # GSAB inside SAA — global spatial attention output
            return model.lstl.saa.gsab

        elif key == "swin_t":
            # Last Swin Transformer stage
            return model.model.layers[-1]

        elif key == "deit_s":
            # Last transformer block
            return model.model.blocks[-1]

        elif key in ("mobilevit_s", "fastvit_sa12"):
            # Last stage of timm model
            return model.model.stages[-1]

        elif key == "mobilenet_v3":
            return model.model.features[-1]

        elif key == "shufflenet_v2":
            return model.model.conv5

        else:
            raise ValueError(f"No default target layer for '{model_name}'.")

    except (AttributeError, IndexError) as e:
        raise AttributeError(
            f"Cannot find default target layer for '{model_name}': {e}\n"
            f"Pass target_layer manually to GradCAM(model, target_layer)."
        ) from e


# ── Denormalise helper ────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation.
    tensor: (C, H, W) float32  →  np.ndarray (H, W, 3) uint8
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)      # (H, W, C)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ── Overlay helper ────────────────────────────────────────────────────────────
def overlay_cam(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto the original image.
    image_np : (H, W, 3) uint8
    cam      : (H, W)    float32 [0,1]
    Returns  : (H, W, 3) uint8
    """
    cmap   = plt.get_cmap("jet")
    heatmap = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3) uint8
    blended = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return blended


# ── Batch GradCAM generation ──────────────────────────────────────────────────
def generate_gradcam_batch(
    model:        nn.Module,
    model_name:   str,
    loader,                         # DataLoader (test)
    device:       torch.device,
    n_images:     int = 16,
    class_labels: list[str] | None = None,
) -> tuple[list, list, list, list]:
    """
    Run Grad-CAM on the first n_images from loader.
    Returns (images_np, true_labels, pred_labels, cams)
    """
    target_layer = get_default_target_layer(model, model_name)
    gcam         = GradCAM(model, target_layer)
    model.eval()

    images_np, true_labels, pred_labels, cams = [], [], [], []

    for inputs, labels in loader:
        for i in range(inputs.size(0)):
            if len(images_np) >= n_images:
                break

            inp = inputs[i:i+1].to(device)   # (1, C, H, W)
            cam = gcam(inp)                   # (H, W)

            images_np.append(denormalise(inputs[i]))
            true_labels.append(int(labels[i]))

            with torch.no_grad():
                pred = int(model(inp).argmax(dim=1).item())
            pred_labels.append(pred)
            cams.append(cam)

        if len(images_np) >= n_images:
            break

    gcam.remove_hooks()
    return images_np, true_labels, pred_labels, cams


# ── Save GradCAM grid ─────────────────────────────────────────────────────────
def save_gradcam_grid(
    images_np:    list,           # list of (H,W,3) uint8
    true_labels:  list[int],
    pred_labels:  list[int],
    cams:         list,           # list of (H,W) float32
    class_labels: list[str],
    save_path:    str,
    n_cols:       int = 4,
    alpha:        float = 0.45,
):
    """
    Save a grid of (original | heatmap overlay) pairs.
    Each cell = original image top, GradCAM overlay bottom.
    Title colour: green = correct, red = wrong.
    """
    n      = len(images_np)
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 6))
    gs  = gridspec.GridSpec(n_rows * 2, n_cols, figure=fig,
                            hspace=0.35, wspace=0.1)

    for idx in range(n):
        row = (idx // n_cols) * 2
        col = idx % n_cols

        img     = images_np[idx]
        cam     = cams[idx]
        overlay = overlay_cam(img, cam, alpha)
        true_l  = class_labels[true_labels[idx]]
        pred_l  = class_labels[pred_labels[idx]]
        correct = true_labels[idx] == pred_labels[idx]

        # Original image
        ax1 = fig.add_subplot(gs[row, col])
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title(f"True: {true_l}", fontsize=7,
                      color="green" if correct else "red", pad=2)

        # Grad-CAM overlay
        ax2 = fig.add_subplot(gs[row + 1, col])
        ax2.imshow(overlay)
        ax2.axis("off")
        ax2.set_title(f"Pred: {pred_l}", fontsize=7,
                      color="green" if correct else "red", pad=2)

    fig.suptitle("Grad-CAM Visualisation", fontsize=12, y=1.01)

    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Grad-CAM grid → {save_path}")