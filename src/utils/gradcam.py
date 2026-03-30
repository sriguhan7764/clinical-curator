"""Gradient-weighted Class Activation Mapping (GradCAM) for CNN interpretability.

GradCAM computes a per-class localisation heatmap by weighting the spatial
activations of a target convolutional layer by the globally pooled gradients
flowing back from the target class neuron.

Reference
---------
Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization." ICCV 2017. https://arxiv.org/abs/1610.02391

Usage
-----
    # Basic usage
    cam = GradCAM(model, layer=model.features.denseblock4)
    heatmap = cam(input_tensor, cls_idx=2)     # Effusion class

    # Full overlay pipeline
    GradCAM.disable_inplace_relu(model)        # required before backward
    heatmap  = cam(x, cls_idx=0)
    overlay  = GradCAM.overlay(img_np, heatmap, alpha=0.5)
    GradCAM.enable_inplace_relu(model)         # restore after

Notes
-----
- Target layer must be a convolutional layer (output has shape B×C×H×W).
- ``disable_inplace_relu`` must be called before computing gradients to avoid
  in-place operation errors with PyTorch's autograd engine.
- For MLP models without conv layers, GradCAM is not applicable.
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = (224, 224)


class GradCAM:
    """Gradient-weighted Class Activation Map generator.

    Registers forward and backward hooks on *layer* to capture activations
    and gradients without modifying the model's ``forward`` method.

    Parameters
    ----------
    model : nn.Module
        The model to explain. Must accept ``(1, C, H, W)`` input.
    layer : nn.Module
        Target convolutional layer. Hooks are registered on this layer.
        Typically the last ``nn.Conv2d`` or ``DenseBlock`` in the network.
    """

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Hook captures the layer's output tensor on every forward pass
        layer.register_forward_hook(
            lambda _m, _i, output: setattr(self, "activations", output.clone())
        )
        # Hook captures gradients flowing back through the layer on backward
        layer.register_full_backward_hook(
            lambda _m, _grad_in, grad_out: setattr(self, "gradients", grad_out[0])
        )

    def __call__(
        self,
        x: torch.Tensor,
        cls_idx: Optional[int] = None,
        img_size: tuple[int, int] = IMG_SIZE,
    ) -> np.ndarray:
        """Compute the GradCAM heatmap.

        Parameters
        ----------
        x : torch.Tensor
            Single image batch of shape ``(1, C, H, W)`` with
            ``requires_grad=True``.
        cls_idx : int, optional
            Class index to explain. Defaults to the predicted class (argmax).
        img_size : tuple[int, int]
            Output heatmap size ``(H, W)``. Defaults to ``(224, 224)``.

        Returns
        -------
        np.ndarray
            Normalised heatmap in ``[0, 1]`` of shape ``img_size``.
        """
        self.model.zero_grad()
        logits = self.model(x)

        if cls_idx is None:
            cls_idx = int(logits[0].argmax().item())

        # Back-propagate a one-hot signal for the target class
        target = torch.zeros_like(logits)
        target[0, cls_idx] = 1.0
        logits.backward(gradient=target, retain_graph=True)

        # Global-average-pool the gradients → channel weights
        assert self.gradients is not None, "Backward hook did not fire"
        assert self.activations is not None, "Forward hook did not fire"
        weights = self.gradients.mean(dim=[0, 2, 3])           # (C,)
        cam = (self.activations[0] * weights[:, None, None]).mean(0)  # (H, W)

        # ReLU + normalise → [0, 1]
        cam = np.maximum(cam.cpu().detach().numpy(), 0)
        cam = cv2.resize(cam, img_size[::-1])                  # cv2 uses (W, H)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def overlay(
        original_img: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend a GradCAM heatmap onto the original image.

        Parameters
        ----------
        original_img : np.ndarray
            Float RGB image in ``[0, 1]`` of shape ``(H, W, 3)``.
        cam : np.ndarray
            Normalised heatmap in ``[0, 1]`` of shape ``(H, W)``.
        alpha : float
            Blend weight for the heatmap (``0`` = original only,
            ``1`` = heatmap only).

        Returns
        -------
        np.ndarray
            Blended RGB image in ``[0, 1]`` of shape ``(H, W, 3)``.
        """
        heatmap_bgr = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB) / 255.0
        return np.clip(alpha * original_img + alpha * heatmap_rgb, 0.0, 1.0)

    @staticmethod
    def disable_inplace_relu(module: nn.Module) -> None:
        """Set all ``nn.ReLU`` layers in *module* to non-inplace mode.

        This is required before calling ``backward()`` through hooks, because
        inplace operations destroy intermediate values needed by autograd.
        """
        for child in module.modules():
            if isinstance(child, nn.ReLU):
                child.inplace = False

    @staticmethod
    def enable_inplace_relu(module: nn.Module) -> None:
        """Restore inplace mode for all ``nn.ReLU`` layers in *module*."""
        for child in module.modules():
            if isinstance(child, nn.ReLU):
                child.inplace = True
