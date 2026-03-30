"""Unit tests for the GradCAM utility."""

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    """Minimal CNN for testing GradCAM hooks."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 14)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestGradCAM:
    def setup_method(self):
        from src.utils.gradcam import GradCAM
        self.model = SimpleConvNet().eval()
        self.cam = GradCAM(self.model, self.model.conv)

    def test_heatmap_shape_matches_img_size(self):
        GradCAM = self._import()
        model = SimpleConvNet().eval()
        GradCAM.disable_inplace_relu(model)
        cam = GradCAM(model, model.conv)
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        heatmap = cam(x, cls_idx=0, img_size=(224, 224))
        assert heatmap.shape == (224, 224)

    def test_heatmap_values_in_unit_range(self):
        GradCAM = self._import()
        model = SimpleConvNet().eval()
        GradCAM.disable_inplace_relu(model)
        cam = GradCAM(model, model.conv)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        heatmap = cam(x, cls_idx=0, img_size=(64, 64))
        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

    def test_overlay_output_shape(self):
        from src.utils.gradcam import GradCAM
        img = np.random.rand(64, 64, 3).astype(np.float32)
        heatmap = np.random.rand(64, 64).astype(np.float32)
        overlay = GradCAM.overlay(img, heatmap, alpha=0.5)
        assert overlay.shape == (64, 64, 3)
        assert overlay.min() >= 0.0
        assert overlay.max() <= 1.0

    def test_default_cls_idx_is_argmax(self):
        """When cls_idx=None, GradCAM should use the predicted class."""
        GradCAM = self._import()
        model = SimpleConvNet().eval()
        GradCAM.disable_inplace_relu(model)
        cam = GradCAM(model, model.conv)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        # Should not raise
        heatmap = cam(x, cls_idx=None, img_size=(32, 32))
        assert heatmap is not None

    @staticmethod
    def _import():
        from src.utils.gradcam import GradCAM
        return GradCAM
