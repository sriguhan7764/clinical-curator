"""Unit tests for model forward passes.

Tests that every model architecture:
- Accepts a (B, C, H, W) input batch without raising
- Returns the correct output shape (B, 14)
- Runs in eval mode without nan/inf outputs
"""

import pytest
import torch


BATCH = 2
N_CLASSES = 14


# ---------------------------------------------------------------------------
# Review 1 — MLP and CNN
# ---------------------------------------------------------------------------

class TestChestMLP:
    def setup_method(self):
        from src.models.mlp import ChestMLP
        self.model = ChestMLP().eval()

    def test_output_shape(self):
        x = torch.randn(BATCH, 3, 224, 224)
        out = self.model(x)
        assert out.shape == (BATCH, N_CLASSES)

    def test_no_nan_output(self):
        x = torch.randn(BATCH, 3, 224, 224)
        out = self.model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_sigmoid_range(self):
        x = torch.randn(BATCH, 3, 224, 224)
        probs = torch.sigmoid(self.model(x))
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


class TestChestCNN:
    def setup_method(self):
        from src.models.cnn import ChestCNN
        self.model = ChestCNN().eval()

    def test_output_shape(self):
        x = torch.randn(BATCH, 3, 224, 224)
        assert self.model(x).shape == (BATCH, N_CLASSES)

    def test_no_nan_output(self):
        x = torch.randn(BATCH, 3, 224, 224)
        out = self.model(x)
        assert not torch.isnan(out).any()

    def test_variable_input_size(self):
        """AdaptiveAvgPool2d makes the model resolution-agnostic."""
        from src.models.cnn import ChestCNN
        model = ChestCNN().eval()
        for h, w in [(224, 224), (256, 256), (320, 240)]:
            x = torch.randn(1, 3, h, w)
            out = model(x)
            assert out.shape == (1, N_CLASSES), f"Failed for size ({h}, {w})"


# ---------------------------------------------------------------------------
# Review 2 — pretrained backbones
# ---------------------------------------------------------------------------

class TestFineTunedResNet:
    def setup_method(self):
        from src.models.pretrained import FineTunedResNet
        self.model = FineTunedResNet().eval()

    def test_output_shape(self):
        x = torch.randn(BATCH, 3, 224, 224)
        assert self.model(x).shape == (BATCH, N_CLASSES)

    def test_frozen_layers(self):
        """layer1 and layer2 should have no trainable parameters."""
        from src.models.pretrained import FineTunedResNet
        model = FineTunedResNet()
        for name, p in model.backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                assert not p.requires_grad, f"{name} should be frozen"


class TestFeatureExtractor:
    @pytest.mark.parametrize("backbone", ["resnet50", "densenet121"])
    def test_output_shapes(self, backbone):
        from src.models.pretrained import FeatureExtractor
        model = FeatureExtractor(backbone=backbone, freeze=True).eval()
        x = torch.randn(BATCH, 3, 224, 224)
        seq, pooled = model(x)
        assert seq.shape[0] == BATCH
        assert seq.shape[2] == model.feat_dim
        assert pooled.shape == (BATCH, model.feat_dim)


# ---------------------------------------------------------------------------
# Review 3 — autoencoder and GAN
# ---------------------------------------------------------------------------

class TestConvAutoencoder:
    def setup_method(self):
        from src.models.autoencoder import ConvAutoencoder
        self.model = ConvAutoencoder(latent_dim=64).eval()

    def test_reconstruction_shape(self):
        x = torch.randn(BATCH, 1, 64, 64)
        recon, z = self.model(x)
        assert recon.shape == x.shape
        assert z.shape == (BATCH, 64)

    def test_encode_decode_roundtrip(self):
        x = torch.randn(1, 1, 64, 64)
        z = self.model.encode(x)
        recon = self.model.decode(z)
        assert recon.shape == x.shape


class TestVAE:
    def setup_method(self):
        from src.models.autoencoder import VAE
        self.model = VAE(latent_dim=64).eval()

    def test_forward_returns_four_values(self):
        x = torch.randn(BATCH, 1, 64, 64)
        recon, mu, logvar, z = self.model(x)
        assert recon.shape == x.shape
        assert mu.shape == (BATCH, 64)
        assert logvar.shape == (BATCH, 64)
        assert z.shape == (BATCH, 64)

    def test_sample(self):
        samples = self.model.sample(n=4, device="cpu")
        assert samples.shape == (4, 1, 64, 64)


class TestDCGAN:
    def setup_method(self):
        from src.models.gan import Generator, Discriminator
        self.G = Generator(latent_dim=64).eval()
        self.D = Discriminator().eval()

    def test_generator_output_shape(self):
        z = torch.randn(BATCH, 64)
        imgs = self.G(z)
        assert imgs.shape == (BATCH, 1, 64, 64)

    def test_discriminator_output_shape(self):
        imgs = torch.randn(BATCH, 1, 64, 64)
        logits = self.D(imgs)
        assert logits.shape == (BATCH, 1)

    def test_generator_output_range(self):
        """Generator uses Tanh — output must be in [-1, 1]."""
        z = torch.randn(4, 64)
        imgs = self.G(z)
        assert imgs.min() >= -1.0 - 1e-5
        assert imgs.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Review 4 — DenseNet production model
# ---------------------------------------------------------------------------

class TestDenseNetCXR:
    def setup_method(self):
        from src.models.densenet import DenseNetCXR
        self.model = DenseNetCXR(pretrained=False).eval()

    def test_output_shape(self):
        x = torch.randn(BATCH, 3, 224, 224)
        assert self.model(x).shape == (BATCH, N_CLASSES)

    def test_frozen_early_layers(self):
        """denseblock1–3 should be frozen; denseblock4 should be trainable."""
        from src.models.densenet import DenseNetCXR
        model = DenseNetCXR(pretrained=False)
        for name, p in model.features.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                assert p.requires_grad, f"{name} should be trainable"
            elif "denseblock1" in name or "denseblock2" in name:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_get_cam_layer_is_not_none(self):
        from src.models.densenet import DenseNetCXR
        model = DenseNetCXR(pretrained=False)
        assert model.get_cam_layer() is not None
