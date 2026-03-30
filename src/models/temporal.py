"""CNN-RNN hybrid with attention mechanisms for temporal X-ray sequences (Review 2)."""

import numpy as np
import torch
import torch.nn as nn

from .pretrained import FeatureExtractor

NUM_CLASSES = 14
HIDDEN_DIM = 256
EMBED_DIM = 128
NUM_RNN_LAYERS = 2


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2).float() * (-np.log(10000) / d)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau-style) attention over RNN hidden states."""

    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.V = nn.Linear(d, 1, bias=False)

    def forward(self, h):
        w = torch.softmax(self.V(torch.tanh(self.W(h))).squeeze(-1), dim=-1)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1), w


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over RNN hidden states."""

    def __init__(self, d: int):
        super().__init__()
        self.Q = nn.Linear(d, d)
        self.K = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.scale = d ** 0.5

    def forward(self, h):
        w = torch.softmax(
            torch.bmm(self.Q(h), self.K(h).transpose(1, 2)) / self.scale, dim=-1
        )
        return torch.bmm(w, self.V(h)).mean(dim=1), w


class MultiHeadAttention(nn.Module):
    """Multi-head attention with residual connection over RNN hidden states."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        out, w = self.mha(h, h, h)
        return self.norm(out + h).mean(dim=1), w


class CNN_RNN_Hybrid(nn.Module):
    """CNN feature extractor + RNN temporal encoder + attention classifier.

    Supports backbones: resnet50, densenet121, efficientnet_b0.
    Supports RNN types: RNN, LSTM, GRU.
    Supports attention: bahdanau, self, multihead.
    Output: raw logits of shape (batch, 14).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        rnn_type: str = "LSTM",
        attn: str = "bahdanau",
        hidden: int = HIDDEN_DIM,
        embed: int = EMBED_DIM,
        layers: int = NUM_RNN_LAYERS,
        bidir: bool = True,
        freeze: bool = True,
        dropout: float = 0.3,
        n_cls: int = NUM_CLASSES,
    ):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.attn_type = attn
        self.bidir = bidir
        self.hidden = hidden

        self.cnn = FeatureExtractor(backbone, freeze=freeze)
        feat = self.cnn.feat_dim

        self.emb = nn.Sequential(
            nn.Linear(feat, embed),
            nn.LayerNorm(embed),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pe = PositionalEncoding(embed)

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(
            embed,
            hidden,
            layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout if layers > 1 else 0,
        )

        out_dim = hidden * (2 if bidir else 1)
        attn_map = {
            "bahdanau": BahdanauAttention,
            "self": SelfAttention,
            "multihead": MultiHeadAttention,
        }
        self.attention = attn_map[attn](out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        # x: (B, T, C, H, W) for temporal input, or (B, C, H, W) for single frame
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            seq_feat, _ = self.cnn(x_flat)  # (B*T, 49, feat)
            # pool spatial tokens
            seq_feat = seq_feat.mean(dim=1)  # (B*T, feat)
            seq_feat = seq_feat.view(B, T, -1)  # (B, T, feat)
        else:
            seq, _ = self.cnn(x)  # (B, 49, feat)
            seq_feat = seq

        seq = self.pe(self.emb(seq_feat))
        out, _ = self.rnn(seq)
        context, _ = self.attention(out)
        return self.classifier(context)
