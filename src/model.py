from __future__ import annotations
import torch
from torch import nn

class PatchTSTLite(nn.Module):
    def __init__(self, n_features: int, seq_len: int = 96, patch_len: int = 12, stride: int = 6,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3, ff_dim: int = 128,
                 dropout: float = 0.2, n_classes: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + max(0, (seq_len - patch_len) // stride)
        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            p = x[:, i:i + self.patch_len, :].reshape(x.size(0), -1)
            patches.append(p)
        x = torch.stack(patches, dim=1)
        x = self.patch_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)
