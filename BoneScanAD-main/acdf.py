"""Anatomy-Conditioned Dynamic Cross-Layer Fusion (ACDF)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyConditionedLayerFusion(nn.Module):
    """Predict per-layer fusion weights conditioned on anatomy.

    Instead of summing the anomaly maps of several ViT blocks with fixed equal
    weights, the fusion weights are predicted dynamically from:

    1. the descriptor of the current image at each selected ViT layer, and
    2. the prototype context of the current anatomical region.

    A softmax over the layer axis keeps the weights normalised, so the fused
    maps stay on the same scale as the individual ones.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.desc_norm = nn.LayerNorm(embed_dim)
        self.desc_proj = nn.Linear(embed_dim, hidden_dim)
        self.region_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
        )
        self.layer_embedding = nn.Parameter(torch.randn(num_layers, hidden_dim) * 0.02)
        self.score = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        layer_descriptors: torch.Tensor,
        region_context: torch.Tensor,
    ) -> torch.Tensor:
        """Map descriptors ``[K, C]`` + region context ``[C]`` to weights ``[K]``."""
        if layer_descriptors.shape[0] != self.num_layers:
            raise ValueError(
                f"ACDF expected {self.num_layers} layers, "
                f"got {layer_descriptors.shape[0]}"
            )

        h = self.desc_proj(self.desc_norm(layer_descriptors))
        r = self.region_proj(region_context).unsqueeze(0)
        h = h + r + self.layer_embedding
        logits = self.score(h).squeeze(-1)
        return F.softmax(logits, dim=0)
