"""Spatial-Aware Cross-Attention (SCA), as used by the official VisualAD.

The learned anomaly/normal tokens are single vectors, so they cannot attend to
the patch grid on their own. SCA supplies a small set of learnable *anchor*
queries that aggregate spatial evidence from the patch tokens and fold the
result back into the tokens through a gated residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBottleneckAttention(nn.Module):
    """Anchor-based spatial aggregation attached to the ViT output tokens.

    Design notes (matching the official implementation):

    1. Learnable anchor queries instead of the token itself as query, which
       avoids the severe query/key imbalance of one query against hundreds of
       patches.
    2. A 2D positional encoding added to the keys, preserving patch layout.
    3. Projections plus LayerNorm performed in the attention space.
    4. Token-guided gating, an output projection, dropout and a learnable
       residual scale (initialised at 0.01 so the discriminative power of the
       original tokens is preserved early in training).
    """

    def __init__(
        self,
        embed_dim: int,
        num_anchors: int = 4,
        dropout: float = 0.1,
        max_patches: int = 512,
        res_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors
        self.max_patches = max_patches

        # 1. Learnable anchor queries (far fewer than the patch count).
        self.anchor_queries = nn.Parameter(
            torch.randn(num_anchors, embed_dim) * 0.02
        )
        # 2. 2D positional encoding applied to the keys.
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_patches, embed_dim) * 0.02
        )
        # 3. Q/K/V projections and normalisation.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        # 4. Token-guided gating over the anchors.
        self.gate = nn.Sequential(nn.Linear(embed_dim, num_anchors), nn.Sigmoid())
        # 5. Output projection and dropout.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 6. Learnable residual scale.
        self.res_scale = nn.Parameter(torch.ones(1) * res_scale_init)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        token_features: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Enhance tokens ``[B, C]`` with patch evidence ``[B, N, C]``."""
        B, N, C = patch_tokens.shape

        # 1. Queries come from the learnable anchors, not from the token.
        anchor_q = self.anchor_queries.unsqueeze(0).expand(B, -1, -1)
        Q = self.q_norm(self.q_proj(anchor_q))  # [B, A, C]

        # 2. Keys are patches plus positional encoding.
        patch_with_pos = patch_tokens + self.pos_encoding[:, :N, :]
        K = self.k_norm(self.k_proj(patch_with_pos))  # [B, N, C]

        # 3. Values are a plain projection of the patches.
        V = self.v_proj(patch_tokens)  # [B, N, C]

        # 4. Scaled dot-product attention and aggregation.
        attn = torch.bmm(Q, K.transpose(1, 2)) * (C**-0.5)  # [B, A, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        aggregated = torch.bmm(attn, V)  # [B, A, C]

        # 5. Token-guided gating: a weighted average over the anchors.
        gate_weights = self.gate(token_features).unsqueeze(-1)  # [B, A, 1]
        gated_agg = (aggregated * gate_weights).sum(dim=1)  # [B, C]

        # 6. Output projection, dropout and learnable residual.
        output = self.dropout(self.out_proj(gated_agg))
        return token_features + self.res_scale * output
