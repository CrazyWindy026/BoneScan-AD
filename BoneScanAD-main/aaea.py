"""Adaptive Anomaly Evidence Aggregation (AAEA)."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEvidenceAggregator(nn.Module):
    """Aggregate patch anomaly scores into one region-level score.

    The official VisualAD reduces the anomaly map with a plain mean over all
    patches. This module instead:

    * attends over patches with a query built from the prototype delta and the
      anatomical region context, so evidence selection is both semantic and
      anatomy-conditioned;
    * adapts the strength of the soft evidence pooling to the spread of the
      current anomaly map;
    * blends the weighted evidence with the global mean through a learnable
      coefficient, which avoids committing to a fixed Top-k fraction and
      suppresses isolated noisy patches.

    ``effective_ratio`` reports ``exp(entropy(weights)) / N``: the fraction of
    spatial evidence the model actually used, which is what the paper's
    analysis of evidence usage is based on.
    """

    def __init__(
        self,
        embed_dim: int,
        evidence_dim: int = 128,
        init_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.patch_key = nn.Linear(embed_dim, evidence_dim)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, evidence_dim),
        )
        self.mix_mlp = nn.Sequential(
            nn.Linear(evidence_dim + 3, max(evidence_dim // 2, 32)),
            nn.GELU(),
            nn.Linear(max(evidence_dim // 2, 32), 1),
        )
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(init_temperature)))
        )

    def forward(
        self,
        patches: torch.Tensor,
        anomaly_map: torch.Tensor,
        prototype_delta: torch.Tensor,
        region_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(weights [N], mix [], effective_ratio [])``.

        The same ``weights``/``mix`` pair is applied to both similarity maps
        outside this module, which keeps the final decision exactly
        ``aggregated_sim_a - aggregated_sim_n``.
        """
        n = anomaly_map.numel()
        if n == 0:
            raise ValueError("AAEA received an empty anomaly map")

        key = F.normalize(self.patch_key(self.patch_norm(patches)), dim=-1)
        query_input = prototype_delta + region_context
        query = F.normalize(self.query_proj(query_input), dim=-1)

        semantic_logits = torch.matmul(key, query)  # [N]

        mean = anomaly_map.mean()
        std = anomaly_map.std(unbiased=False).clamp_min(1e-6)
        maxv = anomaly_map.max()
        score_z = (anomaly_map - mean) / std

        # softplus keeps score_scale positive, so a stronger anomaly response
        # always makes a patch more likely to be selected as evidence.
        score_scale = F.softplus(self.score_scale)
        logits = semantic_logits + score_scale * score_z
        temperature = self.log_temperature.exp().clamp(0.03, 2.0)
        weights = F.softmax(logits / temperature, dim=0)

        stats = torch.stack([mean, maxv, std], dim=0)
        mix_input = torch.cat([query, stats], dim=0)
        mix = torch.sigmoid(self.mix_mlp(mix_input)).squeeze()

        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum()
        effective_ratio = torch.exp(entropy) / float(n)

        return weights, mix, effective_ratio
