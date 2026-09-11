"""Loss terms used to train the anomaly/normal token pair."""

import torch
import torch.nn.functional as F


def contrastive_loss(
    t_a: torch.Tensor,
    t_n: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Push the anomaly and normal tokens apart in cosine space.

    Follows the official VisualAD objective: the two learned tokens should
    describe dissimilar concepts, so their cosine similarity is penalised
    whenever it rises above ``margin``.
    """
    cos_sim = (F.normalize(t_a, dim=-1) * F.normalize(t_n, dim=-1)).sum(dim=-1)
    return F.relu(cos_sim - margin).mean()


def margin_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.20,
) -> torch.Tensor:
    """Require a separation of ``margin`` between positive and negative scores.

    ``scores`` are image-level anomaly logits; ``labels`` are 0/1. The signed
    score (+1 for abnormal, -1 for normal) turns this into a single hinge term
    that is correct for both classes.
    """
    sign = labels * 2.0 - 1.0
    return F.relu(margin - sign * scores).mean()
