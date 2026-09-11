"""Symmetry-Aware Anatomical Relation Enhancement (SARE)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetryAwareRelationEnhancer(nn.Module):
    """Enhance target-region patches with true contralateral discrepancy.

    The reference is the *contralateral crop of the same patient in the same
    projection*, not a horizontal mirror of the target:

        left knee     <-> right knee
        left ankle    <-> right ankle
        left shoulder <-> right shoulder
        left chest    <-> right chest
        spine          -> no SARE (midline structure)

    The contralateral feature grid is flipped horizontally before comparison so
    that anatomically corresponding sides line up.

    The discrepancy is only a residual cue -- the original target
    representation remains the main signal. A sample- and anatomy-conditioned
    gate can suppress misleading symmetry cues, for example in bilateral
    disease or under imperfect patient positioning.
    """

    def __init__(
        self,
        embed_dim: int,
        res_scale_init: float = 0.05,
        bottleneck_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(embed_dim // 4, 64)

        # Shared normalisation so both sides are compared in the same space.
        self.patch_norm = nn.LayerNorm(embed_dim)

        # Map the contralateral discrepancy back to the ViT feature dimension.
        self.diff_proj = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

        # The gate sees both the anatomy context and the current discrepancy,
        # which makes it sample-adaptive rather than only region-adaptive.
        self.region_gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        nn.init.zeros_(self.region_gate[-1].weight)
        nn.init.constant_(self.region_gate[-1].bias, -2.0)

        # Small residual initialisation protects the original VisualAD path.
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))

    @staticmethod
    def _reshape_grid(
        patches: torch.Tensor,
        grid_thw_i: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``[N, C]`` -> ``[T, H, W, C]``, or ``None`` if the grid is inconsistent."""
        if patches.ndim != 2:
            return None

        n, c = patches.shape
        t = int(grid_thw_i[0].item())
        h = int(grid_thw_i[1].item())
        w = int(grid_thw_i[2].item())
        if t <= 0 or h <= 0 or w <= 0 or t * h * w != n:
            return None

        return patches.reshape(t, h, w, c)

    def forward(
        self,
        patches: torch.Tensor,
        paired_patches: torch.Tensor,
        grid_thw_i: torch.Tensor,
        paired_grid_thw_i: torch.Tensor,
        region_context: torch.Tensor,
        valid_pair: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(enhanced [N, C], mean_discrepancy, gate)``.

        Args:
            patches: target patch tokens ``[N, C]``.
            paired_patches: contralateral reference tokens ``[N2, C]``.
            grid_thw_i: target patch grid.
            paired_grid_thw_i: contralateral patch grid.
            region_context: anatomy-conditioned prototype context ``[C]``.
            valid_pair: ``False`` for the spine or an unavailable pairing.
        """
        zero = patches.new_zeros(())
        if not valid_pair:
            return patches, zero, zero

        x = self._reshape_grid(patches, grid_thw_i)
        x_pair = self._reshape_grid(paired_patches, paired_grid_thw_i)
        if x is None or x_pair is None:
            return patches, zero, zero

        t, h, w, c = x.shape
        pt, ph, pw, pc = x_pair.shape
        if pt != t or pc != c:
            return patches, zero, zero

        x_norm = self.patch_norm(x)
        pair_norm = self.patch_norm(x_pair)

        # If the processor produced a slightly different grid for the
        # counterpart, resample it before anatomical alignment.
        if (ph, pw) != (h, w):
            pair_2d = pair_norm.permute(0, 3, 1, 2)  # [T, C, H, W]
            pair_2d = F.interpolate(
                pair_2d, size=(h, w), mode="bilinear", align_corners=False
            )
            pair_norm = pair_2d.permute(0, 2, 3, 1)

        # Mirror the opposite region (not the target itself) to align sides.
        pair_aligned = torch.flip(pair_norm, dims=[2])

        discrepancy = torch.abs(x_norm - pair_aligned)
        relation = self.diff_proj(discrepancy)

        # Sample-adaptive, anatomy-conditioned gate.
        discrepancy_context = discrepancy.mean(dim=(0, 1, 2))
        gate_context = region_context + discrepancy_context
        gate = torch.sigmoid(self.region_gate(gate_context)).squeeze()

        enhanced = x + self.res_scale * gate * relation
        mean_discrepancy = discrepancy.mean()

        return enhanced.reshape(-1, c), mean_discrepancy, gate
