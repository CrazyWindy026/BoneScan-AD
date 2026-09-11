"""BoneScanAD: VisualAD-style anomaly detection on a Qwen3.5 ViT.

The model follows the official VisualAD formulation -- a pair of learnable
``t_a``/``t_n`` tokens is inserted into the ViT patch sequence and patches are
scored by ``cos(patch, t_a) - cos(patch, t_n)`` -- and adds three modules on
top of it:

1. region-conditioned ``t_a``/``t_n`` prototype pairs (one pair per body site);
2. AAEA: Adaptive Anomaly Evidence Aggregation;
3. ACDF: Anatomy-Conditioned Dynamic Cross-Layer Fusion;
4. SARE: Symmetry-Aware Anatomical Relation Enhancement.

Fusion layers use 1-based block indices (default ``6, 12, 18, 27``). Each of
the added modules can be switched off independently for ablation via
``use_aaea`` / ``use_acdf`` / ``use_sare`` / ``use_sca``.

NOTE: the attribute names of the submodules are part of the released
checkpoint format -- renaming any of them (``patch_embed``, ``blocks``,
``anomaly_tokens``, ``normal_tokens``, ``anomaly_pos``, ``normal_pos``,
``vis_sca``, ``layer_fusion``, ``evidence_aggregator``, ``symmetry_enhancer``)
would break loading of the pretrained weights.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.spatial_cross_attention import SpatialBottleneckAttention

from .aaea import AdaptiveEvidenceAggregator
from .acdf import AnatomyConditionedLayerFusion
from .constants import NUM_REGIONS
from .sare import SymmetryAwareRelationEnhancer


class QwenViTVisualAD(nn.Module):
    """Region-conditioned VisualAD head on top of a Qwen3.5 vision encoder."""

    def __init__(
        self,
        vision_model: nn.Module,
        num_regions: int = NUM_REGIONS,
        use_sca: bool = True,
        sca_anchors: int = 4,
        fusion_layers: Sequence[int] = (6, 12, 18, 27),
        use_acdf: bool = True,
        fusion_dim: int = 256,
        use_aaea: bool = True,
        evidence_dim: int = 128,
        evidence_temperature: float = 0.25,
        use_sare: bool = True,
        sare_res_scale: float = 0.05,
        sare_detach_ref: bool = False,
    ) -> None:
        super().__init__()

        # Share the Qwen ViT submodules directly (no weight copy).
        self.patch_embed = vision_model.patch_embed
        self.pos_embed = vision_model.pos_embed
        self.blocks = vision_model.blocks
        self.rotary_pos_emb = vision_model.rotary_pos_emb

        self.spatial_merge_size = getattr(vision_model.config, "spatial_merge_size", 2)
        self.num_grid_per_side = getattr(vision_model, "num_grid_per_side", None)
        self.hidden_size = vision_model.config.hidden_size

        self.use_sca = use_sca
        self.use_acdf = use_acdf
        self.use_aaea = use_aaea
        self.use_sare = use_sare

        num_blocks = len(self.blocks)
        self.fusion_layers = tuple(int(x) for x in fusion_layers)
        if not self.fusion_layers:
            raise ValueError("fusion_layers must contain at least one block index")
        if len(set(self.fusion_layers)) != len(self.fusion_layers):
            raise ValueError(f"fusion_layers contains duplicates: {self.fusion_layers}")
        if any(x < 1 or x > num_blocks for x in self.fusion_layers):
            raise ValueError(
                f"fusion_layers must be within [1, {num_blocks}], "
                f"got {self.fusion_layers}"
            )

        # Helper methods of the Qwen3.5 vision model. In some transformers
        # builds these are methods (fast_pos_embed_interpolate / rot_pos_emb)
        # rather than free functions, so they are resolved from the instance.
        self.fast_pos_embed = vision_model.fast_pos_embed_interpolate
        self.rot_pos_emb = vision_model.rot_pos_emb
        self.sare_detach_ref = sare_detach_ref and use_sare

        self.register_buffer("_dummy", torch.zeros(0), persistent=False)

        # -- Innovation 1: region-conditioned prototype pairs --
        self.anomaly_tokens = nn.Parameter(
            torch.randn(num_regions, 1, self.hidden_size) * 0.1
        )
        self.normal_tokens = nn.Parameter(
            torch.randn(num_regions, 1, self.hidden_size) * 0.1
        )
        self.anomaly_pos = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.normal_pos = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)

        if use_sca:
            self.vis_sca = SpatialBottleneckAttention(
                embed_dim=self.hidden_size,
                num_anchors=sca_anchors,
                dropout=0.1,
                max_patches=512,
                res_scale_init=0.01,
            )

        if use_acdf:
            self.layer_fusion = AnatomyConditionedLayerFusion(
                embed_dim=self.hidden_size,
                num_layers=len(self.fusion_layers),
                hidden_dim=fusion_dim,
                dropout=0.1,
            )

        if use_aaea:
            self.evidence_aggregator = AdaptiveEvidenceAggregator(
                embed_dim=self.hidden_size,
                evidence_dim=evidence_dim,
                init_temperature=evidence_temperature,
            )

        if use_sare:
            self.symmetry_enhancer = SymmetryAwareRelationEnhancer(
                embed_dim=self.hidden_size,
                res_scale_init=sare_res_scale,
            )

    def train(self, mode: bool = True) -> "QwenViTVisualAD":
        """Propagate train/eval mode into the borrowed ViT submodules."""
        super().train(mode)
        self.patch_embed.train(mode)
        self.blocks.train(mode)
        return self

    def _region_context(self, region_id: torch.Tensor) -> torch.Tensor:
        """Anatomy context derived from the learned prototype pair of a region."""
        t_a0 = self.anomaly_tokens[region_id].squeeze(0).squeeze(0)
        t_n0 = self.normal_tokens[region_id].squeeze(0).squeeze(0)
        return F.normalize(0.5 * (t_a0 + t_n0).float(), dim=-1)

    def _aggregate_similarity(
        self,
        sim_map: torch.Tensor,
        evidence_weights: torch.Tensor,
        evidence_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Blend AAEA-weighted evidence with the plain patch mean."""
        mean_score = sim_map.mean()
        weighted_score = torch.sum(evidence_weights * sim_map)
        return evidence_mix * weighted_score + (1.0 - evidence_mix) * mean_score

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        region_ids: torch.Tensor,
        primary_count: Optional[int] = None,
        symmetry_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Score every target region in the batch.

        Args:
            pixel_values: images preprocessed by the Qwen processor.
            grid_thw: ``[B, 3]`` = (temporal, height, width) in patches.
            region_ids: ``[B]`` body region id in ``0..8``.
            primary_count: number of scored images. When SARE is enabled the
                batch is ``[targets] + [contralateral references]``, so only
                the first ``primary_count`` images are classified.
            symmetry_valid: ``[primary_count]`` booleans marking the samples
                with a true contralateral pair.

        Returns:
            A dict with the region-level score/probability plus the module
            diagnostics used for analysis and ablation: ``layer_weights``,
            ``evidence_ratio``, ``evidence_mix``, ``symmetry_discrepancy`` and
            ``symmetry_gate``.
        """
        device = pixel_values.device
        b_total = int(grid_thw.shape[0])
        b_primary = int(primary_count) if primary_count is not None else b_total

        if b_primary <= 0 or b_primary > b_total:
            raise ValueError(
                f"Invalid primary_count={b_primary} for total images={b_total}"
            )

        if self.use_sare and primary_count is not None:
            if b_total != 2 * b_primary:
                raise ValueError(
                    "Contralateral SARE expects the processor batch to be "
                    "[primary images] + [paired images], so the total must be "
                    "2 * primary_count. "
                    f"Got total={b_total}, primary_count={b_primary}."
                )
            if symmetry_valid is None or symmetry_valid.numel() != b_primary:
                raise ValueError(
                    "symmetry_valid must have one boolean entry per primary image"
                )

        # -- 1. Patch embedding + absolute position embeddings --
        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self.fast_pos_embed(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # -- 2. Split per image and insert the region-specific t_a/t_n --
        n_patches = (grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0]).tolist()
        patch_splits = hidden_states.split(n_patches)

        vit_dtype = hidden_states.dtype
        new_hidden: List[torch.Tensor] = []
        for i in range(b_total):
            t_a = (self.anomaly_tokens[region_ids[i]] + self.anomaly_pos).squeeze(1)
            t_n = (self.normal_tokens[region_ids[i]] + self.normal_pos).squeeze(1)
            new_hidden.extend([t_a.to(vit_dtype), t_n.to(vit_dtype), patch_splits[i]])

        hidden_states = torch.cat(new_hidden, dim=0)

        # -- 3. Rotary position embeddings (tokens reuse the first patch's) --
        rotary_emb = self.rot_pos_emb(grid_thw)
        new_rotary: List[torch.Tensor] = []
        offset = 0
        for i in range(b_total):
            n = n_patches[i]
            first_patch_rope = rotary_emb[offset : offset + 1]
            new_rotary.extend(
                [
                    first_patch_rope,
                    first_patch_rope,
                    rotary_emb[offset : offset + n],
                ]
            )
            offset += n
        rotary_pos_emb = torch.cat(new_rotary, dim=0)

        # -- 4. cu_seqlens for the packed attention --
        cu_list = [0]
        for i in range(b_total):
            cu_list.append(cu_list[-1] + n_patches[i] + 2)
        cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)

        # -- 5. RoPE cos/sin --
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_2d = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat([rotary_2d, rotary_2d], dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # -- 6. Transformer blocks, capturing the selected layers --
        selected = set(self.fusion_layers)
        layer_states: Dict[int, torch.Tensor] = {}

        with torch.autocast(
            device_type="cuda" if hidden_states.is_cuda else "cpu",
            dtype=torch.bfloat16 if hidden_states.is_cuda else torch.float32,
            enabled=True,
        ):
            for layer_no, blk in enumerate(self.blocks, start=1):
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )
                if layer_no in selected:
                    layer_states[layer_no] = hidden_states

        missing = [x for x in self.fusion_layers if x not in layer_states]
        if missing:
            raise RuntimeError(f"Failed to capture fusion layers: {missing}")

        # -- 7. Per-image multi-layer scoring --
        outputs = {
            "t_a": [],
            "t_n": [],
            "sim_abnormal": [],
            "sim_normal": [],
            "score": [],
            "layer_weights": [],
            "evidence_ratio": [],
            "evidence_mix": [],
            "symmetry_discrepancy": [],
            "symmetry_gate": [],
            "symmetry_valid": [],
        }

        # Only the first b_primary images are classification targets; when SARE
        # is enabled the images in [b_primary, 2*b_primary) are references.
        for i in range(b_primary):
            s = cu_list[i]
            e = cu_list[i + 1]
            region_context = self._region_context(region_ids[i])

            layer_t_a: List[torch.Tensor] = []
            layer_t_n: List[torch.Tensor] = []
            layer_patches: List[torch.Tensor] = []
            layer_sim_a: List[torch.Tensor] = []
            layer_sim_n: List[torch.Tensor] = []
            layer_anomaly: List[torch.Tensor] = []
            layer_desc: List[torch.Tensor] = []
            layer_sym_discrepancy: List[torch.Tensor] = []
            layer_sym_gate: List[torch.Tensor] = []

            for layer_no in self.fusion_layers:
                img_out = layer_states[layer_no][s:e].float()
                t_a_out = img_out[0:1]
                t_n_out = img_out[1:2]
                patches_out = img_out[2:]

                # SARE: the paired reference sits in the second half of the batch.
                pair_is_valid = (
                    self.use_sare
                    and primary_count is not None
                    and symmetry_valid is not None
                    and bool(symmetry_valid[i].item())
                )

                if pair_is_valid:
                    pair_idx = i + b_primary
                    ps = cu_list[pair_idx]
                    pe = cu_list[pair_idx + 1]
                    pair_img_out = layer_states[layer_no][ps:pe].float()
                    paired_patches_out = pair_img_out[2:]

                    # Gradient-isolated SARE: detach the reference branch so it
                    # does not backpropagate into the shared ViT. The SARE
                    # module itself still trains through the target-side path.
                    if self.sare_detach_ref:
                        paired_patches_out = paired_patches_out.detach()

                    patches_out, sym_discrepancy, sym_gate = self.symmetry_enhancer(
                        patches=patches_out,
                        paired_patches=paired_patches_out,
                        grid_thw_i=grid_thw[i],
                        paired_grid_thw_i=grid_thw[pair_idx],
                        region_context=region_context,
                        valid_pair=True,
                    )
                else:
                    sym_discrepancy = patches_out.new_zeros(())
                    sym_gate = patches_out.new_zeros(())

                # SCA consumes the relation-enhanced patch evidence.
                if self.use_sca:
                    t_a_enh = self.vis_sca(t_a_out, patches_out.unsqueeze(0))
                    t_n_enh = self.vis_sca(t_n_out, patches_out.unsqueeze(0))
                else:
                    t_a_enh, t_n_enh = t_a_out, t_n_out

                t_a_norm = F.normalize(t_a_enh, dim=-1).squeeze(0)
                t_n_norm = F.normalize(t_n_enh, dim=-1).squeeze(0)
                patches_norm = F.normalize(patches_out, dim=-1)

                sim_a_map = torch.matmul(patches_norm, t_a_norm)
                sim_n_map = torch.matmul(patches_norm, t_n_norm)
                anomaly_map = sim_a_map - sim_n_map

                # Descriptor combines the local patch context with the
                # prototype state, and drives the ACDF layer weights.
                descriptor = patches_norm.mean(dim=0) + 0.5 * (t_a_norm + t_n_norm)

                layer_t_a.append(t_a_norm)
                layer_t_n.append(t_n_norm)
                layer_patches.append(patches_norm)
                layer_sim_a.append(sim_a_map)
                layer_sim_n.append(sim_n_map)
                layer_anomaly.append(anomaly_map)
                layer_desc.append(descriptor)
                layer_sym_discrepancy.append(sym_discrepancy)
                layer_sym_gate.append(sym_gate)

            desc_stacked = torch.stack(layer_desc, dim=0)  # [K, C]

            if self.use_acdf:
                layer_weights = self.layer_fusion(desc_stacked, region_context)
            else:
                layer_weights = torch.full(
                    (len(self.fusion_layers),),
                    1.0 / len(self.fusion_layers),
                    device=desc_stacked.device,
                    dtype=desc_stacked.dtype,
                )

            def fuse_vector_list(vectors: List[torch.Tensor]) -> torch.Tensor:
                stacked = torch.stack(vectors, dim=0)
                view_shape = [layer_weights.shape[0]] + [1] * (stacked.ndim - 1)
                return torch.sum(layer_weights.view(*view_shape) * stacked, dim=0)

            fused_t_a = F.normalize(fuse_vector_list(layer_t_a), dim=-1)
            fused_t_n = F.normalize(fuse_vector_list(layer_t_n), dim=-1)
            fused_patches = F.normalize(fuse_vector_list(layer_patches), dim=-1)
            fused_sim_a_map = fuse_vector_list(layer_sim_a)
            fused_sim_n_map = fuse_vector_list(layer_sim_n)
            fused_anomaly_map = fuse_vector_list(layer_anomaly)

            if self.use_aaea:
                prototype_delta = F.normalize(fused_t_a - fused_t_n, dim=-1)
                evidence_weights, evidence_mix, evidence_ratio = (
                    self.evidence_aggregator(
                        fused_patches,
                        fused_anomaly_map,
                        prototype_delta,
                        region_context,
                    )
                )
            else:
                n = fused_anomaly_map.numel()
                evidence_weights = torch.full(
                    (n,),
                    1.0 / float(n),
                    device=fused_anomaly_map.device,
                    dtype=fused_anomaly_map.dtype,
                )
                evidence_mix = fused_anomaly_map.new_zeros(())
                evidence_ratio = fused_anomaly_map.new_ones(())

            region_sim_abnormal = self._aggregate_similarity(
                fused_sim_a_map, evidence_weights, evidence_mix
            )
            region_sim_normal = self._aggregate_similarity(
                fused_sim_n_map, evidence_weights, evidence_mix
            )
            region_score = region_sim_abnormal - region_sim_normal

            outputs["t_a"].append(fused_t_a)
            outputs["t_n"].append(fused_t_n)
            outputs["sim_abnormal"].append(region_sim_abnormal.unsqueeze(0))
            outputs["sim_normal"].append(region_sim_normal.unsqueeze(0))
            outputs["score"].append(region_score.unsqueeze(0))
            outputs["layer_weights"].append(layer_weights)
            outputs["evidence_ratio"].append(evidence_ratio.unsqueeze(0))
            outputs["evidence_mix"].append(evidence_mix.unsqueeze(0))
            outputs["symmetry_discrepancy"].append(
                torch.stack(layer_sym_discrepancy).mean().unsqueeze(0)
            )
            outputs["symmetry_gate"].append(
                torch.stack(layer_sym_gate).mean().unsqueeze(0)
            )

            valid_value = (
                float(bool(symmetry_valid[i].item()))
                if symmetry_valid is not None
                else 0.0
            )
            outputs["symmetry_valid"].append(region_score.new_tensor([valid_value]))

        t_a_stacked = torch.stack(outputs["t_a"], dim=0)
        t_n_stacked = torch.stack(outputs["t_n"], dim=0)
        region_scores = torch.cat(outputs["score"], dim=0)

        return {
            "score": region_scores,
            "probability": torch.sigmoid(region_scores),
            "sim_abnormal": torch.cat(outputs["sim_abnormal"], dim=0),
            "sim_normal": torch.cat(outputs["sim_normal"], dim=0),
            "t_a": t_a_stacked,
            "t_n": t_n_stacked,
            "cosine_sim": (t_a_stacked * t_n_stacked).sum(dim=-1),
            "layer_weights": torch.stack(outputs["layer_weights"], dim=0),
            "evidence_ratio": torch.cat(outputs["evidence_ratio"], dim=0),
            "evidence_mix": torch.cat(outputs["evidence_mix"], dim=0),
            "symmetry_discrepancy": torch.cat(outputs["symmetry_discrepancy"], dim=0),
            "symmetry_gate": torch.cat(outputs["symmetry_gate"], dim=0),
            "symmetry_valid": torch.cat(outputs["symmetry_valid"], dim=0),
        }
