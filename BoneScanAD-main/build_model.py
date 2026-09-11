"""Factory assembling a :class:`QwenViTVisualAD` from parsed arguments."""

from typing import Sequence, Union

import torch.nn as nn

from .VisualAD import QwenViTVisualAD


def parse_fusion_layers(spec: Union[str, Sequence[int]]) -> tuple:
    """Parse ``"6,12,18,27"`` into ``(6, 12, 18, 27)``."""
    if isinstance(spec, str):
        layers = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    else:
        layers = tuple(int(x) for x in spec)

    if not layers:
        raise ValueError("No fusion layers given; expected e.g. '6,12,18,27'.")
    return layers


def build_model(args, vision_model: nn.Module) -> QwenViTVisualAD:
    """Build the model with the ablation switches taken from ``args``.

    Each ``--no-*`` flag disables exactly one component, which is what the
    ablation table in the paper is produced with.
    """
    return QwenViTVisualAD(
        vision_model,
        use_sca=not args.no_sca,
        sca_anchors=args.sca_anchors,
        fusion_layers=parse_fusion_layers(args.fusion_layers),
        use_acdf=not args.no_acdf,
        fusion_dim=args.fusion_dim,
        use_aaea=not args.no_aaea,
        evidence_dim=args.evidence_dim,
        evidence_temperature=args.evidence_temperature,
        use_sare=not args.no_sare,
        sare_res_scale=args.sare_res_scale,
        sare_detach_ref=args.sare_detach_ref,
    )
