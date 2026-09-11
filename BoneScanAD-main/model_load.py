"""Loading of the Qwen3.5 vision backbone and its image processor."""

from typing import Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoProcessor

from .transform import ViTProcessor


def load_backbone(
    backbone_path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.nn.Module, ViTProcessor]:
    """Load the Qwen3.5 VLM and return its vision tower plus a processor.

    ``backbone_path`` may be either a local directory or a HuggingFace hub
    model id. Only the vision tower is used downstream; the language model is
    never called, but it is loaded because the vision weights ship inside the
    same checkpoint.

    Placement is decided by ``device_map="auto"``. Passing ``device`` moves the
    vision tower to that device explicitly, which is only safe when the tower
    already sits on a single device.
    """
    qwen = AutoModel.from_pretrained(
        backbone_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        backbone_path,
        trust_remote_code=True,
    )

    vision_model = qwen.visual
    vit_device = next(vision_model.parameters()).device

    if device is not None and torch.device(device) != vit_device:
        vision_model = vision_model.to(torch.device(device))
        vit_device = torch.device(device)

    return vision_model, ViTProcessor(processor, vit_device)
