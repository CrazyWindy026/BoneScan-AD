"""Batched image preprocessing for the Qwen3.5 vision encoder."""

from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoProcessor


class ViTProcessor:
    """Wrap the HuggingFace processor and move its output to the device."""

    def __init__(self, processor: AutoProcessor, device: torch.device) -> None:
        self.processor = processor
        self.device = device

    def __call__(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of region crops.

        Returns:
            ``pixel_values`` flattened over all patches, and ``grid_thw`` of
            shape ``[B, 3]``.
        """
        inputs = self.processor(
            images=images, text=[""] * len(images), return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(torch.bfloat16).to(self.device)
        grid_thw = inputs["image_grid_thw"].to(self.device)
        return pixel_values, grid_thw
