"""Image transforms applied to each cropped region before the ViT."""

from typing import Optional

import cv2
import numpy as np

from VisualAD_lib.constants import (
    CONTRALATERAL_REGION,
    DEFAULT_CROP_SIZE,
    DEFAULT_VIEW,
    VIEW_BY_REGION,
)


def view_for_region(region_id: int) -> str:
    """Return the whole-body projection (``ant``/``post``) used for a region."""
    return VIEW_BY_REGION.get(int(region_id), DEFAULT_VIEW)


def paired_region_for(region_id: int) -> Optional[int]:
    """Return the true contralateral region id, or ``None`` for midline regions."""
    return CONTRALATERAL_REGION.get(int(region_id))


def letterbox_crop(crop: np.ndarray, target: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    """Resize preserving aspect ratio and zero-pad to ``target x target``.

    A plain ``cv2.resize`` to a square would stretch every rectangular ROI and
    distort the long-axis / short-axis ratio of the bone. Scaling the long edge
    and centring the result keeps the intra-region anatomy proportional; the
    padding is black, matching the scan background, so it introduces no
    spurious structure.
    """
    h, w = crop.shape[:2]
    scale = target / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(
        crop,
        (nw, nh),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    canvas = np.zeros((target, target), dtype=crop.dtype)
    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def to_rgb(crop: np.ndarray) -> np.ndarray:
    """Grayscale -> 3-channel, since the ViT expects three input channels."""
    return np.stack([crop, crop, crop], axis=-1)
