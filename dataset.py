"""Dataset for nine-region bone-scan anomaly detection.

One sample is a single body region of a single patient together with that
region's true contralateral counterpart, which SARE uses as an anatomical
reference. Targets can optionally be oversampled to a fixed per-region
positive count.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from torch.utils.data import Dataset

from VisualAD_lib.constants import (
    DEFAULT_CROP_SIZE,
    NUM_REGIONS,
    REGION_NAMES,
)
from utils.crop import CropTemplate
from utils.transforms import letterbox_crop, paired_region_for, to_rgb, view_for_region

ANT_SUBDIR = "ant"
POST_SUBDIR = "post"


def load_patient_crops(
    patient_id: str,
    ant_dir: Path,
    post_dir: Path,
    crop_boxes: Union[str, Path],
    target_size: int = DEFAULT_CROP_SIZE,
) -> Optional[np.ndarray]:
    """Load one whole-body scan pair and crop all nine regions.

    Returns:
        ``[9, target_size, target_size, 3]`` uint8, or ``None`` when either
        projection is missing for this patient.
    """
    cropper = CropTemplate(crop_boxes)

    images = {
        ANT_SUBDIR: cv2.imread(
            str(ant_dir / f"{patient_id}.png"), cv2.IMREAD_GRAYSCALE
        ),
        POST_SUBDIR: cv2.imread(
            str(post_dir / f"{patient_id}.png"), cv2.IMREAD_GRAYSCALE
        ),
    }
    if images[ANT_SUBDIR] is None or images[POST_SUBDIR] is None:
        return None

    crops = []
    for region_id in range(NUM_REGIONS):
        view = view_for_region(region_id)
        crop = cropper.crop(images[view], view, region_id)
        crops.append(to_rgb(letterbox_crop(crop, target_size)))

    return np.stack(crops, axis=0)


class RegionImageDataset(Dataset):
    """Each sample is one target region plus its contralateral reference.

    Yields:
        patient id, target crop, paired crop, target region id, paired region
        id, whether the pair is anatomically valid, and the target label.

    The paired crop is an internal anatomical reference only -- its own label
    is never used. For the spine (region 6) SARE is disabled through
    ``symmetry_valid=False`` and the crop falls back to the target itself so
    that batch shapes stay uniform.
    """

    def __init__(
        self,
        patient_ids: Sequence[str],
        label_map: Mapping[str, Mapping[int, Optional[int]]],
        ant_dir: Path,
        crop_boxes: Union[str, Path],
        post_dir: Optional[Path] = None,
        target_size: int = DEFAULT_CROP_SIZE,
        oversample_target: Optional[int] = None,
    ) -> None:
        self.ant_dir = Path(ant_dir)
        self.post_dir = Path(post_dir) if post_dir is not None else self.ant_dir
        self.target_size = target_size
        self.cropper = CropTemplate(crop_boxes)

        # (patient_id, region_id, label)
        self.samples: List[Tuple[str, int, int]] = []

        for patient_id in patient_ids:
            for region_id in range(NUM_REGIONS):
                label = label_map.get(patient_id, {}).get(region_id)
                if label in (0, 1):
                    self.samples.append((patient_id, region_id, int(label)))

        if oversample_target is not None and oversample_target > 0:
            self._oversample_positives(int(oversample_target))

    def _oversample_positives(self, oversample_target: int) -> None:
        """Top up each region's positives to an exact target count.

        Per region independently: if a region has fewer than ``oversample_target``
        positives, exactly ``oversample_target - current`` samples are drawn
        once with replacement. The count therefore can never exceed the target.

        NOTE: an earlier implementation appended the remainder inside the loop
        over positives, so a region with 101 positives and a target of 187 grew
        to 101 * 86 samples. Keep this single-draw structure.
        """
        positives_by_region: Dict[int, List[Tuple[str, int, int]]] = defaultdict(list)
        for sample in self.samples:
            _, region_id, label = sample
            if label == 1:
                positives_by_region[int(region_id)].append(sample)

        extras: List[Tuple[str, int, int]] = []
        report: List[str] = []

        for region_id in range(NUM_REGIONS):
            region_positives = positives_by_region.get(region_id, [])
            current = len(region_positives)

            # Cannot synthesise a positive class when none exists.
            if current == 0 or current >= oversample_target:
                continue

            need = oversample_target - current
            extras.extend(random.choices(region_positives, k=need))
            report.append(
                f"{REGION_NAMES[region_id]}: {current} -> {current + need} (+{need})"
            )

        if extras:
            self.samples.extend(extras)
            random.shuffle(self.samples)
            print(
                f"    Oversample: added {len(extras)} positives total "
                f"(exact target={oversample_target})"
            )
            for message in report:
                print(f"      {message}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_region_crop(
        self,
        patient_id: str,
        region_id: int,
        *,
        view_override: Optional[str] = None,
    ) -> np.ndarray:
        """Load one region using the requested projection and its crop box."""
        view = view_override if view_override is not None else view_for_region(region_id)
        img_dir = self.ant_dir if view == ANT_SUBDIR else self.post_dir
        image = cv2.imread(str(img_dir / f"{patient_id}.png"), cv2.IMREAD_GRAYSCALE)

        if image is None:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        crop = self.cropper.crop(image, view, region_id)
        return to_rgb(letterbox_crop(crop, self.target_size))

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray, int, int, bool, int]:
        patient_id, region_id, label = self.samples[idx]

        target_view = view_for_region(region_id)
        crop = self._load_region_crop(patient_id, region_id, view_override=target_view)

        paired_region_id = paired_region_for(region_id)
        symmetry_valid = paired_region_id is not None

        if symmetry_valid:
            # Critical: use the counterpart's box in the SAME projection as the
            # target. The feature grid is mirrored later, inside SARE.
            paired_crop = self._load_region_crop(
                patient_id, int(paired_region_id), view_override=target_view
            )
            paired_region_id_for_model = int(paired_region_id)
        else:
            # Keeps batch shapes uniform; the reference is never consumed
            # because symmetry_valid=False, so the spine gets no residual.
            paired_crop = crop.copy()
            paired_region_id_for_model = int(region_id)

        return (
            patient_id,
            crop,
            paired_crop,
            int(region_id),
            paired_region_id_for_model,
            bool(symmetry_valid),
            int(label),
        )
