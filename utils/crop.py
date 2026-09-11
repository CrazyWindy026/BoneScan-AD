"""Region-of-interest cropping driven by a user-supplied JSON template.

BoneScanAD does not ship a crop template: the ROI coordinates depend on the
scan protocol, the camera field of view and the patient positioning, so they
must be measured on your own data. Write a template in the format below and
pass it with ``--crop-boxes``.

Template format -- keyed by projection (``ant``/``post``) and region id
(``0``-``8``, see ``VisualAD_lib.constants.REGION_NAMES``)::

    {
      "ant": {
        "0": [x1, y1, x2, y2],
        "1": [x1, y1, x2, y2],
        ...
        "7": {"box": [x1, y1, x2, y2], "polygon": [[x, y], ...]},
        "8": {"box": [x1, y1, x2, y2], "polygon": [[x, y], ...]}
      },
      "post": {
        "0": [x1, y1, x2, y2],
        ...
      }
    }

Boxes may be either a bare ``[x1, y1, x2, y2]`` list or a mapping carrying
``box`` (``bbox``/``crop`` also work) plus an optional ``polygon``. A polygon
masks out neighbouring anatomy inside the box -- useful for regions such as
the chest, where the ROI would otherwise overlap the ribs. Polygon points may
be in whole-image coordinates or already local to the box.

Polygons can alternatively live in a separate top-level ``polygons`` section::

    {"ant": {...}, "post": {...}, "polygons": {"ant": {"7": [[x, y], ...]}}}

Coordinates are pixels in the whole-body image, validated against its size at
crop time. Regions without a box raise ``KeyError``, so a template can be
filled in incrementally.
"""

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from VisualAD_lib.constants import EXPECTED_REGION_NAMES, REGION_NAMES


class CropTemplate:
    """Loads ROI definitions and crops them out of a whole-body scan."""

    def __init__(self, json_path: Union[str, Path]) -> None:
        self.path = Path(json_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Crop template not found: {self.path}\n"
                "BoneScanAD does not ship ROI coordinates -- they depend on "
                "your scan protocol. Write your own template in the format "
                "documented in utils/crop.py and pass it with --crop-boxes."
            )

        with self.path.open("r", encoding="utf-8") as handle:
            self.data = json.load(handle)

    @staticmethod
    def _view_keys(view: str) -> Sequence[str]:
        return (view, view.lower(), view.upper())

    def _view_mapping(self, container: Mapping, view: str) -> Optional[Mapping]:
        for key in self._view_keys(view):
            if key in container:
                return container[key]
        return None

    def _lookup(self, view: str, part_id: int):
        part_key = str(part_id)

        boxes_container = self.data.get("boxes", self.data)
        boxes_view = self._view_mapping(boxes_container, view)
        if boxes_view is None or part_key not in boxes_view:
            raise KeyError(
                f"Missing crop box for view={view}, region={part_id} "
                f"({REGION_NAMES[int(part_id)]}) in {self.path}"
            )

        raw = boxes_view[part_key]
        polygon = None

        if isinstance(raw, Mapping):
            box = raw.get("box") or raw.get("bbox") or raw.get("crop")
            polygon = raw.get("polygon")
        else:
            box = raw

        # Polygons may live in a separate top-level section of the template.
        polygons_container = self.data.get("polygons")
        if polygon is None and isinstance(polygons_container, Mapping):
            poly_view = self._view_mapping(polygons_container, view)
            if poly_view is not None:
                polygon = poly_view.get(part_key)

        if box is None or len(box) != 4:
            raise ValueError(
                f"Crop box must contain [x1, y1, x2, y2], got {box!r}"
            )
        return [int(round(v)) for v in box], polygon

    def has_region(self, view: str, part_id: int) -> bool:
        """True when the template defines a usable box for this view/region."""
        try:
            self._lookup(view, part_id)
        except (KeyError, ValueError):
            return False
        return True

    def crop(self, image: np.ndarray, view: str, part_id: int) -> np.ndarray:
        """Crop one region out of ``image``, applying the polygon mask if any."""
        box, polygon = self._lookup(view, part_id)
        x1, y1, x2, y2 = box

        height, width = image.shape
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, x1 + 1, width))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, y1 + 1, height))

        crop = image[y1:y2, x1:x2].copy()

        if polygon is not None:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
            # Accept either global or already crop-local coordinates.
            if (
                points[:, 0].max(initial=0) > crop.shape[1]
                or points[:, 1].max(initial=0) > crop.shape[0]
            ):
                points = points - np.array([x1, y1], dtype=np.int32)

            mask = np.zeros(crop.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points], color=1)
            crop = crop * mask

        return crop


def validate_region_layout() -> None:
    """Fail fast if the region order no longer matches the pairing protocol.

    The contralateral pairs, the crop-box indices and every released
    checkpoint all assume the canonical region order, so a silent reorder
    would corrupt them without raising anywhere else.
    """
    actual = tuple(str(name) for name in REGION_NAMES)
    if actual != EXPECTED_REGION_NAMES:
        raise RuntimeError(
            "REGION_NAMES order does not match the contralateral pairing "
            "protocol. Reordering this list invalidates the crop-box indices "
            "and any pretrained checkpoint.\n"
            f"Expected: {EXPECTED_REGION_NAMES}\n"
            f"Actual:   {actual}"
        )
