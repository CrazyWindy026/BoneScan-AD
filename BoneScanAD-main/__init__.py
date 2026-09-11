"""BoneScanAD model library.

This package mirrors the layout of the official VisualAD release: the core
anomaly-detection model lives in :mod:`VisualAD_lib.VisualAD`, the backbone
loading helpers in :mod:`VisualAD_lib.model_load`, and the lightweight
task-specific heads in their own modules so that each one can be ablated
independently.
"""

from .constants import (
    CONTRALATERAL_REGION,
    EXPECTED_REGION_NAMES,
    LABEL_SOURCES,
    NUM_REGIONS,
    REGION_NAMES,
    SYMMETRIC_REGION_IDS,
    VIEW_BY_REGION,
)

__all__ = [
    "CONTRALATERAL_REGION",
    "EXPECTED_REGION_NAMES",
    "LABEL_SOURCES",
    "NUM_REGIONS",
    "REGION_NAMES",
    "SYMMETRIC_REGION_IDS",
    "VIEW_BY_REGION",
]
