"""Shared constants describing the nine-region bone-scan anatomy layout."""

from typing import Dict, Set, Tuple

# Canonical region order used by the classifier heads and by every exported
# CSV/JSON. The contralateral pairing below depends on this exact order, so
# ``validate_region_layout`` re-checks it at startup.
REGION_NAMES = [
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
    "left shoulder",
    "right shoulder",
    "spine",
    "left chest",
    "right chest",
]
NUM_REGIONS = len(REGION_NAMES)

# Deliberate tripwire: kept as a literal copy of REGION_NAMES rather than
# derived from it, so that silently reordering the list above (which would
# misalign the contralateral pairs, the crop-box indices and every released
# checkpoint) fails fast at startup instead of degrading quietly.
EXPECTED_REGION_NAMES: Tuple[str, ...] = (
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
    "left shoulder",
    "right shoulder",
    "spine",
    "left chest",
    "right chest",
)

# Label sources: (source name, region ids, column prefix in the source CSV).
# The prefix is also the name of the sub-directory holding the split CSVs.
LABEL_SOURCES = [
    ("limbs", range(6), "loc_"),
    ("spine", [6], "label_spine"),
    ("chest", [7, 8], "loc_"),
]

# Whole-body projection used to crop each region.
#   0/1 knee, 2/3 ankle, 4/5 shoulder, 7/8 chest -> anterior
#   6 spine                                        -> posterior
# The crop template holds a separate box per (projection, region).
VIEW_BY_REGION: Dict[int, str] = {6: "post"}  # defaults to "ant"
DEFAULT_VIEW = "ant"

# True contralateral anatomy. The spine is a midline structure and is
# deliberately excluded, so it receives no symmetry-based enhancement.
CONTRALATERAL_REGION: Dict[int, int] = {
    0: 1,
    1: 0,  # knees
    2: 3,
    3: 2,  # ankles
    4: 5,
    5: 4,  # shoulders
    7: 8,
    8: 7,  # chest
}
SYMMETRIC_REGION_IDS: Set[int] = set(CONTRALATERAL_REGION.keys())

# Default crop size fed to the ViT.
DEFAULT_CROP_SIZE = 128
