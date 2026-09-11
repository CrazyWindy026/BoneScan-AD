"""Patient-level label loading and train/val/test splitting."""

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import numpy as np
import torch

from VisualAD_lib.constants import LABEL_SOURCES, REGION_NAMES

SPLIT_NAMES = ("train", "val", "test")


def set_seed(seed: int) -> None:
    """Seed every RNG used by the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_patient_id(value: object) -> str:
    """Canonical patient identifier: the image file stem, stripped."""
    return str(value).strip()


def parse_binary_label(raw_value: object) -> Optional[int]:
    """Map a raw CSV cell to a binary label.

    ``0`` is normal, any positive value is abnormal, and blank/negative/
    unparseable values are treated as missing rather than as negatives.
    """
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text:
        return None

    try:
        raw = int(float(text))
    except (TypeError, ValueError):
        return None

    if raw < 0:
        return None

    return 0 if raw == 0 else 1


def load_region_labels(
    label_dir: Union[str, Path],
    subdirs: Optional[Mapping[str, str]] = None,
) -> Tuple[
    Dict[str, Dict[int, Optional[int]]],
    Dict[str, Set[str]],
    List[Dict[str, object]],
]:
    """Read every region CSV and merge them into one patient -> region map.

    ``label_dir`` holds one sub-directory per label source, each containing
    ``train.csv``, ``val.csv`` and ``test.csv``. The source -> sub-directory
    mapping defaults to the source names themselves.

    When the same patient/region pair carries conflicting labels across
    sources, the region is marked ``None`` (unknown) instead of silently
    keeping whichever file happened to be read last.
    """
    label_dir = Path(label_dir)
    subdirs = dict(subdirs or {})

    labels: Dict[str, Dict[int, Optional[int]]] = defaultdict(dict)
    split_membership: Dict[str, Set[str]] = {name: set() for name in SPLIT_NAMES}

    conflicts: List[Dict[str, object]] = []
    conflict_keys: Set[Tuple[str, int]] = set()

    for source_name, region_ids, column_prefix in LABEL_SOURCES:
        subdir = subdirs.get(source_name, source_name)

        for split_name in SPLIT_NAMES:
            csv_path = label_dir / subdir / f"{split_name}.csv"

            if not csv_path.exists():
                print(f"[warning] label file not found: {csv_path}")
                continue

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)

                for row in reader:
                    patient_id = normalize_patient_id(row.get("patient_id", ""))
                    if not patient_id:
                        continue

                    split_membership[split_name].add(patient_id)

                    for region_id in region_ids:
                        if column_prefix == "label_spine":
                            column_name = "label_spine"
                        else:
                            column_name = f"{column_prefix}{region_id}"

                        new_label = parse_binary_label(row.get(column_name))
                        if new_label is None:
                            continue

                        key = (patient_id, region_id)
                        if key in conflict_keys:
                            continue

                        old_label = labels[patient_id].get(region_id)

                        if old_label is None:
                            labels[patient_id][region_id] = new_label
                        elif old_label != new_label:
                            labels[patient_id][region_id] = None
                            conflict_keys.add(key)

                            conflicts.append(
                                {
                                    "patient_id": patient_id,
                                    "region_id": region_id,
                                    "region": REGION_NAMES[region_id],
                                    "old_label": old_label,
                                    "new_label": new_label,
                                    "source": source_name,
                                    "split": split_name,
                                }
                            )

    return dict(labels), split_membership, conflicts


def patient_has_valid_label(
    patient_id: str,
    label_map: Mapping[str, Mapping[int, Optional[int]]],
) -> bool:
    """True when the patient has at least one region with a resolved label."""
    patient_labels = label_map.get(patient_id, {})
    return any(value in (0, 1) for value in patient_labels.values())


def stable_hash_fraction(patient_id: str, seed: int) -> float:
    """Deterministic pseudo-random value in [0, 1) derived from the patient id."""
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def build_patient_splits(
    cache_patient_ids: Iterable[str],
    label_map: Mapping[str, Mapping[int, Optional[int]]],
    split_membership: Mapping[str, Set[str]],
    split_mode: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[str]]:
    """Split eligible patients into train/val/test.

    ``original`` keeps the cohort's own train/val/test membership, resolving
    cross-source conflicts conservatively with priority test > val > train.
    ``cache-hash`` re-splits by a stable hash of the patient id instead, which
    is useful when a subset needs to be re-partitioned; the CSV split columns
    then only serve as a label source.
    """
    eligible = {
        normalize_patient_id(patient_id)
        for patient_id in cache_patient_ids
        if patient_has_valid_label(normalize_patient_id(patient_id), label_map)
    }

    if split_mode == "cache-hash":
        if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
            raise ValueError(
                "train_ratio and val_ratio must be positive and sum to less than 1."
            )

        splits: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}

        for patient_id in sorted(eligible):
            value = stable_hash_fraction(patient_id, seed)

            if value < train_ratio:
                split_name = "train"
            elif value < train_ratio + val_ratio:
                split_name = "val"
            else:
                split_name = "test"

            splits[split_name].append(patient_id)

        print(
            "[warning] cache-hash split in use: the CSV train/val/test columns "
            "only provide labels and no longer reflect the original protocol."
        )

    else:
        original_train = eligible & set(split_membership["train"])
        original_val = eligible & set(split_membership["val"])
        original_test = eligible & set(split_membership["test"])

        # Conservative patient-level priority: test > val > train.
        test_ids = original_test
        val_ids = original_val - test_ids
        train_ids = original_train - val_ids - test_ids

        conflict_count = len(
            (original_train & original_val)
            | (original_train & original_test)
            | (original_val & original_test)
        )
        print(
            "Cross-source split conflicts resolved conservatively: "
            f"{conflict_count}"
        )

        splits = {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
        }

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Patient splits are not mutually exclusive.")

    if not splits["train"]:
        raise RuntimeError(
            "No eligible patients left in the train split. Check that the "
            "label files and the image directory cover the same patients."
        )

    if not splits["val"]:
        raise RuntimeError("No eligible patients left in the val split.")

    return splits
