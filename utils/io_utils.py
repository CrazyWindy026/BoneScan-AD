"""CSV/JSON writers for predictions, diagnostics and run metadata."""

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np

from VisualAD_lib.constants import CONTRALATERAL_REGION, REGION_NAMES


def save_json(path: Union[str, Path], payload: Any) -> None:
    """Write ``payload`` as UTF-8 JSON with a stable layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def save_prediction_csv(
    path: Union[str, Path],
    predictions: Mapping[str, np.ndarray],
    threshold: float,
) -> None:
    """Write the per-sample predictions used to compute the reported metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patient_id",
                "region_id",
                "region",
                "label",
                "sim_normal",
                "sim_abnormal",
                "delta",
                "probability",
                "threshold",
                "prediction",
            ],
        )
        writer.writeheader()

        for index in range(len(predictions["label"])):
            region_id = int(predictions["region_id"][index])
            probability = float(predictions["probability"][index])

            writer.writerow(
                {
                    "patient_id": predictions["patient_id"][index],
                    "region_id": region_id,
                    "region": REGION_NAMES[region_id],
                    "label": int(predictions["label"][index]),
                    "sim_normal": float(predictions["sim_normal"][index]),
                    "sim_abnormal": float(predictions["sim_abnormal"][index]),
                    "delta": float(predictions["delta"][index]),
                    "probability": probability,
                    "threshold": threshold,
                    "prediction": int(probability >= threshold),
                }
            )


def save_innovation_diagnostics(
    path: Union[str, Path],
    predictions: Mapping[str, np.ndarray],
    fusion_layers: Sequence[int],
) -> None:
    """Export per-sample module diagnostics (AAEA / ACDF / SARE) for analysis.

    These columns are what the ablation and interpretability tables in the
    paper are computed from: the effective spatial evidence ratio, the
    adaptive pooling coefficient, the contralateral discrepancy and gate, and
    the per-layer fusion weights.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layer_weights = predictions["layer_weights"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "patient_id",
            "region_id",
            "label",
            "score",
            "probability",
            "evidence_ratio",
            "evidence_mix",
            "symmetry_discrepancy",
            "symmetry_gate",
            "symmetry_valid",
            "paired_region_id",
            "paired_region_name",
        ] + [f"layer_{layer}_weight" for layer in fusion_layers]
        writer.writerow(header)

        for index in range(len(predictions["label"])):
            region_id = int(predictions["region_id"][index])
            paired_id = CONTRALATERAL_REGION.get(region_id)

            row = [
                predictions["patient_id"][index],
                region_id,
                int(predictions["label"][index]),
                float(predictions["score"][index]),
                float(predictions["probability"][index]),
                float(predictions["evidence_ratio"][index]),
                float(predictions["evidence_mix"][index]),
                float(predictions["symmetry_discrepancy"][index]),
                float(predictions["symmetry_gate"][index]),
                float(predictions["symmetry_valid"][index]),
                paired_id if paired_id is not None else -1,
                REGION_NAMES[paired_id] if paired_id is not None else "none",
            ]

            if layer_weights.shape[0] > index:
                row.extend(float(x) for x in layer_weights[index].tolist())
            writer.writerow(row)
