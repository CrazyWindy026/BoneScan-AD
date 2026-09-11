"""Training loop, evaluation loop and parameter-group construction."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from VisualAD_lib.transform import ViTProcessor

from .loss import contrastive_loss, margin_ranking_loss


def build_param_groups(model: nn.Module, args) -> List[dict]:
    """Split parameters into backbone / prototype-token / head groups.

    The lightweight modules added on top of VisualAD (SCA, ACDF, AAEA, SARE)
    and the prototype tokens use ``args.token_lr``; the Qwen ViT backbone uses
    the smaller ``args.lr``.
    """
    head_keywords = (
        "vis_sca",
        "layer_fusion",
        "evidence_aggregator",
        "symmetry_enhancer",
    )

    vit_group = []
    token_group = []
    head_group = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if any(keyword in name for keyword in head_keywords):
            head_group.append(parameter)
        elif "token" in name or name.endswith("_pos") or ".pos_" in name:
            token_group.append(parameter)
        else:
            vit_group.append(parameter)

    param_groups = []
    if vit_group:
        param_groups.append({"params": vit_group, "lr": args.lr})
    if token_group:
        param_groups.append({"params": token_group, "lr": args.token_lr})
    if head_group:
        param_groups.append({"params": head_group, "lr": args.token_lr})
    return param_groups


def prepare_batch(
    model: nn.Module,
    batch,
    processor: ViTProcessor,
):
    """Turn one collated batch into model inputs.

    When SARE is enabled, the target crops and their contralateral references
    are concatenated into a single processor call so the ViT runs once, and
    only the first half of the batch is scored.
    """
    (
        pids,
        crops,
        paired_crops,
        region_ids_np,
        paired_region_ids_np,
        symmetry_valid_np,
        labels_np,
    ) = batch

    if model.use_sare:
        pixel_values, grid_thw = processor(list(crops) + list(paired_crops))
        region_ids = torch.tensor(
            list(region_ids_np) + list(paired_region_ids_np),
            dtype=torch.long,
            device=processor.device,
        )
        symmetry_valid = torch.tensor(
            symmetry_valid_np, dtype=torch.bool, device=processor.device
        )
        primary_count: Optional[int] = len(crops)
    else:
        pixel_values, grid_thw = processor(list(crops))
        region_ids = torch.tensor(
            region_ids_np, dtype=torch.long, device=processor.device
        )
        symmetry_valid = None
        primary_count = None

    labels = torch.tensor(labels_np, dtype=torch.float32, device=processor.device)

    return pids, labels, {
        "pixel_values": pixel_values,
        "grid_thw": grid_thw,
        "region_ids": region_ids,
        "primary_count": primary_count,
        "symmetry_valid": symmetry_valid,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    processor: ViTProcessor,
    positive_weight: torch.Tensor,
    args,
    epoch: int,
) -> Dict[str, float]:
    """Run a single training epoch and return the mean loss terms."""
    model.train()

    total_bce = 0.0
    total_contrast = 0.0
    total_margin = 0.0
    total_valid = 0

    progress = tqdm(loader, desc=f"Train E{epoch}", leave=False)

    for batch in progress:
        _, labels, model_inputs = prepare_batch(model, batch, processor)

        optimizer.zero_grad(set_to_none=True)

        output = model(**model_inputs)

        bce_loss = F.binary_cross_entropy_with_logits(
            output["score"], labels, pos_weight=positive_weight
        )
        contrast = contrastive_loss(output["t_a"], output["t_n"])
        margin_loss = margin_ranking_loss(output["score"], labels, args.margin)

        loss = (
            args.lambda_bce * bce_loss
            + args.lambda_contrast * contrast
            + args.lambda_margin * margin_loss
        )

        loss.backward()

        trainable = [
            p for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

        optimizer.step()

        bs = labels.shape[0]
        total_bce += float(bce_loss.detach().cpu()) * bs
        total_contrast += float(contrast.detach().cpu()) * bs
        total_margin += float(margin_loss.detach().cpu()) * bs
        total_valid += bs

        progress.set_postfix(
            bce=f"{bce_loss.item():.4f}", cont=f"{contrast.item():.4f}"
        )

    n = max(total_valid, 1)
    return {
        "bce": total_bce / n,
        "contrast": total_contrast / n,
        "margin": total_margin / n,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    processor: ViTProcessor,
) -> Dict[str, np.ndarray]:
    """Collect every prediction and diagnostic for metric computation."""
    model.eval()

    collected: Dict[str, List] = {
        "patient_id": [],
        "region_id": [],
        "label": [],
        "score": [],
        "probability": [],
        "sim_normal": [],
        "sim_abnormal": [],
        "evidence_ratio": [],
        "evidence_mix": [],
        "symmetry_discrepancy": [],
        "symmetry_gate": [],
        "symmetry_valid": [],
        "layer_weights": [],
    }

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        pids, labels, model_inputs = prepare_batch(model, batch, processor)
        output = model(**model_inputs)

        scores = output["score"].cpu().numpy()
        probs = output["probability"].cpu().numpy()
        sim_n = output["sim_normal"].cpu().numpy()
        sim_a = output["sim_abnormal"].cpu().numpy()
        evidence_ratio = output["evidence_ratio"].cpu().numpy()
        evidence_mix = output["evidence_mix"].cpu().numpy()
        symmetry_discrepancy = output["symmetry_discrepancy"].cpu().numpy()
        symmetry_gate = output["symmetry_gate"].cpu().numpy()
        symmetry_valid_out = output["symmetry_valid"].cpu().numpy()
        layer_weights = output["layer_weights"].cpu().numpy()

        region_ids_np = model_inputs["region_ids"][: len(pids)].cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i in range(len(pids)):
            collected["patient_id"].append(str(pids[i]))
            collected["region_id"].append(int(region_ids_np[i]))
            collected["label"].append(int(labels_np[i]))
            collected["score"].append(float(scores[i]))
            collected["probability"].append(float(probs[i]))
            collected["sim_normal"].append(float(sim_n[i]))
            collected["sim_abnormal"].append(float(sim_a[i]))
            collected["evidence_ratio"].append(float(evidence_ratio[i]))
            collected["evidence_mix"].append(float(evidence_mix[i]))
            collected["symmetry_discrepancy"].append(float(symmetry_discrepancy[i]))
            collected["symmetry_gate"].append(float(symmetry_gate[i]))
            collected["symmetry_valid"].append(float(symmetry_valid_out[i]))
            collected["layer_weights"].append(
                np.asarray(layer_weights[i], dtype=np.float64)
            )

    predictions = {
        "patient_id": np.asarray(collected["patient_id"], dtype=object),
        "region_id": np.asarray(collected["region_id"], dtype=np.int64),
        "label": np.asarray(collected["label"], dtype=np.int64),
        "score": np.asarray(collected["score"], dtype=np.float64),
        "probability": np.asarray(collected["probability"], dtype=np.float64),
        "sim_normal": np.asarray(collected["sim_normal"], dtype=np.float64),
        "sim_abnormal": np.asarray(collected["sim_abnormal"], dtype=np.float64),
        "evidence_ratio": np.asarray(collected["evidence_ratio"], dtype=np.float64),
        "evidence_mix": np.asarray(collected["evidence_mix"], dtype=np.float64),
        "symmetry_discrepancy": np.asarray(
            collected["symmetry_discrepancy"], dtype=np.float64
        ),
        "symmetry_gate": np.asarray(collected["symmetry_gate"], dtype=np.float64),
        "symmetry_valid": np.asarray(collected["symmetry_valid"], dtype=np.float64),
        "layer_weights": (
            np.stack(collected["layer_weights"], axis=0)
            if collected["layer_weights"]
            else np.zeros((0, len(model.fusion_layers)), dtype=np.float64)
        ),
    }
    # The score is also the decision margin, kept under "delta" so that the
    # exported prediction CSV keeps its documented column name.
    predictions["delta"] = predictions["score"]
    return predictions
