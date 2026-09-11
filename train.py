#!/usr/bin/env python3
"""Training entry point for BoneScanAD.

BoneScanAD applies the VisualAD formulation -- a learnable anomaly/normal
token pair inserted into the ViT patch sequence, scored by
``cos(patch, t_a) - cos(patch, t_n)`` -- to whole-body bone SPECT scans, with
one token pair per anatomical region. Three modules are added on top:

  AAEA  Adaptive Anomaly Evidence Aggregation
  ACDF  Anatomy-Conditioned Dynamic Cross-Layer Fusion
  SARE  Symmetry-Aware Anatomical Relation Enhancement

Each one can be disabled with ``--no-aaea`` / ``--no-acdf`` / ``--no-sare``
(and the shared SCA with ``--no-sca``) to reproduce the ablation table.

Example:
    python train.py \
        --data-dir data --label-dir dataset \
        --backbone-path /path/to/Qwen3.5-VL \
        --output-dir outputs/bonescanad --evaluate-test
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ANT_SUBDIR, POST_SUBDIR, RegionImageDataset
from utils.crop import CropTemplate, validate_region_layout
from utils.data_utils import (
    build_patient_splits,
    load_region_labels,
    set_seed,
)
from utils.io_utils import (
    save_innovation_diagnostics,
    save_json,
    save_prediction_csv,
)
from utils.logger import configure_log_file, get_logger
from utils.metrics import (
    choose_threshold,
    compute_binary_metrics,
    compute_region_metrics,
)
from utils.training_utils import build_param_groups, evaluate, train_one_epoch
from utils.transforms import view_for_region
from VisualAD_lib.build_model import build_model
from VisualAD_lib.constants import CONTRALATERAL_REGION, NUM_REGIONS, REGION_NAMES
from VisualAD_lib.model_load import load_backbone

REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BoneScanAD on nine-region bone SPECT scans."
    )

    # -- Paths --
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing the 'ant' and 'post' image sub-directories.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="Directory holding one sub-directory per label source.",
    )
    parser.add_argument(
        "--label-subdirs",
        type=str,
        default="limbs,spine,chest",
        help=(
            "Comma-separated sub-directory names under --label-dir, in the "
            "order limbs,spine,chest, each containing train/val/test.csv."
        ),
    )
    parser.add_argument(
        "--crop-boxes",
        type=Path,
        default=None,
        help=(
            "JSON template with your own per-region ROI boxes/polygons. "
            "Defaults to <data-dir>/crop_boxes.json. The format is documented "
            "in utils/crop.py."
        ),
    )
    parser.add_argument(
        "--backbone-path",
        type=str,
        default=str(REPO_ROOT / "backbone"),
        help=(
            "Local directory or HuggingFace model id of the Qwen3.5 backbone "
            "whose vision tower is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Where best.pt, prediction CSVs and metric JSONs are written.",
    )

    # -- Runtime --
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Move the vision tower to this device (e.g. 'cuda:1'). By default "
            "placement is left to device_map='auto'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)

    # -- Optimization --
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--token-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # -- Loss weights --
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-contrast", type=float, default=0.10)
    parser.add_argument("--lambda-margin", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.20)

    # -- ViT fine-tuning strategy --
    parser.add_argument(
        "--freeze-vit",
        action="store_true",
        help="Freeze the whole ViT and train only the t_a/t_n tokens.",
    )
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=-1,
        help="Unfreeze the last N ViT blocks (0 = freeze all, -1 = all trainable).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Epochs of token-only training before the ViT is unfrozen.",
    )

    # -- SCA (shared with the official VisualAD) --
    parser.add_argument(
        "--no-sca",
        action="store_true",
        help="Disable SCA to run the plain ViTAD baseline.",
    )
    parser.add_argument(
        "--sca-anchors",
        type=int,
        default=4,
        help="Number of SCA anchor queries (the official train.py uses 4).",
    )

    # -- AAEA --
    parser.add_argument(
        "--no-aaea",
        action="store_true",
        help="Disable AAEA and average the patch anomaly scores instead.",
    )
    parser.add_argument(
        "--evidence-dim",
        type=int,
        default=128,
        help="Internal evidence-attention dimension of AAEA.",
    )
    parser.add_argument(
        "--evidence-temperature",
        type=float,
        default=0.25,
        help="Initial soft-evidence temperature of AAEA.",
    )

    # -- ACDF --
    parser.add_argument(
        "--no-acdf",
        action="store_true",
        help="Disable ACDF and use an equal-weight average over the layers.",
    )
    parser.add_argument(
        "--fusion-layers",
        type=str,
        default="6,12,18,27",
        help="Qwen ViT block indices (1-based) used for cross-layer fusion.",
    )
    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=256,
        help="Hidden dimension of the ACDF gating network.",
    )

    # -- SARE --
    parser.add_argument(
        "--no-sare",
        action="store_true",
        help="Disable symmetry-aware anatomical relation enhancement.",
    )
    parser.add_argument(
        "--sare-detach-ref",
        action="store_true",
        help=(
            "Gradient-isolate the contralateral branch so it does not "
            "backpropagate into the shared ViT. SARE itself still trains "
            "through the target-side path."
        ),
    )
    parser.add_argument(
        "--sare-res-scale",
        type=float,
        default=0.05,
        help="Initial residual scale of the SARE enhancement.",
    )

    # -- Splitting --
    parser.add_argument("--split-mode", choices=("original", "cache-hash"), default="original")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--threshold-objective",
        choices=("recall_at_spec", "f1", "f2"),
        default="recall_at_spec",
    )
    parser.add_argument("--minimum-specificity", type=float, default=0.90)

    # -- Class imbalance --
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help=(
            "Positive weight for BCE. The default 1.0 keeps the natural "
            "training distribution; a value <= 0 derives a global neg/pos "
            "weight from the training set instead."
        ),
    )
    parser.add_argument("--maximum-pos-weight", type=float, default=10.0)

    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--oversample",
        type=int,
        default=0,
        help=(
            "Per-region positive oversampling target. 0 disables it "
            "(recommended for a clean run); N > 0 tops every region's "
            "positives up to exactly N."
        ),
    )

    return parser.parse_args()


def build_collate():
    """Keep variable-size crops as Python lists; the processor batches them."""

    def collate(batch):
        (
            pids,
            crops,
            paired_crops,
            region_ids,
            paired_region_ids,
            symmetry_valid,
            labels,
        ) = zip(*batch)
        return (
            list(pids),
            list(crops),
            list(paired_crops),
            list(region_ids),
            list(paired_region_ids),
            list(symmetry_valid),
            list(labels),
        )

    return collate


def count_by_region(dataset: RegionImageDataset):
    """Number of positive and negative samples per region."""
    positives = defaultdict(int)
    negatives = defaultdict(int)
    for _, region_id, label in dataset.samples:
        if label == 1:
            positives[region_id] += 1
        else:
            negatives[region_id] += 1
    return positives, negatives


def check_crop_template(crop_boxes: Path, logger) -> None:
    """Verify the ROI template covers every region before the backbone loads.

    Loading the backbone takes minutes, so a missing or incomplete template is
    worth reporting up front rather than on the first training batch.
    """
    template = CropTemplate(crop_boxes)
    missing = [
        f"  {view_for_region(region_id)}/{region_id} ({REGION_NAMES[region_id]})"
        for region_id in range(NUM_REGIONS)
        if not template.has_region(view_for_region(region_id), region_id)
    ]

    if missing:
        raise RuntimeError(
            f"Crop template {crop_boxes} does not cover every region:\n"
            + "\n".join(missing)
        )
    logger.info("Crop template OK: %s", crop_boxes)


def apply_freeze_strategy(model, vision_model, args, logger) -> None:
    """Configure which ViT parameters are trainable for this run."""
    if args.freeze_vit or args.unfreeze_last_n == 0:
        logger.info("Freezing the whole ViT; only the tokens are trainable.")
        for parameter in vision_model.parameters():
            parameter.requires_grad_(False)
    elif args.unfreeze_last_n > 0:
        n = args.unfreeze_last_n
        logger.info("Unfreezing the last %d ViT blocks.", n)
        for parameter in vision_model.parameters():
            parameter.requires_grad_(False)
        for block in vision_model.blocks[-n:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        # patch_embed and pos_embed are unfrozen as well.
        for parameter in vision_model.patch_embed.parameters():
            parameter.requires_grad_(True)
        for parameter in vision_model.pos_embed.parameters():
            parameter.requires_grad_(True)
    else:
        logger.info("All ViT parameters are trainable.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    validate_region_layout()

    # No crop template ships with the repo: the ROI coordinates depend on the
    # scan protocol, so each user supplies their own. Default to the obvious
    # location next to the images.
    if args.crop_boxes is None:
        args.crop_boxes = args.data_dir / "crop_boxes.json"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    configure_log_file(args.output_dir / "train.log")
    save_json(args.output_dir / "run_config.json", vars(args))

    # -- Labels --
    check_crop_template(args.crop_boxes, logger)

    logger.info("Loading region labels from %s ...", args.label_dir)
    subdir_names = [x.strip() for x in args.label_subdirs.split(",") if x.strip()]
    subdirs = dict(zip(("limbs", "spine", "chest"), subdir_names))
    label_map, split_membership, conflicts = load_region_labels(
        args.label_dir, subdirs=subdirs
    )
    logger.info(
        "%d patients with labels, %d conflicting region labels.",
        len(label_map),
        len(conflicts),
    )

    # -- Backbone --
    logger.info("Loading the Qwen vision backbone from %s ...", args.backbone_path)
    vision_model, vit_processor = load_backbone(str(args.backbone_path), args.device)
    vit_device = vit_processor.device
    logger.info(
        "ViT: %d-dim, %d blocks, device=%s",
        vision_model.config.hidden_size,
        len(vision_model.blocks),
        vit_device,
    )

    # -- Patients with both images and labels --
    ant_dir = args.data_dir / ANT_SUBDIR
    post_dir = args.data_dir / POST_SUBDIR
    ant_patients = {p.stem for p in ant_dir.glob("*.png")}
    post_patients = {p.stem for p in post_dir.glob("*.png")}
    # The spine is cropped from the POST view and the bilateral SARE pairs
    # from the ANT view, so both projections are required. Intersecting here
    # prevents a silent zero-crop fallback later.
    all_image_patients = ant_patients & post_patients
    logger.info(
        "Patients with ANT=%d, POST=%d, both=%d",
        len(ant_patients),
        len(post_patients),
        len(all_image_patients),
    )

    cache_from_images = {pid: {} for pid in all_image_patients if pid in label_map}
    logger.info("Patients with both labels and images: %d", len(cache_from_images))

    split_patient_ids = build_patient_splits(
        cache_patient_ids=cache_from_images.keys(),
        label_map=label_map,
        split_membership=split_membership,
        split_mode=args.split_mode,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    logger.info(
        "Splits: train=%d, val=%d, test=%d",
        len(split_patient_ids["train"]),
        len(split_patient_ids["val"]),
        len(split_patient_ids["test"]),
    )

    # -- Oversampling (0 means genuinely off) --
    oversample_target: Optional[int]
    if args.oversample <= 0:
        oversample_target = None
        logger.info("Oversampling: OFF (natural training distribution)")
    else:
        oversample_target = int(args.oversample)
        logger.info("Oversampling: ON, exact per-region target=%d", oversample_target)

    # -- Datasets --
    dataset_kwargs = {
        "ant_dir": ant_dir,
        "post_dir": post_dir,
        "crop_boxes": args.crop_boxes,
    }
    train_dataset = RegionImageDataset(
        split_patient_ids["train"],
        label_map,
        oversample_target=oversample_target,
        **dataset_kwargs,
    )
    val_dataset = RegionImageDataset(
        split_patient_ids["val"], label_map, **dataset_kwargs
    )
    test_dataset = RegionImageDataset(
        split_patient_ids["test"], label_map, **dataset_kwargs
    )

    logger.info(
        "Samples: train=%d, val=%d, test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_pos_reg, train_neg_reg = count_by_region(train_dataset)
    logger.info("Train per region (pos/neg):")
    for region_id in range(NUM_REGIONS):
        positives = train_pos_reg[region_id]
        negatives = train_neg_reg[region_id]
        logger.info(
            "  %14s: %5d pos / %5d neg (%5d total)",
            REGION_NAMES[region_id],
            positives,
            negatives,
            positives + negatives,
        )

    # Fail fast if a future edit reintroduces runaway oversampling.
    if oversample_target is not None:
        exceeded = {
            region_id: train_pos_reg[region_id]
            for region_id in range(NUM_REGIONS)
            if train_pos_reg[region_id] > oversample_target
        }
        if exceeded:
            raise RuntimeError(
                "Oversampling sanity check failed: positive count exceeded "
                f"target={oversample_target}: {exceeded}"
            )

    train_pos = sum(train_pos_reg.values())
    train_neg = sum(train_neg_reg.values())
    logger.info("Train total: positive=%d, negative=%d", train_pos, train_neg)

    if train_pos == 0 or train_neg == 0:
        raise RuntimeError("Training must have both positive and negative samples.")

    # Positive weight for BCE. The clean-distribution default is 1.0; only when
    # --pos-weight <= 0 is a global neg/pos weight derived from the train set.
    if args.pos_weight <= 0:
        pos_weight_value = min(
            train_neg / max(train_pos, 1), args.maximum_pos_weight
        )
        logger.info(
            "Positive BCE weight: %.4f (auto global neg/pos)", pos_weight_value
        )
    else:
        pos_weight_value = float(args.pos_weight)
        logger.info("Positive BCE weight: %.4f (explicit)", pos_weight_value)

    # -- Data loaders --
    collate = build_collate()
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # -- Model --
    logger.info("Building BoneScanAD ...")
    model = build_model(args, vision_model).to(vit_device)

    logger.info(
        "Innovations: AAEA=%s, ACDF=%s layers=%s, SARE=%s%s",
        model.use_aaea,
        model.use_acdf,
        model.fusion_layers,
        model.use_sare,
        f" (gradient-isolated reference={model.sare_detach_ref})"
        if model.use_sare
        else "",
    )

    if model.use_sare:
        logger.info("SARE contralateral pairs:")
        printed = set()
        for left, right in CONTRALATERAL_REGION.items():
            key = tuple(sorted((left, right)))
            if key in printed:
                continue
            printed.add(key)
            a, b = key
            logger.info("  %d:%s <-> %d:%s", a, REGION_NAMES[a], b, REGION_NAMES[b])
        logger.info("  6:%s -> no contralateral SARE", REGION_NAMES[6])

    total_params = sum(p.numel() for p in model.parameters())
    token_params = sum(
        p.numel()
        for p in [
            model.anomaly_tokens,
            model.normal_tokens,
            model.anomaly_pos,
            model.normal_pos,
        ]
    )
    sca_params = sum(p.numel() for p in model.vis_sca.parameters()) if model.use_sca else 0
    innovation_params = 0
    if model.use_acdf:
        innovation_params += sum(p.numel() for p in model.layer_fusion.parameters())
    if model.use_aaea:
        innovation_params += sum(p.numel() for p in model.evidence_aggregator.parameters())
    if model.use_sare:
        innovation_params += sum(p.numel() for p in model.symmetry_enhancer.parameters())
    vit_params = total_params - token_params - sca_params - innovation_params

    logger.info("Total parameters:      %s", f"{total_params:,}")
    logger.info("ViT parameters:        %s", f"{vit_params:,}")
    logger.info("Token parameters:      %s", f"{token_params:,}")
    logger.info("SCA parameters:        %s (use_sca=%s)", f"{sca_params:,}", model.use_sca)
    logger.info("Innovation parameters: %s", f"{innovation_params:,}")

    # -- Freeze strategy --
    apply_freeze_strategy(model, vision_model, args, logger)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{trainable_params:,}")

    # -- Optimizer --
    param_groups = build_param_groups(model, args)
    if not param_groups:
        raise RuntimeError("No trainable parameters!")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    positive_weight = torch.tensor(
        pos_weight_value, dtype=torch.float32, device=vit_device
    )

    # -- Training loop --
    best_monitor = -float("inf")
    no_improvement = 0
    best_checkpoint_path = args.output_dir / "best.pt"

    logger.info("Starting training ...")

    for epoch in range(args.epochs):
        # Warmup: when the ViT was fully frozen via --unfreeze-last-n 0, train
        # the tokens alone for a few epochs before unfreezing everything.
        if (
            epoch == args.warmup_epochs
            and args.freeze_vit is False
            and args.unfreeze_last_n == 0
        ):
            logger.info("Epoch %d: unfreezing all ViT parameters.", epoch)
            for parameter in vision_model.parameters():
                parameter.requires_grad_(True)

            param_groups = build_param_groups(model, args)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info("New trainable parameters: %s", f"{trainable_params:,}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            processor=vit_processor,
            positive_weight=positive_weight,
            args=args,
            epoch=epoch,
        )

        val_predictions = evaluate(model, val_loader, vit_processor)
        val_threshold, val_metrics = choose_threshold(
            labels=val_predictions["label"],
            probabilities=val_predictions["probability"],
            objective=args.threshold_objective,
            minimum_specificity=args.minimum_specificity,
        )

        t_a_norm = F.normalize(model.anomaly_tokens.detach().float(), dim=-1)
        t_n_norm = F.normalize(model.normal_tokens.detach().float(), dim=-1)
        proto_cos = (t_a_norm * t_n_norm).sum(dim=-1).mean().item()

        logger.info(
            "Epoch %02d: bce=%.4f cont=%.4f mar=%.4f | "
            "AUC=%.4f AP=%.4f prec=%.3f rec=%.3f spec=%.3f F1=%.3f "
            "thr=%.3f | cos(ta,tn)=%.3f",
            epoch,
            train_metrics["bce"],
            train_metrics["contrast"],
            train_metrics["margin"],
            val_metrics["auc"],
            val_metrics["ap"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["specificity"],
            val_metrics["f1"],
            val_threshold,
            proto_cos,
        )

        # Monitor average precision, falling back to AUC and F2 when a
        # validation split has only one class present.
        monitor = val_metrics["ap"]
        if math.isnan(monitor):
            monitor = val_metrics["auc"]
        if math.isnan(monitor):
            monitor = val_metrics["f2"]

        if monitor > best_monitor:
            best_monitor = monitor
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    # Full state: ViT + prototypes + SCA + AAEA + ACDF + SARE.
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "threshold": val_threshold,
                    "validation_metrics": dict(val_metrics),
                    "region_names": REGION_NAMES,
                    "fusion_layers": model.fusion_layers,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )

            save_prediction_csv(
                args.output_dir / "val_predictions_best.csv",
                val_predictions,
                val_threshold,
            )
            save_innovation_diagnostics(
                args.output_dir / "val_innovation_diagnostics_best.csv",
                val_predictions,
                model.fusion_layers,
            )
            save_json(
                args.output_dir / "val_region_metrics_best.json",
                compute_region_metrics(val_predictions, val_threshold),
            )

            logger.info("Saved the best model to %s", best_checkpoint_path)

        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                logger.info(
                    "Early stopping: %d epochs without improvement.", args.patience
                )
                break

    # -- Final validation --
    if not best_checkpoint_path.exists():
        logger.warning(
            "No best.pt was saved (no epoch improved, or training was "
            "interrupted); skipping the final evaluation."
        )
        return

    logger.info("Loading the best checkpoint from %s", best_checkpoint_path)
    best_ckpt = torch.load(
        best_checkpoint_path, map_location=vit_device, weights_only=False
    )

    if "model" not in best_ckpt:
        raise RuntimeError(
            "Checkpoint format is from an older script and does not contain "
            "the AAEA/ACDF/SARE states. Please retrain with this script."
        )
    if tuple(best_ckpt.get("fusion_layers", ())) != tuple(model.fusion_layers):
        raise RuntimeError(
            f"Checkpoint fusion_layers={best_ckpt.get('fusion_layers')} does not "
            f"match model.fusion_layers={model.fusion_layers}. Retrain with the "
            "same --fusion-layers, or change the argument to match."
        )
    model.load_state_dict(best_ckpt["model"])

    best_threshold = float(best_ckpt["threshold"])

    final_val_predictions = evaluate(model, val_loader, vit_processor)
    final_val_metrics = compute_binary_metrics(
        final_val_predictions["label"],
        final_val_predictions["probability"],
        best_threshold,
    )

    logger.info(
        "Best validation: AUC=%.4f, AP=%.4f, Prec=%.4f, Rec=%.4f, "
        "Spec=%.4f, F1=%.4f, F2=%.4f, threshold=%.4f",
        final_val_metrics["auc"],
        final_val_metrics["ap"],
        final_val_metrics["precision"],
        final_val_metrics["recall"],
        final_val_metrics["specificity"],
        final_val_metrics["f1"],
        final_val_metrics["f2"],
        best_threshold,
    )

    # -- Test --
    if args.evaluate_test and len(test_dataset) > 0:
        test_predictions = evaluate(model, test_loader, vit_processor)

        test_metrics = compute_binary_metrics(
            test_predictions["label"],
            test_predictions["probability"],
            best_threshold,
        )

        save_prediction_csv(
            args.output_dir / "test_predictions.csv", test_predictions, best_threshold
        )
        save_innovation_diagnostics(
            args.output_dir / "test_innovation_diagnostics.csv",
            test_predictions,
            model.fusion_layers,
        )
        save_json(
            args.output_dir / "test_region_metrics.json",
            compute_region_metrics(test_predictions, best_threshold),
        )
        save_json(args.output_dir / "test_metrics.json", test_metrics)

        logger.info(
            "Test metrics: AUC=%.4f, AP=%.4f, Pr=%.4f, Re=%.4f, Sp=%.4f, "
            "F1=%.4f, F2=%.4f, threshold=%.4f",
            test_metrics["auc"],
            test_metrics["ap"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["specificity"],
            test_metrics["f1"],
            test_metrics["f2"],
            best_threshold,
        )

    logger.info("Training finished. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
