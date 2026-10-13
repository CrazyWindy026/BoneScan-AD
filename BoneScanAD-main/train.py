from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import QwenViTVisualAD

REGION_NAMES = (
    "left knee", "right knee", "left ankle", "right ankle",
    "left shoulder", "right shoulder", "spine", "left chest", "right chest",
)
PAIRS = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 7: 8, 8: 7}
LABEL_COLUMNS = {
    "limbs": {i: (f"loc_{i}",) for i in range(6)},
    "spine": {6: ("label_spine", "loc_6")},
    "chest": {7: ("loc_7",), 8: ("loc_8",)},
}


def arguments():
    p = argparse.ArgumentParser(description="Train BoneScanAD on region crops")
    p.add_argument("--backbone-path", default="./backbone")
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--label-dir", type=Path, default=Path("./dataset"))
    p.add_argument("--crop-boxes", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--token-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=1.0, help="<=0: use train neg/pos")
    p.add_argument("--max-pos-weight", type=float, default=10.0)
    p.add_argument("--oversample", type=int, default=0, help="Per-region positive target; 0 disables")
    p.add_argument("--lambda-bce", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=0.10)
    p.add_argument("--lambda-margin", type=float, default=0.20)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--fusion-layers", default="6,12,18,27")
    p.add_argument("--no-sca", action="store_true")
    p.add_argument("--no-aaea", action="store_true")
    p.add_argument("--no-acdf", action="store_true")
    p.add_argument("--no-sare", action="store_true")
    p.add_argument("--sare-detach-ref", action="store_true")
    p.add_argument("--freeze-vit", action="store_true")
    p.add_argument("--unfreeze-last-n", type=int, default=-1, help="-1 all, 0 none, N last blocks")
    p.add_argument("--warmup-epochs", type=int, default=0, help="Optional token-only warmup")
    p.add_argument("--threshold-objective", choices=("recall_at_spec", "f1", "f2"), default="recall_at_spec")
    p.add_argument("--minimum-specificity", type=float, default=0.90)
    p.add_argument("--evaluate-test", action="store_true")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label(value):
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v < 0:
        return None
    return int(v > 0)


def normalize_patient_id(value):
    pid = str(value or "").strip()
    if not pid or pid.lower() in {"nan", "none"}:
        return ""
    # Excel-exported numeric identifiers may end with '.0'.
    return re.sub(r"\.0+$", "", pid) if re.fullmatch(r"\d+\.0+", pid) else pid


def read_splits(label_dir):
    """Use supplied patient-level CSV splits; prevent patient leakage."""
    splits = {split: {} for split in ("train", "val", "test")}
    for source, region_cols in LABEL_COLUMNS.items():
        for split in splits:
            path = label_dir / source / f"{split}.csv"
            if not path.is_file():
                if split == "test":
                    continue
                raise FileNotFoundError(f"Missing required labels: {path}")
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                pid_col = next((c for c in ("patient_id", "patient", "pid") if c in columns), None)
                if pid_col is None:
                    raise ValueError(f"No patient_id column: {path}")
                resolved = {}
                for region, aliases in region_cols.items():
                    col = next((x for x in aliases if x in columns), None)
                    if col is None:
                        raise ValueError(f"Missing region {region} label column in {path}")
                    resolved[region] = col
                for row in reader:
                    pid = normalize_patient_id(row[pid_col])
                    if not pid:
                        continue
                    for region, col in resolved.items():
                        label = parse_label(row[col])
                        if label is None:
                            continue
                        record = splits[split].setdefault(pid, {})
                        if region in record and record[region] != label:
                            raise ValueError(f"Conflicting label: {pid}, region {region}, {path}")
                        record[region] = label
    members = defaultdict(set)
    for split, entries in splits.items():
        for pid in entries:
            members[pid].add(split)
    overlap = {pid: src for pid, src in members.items() if len(src) > 1}
    if overlap:
        raise ValueError(f"Patient overlap across splits, e.g. {next(iter(overlap.items()))}")
    return splits


def load_boxes(path):
    """Region JSON: {"ant": {"0": [x1,y1,x2,y2], ...}, "post": {...}}."""
    with path.open(encoding="utf-8") as f:
        boxes = json.load(f)
    for region in range(9):
        view = "post" if region == 6 else "ant"
        entry = boxes.get(view, {}).get(str(region))
        coords = entry.get("box") if isinstance(entry, dict) else entry
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError(f"Missing/invalid crop box for {view}/{region} in {path}")
        x1, y1, x2, y2 = map(int, coords)
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            raise ValueError(f"Invalid crop coordinates for {view}/{region}: {coords}")
    return boxes


def crop_image(image, region, view, boxes, size=128):
    entry = boxes[view][str(region)]
    coords = entry["box"] if isinstance(entry, dict) else entry
    x1, y1, x2, y2 = map(int, coords)
    if x2 > image.shape[1] or y2 > image.shape[0]:
        raise ValueError(f"Crop ROI {view}/{region} exceeds image shape {image.shape}")
    cropped = image[y1:y2, x1:x2].copy()
    if isinstance(entry, dict) and entry.get("polygon"):
        poly = np.asarray(entry["polygon"], dtype=np.int32) - [x1, y1]
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
    h, w = cropped.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size - nh)//2:(size - nh)//2 + nh, (size - nw)//2:(size - nw)//2 + nw] = resized
    return np.repeat(canvas[:, :, None], 3, axis=2)


class RegionDataset(Dataset):
    def __init__(self, split_labels, data_dir, boxes, oversample=0):
        self.data_dir, self.boxes = data_dir, boxes
        self.samples = []
        for pid, region_labels in split_labels.items():
            if not all((data_dir / view / f"{pid}.png").is_file() for view in ("ant", "post")):
                continue
            self.samples.extend((pid, int(rid), int(label)) for rid, label in region_labels.items())
        by_region = defaultdict(list)
        for sample in self.samples:
            if sample[2] == 1:
                by_region[sample[1]].append(sample)
        if oversample > 0:
            for region in range(9):
                positive = by_region[region]
                if 0 < len(positive) < oversample:
                    self.samples.extend(random.choices(positive, k=oversample - len(positive)))
        if oversample > 0:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _roi(self, pid, region, view):
        path = self.data_dir / view / f"{pid}.png"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return crop_image(image, region, view, self.boxes)

    def __getitem__(self, index):
        pid, region, label = self.samples[index]
        view = "post" if region == 6 else "ant"
        image = self._roi(pid, region, view)
        paired_region = PAIRS.get(region, region)
        # Reference is the patient's TRUE contralateral ROI in the same view.
        pair = self._roi(pid, paired_region, view) if paired_region != region else image.copy()
        return pid, image, pair, region, paired_region, region in PAIRS, label


def collate(batch):
    return tuple(list(items) for items in zip(*batch))


def make_inputs(batch, processor, device, use_sare):
    patients, images, pairs, regions, paired_regions, valid, labels = batch
    primary = len(images)
    image_list = images + pairs if use_sare else images
    inputs = processor(images=image_list, text=[""] * len(image_list), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)
    grid_thw = inputs["image_grid_thw"].to(device=device)
    rid = torch.tensor(regions + paired_regions if use_sare else regions, device=device, dtype=torch.long)
    valid = torch.tensor(valid, device=device, dtype=torch.bool) if use_sare else None
    labels = torch.tensor(labels, device=device, dtype=torch.float32)
    kwargs = dict(pixel_values=pixel_values, grid_thw=grid_thw, region_ids=rid,
                  primary_count=primary if use_sare else None, symmetry_valid=valid)
    return kwargs, labels


def contrastive_loss(t_a, t_n, margin=0.5):
    similarity = (F.normalize(t_a, dim=-1) * F.normalize(t_n, dim=-1)).sum(dim=-1)
    return F.relu(similarity - margin).mean()


def param_groups(model, args):
    groups = {"vit": [], "token": [], "aux": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(x in name for x in ("vis_sca", "layer_fusion", "evidence_aggregator", "symmetry_enhancer")):
            groups["aux"].append(p)
        elif "token" in name or name.endswith("_pos") or ".pos_" in name:
            groups["token"].append(p)
        else:
            groups["vit"].append(p)
    return [{"params": v, "lr": args.lr if k == "vit" else args.token_lr}
            for k, v in groups.items() if v]


def set_trainable_vision(vision, freeze, last_n):
    for p in vision.parameters():
        p.requires_grad_(not freeze and last_n == -1)
    if not freeze and last_n > 0:
        for block in vision.blocks[-last_n:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for name in ("patch_embed", "pos_embed"):
            module = getattr(vision, name, None)
            if hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad_(True)


def train_epoch(model, loader, processor, optimizer, args, pos_weight, device, epoch):
    model.train()
    running = Counter()
    for batch in tqdm(loader, desc=f"Epoch {epoch:02d}", leave=False):
        kwargs, labels = make_inputs(batch, processor, device, model.use_sare)
        optimizer.zero_grad(set_to_none=True)
        output = model(**kwargs)
        bce = F.binary_cross_entropy_with_logits(output["score"], labels, pos_weight=pos_weight)
        contrast = contrastive_loss(output["t_a"], output["t_n"])
        sign = labels * 2.0 - 1.0
        margin = F.relu(args.margin - sign * output["score"]).mean()
        loss = args.lambda_bce * bce + args.lambda_contrast * contrast + args.lambda_margin * margin
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        n = labels.numel()
        running["bce"] += bce.item() * n
        running["contrast"] += contrast.item() * n
        running["margin"] += margin.item() * n
        running["samples"] += n
    if not running["samples"]:
        raise ValueError("Empty training set")
    return {k: running[k] / running["samples"] for k in ("bce", "contrast", "margin")}


@torch.no_grad()
def predict(model, loader, processor, device):
    model.eval()
    result = []
    for batch in tqdm(loader, desc="Evaluate", leave=False):
        kwargs, labels = make_inputs(batch, processor, device, model.use_sare)
        output = model(**kwargs)
        pids, _, _, regions, _, _, _ = batch
        for pid, region, label, score, prob in zip(pids, regions, labels.tolist(),
                                                    output["score"].float().cpu().tolist(),
                                                    output["probability"].float().cpu().tolist()):
            result.append({"patient_id": pid, "region_id": int(region),
                           "label": int(label), "score": score, "probability": prob})
    return result


def metrics(rows, threshold):
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    p = np.array([r["probability"] for r in rows], dtype=float)
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tn = int(np.sum((y == 0) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {"auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
            "ap": float(average_precision_score(y, p)) if any(y) else float("nan"),
            "precision": precision, "recall": recall, "specificity": spec,
            "f1": f1, "f2": f2, "accuracy": (tp + tn) / max(len(y), 1)}


def choose_threshold(rows, objective, min_specificity):
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    p = np.array([r["probability"] for r in rows], dtype=float)
    if not len(p):
        raise ValueError("Empty validation set")
    if len(set(y)) != 2:
        return 0.5, metrics(rows, 0.5)
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    tp = tpr * y.sum()
    fp = fpr * (len(y) - y.sum())
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(2 * precision * tpr, precision + tpr,
                   out=np.zeros_like(tpr), where=(precision + tpr) > 0)
    f2 = np.divide(5 * precision * tpr, 4 * precision + tpr,
                   out=np.zeros_like(tpr), where=(4 * precision + tpr) > 0)
    if objective == "recall_at_spec":
        eligible = (1 - fpr) >= min_specificity
        ranking = tpr + 1e-5 * f1
        ranking[~eligible] = -1
    else:
        ranking = f1 if objective == "f1" else f2
    index = int(np.argmax(ranking))
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(p.max(), np.inf))
    return threshold, metrics(rows, threshold)


def save_region_metrics(path, rows, threshold):
    by_region = defaultdict(list)
    for row in rows:
        by_region[int(row["region_id"])].append(row)
    report = {REGION_NAMES[r]: {"samples": len(by_region[r]),
                                **metrics(by_region[r], threshold)}
              for r in range(len(REGION_NAMES)) if by_region[r]}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_predictions(path, rows, threshold):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region_id", "label", "score", "probability", "prediction"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "prediction": int(row["probability"] >= threshold)})


def main():
    args = arguments()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Qwen3.5 ViT training requires a compatible GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    box_path = args.crop_boxes or args.data_dir / "crop_boxes.json"
    boxes = load_boxes(box_path)
    splits = read_splits(args.label_dir)
    datasets = {split: RegionDataset(labels, args.data_dir, boxes,
                                     args.oversample if split == "train" else 0)
                for split, labels in splits.items()}
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise ValueError("Train/validation sets must contain labeled scans with both views")
    print("Region samples:", {key: len(ds) for key, ds in datasets.items()})
    loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                             num_workers=args.num_workers, collate_fn=collate)
               for s, ds in datasets.items()}
    counts = Counter(label for _, _, label in datasets["train"].samples)
    if not counts[0] or not counts[1]:
        raise ValueError("Training requires both normal and abnormal labels")
    pos_value = (min(counts[0] / counts[1], args.max_pos_weight)
                 if args.pos_weight <= 0 else args.pos_weight)

    # Import only at execution time so data functions can be inspected without transformers.
    from transformers import AutoModel, AutoProcessor
    device = torch.device(args.device)
    qwen = AutoModel.from_pretrained(args.backbone_path, torch_dtype=torch.bfloat16,
                                     device_map="auto" if device.type == "cuda" else None,
                                     trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.backbone_path, trust_remote_code=True)
    vision = qwen.visual
    vision_device = next(vision.parameters()).device
    if vision_device != device:
        raise RuntimeError(f"Vision tower is on {vision_device}, requested {device}. Use a matching --device")
    fusion_layers = tuple(int(x.strip()) for x in args.fusion_layers.split(",") if x.strip())
    model = QwenViTVisualAD(
        vision_model=vision, use_sca=not args.no_sca, use_aaea=not args.no_aaea,
        use_acdf=not args.no_acdf, use_sare=not args.no_sare,
        sare_detach_ref=args.sare_detach_ref, fusion_layers=fusion_layers,
    ).to(device)

    # Default is full ViT fine-tuning (as in the original script).
    warmup = args.warmup_epochs > 0 and not args.freeze_vit and args.unfreeze_last_n != 0
    set_trainable_vision(vision, args.freeze_vit or warmup, args.unfreeze_last_n)
    optimizer = torch.optim.AdamW(param_groups(model, args), weight_decay=args.weight_decay)
    positive_weight = torch.tensor(float(pos_value), device=device)
    best_ap, best_path, stale = -float("inf"), args.output_dir / "best.pt", 0
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    for epoch in range(args.epochs):
        if warmup and epoch == args.warmup_epochs:
            set_trainable_vision(vision, False, args.unfreeze_last_n)
            optimizer = torch.optim.AdamW(param_groups(model, args), weight_decay=args.weight_decay)
        train_stats = train_epoch(model, loaders["train"], processor, optimizer, args,
                                  positive_weight, device, epoch)
        val_rows = predict(model, loaders["val"], processor, device)
        threshold, val_metrics = choose_threshold(val_rows, args.threshold_objective, args.minimum_specificity)
        score = val_metrics["ap"]
        if not np.isfinite(score):
            score = val_metrics["auc"]
        if not np.isfinite(score):
            score = val_metrics["f2"]
        print(f"Epoch {epoch:03d} | BCE {train_stats['bce']:.4f} | "
              f"AP {val_metrics['ap']:.4f} | AUROC {val_metrics['auc']:.4f} | "
              f"F1 {val_metrics['f1']:.4f} | threshold {threshold:.4f}")
        if score > best_ap:
            best_ap, stale = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "threshold": threshold,
                        "fusion_layers": fusion_layers, "validation_metrics": val_metrics,
                        "args": vars(args)}, best_path)
            save_predictions(args.output_dir / "val_predictions.csv", val_rows, threshold)
            save_region_metrics(args.output_dir / "val_region_metrics.json", val_rows, threshold)
            (args.output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if args.evaluate_test and len(datasets["test"]):
        test_rows = predict(model, loaders["test"], processor, device)
        test_metrics = metrics(test_rows, checkpoint["threshold"])
        save_predictions(args.output_dir / "test_predictions.csv", test_rows, checkpoint["threshold"])
        save_region_metrics(args.output_dir / "test_region_metrics.json", test_rows, checkpoint["threshold"])
        (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print("Test:", test_metrics)
    print(f"Saved checkpoint: {best_path}")


if __name__ == "__main__":
    main()
