import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from .preprocess import get_transforms
from .train import save_eval_artifacts
import torch.nn as nn
from torchvision import models
from .data_utils import build_species_mapping, build_disease_mapping_for_species


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an existing checkpoint and save artifacts")
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml"])
    p.add_argument("--species", type=str, default="", help="If evaluating a disease model, specify species name")
    p.add_argument("--output_dir", type=str, default="", help="Where to save artifacts; default is ckpt dir")
    p.add_argument("--save_feats", action="store_true", help="Also extract and save per-image features and per-class prototypes")
    return p.parse_args()


def select_device(name: str):
    if name == "cpu":
        return "cpu", torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if name == "dml":
        try:
            import torch_directml

            return "dml", torch_directml.device()
        except Exception:
            return "cpu", torch.device("cpu")
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    try:
        import torch_directml

        return "dml", torch_directml.device()
    except Exception:
        return "cpu", torch.device("cpu")


def _val_dir(data_dir: Path) -> Path:
    for n in ["val", "valid", "validation"]:
        cand = data_dir / n
        if cand.exists():
            return cand
    raise FileNotFoundError("Validation directory not found (val/valid/validation)")


def create_val_loader_for_species(data_dir: Path, img_size: int, batch_size: int = 64, num_workers: int = 8):
    _, val_tfms = get_transforms(img_size)
    val_ds = datasets.ImageFolder(root=str(_val_dir(data_dir)), transform=val_tfms)
    species_list, class_to_species_idx = build_species_mapping(val_ds.classes)
    # Remap targets to species index
    val_ds.samples = [(p, class_to_species_idx[t]) for p, t in val_ds.samples]
    val_ds.targets = [t for _, t in val_ds.samples]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return val_loader, species_list


def create_val_loader_for_disease(data_dir: Path, species: str, img_size: int, batch_size: int = 64, num_workers: int = 8):
    _, val_tfms = get_transforms(img_size)
    base = datasets.ImageFolder(root=str(_val_dir(data_dir)))
    diseases, mapping = build_disease_mapping_for_species(base.classes, species)

    # Recreate with transform and filter to this species
    val_ds = datasets.ImageFolder(root=str(_val_dir(data_dir)), transform=val_tfms)
    val_ds.samples = [(p, mapping[t]) for p, t in val_ds.samples if mapping[t] >= 0]
    val_ds.targets = [t for _, t in val_ds.samples]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return val_loader, diseases


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    import numpy as np
    val_loss = running / max(1, total)
    val_acc = correct / max(1, total)
    val_preds_np = np.concatenate(all_preds, axis=0) if len(all_preds) > 0 else np.array([])
    val_labels_np = np.concatenate(all_labels, axis=0) if len(all_labels) > 0 else np.array([])
    return val_loss, val_acc, val_preds_np, val_labels_np


def build_mnv3_linear(num_classes: int) -> nn.Module:
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Identity()
    return nn.Sequential(backbone, nn.Linear(in_features, num_classes))


def _save_confusion_matrix_large(labels_np, preds_np, class_names, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        num_classes = len(class_names)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for y_true, y_pred in zip(labels_np, preds_np):
            if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
                cm[y_true, y_pred] += 1
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes), xticklabels=class_names, yticklabels=class_names, ylabel="True", xlabel="Pred")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Add cell numbers
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
        ax.set_title("Confusion Matrix (Large)")
        fig.tight_layout()
        plt.savefig(out_dir / "confusion_matrix_large.png", dpi=220)
        plt.close(fig)
    except Exception:
        pass


def main():
    args = parse_args()
    device_name, device = select_device(args.device)
    print(f"Using device: {device_name}")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes")
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes'")

    # Auto-detect model type: species vs disease (based on provided --species)
    is_disease = bool(args.species)
    if is_disease:
        val_loader, class_names = create_val_loader_for_disease(Path(args.data_dir), args.species, args.img_size)
    else:
        val_loader, class_names = create_val_loader_for_species(Path(args.data_dir), args.img_size)

    # Build MobileNetV3+Linear head to match training scripts
    model = build_mnv3_linear(num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])  # assumes same architecture
    model = model.to(device)

    val_loss, val_acc, preds_np, labels_np = evaluate_model(model, val_loader, device)
    print(f"Eval: loss={val_loss:.4f} acc={val_acc:.4f}")

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    artifacts = {
        "eval_loss": val_loss,
        "eval_acc": val_acc,
        "classes": class_names,
    }
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"val_loss": [val_loss], "val_acc": [val_acc]}, f, ensure_ascii=False, indent=2)
    if labels_np.size > 0 and preds_np.size > 0:
        save_eval_artifacts(labels_np, preds_np, class_names, out_dir)
        _save_confusion_matrix_large(labels_np, preds_np, class_names, out_dir)
        print("Saved confusion_matrix.png, confusion_matrix_large.png and eval_artifacts.json")

    # Optional: extract features (embeddings) and per-class prototypes
    if args.save_feats:
        model.eval()
        feats_list = []
        lbls_list = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                if isinstance(model, nn.Sequential):
                    backbone = model[0]
                    feats = backbone(imgs)
                else:
                    feats = getattr(model, "backbone")(imgs)
                feats_list.append(feats.cpu())
                lbls_list.append(labels.cpu())
        feats_all = torch.cat(feats_list, dim=0).numpy()
        lbls_all = torch.cat(lbls_list, dim=0).numpy()
        prototypes = []
        counts = []
        for ci in range(len(class_names)):
            mask = (lbls_all == ci)
            if mask.any():
                prototypes.append(feats_all[mask].mean(axis=0))
                counts.append(int(mask.sum()))
            else:
                prototypes.append(np.zeros((feats_all.shape[1],), dtype=feats_all.dtype))
                counts.append(0)
        np.save(out_dir / "features.npy", feats_all)
        np.save(out_dir / "labels.npy", lbls_all)
        np.savez(out_dir / "prototypes.npz", prototypes=np.stack(prototypes, axis=0), counts=np.array(counts, dtype=np.int64))
        with open(out_dir / "prototypes_meta.json", "w", encoding="utf-8") as f:
            json.dump({"classes": class_names, "counts": counts}, f, ensure_ascii=False, indent=2)
        print("Saved features.npy, labels.npy, prototypes.npz, prototypes_meta.json")


if __name__ == "__main__":
    main()


