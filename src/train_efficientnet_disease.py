#!/usr/bin/env python3
"""
Train EfficientNet-B3 Disease Classifier cho từng species
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .efficientnet_model import create_disease_model
from .preprocess import get_transforms
from .utils import set_global_seed, ensure_dir, save_class_names
from .data_utils import build_disease_mapping_for_species


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 Disease Classifier")
    parser.add_argument("--data_dir", type=str, default="data_masked", help="Dataset directory")
    parser.add_argument("--species", type=str, required=True, help="Target species")
    parser.add_argument("--output_dir", type=str, default="models/efficientnet_disease", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml"])
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--use_cutmix", action="store_true", help="Use CutMix augmentation")
    parser.add_argument("--cutmix_alpha", type=float, default=0.8, help="CutMix alpha")
    return parser.parse_args()


def select_device(device_arg: str) -> Tuple[str, torch.device]:
    """Select device"""
    if device_arg == "cpu":
        return "cpu", torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if device_arg == "dml":
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


def create_dataloaders(data_dir: Path, species: str, img_size: int, batch_size: int, 
                      num_workers: int, val_split: float = 0.2, seed: int = 42) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create data loaders for disease classification"""
    train_tfms, val_tfms = get_transforms(img_size)
    
    train_dir = data_dir / "train"
    val_dir_candidates = [data_dir / "val", data_dir / "valid", data_dir / "validation"]
    val_dir = next((p for p in val_dir_candidates if p.exists()), None)
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    if val_dir is not None and val_dir.exists():
        # Use existing validation split
        train_ds_full = datasets.ImageFolder(root=str(train_dir))
        val_ds_full = datasets.ImageFolder(root=str(val_dir))
        diseases, map_train = build_disease_mapping_for_species(train_ds_full.classes, species)
        _, map_val = build_disease_mapping_for_species(val_ds_full.classes, species)
        
        def filter_and_remap(samples, mapping):
            out = []
            for p, t in samples:
                idx = mapping.get(t, -1)
                if idx >= 0:
                    out.append((p, idx))
            return out
        
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        train_ds.samples = filter_and_remap(train_ds.samples, map_train)
        train_ds.targets = [t for _, t in train_ds.samples]
        
        val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
        val_ds.samples = filter_and_remap(val_ds.samples, map_val)
        val_ds.targets = [t for _, t in val_ds.samples]
        
        print(f"✅ Using existing validation directory: {val_dir}")
    else:
        # Auto-split from training data
        base_ds = datasets.ImageFolder(root=str(train_dir))
        diseases, mapping = build_disease_mapping_for_species(base_ds.classes, species)
        
        num_items = len(base_ds)
        rng = np.random.default_rng(seed)
        indices = np.arange(num_items)
        rng.shuffle(indices)
        val_size = int(max(1, round(val_split * num_items)))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        def remap_subset(transform, idxs):
            sub = datasets.ImageFolder(root=str(train_dir), transform=transform)
            sub.samples = [(p, mapping[t]) for p, t in sub.samples if mapping[t] >= 0]
            sub.targets = [t for _, t in sub.samples]
            return Subset(sub, idxs.tolist())
        
        train_ds = remap_subset(train_tfms, train_indices)
        val_ds = remap_subset(val_tfms, val_indices)
        
        print(f"✅ Auto-split: train={len(train_indices)}, val={len(val_indices)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, diseases


def apply_cutmix(images: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float):
    """Apply CutMix augmentation"""
    if alpha <= 0:
        return images, labels, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = images.size()
    index = torch.randperm(batch_size, device=images.device)
    
    # Generate cut coordinates
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)
    
    # Apply cut
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual cut area
    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    
    # Create soft labels
    y_a = torch.zeros(batch_size, num_classes, device=images.device)
    y_b = torch.zeros(batch_size, num_classes, device=images.device)
    y_a.scatter_(1, labels.view(-1, 1), 1.0)
    y_b.scatter_(1, labels[index].view(-1, 1), 1.0)
    mixed_labels = lam_adj * y_a + (1 - lam_adj) * y_b
    
    return mixed_images, mixed_labels, lam_adj


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, 
                   device: torch.device, num_classes: int, use_cutmix: bool, cutmix_alpha: float) -> Tuple[float, float]:
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply CutMix if enabled
        if use_cutmix and np.random.random() < 0.5:
            images, soft_labels, lam = apply_cutmix(images, labels, num_classes, cutmix_alpha)
            optimizer.zero_grad()
            logits = model(images)
            loss = -torch.sum(torch.log_softmax(logits, dim=1) * soft_labels, dim=1).mean()
        else:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct/total:.4f}"
        })
    
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Validation", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct/total:.4f}"
        })
    
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    
    all_preds = np.concatenate(all_preds, axis=0) if len(all_preds) > 0 else np.array([])
    all_labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 0 else np.array([])
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: List[str], 
                         output_dir: Path, species: str, epoch: int = None):
    """Save confusion matrix with numbers"""
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for y_true, y_pred in zip(labels, preds):
        if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
            cm[y_true, y_pred] += 1
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {species}{" - Epoch " + str(epoch) if epoch else ""}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    filename = f"confusion_matrix_{species}_epoch_{epoch}.png" if epoch else f"confusion_matrix_{species}_final.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    per_class_acc = []
    for i in range(num_classes):
        row_sum = cm[i].sum()
        acc_i = (cm[i, i] / row_sum) if row_sum > 0 else 0.0
        per_class_acc.append(acc_i)
    
    return cm, per_class_acc


def main():
    """Main training function"""
    args = parse_args()
    set_global_seed(args.seed)
    
    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    species_dir = output_dir / args.species
    ensure_dir(species_dir)
    
    device_name, device = select_device(args.device)
    print(f"🔧 Using device: {device_name}")
    print(f"🌿 Training disease classifier for: {args.species}")
    
    # Create data loaders
    print("📊 Loading dataset...")
    train_loader, val_loader, disease_list = create_dataloaders(
        data_dir, args.species, args.img_size, args.batch_size, 
        args.num_workers, args.val_split, args.seed
    )
    
    print(f"✅ Dataset loaded:")
    print(f"   - Diseases: {len(disease_list)}")
    print(f"   - Train samples: {len(train_loader.dataset)}")
    print(f"   - Val samples: {len(val_loader.dataset)}")
    
    # Save class names
    save_class_names(species_dir / "classes.json", disease_list)
    
    # Create model
    print("🏗️  Creating EfficientNet-B3 disease model...")
    model = create_disease_model(num_disease_classes=len(disease_list)).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": []
    }
    
    best_val_acc = 0.0
    best_state_dict = None
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   - CutMix: {args.use_cutmix}")
    print(f"   - Learning rate: {args.lr}")
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, len(disease_list),
            args.use_cutmix, args.cutmix_alpha
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"   📊 LR:    {current_lr:.6f}")
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"   🎉 New best validation accuracy: {best_val_acc:.4f}")
            
            # Save confusion matrix for best model
            cm, per_class_acc = save_confusion_matrix(
                val_labels, val_preds, disease_list, species_dir, args.species, epoch
            )
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {training_time/60:.1f} minutes")
    
    # Save final model
    final_model = {
        "state_dict": model.state_dict(),
        "classes": disease_list,
        "best_val_acc": best_val_acc,
        "training_history": history,
        "species": args.species
    }
    torch.save(final_model, species_dir / "model_final.pt")
    
    # Save best model
    if best_state_dict is not None:
        best_model = {
            "state_dict": best_state_dict,
            "classes": disease_list,
            "best_val_acc": best_val_acc,
            "training_history": history,
            "species": args.species
        }
        torch.save(best_model, species_dir / "model_best.pt")
        print(f"💾 Saved best model with accuracy: {best_val_acc:.4f}")
    
    # Save final confusion matrix
    final_cm, final_per_class_acc = save_confusion_matrix(
        val_labels, val_preds, disease_list, species_dir, args.species
    )
    
    # Save evaluation artifacts
    artifacts = {
        "species": args.species,
        "classes": disease_list,
        "confusion_matrix": final_cm.tolist(),
        "per_class_accuracy": final_per_class_acc,
        "overall_accuracy": float((val_preds == val_labels).mean()) if val_preds.size > 0 else 0.0,
        "training_history": history,
        "best_validation_accuracy": best_val_acc,
        "training_time_minutes": training_time / 60
    }
    
    with open(species_dir / "eval_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Disease training completed successfully!")
    print(f"📁 Results saved to: {species_dir}")
    print(f"📊 Best validation accuracy: {best_val_acc:.4f}")
    print(f"📊 Confusion matrix: {species_dir}/confusion_matrix_{args.species}_final.png")


if __name__ == "__main__":
    main()


