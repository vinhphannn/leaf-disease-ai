#!/usr/bin/env python3
"""
Train EfficientNet-B3 Species Classifier với đầy đủ tính năng
- Progress tracking với tqdm
- Confusion matrix với số liệu
- Eval artifacts
- Model saving
"""

import argparse
import json
import os
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

from .efficientnet_model import create_species_model
from .preprocess import get_transforms
from .utils import set_global_seed, ensure_dir, save_class_names
from .data_utils import build_species_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 Species Classifier")
    parser.add_argument("--data_dir", type=str, default="data_masked", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="models/efficientnet_species", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml"])
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--use_mixup", action="store_true", help="Use Mixup augmentation")
    parser.add_argument("--use_cutmix", action="store_true", help="Use CutMix augmentation")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha")
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
            dml_device = torch_directml.device()
            print(f"🔧 DML device initialized: {dml_device}")
            return "dml", dml_device
        except Exception as e:
            print(f"⚠️  DML failed: {e}, falling back to CPU")
            return "cpu", torch.device("cpu")
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"🔧 DML device initialized: {dml_device}")
        return "dml", dml_device
    except Exception as e:
        print(f"⚠️  DML failed: {e}, falling back to CPU")
        return "cpu", torch.device("cpu")


def create_dataloaders(data_dir: Path, img_size: int, batch_size: int, num_workers: int, 
                      val_split: float = 0.2, seed: int = 42) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create data loaders"""
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
        species_list, class_to_species_idx = build_species_mapping(train_ds_full.classes)
        
        def map_targets(samples, mapping):
            return [(p, mapping[t]) for p, t in samples]
        
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        train_ds.samples = map_targets(train_ds.samples, class_to_species_idx)
        train_ds.targets = [t for _, t in train_ds.samples]
        
        val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
        _, class_to_species_idx_val = build_species_mapping(val_ds_full.classes)
        val_ds.samples = map_targets(val_ds.samples, class_to_species_idx_val)
        val_ds.targets = [t for _, t in val_ds.samples]
        
        print(f"✅ Using existing validation directory: {val_dir}")
    else:
        # Auto-split from training data
        base_ds = datasets.ImageFolder(root=str(train_dir))
        species_list, class_to_species_idx = build_species_mapping(base_ds.classes)
        
        num_items = len(base_ds)
        rng = np.random.default_rng(seed)
        indices = np.arange(num_items)
        rng.shuffle(indices)
        val_size = int(max(1, round(val_split * num_items)))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        def remap_subset(ds, transform, idxs):
            sub = datasets.ImageFolder(root=str(train_dir), transform=transform)
            sub.samples = [(p, class_to_species_idx[t]) for p, t in sub.samples]
            sub.targets = [t for _, t in sub.samples]
            return Subset(sub, idxs.tolist())
        
        train_ds = remap_subset(base_ds, train_tfms, train_indices)
        val_ds = remap_subset(base_ds, val_tfms, val_indices)
        
        print(f"✅ Auto-split: train={len(train_indices)}, val={len(val_indices)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, species_list


def apply_mixup_cutmix(images: torch.Tensor, labels: torch.Tensor, num_classes: int, 
                      use_mixup: bool, use_cutmix: bool, mixup_alpha: float, cutmix_alpha: float):
    """Apply Mixup or CutMix augmentation"""
    if use_cutmix and np.random.random() < 0.5:
        return apply_cutmix(images, labels, num_classes, cutmix_alpha)
    elif use_mixup and np.random.random() < 0.5:
        return apply_mixup(images, labels, num_classes, mixup_alpha)
    else:
        # Return original labels as one-hot for consistency
        batch_size = labels.size(0)
        one_hot_labels = torch.zeros(batch_size, num_classes, device=labels.device)
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1.0)
        return images, one_hot_labels, 1.0


def apply_mixup(images: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float):
    """Apply Mixup augmentation"""
    if alpha <= 0:
        return images, labels, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    
    # Create soft labels
    y_a = torch.zeros(batch_size, num_classes, device=images.device)
    y_b = torch.zeros(batch_size, num_classes, device=images.device)
    y_a.scatter_(1, labels.view(-1, 1), 1.0)
    y_b.scatter_(1, labels[index].view(-1, 1), 1.0)
    mixed_labels = lam * y_a + (1 - lam) * y_b
    
    return mixed_images, mixed_labels, lam


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
                   device: torch.device, num_classes: int, use_mixup: bool, use_cutmix: bool, 
                   mixup_alpha: float, cutmix_alpha: float) -> Tuple[float, float]:
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply augmentation
        if use_mixup or use_cutmix:
            images, soft_labels, lam = apply_mixup_cutmix(
                images, labels, num_classes, use_mixup, use_cutmix, mixup_alpha, cutmix_alpha
            )
        
        optimizer.zero_grad()
        logits = model(images)
        
        if use_mixup or use_cutmix:
            # Use soft labels for loss - ensure dimensions match
            if soft_labels.shape[1] != logits.shape[1]:
                print(f"⚠️  Dimension mismatch: soft_labels {soft_labels.shape} vs logits {logits.shape}")
                # Fallback to regular loss
                loss = criterion(logits, labels)
            else:
                loss = -torch.sum(torch.log_softmax(logits, dim=1) * soft_labels, dim=1).mean()
        else:
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
                         output_dir: Path, epoch: int = None):
    """Save confusion matrix with numbers"""
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for y_true, y_pred in zip(labels, preds):
        if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
            cm[y_true, y_pred] += 1
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix{" - Epoch " + str(epoch) if epoch else ""}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    filename = f"confusion_matrix_epoch_{epoch}.png" if epoch else "confusion_matrix_final.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    per_class_acc = []
    for i in range(num_classes):
        row_sum = cm[i].sum()
        acc_i = (cm[i, i] / row_sum) if row_sum > 0 else 0.0
        per_class_acc.append(acc_i)
    
    return cm, per_class_acc


def save_training_plots(history: Dict, output_dir: Path):
    """Save training plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function"""
    args = parse_args()
    set_global_seed(args.seed)
    
    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    device_name, device = select_device(args.device)
    print(f"🔧 Using device: {device_name}")
    
    # Create data loaders
    print("📊 Loading dataset...")
    train_loader, val_loader, species_list = create_dataloaders(
        data_dir, args.img_size, args.batch_size, args.num_workers, args.val_split, args.seed
    )
    
    print(f"✅ Dataset loaded:")
    print(f"   - Species: {len(species_list)}")
    print(f"   - Train samples: {len(train_loader.dataset)}")
    print(f"   - Val samples: {len(val_loader.dataset)}")
    
    # Save class names
    save_class_names(output_dir / "classes.json", species_list)
    
    # Create model
    print("🏗️  Creating EfficientNet-B3 model...")
    model = create_species_model(num_classes=len(species_list)).to(device)
    
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
    print(f"   - Mixup: {args.use_mixup}")
    print(f"   - CutMix: {args.use_cutmix}")
    print(f"   - Learning rate: {args.lr}")
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, len(species_list),
            args.use_mixup, args.use_cutmix, args.mixup_alpha, args.cutmix_alpha
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
                val_labels, val_preds, species_list, output_dir, epoch
            )
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": species_list,
                "history": history
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {training_time/60:.1f} minutes")
    
    # Save final model
    final_model = {
        "state_dict": model.state_dict(),
        "classes": species_list,
        "best_val_acc": best_val_acc,
        "training_history": history,
        "model_architecture": model
    }
    torch.save(final_model, output_dir / "model_final.pt")
    
    # Save best model
    if best_state_dict is not None:
        best_model = {
            "state_dict": best_state_dict,
            "classes": species_list,
            "best_val_acc": best_val_acc,
            "training_history": history
        }
        torch.save(best_model, output_dir / "model_best.pt")
        print(f"💾 Saved best model with accuracy: {best_val_acc:.4f}")
    
    # Save final confusion matrix
    final_cm, final_per_class_acc = save_confusion_matrix(
        val_labels, val_preds, species_list, output_dir
    )
    
    # Save training plots
    save_training_plots(history, output_dir)
    
    # Save evaluation artifacts
    artifacts = {
        "classes": species_list,
        "confusion_matrix": final_cm.tolist(),
        "per_class_accuracy": final_per_class_acc,
        "overall_accuracy": float((val_preds == val_labels).mean()) if val_preds.size > 0 else 0.0,
        "training_history": history,
        "best_validation_accuracy": best_val_acc,
        "training_time_minutes": training_time / 60
    }
    
    with open(output_dir / "eval_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Best validation accuracy: {best_val_acc:.4f}")
    print(f"📈 Training plots: {output_dir}/training_plots.png")
    print(f"📊 Confusion matrix: {output_dir}/confusion_matrix_final.png")


if __name__ == "__main__":
    main()
