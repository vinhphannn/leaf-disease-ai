import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from tqdm import tqdm

from .preprocess import get_transforms
from .utils import set_global_seed, ensure_dir, save_class_names
from .train import save_eval_artifacts
from .data_utils import build_species_mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train species classifier")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="models/species")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml"])
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


def create_species_loaders(data_dir: Path, img_size: int, batch_size: int, num_workers: int):
    train_tfms, val_tfms = get_transforms(img_size)
    train_dir = data_dir / "train"
    val_dir_candidates = [data_dir / "val", data_dir / "valid", data_dir / "validation"]
    val_dir = next((p for p in val_dir_candidates if p.exists()), None)

    if not train_dir.exists():
        raise FileNotFoundError(train_dir)

    if val_dir is not None and val_dir.exists():
        train_ds_full = datasets.ImageFolder(root=str(train_dir))
        val_ds_full = datasets.ImageFolder(root=str(val_dir))
        species_list, class_to_species_idx = build_species_mapping(train_ds_full.classes)

        def map_targets(samples, mapping):
            return [(p, mapping[t]) for p, t in samples]

        # Remap by replacing samples/targets
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        train_ds.samples = map_targets(train_ds.samples, class_to_species_idx)
        train_ds.targets = [t for _, t in train_ds.samples]

        val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
        # Build mapping for val too (classes identical order assumption)
        _, class_to_species_idx_val = build_species_mapping(val_ds_full.classes)
        val_ds.samples = map_targets(val_ds.samples, class_to_species_idx_val)
        val_ds.targets = [t for _, t in val_ds.samples]
    else:
        base_ds = datasets.ImageFolder(root=str(train_dir))
        species_list, class_to_species_idx = build_species_mapping(base_ds.classes)
        num_items = len(base_ds)
        rng = np.random.default_rng(42)
        indices = np.arange(num_items)
        rng.shuffle(indices)
        val_size = int(max(1, round(0.2 * num_items)))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        def remap_subset(ds, transform, idxs):
            sub = datasets.ImageFolder(root=str(train_dir), transform=transform)
            # apply remap then subset
            sub.samples = [(p, class_to_species_idx[t]) for p, t in sub.samples]
            sub.targets = [t for _, t in sub.samples]
            return Subset(sub, idxs.tolist())

        train_ds = remap_subset(base_ds, train_tfms, train_indices)
        val_ds = remap_subset(base_ds, val_tfms, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, species_list


def main():
    args = parse_args()
    set_global_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    device_name, device = select_device(args.device)
    print(f"Using device: {device_name}")

    train_loader, val_loader, species_list = create_species_loaders(data_dir, args.img_size, args.batch_size, args.num_workers)
    save_class_names(out_dir / "classes.json", species_list)

    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Identity()
    model = nn.Sequential(
        backbone,
        nn.Linear(in_features, len(species_list)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Mixup/CutMix via env flags to avoid CLI breaking
    import os
    use_mixup = bool(int(os.environ.get("MIXUP", "0")))
    use_cutmix = bool(int(os.environ.get("CUTMIX", "0")))
    mixup_alpha = float(os.environ.get("MIXUP_ALPHA", "0.2"))
    cutmix_alpha = float(os.environ.get("CUTMIX_ALPHA", "0.8"))

    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    def one_hot(y: torch.Tensor, n: int) -> torch.Tensor:
        z = torch.zeros((y.size(0), n), device=y.device, dtype=torch.float32)
        z.scatter_(1, y.view(-1, 1), 1.0)
        return z

    def mixup(x, y, n, alpha=0.2):
        if alpha <= 0:
            return x, one_hot(y, n)
        import numpy as _np
        lam = _np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], lam * one_hot(y, n) + (1 - lam) * one_hot(y[idx], n)

    def cutmix(x, y, n, alpha=0.8):
        if alpha <= 0:
            return x, one_hot(y, n)
        import numpy as _np
        lam = _np.random.beta(alpha, alpha)
        b, _, h, w = x.size()
        idx = torch.randperm(b, device=x.device)
        cx = _np.random.randint(w)
        cy = _np.random.randint(h)
        cw = int(w * (_np.sqrt(1 - lam)))
        ch = int(h * (_np.sqrt(1 - lam)))
        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(w, cx + cw // 2)
        y2 = min(h, cy + ch // 2)
        xm = x.clone()
        xm[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        return xm, lam_adj * one_hot(y, n) + (1 - lam_adj) * one_hot(y[idx], n)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Train [{epoch}/{args.epochs}]", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            targets_soft = None
            if use_cutmix:
                imgs, targets_soft = cutmix(imgs, labels, len(species_list), cutmix_alpha)
            elif use_mixup:
                imgs, targets_soft = mixup(imgs, labels, len(species_list), mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            if targets_soft is not None:
                loss = -torch.sum(torch.log_softmax(logits, dim=1) * targets_soft, dim=1).mean()
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = running / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        total = 0
        correct = 0
        running = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
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
        print(f"Epoch {epoch}/{args.epochs} - train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    torch.save({"state_dict": model.state_dict(), "classes": species_list}, out_dir / "model_final.pt")
    if best_state is not None:
        torch.save({"state_dict": best_state, "classes": species_list, "best_val_acc": best_acc}, out_dir / "model_best.pt")
        print(f"Saved best species model acc={best_acc:.4f}")

    # Save history and eval artifacts
    import json
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    try:
        if val_labels_np.size > 0 and val_preds_np.size > 0:
            save_eval_artifacts(val_labels_np, val_preds_np, species_list, out_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()


