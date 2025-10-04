import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

from .preprocess import get_transforms
from .utils import set_global_seed, ensure_dir, save_class_names


class SimpleClassifier(nn.Module):
    """Mạng phân loại đơn giản dùng MobileNetV3 làm backbone.

    - Trích xuất đặc trưng bằng MobileNetV3 Small (pretrained ImageNet)
    - Thay head cuối bằng MLP nhỏ để phân loại num_classes
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Backbone nhẹ từ torchvision (MobileNetV3 Small, dùng weight pretrained)
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Lấy số chiều đầu vào của lớp phân loại gốc
        in_features = backbone.classifier[-1].in_features
        # Bỏ lớp phân loại gốc, giữ lại đặc trưng đầu ra
        backbone.classifier[-1] = nn.Identity()
        self.backbone = backbone
        # Head phân loại mới
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


def parse_args() -> argparse.Namespace:
    # Định nghĩa tham số dòng lệnh cho script train
    parser = argparse.ArgumentParser(description="Train leaf disease classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root containing train/ and val/")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models and logs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export_onnx", action="store_true", help="Export ONNX after training")
    parser.add_argument("--val_split", type=float, default=0.2, help="Tỷ lệ tách validation nếu thiếu data/val")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "dml"],
        help="Chọn thiết bị: auto/cpu/cuda/dml (DirectML cho GPU AMD/NPU trên Windows)",
    )
    return parser.parse_args()


def create_dataloaders(data_dir: Path, img_size: int, batch_size: int, num_workers: int, val_split: float = 0.2, seed: int = 42):
    """Tạo DataLoader cho train/val theo cấu trúc ImageFolder.

    Yêu cầu thư mục:
      data_dir/train/<class_name>/*.jpg
      data_dir/val/<class_name>/*.jpg
    """
    train_tfms, val_tfms = get_transforms(img_size)

    train_dir = data_dir / "train"
    # Hỗ trợ nhiều tên thư mục val phổ biến
    val_dir_candidates = [data_dir / "val", data_dir / "valid", data_dir / "validation"]
    val_dir = next((p for p in val_dir_candidates if p.exists()), None)

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing directory: {train_dir}")

    if val_dir is not None and val_dir.exists():
        # Trường hợp có sẵn thư mục val
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
        class_names = train_ds.classes
        print(f"Using validation directory: {val_dir}")
    else:
        # Không có thư mục val -> tự động tách từ train theo val_split
        base_ds = datasets.ImageFolder(root=str(train_dir))
        class_names = base_ds.classes
        num_items = len(base_ds)
        rng = np.random.default_rng(seed)
        indices = np.arange(num_items)
        rng.shuffle(indices)
        val_size = int(max(1, round(val_split * num_items))) if num_items > 1 else 0
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        # Tạo hai dataset với transform tương ứng và subset theo indices
        train_full = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
        val_full = datasets.ImageFolder(root=str(train_dir), transform=val_tfms)
        train_ds = Subset(train_full, train_indices.tolist())
        val_ds = Subset(val_full, val_indices.tolist())
        print(f"Auto-split train into train/val with ratio={val_split} -> train={len(train_indices)}, val={len(val_indices)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_names


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = torch.zeros((labels.size(0), num_classes), device=labels.device, dtype=torch.float32)
    y.scatter_(1, labels.view(-1, 1), 1.0)
    return y


def _mixup_data(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float = 0.2):
    if alpha <= 0:
        return x, _one_hot(y, num_classes), 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a = _one_hot(y, num_classes)
    y_b = _one_hot(y[index], num_classes)
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y, lam


def _cutmix_data(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float = 0.8):
    if alpha <= 0:
        return x, _one_hot(y, num_classes), 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size, device=x.device)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a = _one_hot(y, num_classes)
    y_b = _one_hot(y[index], num_classes)
    # Adjust lambda based on the actually replaced area
    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    mixed_y = lam_adj * y_a + (1 - lam_adj) * y_b
    return mixed_x, mixed_y, lam_adj


def _count_per_class(targets: list, num_classes: int):
    """Đếm số lượng ảnh theo lớp từ danh sách nhãn (class index)."""
    counts = np.bincount(np.array(targets, dtype=np.int64), minlength=num_classes)
    return counts.tolist()


def _sample_image_sizes(sample_paths: list, max_images: int = 64):
    """Lấy kích thước ảnh (width, height) của một mẫu nhỏ để thống kê nhanh."""
    sizes = []
    for path in sample_paths[:max_images]:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)  # (width, height)
        except Exception:
            continue
    return sizes


def _extract_samples_and_targets(ds, class_names):
    """Trả về danh sách (path, target) và danh sách targets, hỗ trợ ImageFolder hoặc Subset(ImageFolder)."""
    # Subset
    if isinstance(ds, Subset):
        base = ds.dataset
        indices = ds.indices
        if hasattr(base, "samples"):
            samples = [base.samples[i] for i in indices]
        else:
            # fallback: tự dựng từ imgs
            samples = [base.imgs[i] for i in indices] if hasattr(base, "imgs") else []
        targets = [t for _, t in samples]
        return samples, targets
    # ImageFolder
    if hasattr(ds, "samples"):
        samples = ds.samples
        targets = getattr(ds, "targets", [t for _, t in samples])
        return samples, targets
    return [], []


def compute_and_save_dataset_stats(train_ds, val_ds, class_names, out_dir: Path):
    """Tính toán thống kê cơ bản và lưu ra JSON: số ảnh mỗi lớp, mẫu kích thước ảnh."""
    num_classes = len(class_names)
    stats = {}

    # Train stats
    train_samples, train_targets = _extract_samples_and_targets(train_ds, class_names)
    train_counts = _count_per_class(train_targets, num_classes)
    train_sizes = _sample_image_sizes([p for p, _ in train_samples], max_images=128)
    stats["train"] = {"counts": train_counts, "sizes_sample": train_sizes}

    # Val stats
    val_samples, val_targets = _extract_samples_and_targets(val_ds, class_names)
    val_counts = _count_per_class(val_targets, num_classes)
    val_sizes = _sample_image_sizes([p for p, _ in val_samples], max_images=128)
    stats["val"] = {"counts": val_counts, "sizes_sample": val_sizes}

    # Map index -> class name
    stats["classes"] = list(class_names)

    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # In nhanh tổng quan
    print("Dataset stats (per split - counts by class index):")
    print("  classes:", class_names)
    print("  train counts:", train_counts)
    print("  val counts:", val_counts)


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes: int, use_mixup: bool = False, mixup_alpha: float = 0.2, use_cutmix: bool = False, cutmix_alpha: float = 0.8):
    """Train 1 epoch: forward + backward + update trọng số."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        targets_soft = None
        if use_cutmix:
            images, targets_soft, _ = _cutmix_data(images, labels, num_classes, alpha=cutmix_alpha)
        elif use_mixup:
            images, targets_soft, _ = _mixup_data(images, labels, num_classes, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if targets_soft is not None:
            loss = -torch.sum(torch.log_softmax(logits, dim=1) * targets_soft, dim=1).mean()
        else:
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Đánh giá trên tập val (không tính gradient)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in loader:
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
    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    # Ghép lại để dùng làm artifact nếu cần
    if len(all_preds) > 0:
        concat_preds = np.concatenate(all_preds, axis=0)
        concat_labels = np.concatenate(all_labels, axis=0)
    else:
        concat_preds = np.array([])
        concat_labels = np.array([])
    return epoch_loss, epoch_acc, concat_preds, concat_labels


def save_eval_artifacts(labels_np: np.ndarray, preds_np: np.ndarray, class_names, out_dir: Path):
    """Lưu confusion matrix, per-class accuracy và hình ảnh ma trận nhầm lẫn."""
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels_np, preds_np):
        if 0 <= y_true < num_classes and 0 <= y_pred < num_classes:
            cm[y_true, y_pred] += 1

    # Per-class accuracy = TP / (tổng theo hàng)
    per_class_acc = []
    for i in range(num_classes):
        row_sum = cm[i].sum()
        acc_i = (cm[i, i] / row_sum) if row_sum > 0 else 0.0
        per_class_acc.append(acc_i)

    artifacts = {
        "classes": list(class_names),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_acc,
        "overall_accuracy": float((preds_np == labels_np).mean()) if labels_np.size > 0 else 0.0,
    }
    with open(out_dir / "eval_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)

    # Lưu hình ảnh confusion matrix
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes), xticklabels=class_names, yticklabels=class_names, ylabel="True", xlabel="Pred")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Could not save confusion matrix image: {e}")


def main():
    """Luồng chính: parse args -> dựng dữ liệu -> model -> train -> lưu."""
    args = parse_args()
    set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Chọn thiết bị theo tham số --device
    selected_device = None
    if args.device == "cpu":
        selected_device = ("cpu", torch.device("cpu"))
    elif args.device == "cuda":
        if torch.cuda.is_available():
            selected_device = ("cuda", torch.device("cuda"))
        else:
            print("CUDA không khả dụng, fallback về CPU.")
            selected_device = ("cpu", torch.device("cpu"))
    elif args.device == "dml":
        try:
            import torch_directml
            dml_device = torch_directml.device()
            selected_device = ("dml", dml_device)
        except Exception as e:
            print(f"DirectML không khả dụng ({e}), fallback về CPU.")
            selected_device = ("cpu", torch.device("cpu"))
    else:  # auto
        if torch.cuda.is_available():
            selected_device = ("cuda", torch.device("cuda"))
        else:
            try:
                import torch_directml
                dml_device = torch_directml.device()
                selected_device = ("dml", dml_device)
            except Exception:
                selected_device = ("cpu", torch.device("cpu"))

    device_name, device = selected_device
    print(f"Using device: {device_name}")

    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Mixup/CutMix flags (optional via env to avoid arg breaking changes)
    use_mixup = bool(int(os.environ.get("MIXUP", "0")))
    use_cutmix = bool(int(os.environ.get("CUTMIX", "0")))
    mixup_alpha = float(os.environ.get("MIXUP_ALPHA", "0.2"))
    cutmix_alpha = float(os.environ.get("CUTMIX_ALPHA", "0.8"))

    num_classes = len(class_names)
    save_class_names(output_dir / "classes.json", class_names)

    model = SimpleClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_state_dict = None

    # Tính và lưu thống kê dataset trước khi train
    compute_and_save_dataset_stats(train_loader.dataset, val_loader.dataset, class_names, output_dir)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes=num_classes,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            use_cutmix=use_cutmix,
            cutmix_alpha=cutmix_alpha,
        )
        val_loss, val_acc, val_preds_np, val_labels_np = validate(model, val_loader, criterion, device)
        print(f"  train: loss={train_loss:.4f} acc={train_acc:.4f} | val: loss={val_loss:.4f} acc={val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Lưu model cuối cùng và model tốt nhất trên val
    final_path = output_dir / "model_final.pt"
    torch.save({"state_dict": model.state_dict(), "classes": class_names}, final_path)
    print(f"Saved final model to: {final_path}")

    if best_state_dict is not None:
        best_path = output_dir / "model_best.pt"
        torch.save({"state_dict": best_state_dict, "classes": class_names, "best_val_acc": best_val_acc}, best_path)
        print(f"Saved best model to: {best_path} (acc={best_val_acc:.4f})")

    # Lưu lịch sử loss/acc để phân tích sau
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Lưu artifact đánh giá (confusion matrix, per-class accuracy)
    try:
        if val_labels_np.size > 0 and val_preds_np.size > 0:
            save_eval_artifacts(val_labels_np, val_preds_np, class_names, output_dir)
            print("Saved eval artifacts (confusion matrix, per-class acc)")
    except Exception as e:
        print(f"Could not save eval artifacts: {e}")

    # Tùy chọn: xuất ONNX để suy luận đa nền tảng
    if args.export_onnx:
        onnx_path = output_dir / "model.onnx"
        model.eval()
        # Đảm bảo export an toàn cả khi đang ở DML: export trên CPU
        model_cpu = SimpleClassifier(num_classes=num_classes)
        model_cpu.load_state_dict(model.state_dict() if device_name != "dml" else {k: v.cpu() for k, v in model.state_dict().items()})
        model_cpu.eval()
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=torch.device("cpu"))
        torch.onnx.export(
            model_cpu,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"Exported ONNX to: {onnx_path}")


if __name__ == "__main__":
    main()


