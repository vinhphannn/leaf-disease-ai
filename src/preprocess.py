from typing import Tuple
from torchvision import transforms


def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Khởi tạo tập biến đổi ảnh cho train/val.

    - Train: có augment nhẹ (flip, rotate, color jitter) để tăng khả năng tổng quát
    - Val: chỉ resize + normalize để đánh giá ổn định
    """
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tfms, val_tfms


