import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Cố định seed để tái lập kết quả giữa các lần chạy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Đảm bảo hành vi xác định với CUDNN (đổi lại có thể chậm hơn chút)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    """Tạo thư mục nếu chưa tồn tại (bao gồm cả thư mục cha)."""
    path.mkdir(parents=True, exist_ok=True)


def save_class_names(path: Path, class_names: Iterable[str]) -> None:
    """Lưu danh sách tên lớp ra file JSON để dùng khi suy luận."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, ensure_ascii=False, indent=2)


