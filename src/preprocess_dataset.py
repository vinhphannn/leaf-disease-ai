import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def segment_and_crop(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (masked_rgb, cropped_rgb) using HSV mask → dilate → GrabCut with mask.

    - Dilation nới rộng biên để tránh cắt lẹm vào lá
    - GrabCut khởi tạo bằng mask để bám sát mép lá hơn so với rect
    """
    h0, w0 = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 30, 20])
    upper = np.array([100, 255, 255])
    mask0 = cv2.inRange(hsv, lower, upper)
    mask0 = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask0 = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Nới rộng biên để tránh cắt lẹm
    mask_dil = cv2.dilate(mask0, np.ones((7, 7), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, img
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = int(0.06 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w0, x + w + pad), min(h0, y + h + pad)
    roi = img[y0:y1, x0:x1].copy()

    # GrabCut with init mask (probable FG from dilated HSV mask)
    init_mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
    msub = mask_dil[y0:y1, x0:x1]
    init_mask[msub > 0] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, init_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        fg = np.where((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    except Exception:
        fg = (cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)[:, :, 1] > 40).astype("uint8")
    masked = (roi * fg[..., None]).astype(np.uint8)
    return masked, roi


def pad_to_square(img: np.ndarray, pad_value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    out = np.full((size, size, 3), pad_value, dtype=img.dtype)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def add_edge_overlay(img: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(img, 1.0, edges_rgb, alpha, 0)
    return overlay


def white_balance_grayworld(img: np.ndarray) -> np.ndarray:
    # Gray-world assumption: scale channels to equalize mean
    mean = img.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1e-6
    gain = mean.mean() / mean
    out = np.clip(img.astype(np.float32) * gain[None, None, :], 0, 255).astype(np.uint8)
    return out


def apply_clahe_v(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    # CLAHE nhẹ trên kênh V để tăng tương phản mà không đổi màu mạnh
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def deglare_specular(img: np.ndarray) -> np.ndarray:
    # Phát hiện vùng chói: S thấp, V rất cao, rồi inpaint nhẹ
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    glare = ((s < 35) & (v > 220)).astype(np.uint8) * 255
    if glare.sum() == 0:
        return img
    kernel = np.ones((3, 3), np.uint8)
    glare = cv2.dilate(glare, kernel, iterations=1)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fixed = cv2.inpaint(bgr, glare, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)


def process_split(src_dir: Path, dst_dir: Path, img_size: int, add_edges: bool, do_wb: bool, do_deglare: bool, do_clahe: bool) -> None:
    for cls_dir in sorted([p for p in src_dir.glob("*") if p.is_dir()]):
        out_cls = dst_dir / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        for img_path in cls_dir.glob("*.jp*g"):
            try:
                pil = Image.open(img_path).convert("RGB")
                img = np.array(pil)
                masked, _ = segment_and_crop(img)
                if do_deglare:
                    masked = deglare_specular(masked)
                if do_wb:
                    masked = white_balance_grayworld(masked)
                if do_clahe:
                    masked = apply_clahe_v(masked)
                sq = pad_to_square(masked, pad_value=0)
                if add_edges:
                    sq = add_edge_overlay(sq)
                resized = cv2.resize(sq, (img_size, img_size), interpolation=cv2.INTER_AREA)
                Image.fromarray(resized).save(out_cls / img_path.name)
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser(description="Preprocess dataset: background removal + pad + resize")
    ap.add_argument("--src", type=str, default="data", help="Source dataset root containing train/ and valid/")
    ap.add_argument("--dst", type=str, default="data_masked", help="Destination root for masked dataset")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--add_edges", action="store_true", help="Overlay edges to emphasize leaf shape")
    ap.add_argument("--wb", action="store_true", help="Apply gray-world white balance to stabilize color")
    ap.add_argument("--deglare", action="store_true", help="Reduce specular highlights via inpainting")
    ap.add_argument("--clahe", action="store_true", help="Light CLAHE on V channel for contrast")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    for split in ["train", "valid", "val", "validation"]:
        src_split = src / split
        if src_split.exists():
            dst_split = dst / ("valid" if split in ["val", "validation"] else split)
            process_split(src_split, dst_split, args.img_size, args.add_edges, args.wb, args.deglare, args.clahe)
            print(f"Processed {split} -> {dst_split}")


if __name__ == "__main__":
    main()


