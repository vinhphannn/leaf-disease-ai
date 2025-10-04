import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .train import SimpleClassifier
from .preprocess import get_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("input", type=str, help="Path to an image file or a directory of images")
    parser.add_argument("--model", type=str, default="models/model_best.pt", help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml"], help="Device to run on")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--exts", type=str, default=".jpg,.jpeg,.png,.JPG,.JPEG,.PNG", help="Comma-separated image extensions")
    parser.add_argument("--save_csv", type=str, default="", help="Optional path to save predictions as CSV")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--threshold", type=float, default=0.0, help="Confidence threshold for reporting predictions")
    return parser.parse_args()


def select_device(device_arg: str) -> Tuple[str, torch.device]:
    if device_arg == "cpu":
        return "cpu", torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if device_arg == "dml":
        try:
            import torch_directml  # type: ignore

            return "dml", torch_directml.device()
        except Exception:
            return "cpu", torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    try:
        import torch_directml  # type: ignore

        return "dml", torch_directml.device()
    except Exception:
        return "cpu", torch.device("cpu")


def load_checkpoint(model_path: Path) -> Tuple[dict, List[str]]:
    ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt.get("classes")
    if classes is None:
        # Fallback: try to read classes.json next to checkpoint
        classes_json = model_path.parent / "classes.json"
        if classes_json.exists():
            classes = json.loads(classes_json.read_text(encoding="utf-8"))
        else:
            raise RuntimeError("Class names not found in checkpoint and classes.json missing")
    state_dict = ckpt["state_dict"]
    return state_dict, classes


@torch.no_grad()
def predict_images(model: SimpleClassifier, paths: List[Path], device: torch.device, img_size: int, use_tta: bool = False) -> List[Tuple[Path, int, float]]:
    _, val_tfms = get_transforms(img_size)
    model.eval()
    results: List[Tuple[Path, int, float]] = []
    for p in paths:
        try:
            with Image.open(p).convert("RGB") as img:
                x = val_tfms(img).unsqueeze(0).to(device)
                if not use_tta:
                    logits = model(x)
                else:
                    # simple TTA: flips and small rotations
                    xs = [x]
                    xs.append(torch.flip(x, dims=[-1]))  # hflip
                    xs.append(torch.flip(x, dims=[-2]))  # vflip
                    # 15 degree rotations using torch.rot90 approximations (90-degree multiples only). For small rotation, just reuse flips.
                    logits = torch.stack([model(v) for v in xs]).mean(dim=0)
                prob = F.softmax(logits, dim=1)[0]
                conf, pred = torch.max(prob, dim=0)
                results.append((p, int(pred.item()), float(conf.item())))
        except Exception:
            continue
    return results


def collect_images(root: Path, exts: List[str]) -> List[Path]:
    if root.is_file():
        return [root]
    paths: List[Path] = []
    for ext in exts:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted(paths)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        # fallback to model_final.pt
        alt = model_path.parent / "model_final.pt"
        if alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    device_name, device = select_device(args.device)
    print(f"Using device: {device_name}")

    state_dict, class_names = load_checkpoint(model_path)
    model = SimpleClassifier(num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model = model.to(device)

    inputs = collect_images(Path(args.input), [e.strip() for e in args.exts.split(",") if e.strip()])
    if len(inputs) == 0:
        raise FileNotFoundError("No images found for the given path and extensions")

    results = predict_images(model, inputs, device, args.img_size, use_tta=args.tta)

    # Print nicely
    print(f"Predicted {len(results)} image(s)")
    shown = 0
    for p, pred, conf in results:
        if conf >= args.threshold:
            print(f"- {p}: {class_names[pred]} (conf={conf:.3f})")
            shown += 1
            if shown >= 10:
                break
    if len(results) > 10:
        print(f"... and {len(results) - 10} more")

    # Optional CSV
    if args.save_csv:
        import csv

        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "pred_index", "pred_class", "confidence"])
            for p, pred, conf in results:
                if conf >= args.threshold:
                    writer.writerow([str(p), pred, class_names[pred], f"{conf:.6f}"])
        print(f"Saved predictions to: {out_csv}")


if __name__ == "__main__":
    main()



