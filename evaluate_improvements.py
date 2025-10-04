#!/usr/bin/env python3
"""
Script đánh giá hiệu quả của pipeline cải tiến:
- So sánh Grad-CAM giữa model cũ và mới
- Test TTA với các threshold khác nhau
- Đánh giá performance trên ảnh thực tế
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Import từ project
from src.train import SimpleClassifier
from src.preprocess import get_transforms

def load_model(model_path: Path, device: torch.device):
    """Load model từ checkpoint"""
    ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt.get("classes", [])
    model = SimpleClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, classes

def generate_gradcam(model, image_tensor, class_idx, device):
    """Tạo Grad-CAM visualization"""
    # Hook để lấy feature maps từ layer cuối của backbone
    feature_maps = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    
    # Đăng ký hooks
    target_layer = model.backbone.features[-1]  # Last conv layer
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    try:
        # Forward pass
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_()
        logits = model(image_tensor)
        
        # Backward pass
        model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        
        # Tạo Grad-CAM
        if gradients is not None and feature_maps is not None:
            # Global average pooling của gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            # Weighted combination của feature maps
            cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
            cam = F.relu(cam)
            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        else:
            return None
            
    finally:
        # Cleanup hooks
        forward_handle.remove()
        backward_handle.remove()

def visualize_gradcam(image_path: Path, model, classes, device, output_dir: Path):
    """Tạo và lưu Grad-CAM visualization"""
    # Load và preprocess image
    _, val_tfms = get_transforms(224)
    
    with Image.open(image_path).convert("RGB") as img:
        image_tensor = val_tfms(img).unsqueeze(0)
        original_img = np.array(img)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate Grad-CAM
    cam = generate_gradcam(model, image_tensor, pred_class, device)
    
    if cam is not None:
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original\n{image_path.name}")
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        im1 = axes[1].imshow(cam_resized, cmap='jet', alpha=0.8)
        axes[1].set_title(f"Grad-CAM\nPred: {classes[pred_class]}\nConf: {confidence:.3f}")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(original_img)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"gradcam_{image_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "image": str(image_path),
            "prediction": classes[pred_class],
            "confidence": confidence,
            "gradcam_saved": str(output_path)
        }
    else:
        return {
            "image": str(image_path),
            "prediction": classes[pred_class],
            "confidence": confidence,
            "gradcam_saved": None,
            "error": "Failed to generate Grad-CAM"
        }

def test_tta_thresholds(model, image_path: Path, classes, device, thresholds: List[float]):
    """Test TTA với các threshold khác nhau"""
    _, val_tfms = get_transforms(224)
    
    results = {}
    
    with Image.open(image_path).convert("RGB") as img:
        x = val_tfms(img).unsqueeze(0).to(device)
    
    # No TTA
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0]
        conf, pred = torch.max(prob, dim=0)
        results["no_tta"] = {
            "prediction": classes[pred.item()],
            "confidence": conf.item()
        }
    
    # With TTA
    with torch.no_grad():
        # Simple TTA: flips
        xs = [x]
        xs.append(torch.flip(x, dims=[-1]))  # hflip
        xs.append(torch.flip(x, dims=[-2]))  # vflip
        
        logits_tta = torch.stack([model(v) for v in xs]).mean(dim=0)
        prob_tta = F.softmax(logits_tta, dim=1)[0]
        conf_tta, pred_tta = torch.max(prob_tta, dim=0)
        results["with_tta"] = {
            "prediction": classes[pred_tta.item()],
            "confidence": conf_tta.item()
        }
    
    # Test thresholds
    for threshold in thresholds:
        for mode in ["no_tta", "with_tta"]:
            conf = results[mode]["confidence"]
            key = f"{mode}_threshold_{threshold}"
            results[key] = {
                "prediction": results[mode]["prediction"] if conf >= threshold else "UNCERTAIN",
                "confidence": conf,
                "above_threshold": conf >= threshold
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Đánh giá hiệu quả pipeline cải tiến")
    parser.add_argument("--test_images", type=str, default="data/test", help="Thư mục ảnh test")
    parser.add_argument("--old_model", type=str, default="models/species/model_best.pt", help="Model cũ")
    parser.add_argument("--new_model", type=str, default="models/species_v2/model_best.pt", help="Model mới")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Thư mục lưu kết quả")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml"])
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Load models
    print("📥 Loading models...")
    old_model, old_classes = load_model(Path(args.old_model), device)
    new_model, new_classes = load_model(Path(args.new_model), device)
    
    # Find test images
    test_dir = Path(args.test_images)
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    test_images = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not test_images:
        print(f"❌ No test images found in {test_dir}")
        return
    
    print(f"🖼️  Found {len(test_images)} test images")
    
    # 1. Grad-CAM Comparison
    print("\n" + "="*60)
    print("1️⃣ GRAD-CAM COMPARISON")
    print("="*60)
    
    gradcam_results = []
    for img_path in test_images[:5]:  # Test 5 ảnh đầu
        print(f"🔍 Analyzing {img_path.name}...")
        
        # Old model
        old_result = visualize_gradcam(img_path, old_model, old_classes, device, output_dir / "old_model")
        old_result["model"] = "old"
        
        # New model
        new_result = visualize_gradcam(img_path, new_model, new_classes, device, output_dir / "new_model")
        new_result["model"] = "new"
        
        gradcam_results.extend([old_result, new_result])
    
    # 2. TTA and Threshold Testing
    print("\n" + "="*60)
    print("2️⃣ TTA AND THRESHOLD TESTING")
    print("="*60)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    tta_results = []
    
    for img_path in test_images[:3]:  # Test 3 ảnh đầu
        print(f"🧪 Testing TTA for {img_path.name}...")
        
        result = {
            "image": str(img_path),
            "old_model": test_tta_thresholds(old_model, img_path, old_classes, device, thresholds),
            "new_model": test_tta_thresholds(new_model, img_path, new_classes, device, thresholds)
        }
        tta_results.append(result)
    
    # Save results
    print("\n" + "="*60)
    print("3️⃣ SAVING RESULTS")
    print("="*60)
    
    # Save Grad-CAM results
    with open(output_dir / "gradcam_comparison.json", "w", encoding="utf-8") as f:
        json.dump(gradcam_results, f, ensure_ascii=False, indent=2)
    
    # Save TTA results
    with open(output_dir / "tta_threshold_results.json", "w", encoding="utf-8") as f:
        json.dump(tta_results, f, ensure_ascii=False, indent=2)
    
    # Summary report
    print("📊 SUMMARY REPORT")
    print("="*60)
    
    # Grad-CAM summary
    old_confidences = [r["confidence"] for r in gradcam_results if r["model"] == "old"]
    new_confidences = [r["confidence"] for r in gradcam_results if r["model"] == "new"]
    
    print(f"🎯 Grad-CAM Analysis:")
    print(f"   Old model avg confidence: {np.mean(old_confidences):.3f}")
    print(f"   New model avg confidence: {np.mean(new_confidences):.3f}")
    
    # TTA summary
    print(f"\n🔄 TTA Analysis:")
    for result in tta_results:
        img_name = Path(result["image"]).name
        old_conf = result["old_model"]["with_tta"]["confidence"]
        new_conf = result["new_model"]["with_tta"]["confidence"]
        print(f"   {img_name}: Old={old_conf:.3f}, New={new_conf:.3f}")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("🔍 Check Grad-CAM visualizations in:")
    print(f"   - {output_dir}/old_model/")
    print(f"   - {output_dir}/new_model/")

if __name__ == "__main__":
    main()


