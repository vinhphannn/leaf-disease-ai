#!/usr/bin/env python3
"""
Script so sánh performance giữa model cũ và mới:
- Accuracy trên validation set
- Performance trên ảnh thực tế
- Confusion matrix comparison
- Inference speed comparison
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_on_dataset(model, data_loader, device, model_name: str):
    """Đánh giá model trên dataset"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            logits = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "avg_inference_time_ms": avg_inference_time * 1000,
        "predictions": all_preds,
        "labels": all_labels
    }

def create_confusion_matrix_comparison(results_old: Dict, results_new: Dict, classes: List[str], output_dir: Path):
    """Tạo so sánh confusion matrix"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Old model confusion matrix
    cm_old = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(results_old["labels"], results_old["predictions"]):
        if 0 <= true_label < len(classes) and 0 <= pred_label < len(classes):
            cm_old[true_label, pred_label] += 1
    
    sns.heatmap(cm_old, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=classes, yticklabels=classes)
    ax1.set_title(f"Old Model (Acc: {results_old['accuracy']:.3f})")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    
    # New model confusion matrix
    cm_new = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(results_new["labels"], results_new["predictions"]):
        if 0 <= true_label < len(classes) and 0 <= pred_label < len(classes):
            cm_new[true_label, pred_label] += 1
    
    sns.heatmap(cm_new, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=classes, yticklabels=classes)
    ax2.set_title(f"New Model (Acc: {results_new['accuracy']:.3f})")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def test_on_real_images(model, classes, test_dir: Path, device, model_name: str):
    """Test trên ảnh thực tế"""
    _, val_tfms = get_transforms(224)
    
    if not test_dir.exists():
        return {"error": f"Test directory not found: {test_dir}"}
    
    test_images = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not test_images:
        return {"error": "No test images found"}
    
    results = []
    inference_times = []
    
    model.eval()
    with torch.no_grad():
        for img_path in test_images:
            try:
                with Image.open(img_path).convert("RGB") as img:
                    x = val_tfms(img).unsqueeze(0).to(device)
                    
                    start_time = time.time()
                    logits = model(x)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    prob = F.softmax(logits, dim=1)[0]
                    conf, pred = torch.max(prob, dim=0)
                    
                    results.append({
                        "image": str(img_path),
                        "prediction": classes[pred.item()],
                        "confidence": conf.item(),
                        "inference_time_ms": inference_time * 1000
                    })
            except Exception as e:
                results.append({
                    "image": str(img_path),
                    "error": str(e)
                })
    
    return {
        "model_name": model_name,
        "total_images": len(test_images),
        "successful_predictions": len([r for r in results if "error" not in r]),
        "avg_confidence": np.mean([r["confidence"] for r in results if "confidence" in r]),
        "avg_inference_time_ms": np.mean(inference_times) * 1000 if inference_times else 0,
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description="So sánh performance giữa model cũ và mới")
    parser.add_argument("--data_dir", type=str, default="data_masked", help="Dataset directory")
    parser.add_argument("--old_model", type=str, default="models/species/model_best.pt", help="Model cũ")
    parser.add_argument("--new_model", type=str, default="models/species_v2/model_best.pt", help="Model mới")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Thư mục ảnh test thực tế")
    parser.add_argument("--output_dir", type=str, default="model_comparison", help="Thư mục lưu kết quả")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml"])
    parser.add_argument("--batch_size", type=int, default=32)
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
    
    print(f"Old model classes: {len(old_classes)}")
    print(f"New model classes: {len(new_classes)}")
    
    # 1. Validation Set Evaluation
    print("\n" + "="*60)
    print("1️⃣ VALIDATION SET EVALUATION")
    print("="*60)
    
    # Create validation dataloader
    _, val_tfms = get_transforms(224)
    val_dir = Path(args.data_dir) / "val"
    if not val_dir.exists():
        val_dir = Path(args.data_dir) / "valid"
    
    if val_dir.exists():
        val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print("🔍 Evaluating old model on validation set...")
        old_val_results = evaluate_on_dataset(old_model, val_loader, device, "Old Model")
        
        print("🔍 Evaluating new model on validation set...")
        new_val_results = evaluate_on_dataset(new_model, val_loader, device, "New Model")
        
        # Create confusion matrix comparison
        create_confusion_matrix_comparison(old_val_results, new_val_results, val_dataset.classes, output_dir)
        
        print(f"📊 Validation Results:")
        print(f"   Old Model:  Acc={old_val_results['accuracy']:.3f}, Time={old_val_results['avg_inference_time_ms']:.1f}ms")
        print(f"   New Model:  Acc={new_val_results['accuracy']:.3f}, Time={new_val_results['avg_inference_time_ms']:.1f}ms")
        print(f"   Improvement: {new_val_results['accuracy'] - old_val_results['accuracy']:+.3f}")
    else:
        print("⚠️  No validation directory found, skipping validation evaluation")
        old_val_results = None
        new_val_results = None
    
    # 2. Real-world Image Testing
    print("\n" + "="*60)
    print("2️⃣ REAL-WORLD IMAGE TESTING")
    print("="*60)
    
    test_dir = Path(args.test_dir)
    if test_dir.exists():
        print("🔍 Testing old model on real images...")
        old_real_results = test_on_real_images(old_model, old_classes, test_dir, device, "Old Model")
        
        print("🔍 Testing new model on real images...")
        new_real_results = test_on_real_images(new_model, new_classes, test_dir, device, "New Model")
        
        print(f"📊 Real-world Results:")
        if "error" not in old_real_results:
            print(f"   Old Model:  Conf={old_real_results['avg_confidence']:.3f}, Time={old_real_results['avg_inference_time_ms']:.1f}ms")
        if "error" not in new_real_results:
            print(f"   New Model:  Conf={new_real_results['avg_confidence']:.3f}, Time={new_real_results['avg_inference_time_ms']:.1f}ms")
    else:
        print("⚠️  No test directory found, skipping real-world testing")
        old_real_results = None
        new_real_results = None
    
    # 3. Save Results
    print("\n" + "="*60)
    print("3️⃣ SAVING RESULTS")
    print("="*60)
    
    comparison_results = {
        "validation_results": {
            "old_model": old_val_results,
            "new_model": new_val_results
        },
        "real_world_results": {
            "old_model": old_real_results,
            "new_model": new_real_results
        },
        "summary": {
            "validation_accuracy_improvement": (new_val_results['accuracy'] - old_val_results['accuracy']) if old_val_results and new_val_results else None,
            "inference_speed_improvement": (old_val_results['avg_inference_time_ms'] - new_val_results['avg_inference_time_ms']) if old_val_results and new_val_results else None
        }
    }
    
    with open(output_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    # 4. Summary Report
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    if old_val_results and new_val_results:
        acc_improvement = new_val_results['accuracy'] - old_val_results['accuracy']
        speed_improvement = old_val_results['avg_inference_time_ms'] - new_val_results['avg_inference_time_ms']
        
        print(f"🎯 Validation Accuracy:")
        print(f"   Old:  {old_val_results['accuracy']:.3f}")
        print(f"   New:  {new_val_results['accuracy']:.3f}")
        print(f"   Change: {acc_improvement:+.3f} ({'✅ Better' if acc_improvement > 0 else '❌ Worse'})")
        
        print(f"\n⚡ Inference Speed:")
        print(f"   Old:  {old_val_results['avg_inference_time_ms']:.1f}ms")
        print(f"   New:  {new_val_results['avg_inference_time_ms']:.1f}ms")
        print(f"   Change: {speed_improvement:+.1f}ms ({'✅ Faster' if speed_improvement > 0 else '❌ Slower'})")
    
    if old_real_results and new_real_results and "error" not in old_real_results and "error" not in new_real_results:
        conf_improvement = new_real_results['avg_confidence'] - old_real_results['avg_confidence']
        print(f"\n🎯 Real-world Confidence:")
        print(f"   Old:  {old_real_results['avg_confidence']:.3f}")
        print(f"   New:  {new_real_results['avg_confidence']:.3f}")
        print(f"   Change: {conf_improvement:+.3f} ({'✅ More confident' if conf_improvement > 0 else '❌ Less confident'})")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("🔍 Check confusion matrix comparison: confusion_matrix_comparison.png")

if __name__ == "__main__":
    main()


