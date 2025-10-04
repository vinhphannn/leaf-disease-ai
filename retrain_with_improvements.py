#!/usr/bin/env python3
"""
Script để retrain models với pipeline cải tiến:
- Augmentation mạnh hơn (RandomResizedCrop, ColorJitter, RandomErasing)
- Mixup/CutMix để chống spurious correlations
- Consistent preprocessing pipeline
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Chạy command và stream output trực tiếp (hiện tqdm/progress)."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        # Kế thừa stdout/stderr để thấy tiến trình theo thời gian thực
        completed = subprocess.run(cmd, check=True)
        print(f"✅ {description} - Thành công!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Thất bại! (exit={e.returncode})")
        return False

def main():
    """Main retraining pipeline"""
    print("🌱 Bắt đầu retrain models với pipeline cải tiến...")
    
    # Đảm bảo có data_masked
    data_masked = Path("data_masked")
    if not data_masked.exists():
        print("❌ Không tìm thấy data_masked/ - cần chạy preprocessing trước!")
        return
    
    # 1. Train Species Classifier với CutMix
    print("\n" + "="*60)
    print("1️⃣ TRAIN SPECIES CLASSIFIER (với CutMix)")
    print("="*60)
    
    env_vars = {
        "CUTMIX": "1",
        "CUTMIX_ALPHA": "0.8",
        "MIXUP": "0"  # Chỉ dùng 1 trong 2
    }
    
    cmd_species = [
        sys.executable, "-m", "src.train_species",
        "--data_dir", "data_masked",
        "--output_dir", "models/species_v2",
        "--epochs", "15",
        "--batch_size", "64",
        "--lr", "1e-3",
        "--img_size", "224"
    ]
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    if not run_command(cmd_species, "Train Species Classifier"):
        return
    
    # 2. Train Disease Classifiers cho các species chính
    print("\n" + "="*60)
    print("2️⃣ TRAIN DISEASE CLASSIFIERS (với Mixup)")
    print("="*60)
    
    # Thay đổi env vars cho disease training
    env_vars.update({
        "CUTMIX": "0",
        "MIXUP": "1",
        "MIXUP_ALPHA": "0.2"
    })
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Train cho các species chính
    main_species = ["Apple", "Tomato", "Potato", "Corn", "Grape"]
    
    for species in main_species:
        print(f"\n🌿 Training disease classifier cho {species}...")
        
        cmd_disease = [
            sys.executable, "-m", "src.train_disease",
            "--data_dir", "data_masked",
            "--species", species,
            "--output_dir", "models/disease_v2",
            "--epochs", "12",
            "--batch_size", "64",
            "--lr", "1e-3",
            "--img_size", "224"
        ]
        
        if not run_command(cmd_disease, f"Train Disease Classifier - {species}"):
            print(f"⚠️  Bỏ qua {species} do lỗi")
            continue
    
    # 3. Test inference với TTA
    print("\n" + "="*60)
    print("3️⃣ TEST INFERENCE VỚI TTA")
    print("="*60)
    
    # Test với một vài ảnh mẫu
    test_images = Path("data/test")
    if test_images.exists():
        test_files = list(test_images.glob("*.JPG"))[:3]  # Test 3 ảnh đầu
        
        for test_img in test_files:
            print(f"\n🔍 Testing {test_img.name}...")
            
            # Test không TTA
            cmd_no_tta = [
                sys.executable, "-m", "src.predict",
                str(test_img),
                "--model", "models/species_v2/model_best.pt",
                "--threshold", "0.5"
            ]
            run_command(cmd_no_tta, f"Predict {test_img.name} (no TTA)")
            
            # Test với TTA
            cmd_tta = [
                sys.executable, "-m", "src.predict",
                str(test_img),
                "--model", "models/species_v2/model_best.pt",
                "--tta",
                "--threshold", "0.5"
            ]
            run_command(cmd_tta, f"Predict {test_img.name} (with TTA)")
    
    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH RETRAINING!")
    print("="*60)
    print("📁 Models mới được lưu tại:")
    print("   - models/species_v2/")
    print("   - models/disease_v2/")
    print("\n🔧 Để test với threshold khác:")
    print("   python -m src.predict IMAGE --model models/species_v2/model_best.pt --tta --threshold 0.7")
    print("\n📊 Để so sánh performance:")
    print("   - Chạy Grad-CAM trên ảnh thực tế")
    print("   - Test trên data_real/ nếu có")

if __name__ == "__main__":
    main()
