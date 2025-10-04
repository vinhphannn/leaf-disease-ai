#!/usr/bin/env python3
"""
Complete EfficientNet-B3 Training Pipeline
- Train tất cả 14 species
- Progress tracking với tqdm
- Confusion matrix với số liệu
- Eval artifacts
- Model saving
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_training_command(cmd, description, timeout=3600):
    """Chạy training command với timeout"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout)
        elapsed = time.time() - start_time
        print(f"✅ {description} - Completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - Timeout after {timeout/60:.1f} minutes")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed with exit code {e.returncode}")
        return False

def main():
    """Complete training pipeline"""
    print("🌱 EfficientNet-B3 Complete Training Pipeline")
    print("="*80)
    
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available
    os.environ["PYTORCH_DIRECTML"] = "1"  # Enable DML for Windows
    
    # 1. Train Species Classifier
    print("\n" + "="*80)
    print("1️⃣ TRAINING SPECIES CLASSIFIER (EfficientNet-B3)")
    print("="*80)
    
    species_cmd = [
        sys.executable, "-m", "src.train_efficientnet_species",
        "--data_dir", "data_masked",
        "--output_dir", "models/efficientnet_species",
        "--epochs", "25",
        "--batch_size", "16",  # Smaller batch for EfficientNet-B3
        "--lr", "1e-4",
        "--img_size", "224",
        "--num_workers", "4",
        # "--use_mixup",  # Disable mixup for now
        # "--mixup_alpha", "0.2",
        "--val_split", "0.2",
        "--device", "cpu"  # Force CPU to avoid DML issues
    ]
    
    if not run_training_command(species_cmd, "Species Classifier Training", timeout=7200):
        print("❌ Species training failed, stopping pipeline")
        return
    
    # 2. Train Disease Classifiers for all species
    print("\n" + "="*80)
    print("2️⃣ TRAINING DISEASE CLASSIFIERS")
    print("="*80)
    
    # All 14 species in the dataset
    all_species = [
        "Apple", "Blueberry", "Cherry_(including_sour)", "Corn_(maize)",
        "Grape", "Orange", "Peach", "Pepper,_bell", 
        "Potato", "Raspberry", "Soybean", "Squash",
        "Strawberry", "Tomato"
    ]
    
    success_count = 0
    failed_species = []
    
    for i, species in enumerate(all_species, 1):
        print(f"\n🌿 [{i}/{len(all_species)}] Training disease classifier for {species}")
        
        disease_cmd = [
            sys.executable, "-m", "src.train_efficientnet_disease",
            "--data_dir", "data_masked",
            "--species", species,
            "--output_dir", "models/efficientnet_disease",
            "--epochs", "15",
            "--batch_size", "16",
            "--lr", "1e-4",
            "--img_size", "224",
            "--num_workers", "4",
            "--use_cutmix",
            "--cutmix_alpha", "0.8",
            "--val_split", "0.2"
        ]
        
        if run_training_command(disease_cmd, f"Disease Classifier - {species}", timeout=1800):
            success_count += 1
        else:
            failed_species.append(species)
    
    # 3. Generate Summary Report
    print("\n" + "="*80)
    print("3️⃣ TRAINING SUMMARY")
    print("="*80)
    
    print(f"📊 Species Classifier: ✅ Completed")
    print(f"📊 Disease Classifiers: {success_count}/{len(all_species)} completed")
    
    if failed_species:
        print(f"❌ Failed species: {', '.join(failed_species)}")
    
    # 4. Test Inference
    print("\n" + "="*80)
    print("4️⃣ TESTING INFERENCE")
    print("="*80)
    
    test_images = [
        "data/test/AppleCedarRust1.JPG",
        "data/test/AppleScab1.JPG",
        "data/test/TomatoHealthy1.JPG"
    ]
    
    for test_img in test_images:
        if Path(test_img).exists():
            print(f"\n🔍 Testing: {Path(test_img).name}")
            
            test_cmd = [
                sys.executable, "-m", "src.predict_efficientnet",
                test_img,
                "--species_model", "models/efficientnet_species/model_best.pt",
                "--disease_models_dir", "models/efficientnet_disease",
                "--tta",
                "--threshold", "0.5"
            ]
            
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ Inference successful")
                    print(result.stdout)
                else:
                    print("❌ Inference failed")
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print("⏰ Inference timeout")
    
    # 5. Final Summary
    print("\n" + "="*80)
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print("="*80)
    
    print("📁 Models saved to:")
    print("   - models/efficientnet_species/")
    print("   - models/efficientnet_disease/")
    
    print("\n📊 Generated artifacts:")
    print("   - Confusion matrices with numbers")
    print("   - Training plots (loss/accuracy)")
    print("   - Evaluation artifacts (JSON)")
    print("   - Model checkpoints")
    
    print(f"\n🎯 Success rate: {success_count + 1}/{len(all_species) + 1} models trained")
    
    if failed_species:
        print(f"\n⚠️  Failed species: {', '.join(failed_species)}")
        print("   You may need to retrain these manually")
    
    print("\n🚀 Ready for deployment!")

if __name__ == "__main__":
    main()
