#!/usr/bin/env python3
"""
Train tất cả 14 species với disease models
"""

import os
import subprocess
import sys
from pathlib import Path

def train_all_species():
    """Train tất cả species với disease models"""
    
    # Tất cả 14 species trong dataset
    all_species = [
        "Apple", "Blueberry", "Cherry_(including_sour)", "Corn_(maize)",
        "Grape", "Orange", "Peach", "Pepper,_bell", 
        "Potato", "Raspberry", "Soybean", "Squash",
        "Strawberry", "Tomato"
    ]
    
    print("🌱 Training tất cả 14 species...")
    
    # Set environment variables cho Mixup
    os.environ["MIXUP"] = "1"
    os.environ["MIXUP_ALPHA"] = "0.2"
    os.environ["CUTMIX"] = "0"
    
    success_count = 0
    
    for species in all_species:
        print(f"\n🔄 Training disease model for {species}...")
        
        cmd = [
            sys.executable, "-m", "src.train_disease",
            "--data_dir", "data_masked",
            "--species", species,
            "--output_dir", "models/disease_v2",
            "--epochs", "8",
            "--batch_size", "32",
            "--lr", "1e-3",
            "--img_size", "224"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {species} - Success!")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ {species} - Failed: {e.stderr}")
            continue
    
    print(f"\n🎉 Completed: {success_count}/{len(all_species)} species trained successfully")

if __name__ == "__main__":
    train_all_species()


