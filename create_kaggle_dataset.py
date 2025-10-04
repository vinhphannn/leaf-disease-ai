#!/usr/bin/env python3
"""
Tạo Kaggle Dataset từ data local
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_kaggle_dataset():
    """Tạo dataset cho Kaggle"""
    
    # Source data
    source_dir = Path("data_masked")
    if not source_dir.exists():
        print("❌ data_masked directory not found!")
        return
    
    # Create kaggle dataset directory
    kaggle_dir = Path("kaggle_dataset")
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy data structure
    print("📁 Creating Kaggle dataset structure...")
    
    # Copy train data
    train_source = source_dir / "train"
    train_dest = kaggle_dir / "train"
    if train_source.exists():
        shutil.copytree(train_source, train_dest, dirs_exist_ok=True)
        print(f"✅ Copied train data: {len(list(train_dest.rglob('*.JPG')))} images")
    
    # Copy validation data
    val_source = source_dir / "valid"
    val_dest = kaggle_dir / "valid"
    if val_source.exists():
        shutil.copytree(val_source, val_dest, dirs_exist_ok=True)
        print(f"✅ Copied validation data: {len(list(val_dest.rglob('*.JPG')))} images")
    
    # Create metadata
    metadata = {
        "title": "Leaf Disease Dataset",
        "description": "Plant leaf disease classification dataset",
        "keywords": ["plant", "disease", "classification", "leaf"],
        "licenses": [{"name": "MIT"}],
        "collaborators": []
    }
    
    with open(kaggle_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create ZIP file
    print("📦 Creating ZIP file...")
    zip_path = "leaf_disease_dataset.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(kaggle_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, kaggle_dir)
                zipf.write(file_path, arcname)
    
    print(f"✅ Created dataset: {zip_path}")
    print(f"📊 Size: {os.path.getsize(zip_path) / 1e6:.1f} MB")
    
    return zip_path

if __name__ == "__main__":
    create_kaggle_dataset()


