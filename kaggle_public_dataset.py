#!/usr/bin/env python3
"""
Kaggle Notebook với Public Dataset
Sử dụng Plant Pathology 2021 dataset
"""

# =============================================================================
# 1. SETUP & IMPORTS
# =============================================================================

import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add missing import
from PIL import Image

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import timm

# Progress bar
from tqdm.auto import tqdm

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# 2. DOWNLOAD & PREPARE DATASET
# =============================================================================

def download_and_prepare_dataset():
    """Download và prepare dataset từ public sources"""
    
    print("📊 Downloading and preparing dataset...")
    
    # Download dataset using kagglehub
    try:
        import kagglehub
        print("🔽 Downloading dataset...")
        data_dir = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
        print(f"✅ Dataset downloaded to: {data_dir}")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("Trying alternative method...")
        
        # Alternative: Use existing dataset if available
        data_dir = "/kaggle/input/new-plant-diseases-dataset"
        if not os.path.exists(data_dir):
            print("❌ Dataset not found!")
            return None
    
    # Check dataset structure
    print(f"📁 Dataset structure:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    return data_dir

# =============================================================================
# 3. CUSTOM DATASET CLASS
# =============================================================================

class PlantPathologyDataset(Dataset):
    """Custom dataset for New Plant Diseases Dataset"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Load samples
        self.samples = []
        self.classes = []
        
        if split == 'train':
            # Load training data - New Plant Diseases Dataset structure
            train_dir = self.data_dir / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train"
            if train_dir.exists():
                # Use ImageFolder to load with proper class structure
                dataset = datasets.ImageFolder(str(train_dir))
                self.samples = dataset.samples
                self.classes = dataset.classes
                print(f"📊 Found {len(self.samples)} training images")
                print(f"📊 Classes: {self.classes}")
            else:
                # Try alternative paths
                alt_paths = [
                    self.data_dir / "train",
                    self.data_dir / "New Plant Diseases Dataset(Augmented)" / "train",
                    self.data_dir / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "train"
                ]
                
                for path in alt_paths:
                    if path.exists():
                        dataset = datasets.ImageFolder(str(path))
                        self.samples = dataset.samples
                        self.classes = dataset.classes
                        print(f"📊 Found training data in: {path}")
                        print(f"📊 Found {len(self.samples)} training images")
                        print(f"📊 Classes: {self.classes}")
                        break
                else:
                    print("❌ Training data not found!")
        else:
            # Load validation data
            val_dir = self.data_dir / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "valid"
            if val_dir.exists():
                dataset = datasets.ImageFolder(str(val_dir))
                self.samples = dataset.samples
                self.classes = dataset.classes
                print(f"📊 Found {len(self.samples)} validation images")
            else:
                # Try alternative paths
                alt_paths = [
                    self.data_dir / "valid",
                    self.data_dir / "val",
                    self.data_dir / "New Plant Diseases Dataset(Augmented)" / "valid",
                    self.data_dir / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)" / "valid"
                ]
                
                for path in alt_paths:
                    if path.exists():
                        dataset = datasets.ImageFolder(str(path))
                        self.samples = dataset.samples
                        self.classes = dataset.classes
                        print(f"📊 Found validation data in: {path}")
                        print(f"📊 Found {len(self.samples)} validation images")
                        break
                else:
                    print("❌ Validation data not found!")
        
        print(f"📊 Loaded {len(self.samples)} samples with {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# =============================================================================
# 4. TRANSFORMS
# =============================================================================

def get_transforms(img_size=224):
    """Get training and validation transforms"""
    
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_tfms, val_tfms

# =============================================================================
# 5. SIMPLE MODEL (for testing)
# =============================================================================

class SimpleEfficientNet(nn.Module):
    """Simple EfficientNet for testing"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Use pre-trained EfficientNet
        self.backbone = timm.create_model(
            'efficientnet_b0',  # Smaller model for testing
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# =============================================================================
# 6. TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct/total:.4f}"
        })
    
    return running_loss / total, correct / total

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct/total:.4f}"
        })
    
    return running_loss / total, correct / total

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("🌱 EfficientNet Leaf Disease Classification")
    print("="*60)
    
    # 1. Download dataset
    data_dir = download_and_prepare_dataset()
    if data_dir is None:
        print("❌ Cannot proceed without dataset!")
        return None, None, None
    
    # 2. Create datasets
    print("\n📊 Creating datasets...")
    train_tfms, val_tfms = get_transforms(224)
    
    train_dataset = PlantPathologyDataset(data_dir, transform=train_tfms, split='train')
    val_dataset = PlantPathologyDataset(data_dir, transform=val_tfms, split='test')
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return None, None, None
    
    # 3. Create data loaders (fix multiprocessing issue in Kaggle)
    batch_size = 32 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Dataset loaded:")
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Val samples: {len(val_dataset)}")
    print(f"   - Classes: {len(train_dataset.classes)}")
    print(f"   - Batch size: {batch_size}")
    
    # 4. Create model
    print("\n🏗️  Creating model...")
    model = SimpleEfficientNet(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # 5. Train model
    print("\n🚀 Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(5):  # Short training for testing
        print(f"\n📅 Epoch {epoch+1}/5")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        print(f"   📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    # 6. Save model
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes,
        'history': history
    }, 'efficientnet_leaf_classifier.pt')
    
    print("🎉 Training completed successfully!")
    print(f"📊 Final validation accuracy: {history['val_acc'][-1]:.4f}")
    
    return model, train_dataset.classes, history

# =============================================================================
# 8. RUN THE NOTEBOOK
# =============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    result = main()
    
    if result[0] is not None:
        model, class_names, history = result
        
        # Additional analysis
        print("\n🔍 Model Analysis:")
        print(f"📈 Final training accuracy: {history['train_acc'][-1]:.4f}")
        print(f"📈 Final validation accuracy: {history['val_acc'][-1]:.4f}")
        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference
        print("\n🧪 Testing inference...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            output = model(test_input)
            prob = F.softmax(output, dim=1)
            pred_class = prob.argmax(dim=1).item()
            confidence = prob[0, pred_class].item()
            
            print(f"✅ Inference test successful!")
            print(f"   Predicted class: {class_names[pred_class]}")
            print(f"   Confidence: {confidence:.4f}")
    else:
        print("❌ Training failed!")
