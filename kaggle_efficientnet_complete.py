#!/usr/bin/env python3
"""
Kaggle Notebook: EfficientNet-B3 Leaf Disease Classification
- Complete pipeline từ data loading đến model training
- GPU acceleration với TPU/GPU
- Multi-branch architecture với attention visualization
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
# 2. DATA LOADING & PREPROCESSING
# =============================================================================

class LeafDiseaseDataset(Dataset):
    """Custom dataset for leaf disease classification"""
    
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Load all images
        self.samples = []
        self.classes = []
        
        print(f"🔍 Looking for data in: {self.data_dir}")
        
        if is_train:
            # Try different train directory names
            train_dirs = ["train", "training", "Train", "Training"]
            for train_dir_name in train_dirs:
                train_dir = self.data_dir / train_dir_name
                if train_dir.exists():
                    print(f"✅ Found training data in: {train_dir}")
                    dataset = datasets.ImageFolder(str(train_dir))
                    self.samples = dataset.samples
                    self.classes = dataset.classes
                    break
        else:
            # Try different validation directory names
            val_dirs = ["valid", "val", "validation", "test", "Valid", "Val", "Validation", "Test"]
            for val_dir_name in val_dirs:
                val_dir = self.data_dir / val_dir_name
                if val_dir.exists():
                    print(f"✅ Found validation data in: {val_dir}")
                    dataset = datasets.ImageFolder(str(val_dir))
                    self.samples = dataset.samples
                    self.classes = dataset.classes
                    break
        
        print(f"📊 Loaded {len(self.samples)} samples with {len(self.classes)} classes")
        
        # If no data found, try to find any image folder
        if len(self.samples) == 0:
            print("🔍 Searching for any image folders...")
            for root, dirs, files in os.walk(self.data_dir):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    print(f"📁 Found images in: {root}")
                    # Try to create dataset from this directory
                    try:
                        dataset = datasets.ImageFolder(root)
                        self.samples = dataset.samples
                        self.classes = dataset.classes
                        print(f"✅ Successfully loaded from: {root}")
                        break
                    except Exception as e:
                        print(f"❌ Failed to load from {root}: {e}")
                        continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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
# 3. EFFICIENTNET-B3 MODEL ARCHITECTURE
# =============================================================================

class EfficientNetLeafClassifier(nn.Module):
    """EfficientNet-B3 với multi-branch architecture"""
    
    def __init__(self, num_classes=14, num_disease_classes=None):
        super().__init__()
        
        # EfficientNet-B3 backbone
        self.backbone = timm.create_model(
            'efficientnet_b3', 
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        self.feature_dim = self.backbone.num_features  # 1536
        self.num_classes = num_classes
        self.num_disease_classes = num_disease_classes
        
        # Multi-branch feature extraction
        self.shape_branch = self._create_branch("shape")
        self.texture_branch = self._create_branch("texture") 
        self.color_branch = self._create_branch("color")
        
        # Species classifier
        self.species_classifier = nn.Sequential(
            nn.Linear(768, 1024),  # 256 * 3 = 768
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Disease classifier (if specified)
        if num_disease_classes:
            self.disease_classifier = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_disease_classes)
            )
    
    def _create_branch(self, branch_type):
        """Tạo branch cho từng loại đặc trưng"""
        return nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract backbone features
        features = self.backbone.forward_features(x)
        
        # Multi-branch processing
        shape_feat = self.shape_branch(features).flatten(1)
        texture_feat = self.texture_branch(features).flatten(1)
        color_feat = self.color_branch(features).flatten(1)
        
        # Combine features
        combined = torch.cat([shape_feat, texture_feat, color_feat], dim=1)
        
        # Species prediction
        species_logits = self.species_classifier(combined)
        
        if return_features:
            return species_logits, combined
        else:
            return species_logits

# =============================================================================
# 4. TRAINING FUNCTIONS
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
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct/total:.4f}"
        })
    
    return running_loss / total, correct / total, all_preds, all_labels

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"   📊 LR:    {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"   🎉 New best validation accuracy: {best_val_acc:.4f}")
    
    return history, best_model_state, val_preds, val_labels

# =============================================================================
# 5. VISUALIZATION & EVALUATION
# =============================================================================

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Data paths (adjust for your Kaggle dataset)
    print("🌱 EfficientNet-B3 Leaf Disease Classification")
    print("="*60)
    
    # 1. Check available datasets
    print("\n📊 Checking available datasets...")
    import os
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        datasets = os.listdir(kaggle_input)
        print(f"Available datasets: {datasets}")
        
        # Try to find the right dataset
        data_dir = None
        for dataset in datasets:
            if "plant" in dataset.lower() or "leaf" in dataset.lower() or "pathology" in dataset.lower():
                data_dir = f"/kaggle/input/{dataset}"
                print(f"✅ Found dataset: {dataset}")
                break
        
        if data_dir is None:
            print("❌ No suitable dataset found. Please add a plant/leaf dataset.")
            print("Available datasets:", datasets)
            return None, None, None
    else:
        print("❌ Kaggle input directory not found. Please add a dataset.")
        return None, None, None
    
    print(f"📁 Using dataset: {data_dir}")
    
    # 2. Load data
    print("\n📊 Loading dataset...")
    train_tfms, val_tfms = get_transforms(224)
    
    # Create datasets
    train_dataset = LeafDiseaseDataset(data_dir, transform=train_tfms, is_train=True)
    val_dataset = LeafDiseaseDataset(data_dir, transform=val_tfms, is_train=False)
    
    # Check if datasets loaded successfully
    if len(train_dataset) == 0:
        print("❌ No training data found. Please check dataset structure.")
        print("Expected structure:")
        print("  dataset/")
        print("    train/")
        print("      class1/")
        print("      class2/")
        print("    valid/ (or val/)")
        print("      class1/")
        print("      class2/")
        return None, None, None
    
    # Create data loaders
    batch_size = 32 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"✅ Dataset loaded:")
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Val samples: {len(val_dataset)}")
    print(f"   - Classes: {len(train_dataset.classes)}")
    print(f"   - Batch size: {batch_size}")
    
    # 2. Create model
    print("\n🏗️  Creating EfficientNet-B3 model...")
    model = EfficientNetLeafClassifier(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # 3. Train model
    print("\n🚀 Starting training...")
    history, best_model_state, val_preds, val_labels = train_model(
        model, train_loader, val_loader, 
        num_epochs=15, lr=1e-4
    )
    
    # 4. Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with accuracy: {max(history['val_acc']):.4f}")
    
    # 5. Visualizations
    print("\n📊 Generating visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(val_labels, val_preds, train_dataset.classes)
    
    # 6. Save model
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes,
        'history': history,
        'best_val_acc': max(history['val_acc'])
    }, 'efficientnet_leaf_classifier.pt')
    
    print("🎉 Training completed successfully!")
    print(f"📊 Best validation accuracy: {max(history['val_acc']):.4f}")
    
    return model, train_dataset.classes, history

# =============================================================================
# 7. RUN THE NOTEBOOK
# =============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    model, class_names, history = main()
    
    # Additional analysis
    print("\n🔍 Model Analysis:")
    print(f"📈 Final training accuracy: {history['train_acc'][-1]:.4f}")
    print(f"📈 Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test inference
    print("\n🧪 Testing inference...")
    model.eval()
    with torch.no_grad():
        # Test with random input
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        prob = F.softmax(output, dim=1)
        pred_class = prob.argmax(dim=1).item()
        confidence = prob[0, pred_class].item()
        
        print(f"✅ Inference test successful!")
        print(f"   Predicted class: {class_names[pred_class]}")
        print(f"   Confidence: {confidence:.4f}")
