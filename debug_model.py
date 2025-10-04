#!/usr/bin/env python3
"""
Debug script để kiểm tra model và test prediction
"""

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import json

from src.train import SimpleClassifier
from src.preprocess import get_transforms

def debug_model():
    """Debug model loading và prediction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # 1. Kiểm tra model path
    model_path = Path("models/species_v2/model_best.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Model found: {model_path}")
    
    # 2. Load model
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        classes = ckpt.get("classes", [])
        print(f"📋 Classes: {classes}")
        print(f"📊 Number of classes: {len(classes)}")
        
        # 3. Tạo model architecture
        model = SimpleClassifier(num_classes=len(classes))
        
        # 4. Load state dict với strict=False
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"⚠️  Missing keys: {len(missing_keys)}")
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys:
            print("First 5 missing keys:", missing_keys[:5])
        
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 5. Test với ảnh mẫu
    test_images = [
        "data/test/AppleCedarRust1.JPG",
        "data/test/AppleScab1.JPG", 
        "data/test/TomatoHealthy1.JPG"
    ]
    
    _, val_tfms = get_transforms(224)
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n🔍 Testing: {img_path}")
            
            try:
                with Image.open(img_path).convert("RGB") as img:
                    x = val_tfms(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = model(x)
                        prob = F.softmax(logits, dim=1)[0]
                        conf, pred = torch.max(prob, dim=0)
                        
                        print(f"   Prediction: {classes[pred.item()]}")
                        print(f"   Confidence: {conf.item():.4f}")
                        print(f"   Top 3 predictions:")
                        
                        # Top 3 predictions
                        top3_conf, top3_idx = torch.topk(prob, 3)
                        for i, (idx, conf_val) in enumerate(zip(top3_idx, top3_conf)):
                            print(f"     {i+1}. {classes[idx.item()]}: {conf_val.item():.4f}")
                            
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ❌ Image not found: {img_path}")

if __name__ == "__main__":
    debug_model()


