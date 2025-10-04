#!/usr/bin/env python3
"""
Test EfficientNet-B3 Model
"""

import torch
from src.efficientnet_model import create_species_model

def test_model():
    """Test model architecture"""
    print("🔧 Testing EfficientNet-B3 model...")
    
    # Create model
    model = create_species_model(num_classes=14)
    print(f"✅ Model created successfully!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    print(f"📥 Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
        print(f"📤 Output shape: {y.shape}")
        print(f"✅ Forward pass successful!")
    
    # Test with different batch sizes
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
            print(f"✅ Batch size {batch_size}: {x.shape} -> {y.shape}")
    
    print("🎉 Model test completed successfully!")

if __name__ == "__main__":
    test_model()


