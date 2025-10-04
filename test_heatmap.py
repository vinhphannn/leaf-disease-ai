#!/usr/bin/env python3
"""
Test heatmap generation
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from src.efficientnet_b0_model import EfficientNetB0LeafClassifier

def test_heatmap():
    """Test heatmap generation"""
    print("🔄 Testing heatmap generation...")
    
    # Load model
    model = EfficientNetB0LeafClassifier(num_classes=38)
    checkpoint = torch.load('models/model_best_efficientnet.pt', map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='green')
    
    # Test feature extraction
    try:
        from src.feature_extraction import extract_features_and_explain
        
        # Get class names
        class_names = checkpoint.get('classes', [f'Class_{i}' for i in range(38)])
        
        # Extract features
        result = extract_features_and_explain(model, dummy_image, class_names)
        
        print("✅ Feature extraction successful!")
        print(f"📊 Predicted class: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.4f}")
        print(f"📊 Heatmap shape: {result['heatmap'].shape}")
        print(f"📊 Regions found: {len(result['regions_info'])}")
        
        # Test heatmap visualization
        heatmap = result['heatmap']
        print(f"📊 Heatmap min: {heatmap.min():.4f}, max: {heatmap.max():.4f}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axes[0].imshow(dummy_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('test_heatmap.png', dpi=150, bbox_inches='tight')
        print("✅ Heatmap saved as test_heatmap.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_heatmap()
    if success:
        print("🎉 Heatmap test completed successfully!")
    else:
        print("❌ Heatmap test failed!")
