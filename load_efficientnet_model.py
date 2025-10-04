#!/usr/bin/env python3
"""
Load and test the EfficientNet model from Kaggle
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os

def load_model_with_flexible_architecture(model_path='models/model_best_efficientnet.pt', num_classes=38):
    """Load model with flexible architecture detection"""
    print("🔄 Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Check if it's a checkpoint with state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"📊 Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        print(f"📚 Classes: {checkpoint.get('classes', 'N/A')}")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"📊 Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        print(f"📚 Classes: {checkpoint.get('classes', 'N/A')}")
    else:
        state_dict = checkpoint
    
    # Analyze the state_dict to determine architecture
    print("🔍 Analyzing model architecture...")
    
    # Check for EfficientNet-B0 architecture
    if any('backbone.conv_stem' in key for key in state_dict.keys()):
        print("📱 Detected: EfficientNet-B0 architecture")
        from src.efficientnet_b0_model import EfficientNetB0LeafClassifier
        model = EfficientNetB0LeafClassifier(num_classes=num_classes)
    # Check for MobileNetV3 architecture  
    elif any('backbone.features' in key for key in state_dict.keys()):
        print("📱 Detected: MobileNetV3 architecture")
        from src.train import SimpleClassifier
        model = SimpleClassifier(num_classes=num_classes)
    else:
        print("❌ Unknown architecture!")
        return None
    
    # Load state_dict with strict=False to handle architecture differences
    try:
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    model.eval()
    return model

def load_class_names(class_file='models/class_names.txt'):
    """Load class names from file"""
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            lines = f.readlines()
        class_names = [line.strip().split(': ')[1] for line in lines]
    else:
        # Fallback class names
        class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
            'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Peach___healthy',
            'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
            'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot',
            'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
        ]
    
    return class_names

def preprocess_image(image_path, image_size=224):
    """Preprocess image for inference"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

def predict_image(model, image_path, class_names, top_k=5):
    """Predict disease from image"""
    print(f"🔍 Analyzing image: {image_path}")
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        print(f"\n📊 Top {top_k} predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            class_name = class_names[idx.item()]
            confidence = prob.item()
            print(f"   {i+1}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Get best prediction
        best_idx = top_indices[0][0].item()
        best_class = class_names[best_idx]
        best_confidence = top_probs[0][0].item()
        
        return best_class, best_confidence

def main():
    """Main function"""
    print("🌱 Leaf Disease Classification Model Test")
    print("=" * 50)
    
    # Load model
    model = load_model_with_flexible_architecture()
    if model is None:
        print("❌ Failed to load model!")
        return
    
    # Load class names
    class_names = load_class_names()
    print(f"📚 Loaded {len(class_names)} classes")
    
    # Test with sample image (if exists)
    test_image = "test_image.jpg"  # Replace with your test image
    if os.path.exists(test_image):
        predicted_class, confidence = predict_image(model, test_image, class_names)
        print(f"\n🎯 Final prediction: {predicted_class}")
        print(f"🎯 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    else:
        print(f"\n❌ Test image '{test_image}' not found")
        print("💡 Place a test image in the project directory and update the path")
    
    print("\n✅ Model test completed!")

if __name__ == "__main__":
    main()
