#!/usr/bin/env python3
"""
Analyze Features và Attention của EfficientNet-B3
- Extract features từ test images
- Visualize attention patterns
- Compare với MobileNetV3
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json

from src.efficientnet_model import create_species_model, create_disease_model
from src.attention_visualization import AttentionVisualizer, visualize_model_attention
from src.preprocess import get_transforms

def analyze_test_images():
    """Analyze features từ test images"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    species_model_path = Path("models/efficientnet_species/model_best.pt")
    if not species_model_path.exists():
        print("❌ Species model not found. Train first!")
        return
    
    # Load model
    ckpt = torch.load(species_model_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("classes", [])
    model = create_species_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    
    print(f"✅ Loaded model with {len(class_names)} classes")
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, device)
    
    # Test images
    test_dir = Path("data/test")
    if not test_dir.exists():
        print("❌ Test directory not found")
        return
    
    test_images = list(test_dir.glob("*.JPG"))[:10]  # First 10 images
    print(f"🔍 Analyzing {len(test_images)} test images...")
    
    # Results storage
    all_features = []
    attention_analysis = {
        'shape_importance': [],
        'texture_importance': [],
        'color_importance': [],
        'confidence_scores': []
    }
    
    # Create output directory
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(test_images):
        print(f"\n📸 [{i+1}/{len(test_images)}] Analyzing: {img_path.name}")
        
        try:
            # Load and preprocess
            _, val_tfms = get_transforms(224)
            with Image.open(img_path).convert("RGB") as img:
                original_image = np.array(img)
                image_tensor = val_tfms(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                logits = model(image_tensor)
                prob = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(prob, dim=1).item()
                confidence = prob[0, pred_class].item()
            
            class_name = class_names[pred_class] if pred_class < len(class_names) else "Unknown"
            print(f"   🎯 Prediction: {class_name} (conf: {confidence:.3f})")
            
            # Extract features
            features = visualizer.extract_features(image_tensor)
            all_features.append({
                'image_path': str(img_path),
                'prediction': class_name,
                'confidence': confidence,
                'features': features
            })
            
            # Analyze attention patterns
            attention_maps = visualizer.generate_attention_maps(image_tensor)
            
            if 'shape' in attention_maps:
                attention_analysis['shape_importance'].append(np.mean(attention_maps['shape']))
            if 'texture' in attention_maps:
                attention_analysis['texture_importance'].append(np.mean(attention_maps['texture']))
            if 'color' in attention_maps:
                attention_analysis['color_importance'].append(np.mean(attention_maps['color']))
            
            attention_analysis['confidence_scores'].append(confidence)
            
            # Generate and save attention visualization
            fig = visualizer.visualize_all_attentions(
                image_tensor, pred_class, class_name, original_image
            )
            
            output_path = output_dir / f"attention_{img_path.stem}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"   💾 Saved attention map: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {img_path.name}: {e}")
            continue
    
    # Generate summary analysis
    print(f"\n📊 Generating summary analysis...")
    
    # 1. Feature importance analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Shape vs Texture vs Color importance
    if attention_analysis['shape_importance']:
        axes[0, 0].scatter(attention_analysis['shape_importance'], 
                          attention_analysis['texture_importance'], 
                          c=attention_analysis['confidence_scores'], 
                          cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel('Shape Importance')
        axes[0, 0].set_ylabel('Texture Importance')
        axes[0, 0].set_title('Shape vs Texture Importance\n(Color = Confidence)')
    
    # Confidence distribution
    axes[0, 1].hist(attention_analysis['confidence_scores'], bins=10, alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Distribution')
    
    # Feature importance comparison
    feature_means = {
        'Shape': np.mean(attention_analysis['shape_importance']) if attention_analysis['shape_importance'] else 0,
        'Texture': np.mean(attention_analysis['texture_importance']) if attention_analysis['texture_importance'] else 0,
        'Color': np.mean(attention_analysis['color_importance']) if attention_analysis['color_importance'] else 0
    }
    
    axes[1, 0].bar(feature_means.keys(), feature_means.values(), 
                   color=['red', 'green', 'blue'], alpha=0.7)
    axes[1, 0].set_ylabel('Mean Importance')
    axes[1, 0].set_title('Average Feature Importance')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Confidence vs Feature Importance
    if attention_analysis['confidence_scores'] and attention_analysis['shape_importance']:
        axes[1, 1].scatter(attention_analysis['confidence_scores'], 
                          attention_analysis['shape_importance'], 
                          alpha=0.7, label='Shape')
        if attention_analysis['texture_importance']:
            axes[1, 1].scatter(attention_analysis['confidence_scores'], 
                             attention_analysis['texture_importance'], 
                             alpha=0.7, label='Texture')
        if attention_analysis['color_importance']:
            axes[1, 1].scatter(attention_analysis['confidence_scores'], 
                             attention_analysis['color_importance'], 
                             alpha=0.7, label='Color')
        
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Feature Importance')
        axes[1, 1].set_title('Confidence vs Feature Importance')
        axes[1, 1].legend()
    
    plt.tight_layout()
    summary_path = output_dir / "feature_analysis_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Save detailed results
    results = {
        'analysis_summary': {
            'total_images': len(test_images),
            'successful_analyses': len(all_features),
            'average_confidence': np.mean(attention_analysis['confidence_scores']),
            'feature_importance': feature_means
        },
        'detailed_results': all_features,
        'attention_analysis': attention_analysis
    }
    
    with open(output_dir / "detailed_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n🎉 Analysis completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Summary plot: {summary_path}")
    print(f"📋 Detailed results: {output_dir}/detailed_analysis.json")
    print(f"📈 Average confidence: {np.mean(attention_analysis['confidence_scores']):.3f}")
    print(f"🔍 Feature importance: {feature_means}")

if __name__ == "__main__":
    analyze_test_images()


