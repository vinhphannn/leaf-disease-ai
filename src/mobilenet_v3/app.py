#!/usr/bin/env python3
"""
Gradio Web App for Leaf Disease Classification using MobileNetV3
"""

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.feature_extraction import extract_features_and_explain

def load_model():
    """Load the trained MobileNetV3 model"""
    print("🔄 Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load('models/model_best_efficientnet.pt', map_location='cpu', weights_only=False)
    
    # Check if it's a checkpoint with state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        classes = checkpoint.get('classes', [])
        best_acc = checkpoint.get('best_val_acc', 0)
        print(f"📊 Best validation accuracy: {best_acc:.4f}")
        print(f"📚 Classes: {len(classes)}")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        classes = checkpoint.get('classes', [])
        best_acc = checkpoint.get('best_val_acc', 0)
        print(f"📊 Best validation accuracy: {best_acc:.4f}")
        print(f"📚 Classes: {len(classes)}")
    else:
        state_dict = checkpoint
        classes = []
    
    # Create model
    from src.efficientnet_b0_model import EfficientNetB0LeafClassifier
    model = EfficientNetB0LeafClassifier(num_classes=38)
    
    # Load state_dict with strict=False
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, classes

def preprocess_image(image):
    """Preprocess image for inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_disease_with_explanation(image):
    """Predict disease from image with feature extraction and explanation"""
    if image is None:
        return "❌ Please upload an image", None, None
    
    try:
        # Extract features and generate explanation
        result = extract_features_and_explain(model, image, class_names)
        
        # Create detailed result text
        result_text = result['explanation']
        
        # Add top 5 predictions
        with torch.no_grad():
            image_tensor = preprocess_image(image)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            result_text += "\n\n📊 **Top 5 Predictions**:\n"
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                class_name = class_names[idx.item()]
                confidence = prob.item()
                result_text += f"{i+1}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)\n"
        
        # Generate heatmap visualization
        heatmap_fig = None
        try:
            heatmap = result['heatmap']
            print(f"🔍 Heatmap shape: {heatmap.shape}, min: {heatmap.min():.4f}, max: {heatmap.max():.4f}")
            
            # Create heatmap visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Ảnh gốc')
            axes[0].axis('off')
            
            # Heatmap only
            im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Heatmap overlay
            import cv2
            image_np = np.array(image)
            heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
            
            # Overlay heatmap
            colormap = plt.get_cmap('jet')
            heatmap_colored = colormap(heatmap_resized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
            
            axes[2].imshow(overlay)
            axes[2].set_title('Kết hợp (Overlay)')
            axes[2].axis('off')
            
            plt.tight_layout()
            heatmap_fig = fig
            print("✅ Heatmap figure created successfully!")
            
        except Exception as e:
            print(f"❌ Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()
            # Create a simple error plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating heatmap:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Heatmap Error')
            heatmap_fig = fig
        
        return result_text, result['predicted_class'], heatmap_fig
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

def create_interface():
    """Create Gradio interface"""
    # Load model
    global model, class_names
    model, class_names = load_model()
    
    # Create interface
    with gr.Blocks(title="🌱 Leaf Disease Classification", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌱 Leaf Disease Classification")
        gr.Markdown("Upload an image of a plant leaf to identify potential diseases.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Leaf Image",
                    type="pil",
                    height=300
                )
                predict_btn = gr.Button("🔍 Analyze", variant="primary")
                
            with gr.Column():
                output_text = gr.Markdown(
                    label="Analysis Results",
                    value="Upload an image to get started..."
                )
                heatmap_output = gr.Plot(
                    label="Grad-CAM Visualization",
                    visible=True
                )
        
        # Examples
        gr.Markdown("## 📸 Example Images")
        gr.Markdown("Upload images of plant leaves to test the model.")
        
        # Event handlers
        predict_btn.click(
            fn=predict_disease_with_explanation,
            inputs=[image_input],
            outputs=[output_text, gr.State(), heatmap_output]
        )
        
        # Auto-predict on image upload
        image_input.change(
            fn=predict_disease_with_explanation,
            inputs=[image_input],
            outputs=[output_text, gr.State(), heatmap_output]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None
    )
