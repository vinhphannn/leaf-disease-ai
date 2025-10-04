#!/usr/bin/env python3
"""
Hugging Face Spaces deployment for Leaf Disease Classification
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
# Configure matplotlib to prevent memory warnings
plt.rcParams['figure.max_open_warning'] = 0
from src.feature_extraction import extract_features_and_explain

# Global variables
model = None
class_names = []
last_image_hash = None
last_heatmap = None

def load_model():
    """Load the trained EfficientNet-B0 model"""
    global model, class_names
    
    if model is not None:
        return model, class_names
    
    print("🔄 Loading model...")
    
    # Load checkpoint
    try:
        checkpoint = torch.load('models/model_best_efficientnet.pt', map_location='cpu', weights_only=False)
        print("✅ Model loaded from local file")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Create dummy model for testing
        from src.efficientnet_b0_model import EfficientNetB0LeafClassifier
        model = EfficientNetB0LeafClassifier(num_classes=38)
        class_names = [f"Class_{i}" for i in range(38)]
        return model, class_names
    
    # Check if it's a checkpoint with state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        class_names = checkpoint.get('classes', [])
        best_acc = checkpoint.get('best_val_acc', 0)
        print(f"📊 Best validation accuracy: {best_acc:.4f}")
        print(f"📚 Classes: {len(class_names)}")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        class_names = checkpoint.get('classes', [])
        best_acc = checkpoint.get('best_val_acc', 0)
        print(f"📊 Best validation accuracy: {best_acc:.4f}")
        print(f"📚 Classes: {len(class_names)}")
    else:
        state_dict = checkpoint
        class_names = []
    
    # Create model
    from src.efficientnet_b0_model import EfficientNetB0LeafClassifier
    model = EfficientNetB0LeafClassifier(num_classes=38)
    
    # Load state_dict with strict=False
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, class_names

def preprocess_image(image, image_size=224):
    """Preprocess image for inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_disease_with_explanation(image):
    """Predict disease from image with feature extraction and explanation"""
    global model, class_names, last_image_hash, last_heatmap
    
    if image is None:
        return "❌ Please upload an image", None, None
    
    try:
        # Load model if not loaded
        if model is None:
            model, class_names = load_model()
        
        # Check if same image (avoid recomputing heatmap)
        import hashlib
        current_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        # Only recompute if image changed
        if current_hash != last_image_hash:
            print("🔄 New image detected, computing...")
            # Extract features and generate explanation
            result = extract_features_and_explain(model, image, class_names)
            
            # Cache the result
            last_image_hash = current_hash
            last_heatmap = result.get('heatmap')
        else:
            print("🔄 Using cached result...")
            # Use cached result but create new figure
            result = {
                'explanation': "🔄 Using cached analysis...",
                'predicted_class': "Cached",
                'heatmap': last_heatmap
            }
        
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
        
        # Generate heatmap visualization with overlay
        heatmap_fig = None
        try:
            heatmap = result['heatmap']
            print(f"🔍 Heatmap shape: {heatmap.shape}, min: {heatmap.min():.4f}, max: {heatmap.max():.4f}")
            
            # Create heatmap visualization with 3 plots
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
            
            # Heatmap overlay - THE COOL PART!
            import cv2
            image_np = np.array(image)
            heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
            
            # Overlay heatmap
            colormap = plt.get_cmap('jet')
            heatmap_colored = colormap(heatmap_resized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
            
            axes[2].imshow(overlay)
            axes[2].set_title('Kết hợp (Overlay) - Vùng quan trọng')
            axes[2].axis('off')
            
            plt.tight_layout()
            heatmap_fig = fig
            print("✅ Heatmap figure with overlay created successfully!")
            
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
        finally:
            # Close any previous figures to prevent memory leak
            plt.close('all')
        
        return result_text, result['predicted_class'], heatmap_fig
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

# Create Gradio interface
with gr.Blocks(title="🌱 Leaf Disease Classification", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌱 Leaf Disease Classification")
    gr.Markdown("Upload an image of a plant leaf to identify potential diseases with AI-powered analysis.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Leaf Image",
                type="pil",
                height=300
            )
            predict_btn = gr.Button("🔍 Analyze", variant="primary", interactive=True)
            
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
    gr.Markdown("## 📸 How to use:")
    gr.Markdown("1. Upload a clear image of a plant leaf")
    gr.Markdown("2. Click 'Analyze' to get AI-powered disease detection")
    gr.Markdown("3. View the heatmap to see where the AI focuses")
    gr.Markdown("4. Read the detailed analysis and disease description")
    
    # Event handlers
    predict_btn.click(
        fn=predict_disease_with_explanation,
        inputs=[image_input],
        outputs=[output_text, gr.State(), heatmap_output]
    )
    
    # Auto-predict on image upload - DISABLED to prevent auto-refresh
    # image_input.change(
    #     fn=predict_disease_with_explanation,
    #     inputs=[image_input],
    #     outputs=[output_text, gr.State(), heatmap_output]
    # )

if __name__ == "__main__":
    demo.launch()
