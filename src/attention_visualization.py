#!/usr/bin/env python3
"""
Attention Visualization và Feature Extraction cho EfficientNet-B3
- Grad-CAM
- Attention maps
- Feature visualization
- Saliency maps
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Optional
import seaborn as sns

from .efficientnet_model import EfficientNetLeafClassifier, EfficientNetDiseaseClassifier


class AttentionVisualizer:
    """Class để visualize attention và features"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook để capture intermediate features
        self.features = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Đăng ký hooks để capture features và gradients"""
        def forward_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        # Hook cho backbone features
        if hasattr(self.model, 'backbone'):
            # Hook cho layer cuối của backbone
            target_layer = self.model.backbone.features[-1]
            target_layer.register_forward_hook(forward_hook('backbone_features'))
            target_layer.register_backward_hook(backward_hook('backbone_features'))
        
        # Hook cho multi-branch features
        if hasattr(self.model, 'shape_branch'):
            self.model.shape_branch[-2].register_forward_hook(forward_hook('shape_features'))
            self.model.texture_branch[-2].register_forward_hook(forward_hook('texture_features'))
            self.model.color_branch[-2].register_forward_hook(forward_hook('color_features'))
    
    def generate_gradcam(self, image: torch.Tensor, class_idx: int, 
                        target_layer: str = 'backbone_features') -> np.ndarray:
        """Generate Grad-CAM visualization"""
        # Forward pass
        image.requires_grad_()
        logits = self.model(image)
        
        # Backward pass
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        
        # Get gradients and features
        if target_layer in self.gradients and target_layer in self.features:
            gradients = self.gradients[target_layer]
            features = self.features[target_layer]
            
            # Global average pooling của gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Weighted combination của feature maps
            cam = torch.sum(weights * features, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Convert to numpy
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam
        else:
            return None
    
    def generate_attention_maps(self, image: torch.Tensor) -> dict:
        """Generate attention maps cho tất cả branches"""
        with torch.no_grad():
            _ = self.model(image)
        
        attention_maps = {}
        
        # Shape attention
        if 'shape_features' in self.features:
            shape_feat = self.features['shape_features']
            shape_attention = torch.mean(shape_feat, dim=1).squeeze().cpu().numpy()
            attention_maps['shape'] = (shape_attention - shape_attention.min()) / (shape_attention.max() - shape_attention.min() + 1e-8)
        
        # Texture attention
        if 'texture_features' in self.features:
            texture_feat = self.features['texture_features']
            texture_attention = torch.mean(texture_feat, dim=1).squeeze().cpu().numpy()
            attention_maps['texture'] = (texture_attention - texture_attention.min()) / (texture_attention.max() - texture_attention.min() + 1e-8)
        
        # Color attention
        if 'color_features' in self.features:
            color_feat = self.features['color_features']
            color_attention = torch.mean(color_feat, dim=1).squeeze().cpu().numpy()
            attention_maps['color'] = (color_attention - color_attention.min()) / (color_attention.max() - color_attention.min() + 1e-8)
        
        return attention_maps
    
    def generate_saliency_map(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate saliency map"""
        image.requires_grad_()
        logits = self.model(image)
        
        # Backward pass
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        
        # Get gradients w.r.t input
        saliency = image.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def visualize_all_attentions(self, image: torch.Tensor, class_idx: int, 
                                class_name: str, original_image: np.ndarray) -> plt.Figure:
        """Visualize tất cả attention maps"""
        
        # Generate all attention maps
        gradcam = self.generate_gradcam(image, class_idx)
        attention_maps = self.generate_attention_maps(image)
        saliency = self.generate_saliency_map(image, class_idx)
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'Original Image\nPredicted: {class_name}')
        axes[0, 0].axis('off')
        
        # Grad-CAM
        if gradcam is not None:
            gradcam_resized = cv2.resize(gradcam, (original_image.shape[1], original_image.shape[0]))
            axes[0, 1].imshow(gradcam_resized, cmap='jet')
            axes[0, 1].set_title('Grad-CAM\n(Overall Attention)')
            axes[0, 1].axis('off')
        
        # Saliency map
        saliency_resized = cv2.resize(saliency, (original_image.shape[1], original_image.shape[0]))
        axes[0, 2].imshow(saliency_resized, cmap='hot')
        axes[0, 2].set_title('Saliency Map\n(Input Sensitivity)')
        axes[0, 2].axis('off')
        
        # Combined overlay
        axes[0, 3].imshow(original_image)
        if gradcam is not None:
            axes[0, 3].imshow(gradcam_resized, cmap='jet', alpha=0.4)
        axes[0, 3].set_title('Overlay\n(Original + Grad-CAM)')
        axes[0, 3].axis('off')
        
        # Multi-branch attention maps
        branch_names = ['shape', 'texture', 'color']
        for i, branch in enumerate(branch_names):
            if branch in attention_maps:
                att_map = attention_maps[branch]
                att_resized = cv2.resize(att_map, (original_image.shape[1], original_image.shape[0]))
                
                axes[1, i].imshow(att_resized, cmap='viridis')
                axes[1, i].set_title(f'{branch.capitalize()} Attention')
                axes[1, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, f'{branch.capitalize()}\nNot Available', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis('off')
        
        # Feature importance
        if len(attention_maps) > 0:
            feature_importance = []
            feature_names = []
            for branch, att_map in attention_maps.items():
                feature_importance.append(np.mean(att_map))
                feature_names.append(branch.capitalize())
            
            axes[1, 3].bar(feature_names, feature_importance, color=['red', 'green', 'blue'])
            axes[1, 3].set_title('Feature Importance')
            axes[1, 3].set_ylabel('Mean Attention')
            axes[1, 3].tick_params(axis='x', rotation=45)
        else:
            axes[1, 3].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                           ha='center', va='center', transform=axes[1, 3].transAxes)
            axes[1, 3].axis('off')
        
        plt.tight_layout()
        return fig
    
    def extract_features(self, image: torch.Tensor) -> dict:
        """Extract và return features từ tất cả branches"""
        with torch.no_grad():
            logits, combined_features = self.model(image, return_features=True)
            prob = F.softmax(logits, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()
            confidence = prob[0, pred_class].item()
        
        features = {
            'logits': logits.cpu().numpy(),
            'probabilities': prob.cpu().numpy(),
            'predicted_class': pred_class,
            'confidence': confidence,
            'combined_features': combined_features.cpu().numpy()
        }
        
        # Add attention maps
        attention_maps = self.generate_attention_maps(image)
        features.update(attention_maps)
        
        return features


def visualize_model_attention(model, image_path: str, class_names: List[str], 
                            device: torch.device, output_path: str = None) -> dict:
    """Main function để visualize attention cho một ảnh"""
    
    # Load và preprocess image
    from .preprocess import get_transforms
    _, val_tfms = get_transforms(224)
    
    with Image.open(image_path).convert("RGB") as img:
        original_image = np.array(img)
        image_tensor = val_tfms(img).unsqueeze(0).to(device)
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(image_tensor)
        prob = F.softmax(logits, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0, pred_class].item()
    
    class_name = class_names[pred_class] if pred_class < len(class_names) else "Unknown"
    
    # Generate visualization
    fig = visualizer.visualize_all_attentions(
        image_tensor, pred_class, class_name, original_image
    )
    
    # Save figure
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Attention visualization saved to: {output_path}")
    
    # Extract features
    features = visualizer.extract_features(image_tensor)
    
    return {
        'prediction': class_name,
        'confidence': confidence,
        'features': features,
        'figure': fig
    }


def batch_visualize_attention(model, image_paths: List[str], class_names: List[str], 
                            device: torch.device, output_dir: str) -> List[dict]:
    """Visualize attention cho nhiều ảnh"""
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        print(f"🔍 Processing {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        try:
            output_path = output_dir / f"attention_{Path(image_path).stem}.png"
            result = visualize_model_attention(
                model, image_path, class_names, device, str(output_path)
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            results.append({'error': str(e), 'image_path': image_path})
    
    return results


