#!/usr/bin/env python3
"""
Feature extraction and visualization for leaf disease classification
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Dict
import os

class FeatureExtractor:
    """Extract and visualize features that the model focuses on"""
    
    def __init__(self, model, target_layer_name="backbone.blocks.6.0"):
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks(target_layer_name)
    
    def _register_hooks(self, layer_name):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            return None
        
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is not None:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_gradcam(self, image_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        self.model.eval()
        image_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(image_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate Grad-CAM
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Normalize
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam
    
    def analyze_heatmap_regions(self, heatmap, image_size=(224, 224)):
        """Analyze heatmap to extract meaningful features"""
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, image_size)
        
        # Threshold to get significant regions
        threshold = 0.3
        significant_regions = heatmap_resized > threshold
        
        # Find contours
        contours, _ = cv2.findContours(
            (significant_regions * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze regions
        regions_info = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small regions
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate region properties
                area = cv2.contourArea(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Position relative to image
                rel_x = center_x / image_size[0]
                rel_y = center_y / image_size[1]
                
                regions_info.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (center_x, center_y),
                    'relative_position': (rel_x, rel_y),
                    'intensity': np.mean(heatmap_resized[y:y+h, x:x+w])
                })
        
        return regions_info
    
    def generate_natural_language_explanation(self, class_name, confidence, regions_info, heatmap):
        """Generate natural language explanation"""
        explanations = []
        
        # Basic prediction
        explanations.append(f"🌱 **Dự đoán**: {class_name.replace('___', ' - ')}")
        explanations.append(f"🎯 **Độ tin cậy**: {confidence:.1%}")
        
        # Analyze regions
        if regions_info:
            explanations.append("\n🔍 **Phân tích đặc trưng:**")
            
            # Sort regions by intensity
            regions_info.sort(key=lambda x: x['intensity'], reverse=True)
            
            for i, region in enumerate(regions_info[:3]):  # Top 3 regions
                rel_x, rel_y = region['relative_position']
                intensity = region['intensity']
                
                # Position description
                if rel_x < 0.3:
                    pos_x = "bên trái"
                elif rel_x > 0.7:
                    pos_x = "bên phải"
                else:
                    pos_x = "giữa"
                
                if rel_y < 0.3:
                    pos_y = "phần trên"
                elif rel_y > 0.7:
                    pos_y = "phần dưới"
                else:
                    pos_y = "phần giữa"
                
                # Intensity description
                if intensity > 0.7:
                    intensity_desc = "rất mạnh"
                elif intensity > 0.5:
                    intensity_desc = "mạnh"
                elif intensity > 0.3:
                    intensity_desc = "trung bình"
                else:
                    intensity_desc = "yếu"
                
                # Detailed region analysis
                region_desc = f"   {i+1}. **Vùng {pos_x} {pos_y}** của lá có tín hiệu {intensity_desc} "
                region_desc += f"(cường độ: {intensity:.2f})"
                
                # Add specific analysis based on position and intensity
                if intensity > 0.7:
                    region_desc += " - Mô hình rất chú ý đến vùng này"
                elif intensity > 0.5:
                    region_desc += " - Mô hình chú ý vừa phải đến vùng này"
                else:
                    region_desc += " - Mô hình ít chú ý đến vùng này"
                
                # Add position-specific insights
                if rel_x < 0.3 and rel_y < 0.3:
                    region_desc += " (góc trên trái)"
                elif rel_x > 0.7 and rel_y < 0.3:
                    region_desc += " (góc trên phải)"
                elif rel_x < 0.3 and rel_y > 0.7:
                    region_desc += " (góc dưới trái)"
                elif rel_x > 0.7 and rel_y > 0.7:
                    region_desc += " (góc dưới phải)"
                elif 0.3 <= rel_x <= 0.7 and 0.3 <= rel_y <= 0.7:
                    region_desc += " (trung tâm lá)"
                
                explanations.append(region_desc)
        else:
            explanations.append("\n🔍 **Phân tích đặc trưng:**")
            explanations.append("   Mô hình tập trung vào toàn bộ lá một cách đồng đều")
        
        # Disease-specific explanations
        disease_explanations = self._get_disease_specific_explanations(class_name)
        if disease_explanations:
            explanations.append(f"\n📋 **Mô tả bệnh:**")
            explanations.extend(disease_explanations)
        
        return "\n".join(explanations)
    
    def _get_disease_specific_explanations(self, class_name):
        """Get disease-specific explanations"""
        explanations = {
            'Apple___Apple_scab': [
                "• **Bệnh ghẻ táo (Apple Scab)**: Bệnh nấm phổ biến nhất trên cây táo",
                "• **Triệu chứng**: Các đốm đen nhỏ, tròn, có viền rõ ràng trên lá",
                "• **Vị trí**: Thường bắt đầu từ mặt dưới lá, sau đó lan ra mặt trên",
                "• **Tiến triển**: Lá bị vàng, khô và rụng sớm, ảnh hưởng đến năng suất",
                "• **Điều kiện**: Phát triển mạnh trong thời tiết ẩm ướt, mưa nhiều"
            ],
            'Apple___Black_rot': [
                "• **Bệnh thối đen (Black Rot)**: Bệnh nấm nghiêm trọng trên cây táo",
                "• **Triệu chứng**: Các đốm nâu đen lớn, không đều, có hình tròn",
                "• **Đặc điểm**: Viền rõ ràng, trung tâm có thể bị khô và nứt",
                "• **Ảnh hưởng**: Lá bị khô, cong, ảnh hưởng đến quá trình quang hợp",
                "• **Phòng ngừa**: Cần cắt tỉa cành bệnh, phun thuốc phòng ngừa"
            ],
            'Apple___Cedar_apple_rust': [
                "• **Bệnh gỉ sắt táo (Cedar Apple Rust)**: Bệnh nấm phức tạp, cần 2 vật chủ",
                "• **Triệu chứng**: Các đốm cam đỏ nhỏ, có thể có viền vàng",
                "• **Đặc điểm**: Thường xuất hiện thành cụm, có thể lan rộng",
                "• **Vòng đời**: Cần cây bách xù (cedar) và cây táo để hoàn thành chu trình",
                "• **Phòng ngừa**: Tránh trồng táo gần cây bách xù, phun thuốc vào mùa xuân"
            ],
            'Tomato___Late_blight': [
                "• Bệnh mốc sương: Đốm nâu đen, lan nhanh",
                "• Thường xuất hiện ở mép lá trước",
                "• Lá có thể bị héo và chết"
            ],
            'Tomato___Early_blight': [
                "• Bệnh đốm sớm: Đốm nâu với vòng tròn đồng tâm",
                "• Thường bắt đầu từ lá già",
                "• Có thể lan ra toàn bộ cây"
            ],
            'Potato___Late_blight': [
                "• Bệnh mốc sương khoai tây: Đốm nâu đen lớn",
                "• Thường lan nhanh trong điều kiện ẩm ướt",
                "• Có thể gây chết toàn bộ cây"
            ],
            'Corn___Common_rust': [
                "• Bệnh gỉ sắt ngô: Các đốm cam đỏ nhỏ",
                "• Thường xuất hiện thành cụm",
                "• Có thể lan ra toàn bộ lá"
            ]
        }
        
        return explanations.get(class_name, [
            "• Mô hình đã phát hiện các dấu hiệu bệnh đặc trưng",
            "• Khuyến nghị kiểm tra thêm để xác định chính xác",
            "• Nên tham khảo chuyên gia nông nghiệp nếu cần"
        ])
    
    def visualize_heatmap(self, image, heatmap, save_path=None):
        """Visualize heatmap overlay on image"""
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Create colormap
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Ảnh gốc')
        axes[0].axis('off')
        
        # Heatmap
        im1 = axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Kết hợp')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

def extract_features_and_explain(model, image, class_names, device='cpu'):
    """Main function to extract features and generate explanations"""
    extractor = FeatureExtractor(model)
    
    # Preprocess image
    if isinstance(image, Image.Image):
        image_tensor = preprocess_image(image)
    else:
        image_tensor = image
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class_idx = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
        predicted_class = class_names[predicted_class_idx]
    
    # Generate Grad-CAM
    heatmap = extractor.generate_gradcam(image_tensor, predicted_class_idx)
    
    # Analyze regions
    regions_info = extractor.analyze_heatmap_regions(heatmap)
    
    # Generate explanation
    explanation = extractor.generate_natural_language_explanation(
        predicted_class, confidence, regions_info, heatmap
    )
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'heatmap': heatmap,
        'regions_info': regions_info,
        'explanation': explanation
    }

def preprocess_image(image, image_size=224):
    """Preprocess image for inference"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor
