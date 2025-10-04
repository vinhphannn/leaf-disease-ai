#!/usr/bin/env python3
"""
Gradio Web App cho EfficientNet-B3 Leaf Disease Classification
- Attention visualization
- Feature extraction
- Multi-branch analysis
"""

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.efficientnet_model import create_species_model, create_disease_model
from src.attention_visualization import AttentionVisualizer, visualize_model_attention
from src.preprocess import get_transforms

class EfficientNetLeafDiseaseApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.class_names = {}
        self.visualizers = {}
        self.load_models()
    
    def load_models(self):
        """Load EfficientNet models"""
        # Load species model
        species_model_path = Path("models/efficientnet_species/model_best.pt")
        if species_model_path.exists():
            ckpt = torch.load(species_model_path, map_location="cpu", weights_only=False)
            self.class_names["species"] = ckpt.get("classes", [])
            model = create_species_model(num_classes=len(self.class_names["species"]))
            model.load_state_dict(ckpt["state_dict"], strict=False)
            self.models["species"] = model.to(self.device)
            self.visualizers["species"] = AttentionVisualizer(model, self.device)
            print(f"✅ Loaded EfficientNet species model: {len(self.class_names['species'])} classes")
        
        # Load disease models
        disease_dir = Path("models/efficientnet_disease")
        if disease_dir.exists():
            for species_dir in disease_dir.iterdir():
                if species_dir.is_dir():
                    model_path = species_dir / "model_best.pt"
                    if model_path.exists():
                        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                        species_name = species_dir.name
                        self.class_names[f"disease_{species_name}"] = ckpt.get("classes", [])
                        model = create_disease_model(num_classes=len(self.class_names[f"disease_{species_name}"]))
                        model.load_state_dict(ckpt["state_dict"], strict=False)
                        self.models[f"disease_{species_name}"] = model.to(self.device)
                        self.visualizers[f"disease_{species_name}"] = AttentionVisualizer(model, self.device)
                        print(f"✅ Loaded EfficientNet disease model for {species_name}: {len(self.class_names[f'disease_{species_name}'])} classes")
    
    def predict_species_with_attention(self, image, use_tta=True, threshold=0.5, show_attention=True):
        """Dự đoán loài cây với attention visualization"""
        if "species" not in self.models:
            return "Species model not found", 0.0, "❌", None
        
        try:
            # Preprocess
            _, val_tfms = get_transforms(224)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            x = val_tfms(image).unsqueeze(0).to(self.device)
            original_image = np.array(image)
            
            with torch.no_grad():
                if use_tta:
                    # TTA: flips
                    xs = [x]
                    xs.append(torch.flip(x, dims=[-1]))  # hflip
                    xs.append(torch.flip(x, dims=[-2]))  # vflip
                    logits = torch.stack([self.models["species"](v) for v in xs]).mean(dim=0)
                else:
                    logits = self.models["species"](x)
                
                prob = F.softmax(logits, dim=1)[0]
                conf, pred = torch.max(prob, dim=0)
                conf_val = conf.item()
                pred_class = self.class_names["species"][pred.item()]
                
                # Debug info
                print(f"🔍 Species - Pred: {pred_class}, Conf: {conf_val:.4f}, Threshold: {threshold}")
                
                # Generate attention visualization
                attention_fig = None
                if show_attention and conf_val >= threshold:
                    try:
                        attention_fig = self.visualizers["species"].visualize_all_attentions(
                            x, pred.item(), pred_class, original_image
                        )
                    except Exception as e:
                        print(f"⚠️  Attention visualization failed: {e}")
                
                # Check threshold
                if conf_val < threshold:
                    return f"Uncertain (conf: {conf_val:.3f})", conf_val, "⚠️", attention_fig
                else:
                    return pred_class, conf_val, "✅", attention_fig
                    
        except Exception as e:
            print(f"❌ Error in predict_species: {e}")
            return f"Error: {str(e)}", 0.0, "❌", None
    
    def predict_disease_with_attention(self, image, species, use_tta=True, threshold=0.5, show_attention=True):
        """Dự đoán bệnh với attention visualization"""
        model_key = f"disease_{species}"
        if model_key not in self.models:
            return f"Disease model for {species} not found", 0.0, "❌", None
        
        try:
            # Preprocess
            _, val_tfms = get_transforms(224)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            x = val_tfms(image).unsqueeze(0).to(self.device)
            original_image = np.array(image)
            
            with torch.no_grad():
                if use_tta:
                    # TTA: flips
                    xs = [x]
                    xs.append(torch.flip(x, dims=[-1]))  # hflip
                    xs.append(torch.flip(x, dims=[-2]))  # vflip
                    logits = torch.stack([self.models[model_key](v) for v in xs]).mean(dim=0)
                else:
                    logits = self.models[model_key](x)
                
                prob = F.softmax(logits, dim=1)[0]
                conf, pred = torch.max(prob, dim=0)
                conf_val = conf.item()
                pred_class = self.class_names[model_key][pred.item()]
                
                # Debug info
                print(f"🔍 Disease - Pred: {pred_class}, Conf: {conf_val:.4f}, Threshold: {threshold}")
                
                # Generate attention visualization
                attention_fig = None
                if show_attention and conf_val >= threshold:
                    try:
                        attention_fig = self.visualizers[model_key].visualize_all_attentions(
                            x, pred.item(), pred_class, original_image
                        )
                    except Exception as e:
                        print(f"⚠️  Attention visualization failed: {e}")
                
                # Check threshold
                if conf_val < threshold:
                    return f"Uncertain (conf: {conf_val:.3f})", conf_val, "⚠️", attention_fig
                else:
                    return pred_class, conf_val, "✅", attention_fig
                    
        except Exception as e:
            print(f"❌ Error in predict_disease: {e}")
            return f"Error: {str(e)}", 0.0, "❌", None

def create_interface():
    app = EfficientNetLeafDiseaseApp()
    
    with gr.Blocks(title="🌱 EfficientNet-B3 Leaf Disease AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌱 EfficientNet-B3 Leaf Disease Classification AI
        **Advanced Model với Multi-Branch Features + Attention Visualization**
        
        Upload ảnh lá cây để phân loại loài và bệnh với mô hình EfficientNet-B3 cải tiến!
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="📸 Upload ảnh lá cây",
                    type="numpy",
                    height=300
                )
                
                with gr.Row():
                    use_tta = gr.Checkbox(
                        label="🔄 Test-Time Augmentation (TTA)",
                        value=True,
                        info="Cải thiện độ ổn định"
                    )
                    threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="🎯 Confidence Threshold",
                        info="Ngưỡng tin cậy tối thiểu"
                    )
                    show_attention = gr.Checkbox(
                        label="👁️ Show Attention",
                        value=True,
                        info="Hiển thị attention maps"
                    )
                
                predict_btn = gr.Button("🔍 Phân tích với Attention", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### 📊 Kết quả phân loại")
                
                with gr.Row():
                    species_result = gr.Textbox(
                        label="🌿 Loài cây",
                        interactive=False
                    )
                    species_conf = gr.Number(
                        label="Confidence",
                        interactive=False,
                        precision=3
                    )
                    species_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                
                with gr.Row():
                    disease_result = gr.Textbox(
                        label="🦠 Tình trạng bệnh",
                        interactive=False
                    )
                    disease_conf = gr.Number(
                        label="Confidence",
                        interactive=False,
                        precision=3
                    )
                    disease_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
        
        # Attention visualization
        gr.Markdown("### 👁️ Attention Visualization")
        attention_plot = gr.Plot(
            label="Attention Maps",
            show_label=True
        )
        
        # Event handlers
        def predict_all_with_attention(image, tta, thresh, show_att):
            if image is None:
                return "Vui lòng upload ảnh", 0.0, "❌", "Vui lòng upload ảnh", 0.0, "❌", None
            
            # Predict species with attention
            species_pred, species_conf, species_stat, species_attention = app.predict_species_with_attention(
                image, tta, thresh, show_att
            )
            
            # Predict disease (if species is certain)
            if species_stat == "✅" and "Uncertain" not in species_pred:
                disease_pred, disease_conf, disease_stat, disease_attention = app.predict_disease_with_attention(
                    image, species_pred, tta, thresh, show_att
                )
                # Use disease attention if available, otherwise species attention
                final_attention = disease_attention if disease_attention is not None else species_attention
            else:
                disease_pred, disease_conf, disease_stat, final_attention = "Cần xác định loài trước", 0.0, "⚠️", species_attention
            
            return (species_pred, species_conf, species_stat, 
                   disease_pred, disease_conf, disease_stat, final_attention)
        
        predict_btn.click(
            fn=predict_all_with_attention,
            inputs=[image_input, use_tta, threshold, show_attention],
            outputs=[species_result, species_conf, species_status, 
                    disease_result, disease_conf, disease_status, attention_plot]
        )
        
        # Examples
        gr.Markdown("### 📁 Ảnh mẫu để test")
        example_images = []
        test_dir = Path("data/test")
        if test_dir.exists():
            example_images = list(test_dir.glob("*.JPG"))[:6]
        
        if example_images:
            gr.Examples(
                examples=[[str(img)] for img in example_images],
                inputs=image_input,
                label="Click để test với ảnh mẫu"
            )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )


