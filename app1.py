#!/usr/bin/env python3
"""
Gradio Web App cho Leaf Disease Classification
Sử dụng mô hình mới với TTA và confidence threshold
"""

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import json
from pathlib import Path
import numpy as np

from src.train import SimpleClassifier
from src.preprocess import get_transforms

class LeafDiseaseApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.class_names = {}
        self.load_models()
    
    def load_models(self):
        """Load các mô hình đã train"""
        # Load species model
        species_model_path = Path("models/species_v2/model_best.pt")
        if species_model_path.exists():
            ckpt = torch.load(species_model_path, map_location="cpu", weights_only=False)
            self.class_names["species"] = ckpt.get("classes", [])
            
            # Load model architecture từ checkpoint
            if "model_architecture" in ckpt:
                # Nếu có architecture info trong checkpoint
                model = ckpt["model_architecture"]
            else:
                # Fallback: tạo SimpleClassifier với số classes đúng
                model = SimpleClassifier(num_classes=len(self.class_names["species"]))
            
            # Load state dict với strict=False để bỏ qua missing keys
            model.load_state_dict(ckpt["state_dict"], strict=False)
            self.models["species"] = model.to(self.device)
            print(f"✅ Loaded species model: {len(self.class_names['species'])} classes")
        
        # Load disease models
        disease_dir = Path("models/disease_v2")
        if disease_dir.exists():
            for species_dir in disease_dir.iterdir():
                if species_dir.is_dir():
                    model_path = species_dir / "model_best.pt"
                    if model_path.exists():
                        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                        species_name = species_dir.name
                        self.class_names[f"disease_{species_name}"] = ckpt.get("classes", [])
                        
                        # Load model architecture từ checkpoint
                        if "model_architecture" in ckpt:
                            model = ckpt["model_architecture"]
                        else:
                            model = SimpleClassifier(num_classes=len(self.class_names[f"disease_{species_name}"]))
                        
                        # Load state dict với strict=False
                        model.load_state_dict(ckpt["state_dict"], strict=False)
                        self.models[f"disease_{species_name}"] = model.to(self.device)
                        print(f"✅ Loaded disease model for {species_name}: {len(self.class_names[f'disease_{species_name}'])} classes")
    
    def predict_species(self, image, use_tta=True, threshold=0.5):
        """Dự đoán loài cây"""
        if "species" not in self.models:
            return "Species model not found", 0.0, "❌"
        
        try:
            # Preprocess
            _, val_tfms = get_transforms(224)
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            x = val_tfms(image).unsqueeze(0).to(self.device)
            
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
                print(f"🔍 Debug - Pred: {pred_class}, Conf: {conf_val:.4f}, Threshold: {threshold}")
                
                # Check threshold
                if conf_val < threshold:
                    return f"Uncertain (conf: {conf_val:.3f})", conf_val, "⚠️"
                else:
                    return pred_class, conf_val, "✅"
                    
        except Exception as e:
            print(f"❌ Error in predict_species: {e}")
            return f"Error: {str(e)}", 0.0, "❌"
    
    def predict_disease(self, image, species, use_tta=True, threshold=0.5):
        """Dự đoán bệnh cho loài cây cụ thể"""
        model_key = f"disease_{species}"
        if model_key not in self.models:
            return f"Disease model for {species} not found", 0.0, "❌"
        
        # Preprocess
        _, val_tfms = get_transforms(224)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        x = val_tfms(image).unsqueeze(0).to(self.device)
        
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
        
        # Check threshold
        if conf_val < threshold:
            return f"Uncertain (conf: {conf_val:.3f})", conf_val, "⚠️"
        else:
            return pred_class, conf_val, "✅"

def create_interface():
    app = LeafDiseaseApp()
    
    with gr.Blocks(title="🌱 Leaf Disease AI - Enhanced Model", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌱 Leaf Disease Classification AI
        **Enhanced Model với CutMix/Mixup + TTA + Confidence Threshold**
        
        Upload ảnh lá cây để phân loại loài và bệnh với mô hình mới được cải tiến!
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
                
                predict_btn = gr.Button("🔍 Phân tích", variant="primary", size="lg")
            
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
        
        # Event handlers
        def predict_all(image, tta, thresh):
            if image is None:
                return "Vui lòng upload ảnh", 0.0, "❌", "Vui lòng upload ảnh", 0.0, "❌"
            
            # Predict species
            species_pred, species_conf, species_stat = app.predict_species(image, tta, thresh)
            
            # Predict disease (if species is certain)
            if species_stat == "✅" and "Uncertain" not in species_pred:
                disease_pred, disease_conf, disease_stat = app.predict_disease(image, species_pred, tta, thresh)
            else:
                disease_pred, disease_conf, disease_stat = "Cần xác định loài trước", 0.0, "⚠️"
            
            return species_pred, species_conf, species_stat, disease_pred, disease_conf, disease_stat
        
        predict_btn.click(
            fn=predict_all,
            inputs=[image_input, use_tta, threshold],
            outputs=[species_result, species_conf, species_status, disease_result, disease_conf, disease_status]
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
