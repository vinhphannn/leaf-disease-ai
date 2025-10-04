---
title: Leaf Disease Classification
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# 🌱 Leaf Disease Classification

An AI-powered web application for identifying plant diseases from leaf images using EfficientNet-B0 deep learning model.

## 🚀 Features

- **AI Disease Detection**: Identify 38 different plant diseases with high accuracy
- **Grad-CAM Visualization**: See exactly where the AI focuses on the leaf
- **Natural Language Explanations**: Get detailed explanations in Vietnamese
- **Real-time Analysis**: Instant predictions with confidence scores
- **Feature Analysis**: Understand which regions of the leaf are most important

## 🔬 Supported Diseases

The model can identify diseases in:
- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust
- **Tomato**: Late Blight, Early Blight, Bacterial Spot, Leaf Mold
- **Potato**: Late Blight, Early Blight
- **Corn**: Common Rust, Northern Leaf Blight, Cercospora Leaf Spot
- **Grape**: Black Rot, Esca, Leaf Blight
- **And many more...**

## 🎯 How to Use

1. **Upload Image**: Take a clear photo of a plant leaf
2. **Get Analysis**: The AI will analyze the image and provide:
   - Disease prediction with confidence score
   - Grad-CAM heatmap showing focus areas
   - Detailed feature analysis
   - Disease-specific information and prevention tips

## 🛠️ Technical Details

- **Model**: EfficientNet-B0 (4.6M parameters)
- **Accuracy**: 99.14% on validation set
- **Framework**: PyTorch + Gradio
- **Visualization**: Grad-CAM for interpretability
- **Language**: Vietnamese explanations

## 📊 Model Performance

- **Training Data**: 70,295 images
- **Validation Data**: 17,572 images
- **Classes**: 38 plant diseases
- **Architecture**: EfficientNet-B0 with custom classifier
- **Training**: Transfer learning from ImageNet

## 🔍 Interpretability

The application provides:
- **Grad-CAM heatmaps** showing where the model focuses
- **Region analysis** with position and intensity descriptions
- **Natural language explanations** of the AI's reasoning
- **Disease-specific information** and prevention tips

## 🚀 Deployment

This application is deployed on Hugging Face Spaces for easy access and sharing.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.