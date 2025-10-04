# 🌱 Leaf Disease AI - Plant Pathology Classification

Dự án phân loại bệnh cây trồng sử dụng Deep Learning với hai kiến trúc model chính: **MobileNet V3** và **EfficientNet-B3**.

## 📁 Cấu trúc dự án

```
leaf_disease_ai/
├── 📊 data/                    # Dataset gốc (bị ignore)
├── 📊 data_masked/             # Dataset đã xử lý (bị ignore)
├── 🤖 models/                  # Thư mục chứa model đã train
│   ├── mobilenet_v3/          # Models MobileNet V3
│   ├── efficientnet_b3/       # Models EfficientNet-B3
│   └── disease/               # Models theo từng loại bệnh
├── 🔧 src/                     # Source code
│   ├── mobilenet_v3/          # Implementation MobileNet V3
│   │   ├── app.py            # Web app cho MobileNet V3
│   │   ├── train.py          # Training script
│   │   ├── train_disease.py  # Train disease classifier
│   │   └── train_species.py  # Train species classifier
│   ├── efficientnet_b3/       # Implementation EfficientNet-B3
│   │   ├── app.py            # Web app cho EfficientNet-B3
│   │   ├── model.py          # Model definition
│   │   ├── model_b0.py       # EfficientNet-B0 variant
│   │   ├── train_disease.py  # Train disease classifier
│   │   └── train_species.py  # Train species classifier
│   └── shared/                # Code chung
│       ├── data_utils.py     # Data utilities
│       ├── preprocess.py     # Data preprocessing
│       ├── utils.py          # Common utilities
│       └── evaluate.py       # Evaluation functions
├── 🌐 app.py                  # Main web application
├── 📋 requirements.txt        # Dependencies
└── 📄 README.md              # This file
```

## 🚀 Cài đặt

```bash
# Clone repository
git clone https://github.com/vinhphannn/leaf-disease-ai.git
cd leaf_disease_ai

# Cài đặt dependencies
pip install -r requirements.txt
```

## 🎯 Sử dụng

### 1. Training Models

#### MobileNet V3 (Nhanh, nhẹ)
```bash
# Train species classifier
python -m src.mobilenet_v3.train_species --data_dir data_masked --epochs 12

# Train disease classifier cho từng loại cây
python -m src.mobilenet_v3.train_disease --data_dir data_masked --species Apple --epochs 12
```

#### EfficientNet-B3 (Chính xác cao)
```bash
# Train species classifier
python -m src.efficientnet_b3.train_species --data_dir data_masked --epochs 15

# Train disease classifier cho từng loại cây
python -m src.efficientnet_b3.train_disease --data_dir data_masked --species Apple --epochs 15
```

### 2. Chạy Web Application

```bash
# Chạy app chính (hỗ trợ cả hai model)
python app.py

# Hoặc chạy riêng từng model
python -m src.mobilenet_v3.app
python -m src.efficientnet_b3.app
```

## 📊 Models

### MobileNet V3
- **Ưu điểm**: Nhanh, nhẹ, phù hợp mobile/edge devices
- **Sử dụng**: Khi cần tốc độ và tiết kiệm tài nguyên
- **Accuracy**: ~85-90%

### EfficientNet-B3
- **Ưu điểm**: Chính xác cao, state-of-the-art performance
- **Sử dụng**: Khi cần độ chính xác tối đa
- **Accuracy**: ~92-95%

## 🌿 Supported Plants & Diseases

### Plants (14 loại)
- Apple, Blueberry, Cherry, Corn (maize)
- Grape, Orange, Peach, Pepper (bell)
- Potato, Raspberry, Soybean, Squash
- Strawberry, Tomato

### Diseases
Mỗi loại cây có các bệnh đặc trưng, ví dụ:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, etc.
- **Potato**: Early blight, Late blight, Healthy

## 🔧 Features

- ✅ **Dual Model Support**: MobileNet V3 + EfficientNet-B3
- ✅ **Web Interface**: Gradio-based UI
- ✅ **Batch Processing**: Xử lý nhiều ảnh cùng lúc
- ✅ **Confidence Scores**: Hiển thị độ tin cậy
- ✅ **Heatmap Visualization**: Giải thích kết quả
- ✅ **Model Comparison**: So sánh hiệu suất
- ✅ **Export Results**: Xuất kết quả CSV

## 📈 Performance

| Model | Accuracy | Speed | Size | Use Case |
|-------|----------|-------|------|----------|
| MobileNet V3 | ~87% | ⚡⚡⚡ | 5MB | Mobile/Edge |
| EfficientNet-B3 | ~94% | ⚡⚡ | 25MB | Desktop/Server |

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 📞 Contact

- GitHub: [@vinhphannn](https://github.com/vinhphannn)
- Project: [Leaf Disease AI](https://github.com/vinhphannn/leaf-disease-ai)

---

**🌱 Giúp nông dân phát hiện bệnh cây trồng sớm và chính xác!**