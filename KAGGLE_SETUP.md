# 🌱 Kaggle EfficientNet-B3 Setup Guide

## 📋 **Bước 1: Tạo Kaggle Notebook**

1. **Vào Kaggle**: https://www.kaggle.com/code
2. **Click "New Notebook"**
3. **Chọn "GPU" accelerator** (quan trọng!)
4. **Paste code** từ `kaggle_efficientnet_complete.py`

## 📊 **Bước 2: Add Dataset**

### **Option A: Plant Pathology Dataset**
```python
# Trong notebook, click "Add Data" và search:
# "Plant Pathology 2021 - FGVC 8"
# Hoặc: "Leaf Disease Dataset"
```

### **Option B: Upload Your Data**
```python
# Upload file ZIP chứa data_masked/
# Extract trong notebook:
!unzip -q /kaggle/input/your-dataset/data_masked.zip
```

## 🔧 **Bước 3: Modify Code**

### **Update Data Path:**
```python
# Thay đổi dòng này trong code:
data_dir = "/kaggle/input/plant-pathology-2021-fgvc8"  # Your dataset path
```

### **Adjust Parameters:**
```python
# Tăng epochs nếu có thời gian:
num_epochs=25  # Thay vì 15

# Tăng batch size nếu GPU mạnh:
batch_size=64  # Thay vì 32
```

## 🚀 **Bước 4: Run Training**

```python
# Chạy toàn bộ notebook
# Sẽ mất 2-4 giờ tùy GPU
```

## 📊 **Bước 5: Download Results**

```python
# Model sẽ được save:
# - efficientnet_leaf_classifier.pt
# - Training plots
# - Confusion matrix
```

## 🎯 **Expected Results:**

- ✅ **Training time**: 2-4 hours
- ✅ **Final accuracy**: 85-95%
- ✅ **GPU utilization**: 80-90%
- ✅ **Memory usage**: 8-12GB

## 🛠️ **Troubleshooting:**

### **Nếu out of memory:**
```python
batch_size = 16  # Giảm batch size
num_workers = 0  # Giảm workers
```

### **Nếu quá chậm:**
```python
num_epochs = 10  # Giảm epochs
```

### **Nếu dataset khác:**
```python
# Update data_dir path
data_dir = "/kaggle/input/your-dataset-name"
```

## 🎉 **Benefits của Kaggle:**

- ⚡ **Free GPU** (Tesla T4/P100)
- 💾 **30GB RAM**
- 🚀 **Fast training** (10x faster than local)
- 📊 **Built-in visualization**
- 💾 **Easy model saving**

## 📱 **Quick Start:**

1. **Copy code** từ `kaggle_efficientnet_complete.py`
2. **Paste vào Kaggle notebook**
3. **Add dataset** (Plant Pathology)
4. **Click "Run All"**
5. **Chờ 2-4 giờ**
6. **Download model**

## 🎯 **Pro Tips:**

- ✅ **Enable GPU** trước khi chạy
- ✅ **Save checkpoint** mỗi 5 epochs
- ✅ **Monitor GPU usage** trong output
- ✅ **Download model** trước khi hết session


