# 🚘 Bangla License Plate Detection using YOLOv11

A deep learning-based object detection system for detecting **Bangla vehicle license plates** using **YOLOv11** and a custom annotated dataset.

---

## 📌 Project Overview

This project focuses on detecting vehicle-related objects from images using a fine-tuned **YOLOv11n** object detection model.

A custom dataset containing annotated Bangla vehicle images was used for training and evaluation.

The model is capable of detecting:

- 🚗 CNG
- 🔢 Bangla Number Plate

---

## 🎯 Objectives

- Detect Bangla vehicle number plates from images
- Detect CNG vehicles
- Fine-tune a pretrained YOLOv11 model
- Utilize GPU acceleration for faster training
- Evaluate model performance using standard object detection metrics
- Perform inference on unseen test images

---

## 🧠 Model

The project uses:

- **YOLOv11n**
- Transfer Learning
- Custom Dataset Fine-tuning
- GPU-accelerated training
- Automatic Mixed Precision (AMP)

Pretrained model:

```text
yolo11n.pt
```

---

## 🏷️ Detection Classes

The dataset contains two object classes:

| Class ID | Class |
|----------|-------|
| 0 | CNG |
| 1 | Number Plate |

---

## 📂 Dataset

The dataset was prepared in YOLO format and contains three major splits:

```text
bangla-license-plate-detection.v4-version1.yolov11/
│
├── train/
├── valid/
├── test/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

### Dataset Splits

- **Train** – Model training
- **Validation** – Model validation during training
- **Test** – Final model evaluation and inference

---

## 💻 Hardware

Training was performed locally using an NVIDIA GPU.

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 6 GB |
| RAM | 16 GB |
| OS | Windows 11 |
| Python | 3.12 |

GPU training was enabled using:

```python
device=0
```

---

## ⚙️ Training Configuration

The YOLOv11 model was fine-tuned using the following configuration:

```python
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    name="bangla_license_plate_detector",
    device=0,
    workers=2,
    amp=True,
    cache=False
)
```

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Model | YOLOv11n |
| Epochs | 50 |
| Image Size | 640 × 640 |
| Batch Size | 8 |
| Device | NVIDIA GPU |
| Workers | 2 |
| AMP | Enabled |
| Cache | Disabled |

---

## 📊 Evaluation Metrics

The trained model was evaluated using standard object detection metrics:

- **Precision**
- **Recall**
- **mAP@50**
- **mAP@50-95**

### Validation Performance

The overall validation performance was:

| Metric | Score |
|--------|-------|
| Precision | 72.03% |
| Recall | 50.41% |
| mAP@50 | 53.28% |
| mAP@50-95 | 45.82% |

The `numberplate` class achieved particularly strong detection performance.

### Number Plate Performance

| Metric | Score |
|--------|-------|
| Precision | 96.2% |
| Recall | 92.5% |
| mAP@50 | 94.0% |
| mAP@50-95 | 82.9% |

The `cng` class showed comparatively lower recall and remains an area for future improvement.

---

## 📈 Training Results

Training results, validation metrics, plots, and model weights are automatically saved by Ultralytics inside the `runs/` directory.

Example:

```text
runs/
└── detect/
    └── bangla_license_plate_detector-3/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        ├── results.png
        ├── confusion_matrix.png
        └── ...
```

---

## 🔍 Inference

After training, the best-performing model can be loaded using:

```python
from ultralytics import YOLO

model = YOLO(
    "runs/detect/bangla_license_plate_detector-3/weights/best.pt"
)
```

Run detection on an image:

```python
results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    device=0
)
```

The detected image will be saved by Ultralytics in the corresponding `runs/detect/` directory.

---

## 🗂️ Project Structure

```text
yolo_model_finetune/
│
├── bangla-license-plate-detection.v4-version1.yolov11/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── data.yaml
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── runs/
│   └── detect/
│       └── bangla_license_plate_detector-3/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── yolo_model_finetune.ipynb
├── yolo11n.pt
├── yolo26n.pt
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- YOLOv11
- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pillow
- CUDA
- NVIDIA GPU

---

## 📦 Installation

Install Ultralytics:

```bash
pip install ultralytics
```

Install additional libraries:

```bash
pip install matplotlib pillow opencv-python
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/therash08/bangla-license-plate-detection-yolov11.git
```

### 2. Open the project

```bash
cd bangla-license-plate-detection-yolov11
```

### 3. Open the notebook

Open:

```text
yolo_model_finetune.ipynb
```

### 4. Select Python Environment

Use a Python 3.12 environment with GPU-enabled PyTorch.

### 5. Run the notebook

Run the notebook cells sequentially:

```text
Setup
   ↓
Dataset Loading
   ↓
Dataset Configuration
   ↓
YOLOv11 Fine-tuning
   ↓
Model Evaluation
   ↓
Test Image Inference
```

---

## 🚀 Future Improvements

Future versions of this project can include:

- 📈 Increasing the training dataset
- 🎯 Improving CNG detection performance
- ⚙️ Hyperparameter optimization
- 🔄 Advanced data augmentation
- 🧠 Testing YOLOv11s / YOLOv11m
- 🎥 Real-time license plate detection
- 🔤 Bangla license plate OCR
- 🚗 Vehicle tracking
- 🌐 Web-based detection application
- 📱 Mobile application integration

---

## 👨‍💻 Author

**Rasidul Hoque Chowdhury**

Computer Science & Engineering

GitHub: [@therash08](https://github.com/therash08)

---

## 📄 License

This project is intended for **educational and research purposes**.

---