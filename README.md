# Bangla License Plate Detection using YOLOv11

A deep learning-based object detection project for detecting Bangla vehicle license plates using YOLOv11.

## Project Overview

This project fine-tunes a pretrained YOLOv11 model to detect vehicle-related objects from Bangla license plate images.

The model is trained using a custom Roboflow dataset containing annotated images.

## Classes

The dataset contains two object classes:

- `cng`
- `numberplate`

## Dataset

The dataset is organized into three splits:

- Training set
- Validation set
- Test set

Dataset format: YOLOv11

Dataset configuration:

```text
bangla-license-plate-detection.v4-version1.yolov11/
├── train/
├── valid/
├── test/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt

Model

The project uses:

YOLOv11n
Transfer Learning
Custom Dataset Fine-tuning
GPU acceleration with NVIDIA CUDA

Pretrained model:

yolo11n.pt
Hardware

The model was trained using an NVIDIA RTX 3050 Laptop GPU.

Training was performed using GPU device:

device=0
Training

The model was fine-tuned using:

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
Training Parameters
Parameter	Value
Model	YOLOv11n
Epochs	50
Image Size	640 × 640
Batch Size	8
Device	NVIDIA GPU
AMP	Enabled
Workers	2
Evaluation

The trained model is evaluated using standard object detection metrics:

Precision
Recall
mAP@50
mAP@50-95

The model achieved strong detection performance for the numberplate class.

Inference

After training, the best model is saved as:

runs/
└── detect/
    └── bangla_license_plate_detector-3/
        └── weights/
            ├── best.pt
            └── last.pt

The trained model can be used for inference on unseen test images.

Example:

from ultralytics import YOLO


model = YOLO(
    "runs/detect/bangla_license_plate_detector-3/weights/best.pt"
)


results = model.predict(
    source="test_image.jpg",
    conf=0.25,
    device=0
)
Project Structure
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
Technologies Used
Python
YOLOv11
Ultralytics
PyTorch
OpenCV
NumPy
Matplotlib
Pillow
CUDA
NVIDIA GPU
Installation

Install the required package:

pip install ultralytics

Additional libraries:

pip install matplotlib pillow opencv-python
How to Run
1. Open the project

Open the project folder in VS Code.

2. Open the notebook
yolo_model_finetune.ipynb
3. Select Python environment

Use Python 3.12 with GPU-enabled PyTorch.

4. Run the notebook

Run the cells sequentially:

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
Results

The trained YOLOv11 model successfully detects Bangla license plates from vehicle images.

The numberplate class achieved particularly strong detection performance during validation.

Future Improvements

Possible improvements include:

Increasing the training dataset
Improving cng detection performance
Hyperparameter tuning
Data augmentation
Training for more epochs
Testing YOLOv11s or YOLOv11m
Real-time license plate detection
Bangla license plate OCR
Integration with a web or mobile application
Author

Rashidul Hoque Chowdhury

Computer Science & Engineering

License

This project is intended for educational and research purposes.# bangla-license-plate-detection-yolov11
