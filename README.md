# 🚗 Indian Traffic AI - Real-time Vehicle Detection System

A professional-grade AI system for real-time vehicle detection and classification specifically optimized for Indian traffic scenarios. Built with YOLOv8 and Streamlit for easy deployment and usage.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Model Accuracy](https://img.shields.io/badge/mAP50-68%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-FGVD-orange)

![image](https://github.com/user-attachments/assets/d693cc5c-9fce-4d49-8b29-e823a9e361c1)
![image](https://github.com/user-attachments/assets/0907bc84-4c36-4859-a8fd-25ac1cd0b96c)

## 📋 Table of Contents
- [Features](#-features)
- [Dataset Information](#-dataset-information)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training Your Own Model](#-training-your-own-model)
- [Usage](#-usage)
- [Supported Vehicles](#-supported-vehicles)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

## ✨ Features

### 🎯 Core Capabilities
- **Real-time Camera Detection** - Live vehicle detection using webcam/camera with 27ms inference
- **Image Upload Analysis** - Upload and analyze traffic images instantly
- **Video Processing** - Frame-by-frame video analysis with progress tracking
- **Multiple Model Support** - Trained YOLO + Demo models for testing

### 📊 Detection & Analytics
- **6 Vehicle Classes** - Auto-rickshaw, Bus, Car, Motorcycle, Scooter, Truck
- **High Accuracy** - 68% mAP50 overall, 81.3% for auto-rickshaws
- **Real-time Counting** - Live vehicle count with session tracking
- **Performance Metrics** - Detailed accuracy statistics and model performance
- **Interactive Charts** - Visual analytics with Plotly integration

### 🎨 User Interface
- **Modern Web Interface** - Built with Streamlit for ease of use
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Professional Styling** - Clean, intuitive user experience
- **Multiple Tabs** - Camera, Image, Video, and Analytics sections

## 📊 Dataset Information

### FGVD Dataset (Fine-Grained Vehicle Detection)
Our model is trained on the comprehensive **FGVD Dataset** specifically designed for Indian traffic scenarios.

**Dataset Details:**
- **Total Size**: 2.6 GB
- **Source**: [IDD Dataset Portal](https://idd.insaan.iiit.ac.in/dataset/download/6a98c8b1-3f18-4391-b578-844c657952c7/)
- **Total Images**: 5,502 high-quality traffic scenes
- **Optimized for**: Indian road conditions and vehicle types

**Dataset Split:**
```
📂 TRAIN folder:
   📸 Images: 3,535 (64.2%)
   📋 Sample files: ['0.jpg', '1.jpg', '10.jpg', ...]

📂 VAL folder:
   📸 Images: 884 (16.1%)
   📋 Sample files: ['1002.jpg', '1013.jpg', '1015.jpg', ...]

📂 TEST folder:
   📸 Images: 1,083 (19.7%)
   📋 Sample files: ['4810.jpg', '4811.jpg', '4812.jpg', ...]

📊 TOTAL SUMMARY:
   📸 Total Images: 5,502
   📝 Total Annotations: 24,450 labeled vehicle instances
   🎯 Vehicle Classes: 6 types (Auto-rickshaw, Bus, Car, Motorcycle, Scooter, Truck)
```

**Dataset Features:**
- **Diverse Scenarios**: Urban, semi-urban, and highway traffic conditions
- **Weather Variations**: Clear, cloudy, and low-light conditions
- **Traffic Density**: Light to heavy traffic situations
- **Camera Angles**: Multiple viewpoints and perspectives
- **Indian Context**: Specifically curated for Indian traffic patterns

## 📈 Model Performance

### 🏆 Validation Results
Our trained YOLOv8 Nano model achieves excellent performance on Indian traffic scenarios:

| Metric | Overall | Auto Rickshaw | Car | Bus | Truck | Scooter | Motorcycle |
|--------|---------|---------------|-----|-----|-------|---------|------------|
| **mAP50** | **68.0%** | **81.3%** | **76.8%** | **65.6%** | **65.9%** | **61.5%** | **56.6%** |
| **mAP50-95** | **49.5%** | **64.6%** | **59.5%** | **51.8%** | **50.6%** | **35.7%** | **34.9%** |
| **Precision** | **71.5%** | **85.4%** | **76.4%** | **68.6%** | **62.1%** | **81.8%** | **54.5%** |
| **Recall** | **60.2%** | **69.8%** | **73.7%** | **61.2%** | **64.8%** | **39.9%** | **52.0%** |

### ⚡ Performance Metrics
- **Inference Speed**: 26.7ms per image
- **Model Size**: 5.9MB (ultra-compact for deployment)
- **Architecture**: YOLOv8 Nano (3M parameters, 8.1 GFLOPs)
- **Training Time**: ~3 hours on CPU
- **Real-time FPS**: 30+ FPS on modern hardware
- **Dataset Utilization**: Trained on 3,535 images, validated on 884 images

### 📊 Training Statistics
- **Total Dataset Images**: 5,502 high-quality scenes from FGVD
- **Training Images**: 3,535 scenes (64.2% of dataset)
- **Validation Images**: 884 scenes (16.1% of dataset)
- **Test Images**: 1,083 scenes (19.7% of dataset)
- **Total Labeled Objects**: 24,450 vehicle instances
- **Training Device**: CPU optimized (Intel Core i7-13700H)
- **Framework**: Ultralytics 8.3.144, PyTorch 2.5.0
- **Dataset Split**: Train (64%) / Val (16%) / Test (20%)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (8GB recommended)
- Webcam/Camera (for real-time detection)
- Modern CPU (GPU optional but recommended)
- Internet connection (for dataset download)

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/indian-traffic-ai.git
cd indian-traffic-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the FGVD Dataset** (optional for training)
```bash
# Download from IDD Portal (2.6 GB)
# https://idd.insaan.iiit.ac.in/dataset/download/6a98c8b1-3f18-4391-b578-844c657952c7/
# Extract to ./data/fgvd/ directory
```

4. **Download the trained model** (optional)
```bash
# Place your trained model as 'best.pt' in the project root
# Or download our pre-trained model:
wget https://github.com/yourusername/indian-traffic-ai/releases/download/v1.0/best.pt
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open in browser**
```
Local URL: http://localhost:8501
```

### Dependencies

```txt
streamlit>=1.28.0
ultralytics>=8.3.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.5.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
matplotlib>=3.7.0
psutil>=5.9.0
pyyaml>=6.0
```

## 🏃 Quick Start

### 1. Real-time Detection (Fastest Way)
```bash
# Start the app
streamlit run app.py

# Go to "📹 Real-time Camera" tab
# Click "📹 Start Camera"
# Point camera at vehicles
# See live detection results!
```

### 2. Test with Sample Image
```bash
# Go to "📸 Image Detection" tab
# Upload a traffic image
# Click "🔍 Detect Vehicles"
# View detailed results
```

### 3. Demo Mode (No Model Required)
If you don't have the trained model, the app automatically runs in demo mode with simulated detections.

## 🏋️ Training Your Own Model

### Dataset Preparation
Before training, ensure you have the FGVD dataset properly structured:

```
data/
├── fgvd/
│   ├── train/          # 3,535 images
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── val/            # 884 images
│   │   ├── 1002.jpg
│   │   ├── 1013.jpg
│   │   └── ...
│   ├── test/           # 1,083 images
│   │   ├── 4810.jpg
│   │   ├── 4811.jpg
│   │   └── ...
│   └── data.yaml       # Dataset configuration
```

### System Requirements for Training
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 5GB free space (3GB for dataset + 2GB for training)
- **CPU**: Modern multi-core processor
- **Time**: 1-3 hours depending on hardware
- **Dataset**: FGVD dataset (2.6 GB download)

### Training Configuration

Our optimized training configuration for CPU using FGVD dataset:

```python
# Ultra-optimized CPU configuration for FGVD dataset
config = {
    'model': 'yolov8n.pt',     # Nano model for speed
    'data': 'data/fgvd/data.yaml',  # FGVD dataset config
    'epochs': 30,              # Balanced training duration
    'batch': 1,                # Memory efficient
    'imgsz': 320,              # Optimized image size
    'patience': 5,             # Early stopping
    'lr0': 0.005,              # Stable learning rate
    'workers': 1,              # Single worker for stability
    'device': 'cpu',           # CPU optimized
    'save_period': 3,          # Frequent checkpoints
}
```

### Memory Management
The training script includes automatic:
- **System resource checking** before training
- **Memory optimization** with garbage collection
- **Progress monitoring** with time estimates
- **Automatic checkpointing** every 3 epochs
- **Graceful error handling** and recovery

### Training Steps

1. **Download FGVD Dataset**
```bash
# Download from: https://idd.insaan.iiit.ac.in/dataset/download/6a98c8b1-3f18-4391-b578-844c657952c7/
# Extract to ./data/fgvd/
```

2. **Verify dataset structure**:
```python
python -c "
import os
print('TRAIN images:', len(os.listdir('data/fgvd/train')))
print('VAL images:', len(os.listdir('data/fgvd/val')))
print('TEST images:', len(os.listdir('data/fgvd/test')))
"
```

3. **Check system resources**:
```python
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/(1024**3):.1f}GB')"
```

4. **Start training**:
```python
from ultralytics import YOLO

# Load and train on FGVD dataset
model = YOLO('yolov8n.pt')
results = model.train(
    data='data/fgvd/data.yaml',
    epochs=30,
    imgsz=320,
    batch=1,
    device='cpu'
)
```

5. **Monitor progress** - Training saves checkpoints every 3 epochs

## 📖 Usage

### 1. Real-time Camera Detection

```python
# Start the app and navigate to Camera tab
streamlit run app.py

# Features:
# - Live video feed with bounding boxes
# - Real-time vehicle counting
# - Session statistics tracking
# - Adjustable confidence threshold
# - Multiple camera source support
```

### 2. Image Analysis

```python
# Upload single images for analysis
# Supported formats: PNG, JPG, JPEG
# Get detailed detection results
# View annotated images with bounding boxes
# Export results and statistics
```

### 3. Video Processing

```python
# Upload video files for batch processing
# Process frame-by-frame with progress tracking
# Configurable frame skipping for speed
# Comprehensive analytics and reporting
# Timeline analysis of vehicle presence
```

### 4. Analytics Dashboard

```python
# View model performance metrics
# Session statistics and trends
# Class-wise accuracy breakdown
# Interactive charts and visualizations
# Export capabilities for reports
```

## 🚗 Supported Vehicles

The system detects 6 vehicle types commonly found in Indian traffic, optimized using the FGVD dataset:

| Vehicle | Icon | Detection Accuracy | Common Use | Dataset Representation |
|---------|------|-------------------|------------|----------------------|
| **Auto Rickshaw** | 🛺 | **81.3%** ⭐ | Public transport, Last-mile connectivity | High (well-represented) |
| **Car** | 🚗 | **76.8%** ⭐ | Private transportation | High (most common) |
| **Truck** | 🚛 | **65.9%** ✅ | Commercial goods transport | Medium |
| **Bus** | 🚌 | **65.6%** ✅ | Mass public transportation | Medium |
| **Scooter** | 🛵 | **61.5%** ✅ | Personal mobility, Food delivery | Medium |
| **Motorcycle** | 🏍️ | **56.6%** ⚠️ | Personal transport, Quick mobility | Lower (challenging detection) |

**Legend**: ⭐ Excellent (>75%) | ✅ Good (60-75%) | ⚠️ Fair (50-60%)

## 🔧 Configuration

### Detection Settings
```python
# Confidence threshold (adjust for accuracy vs detection rate)
CONFIDENCE_THRESHOLD = 0.25    # Default: 25%
LOW_CONF = 0.15               # More detections, some false positives
HIGH_CONF = 0.50              # Fewer but more accurate detections

# Image processing
IMAGE_SIZE = 640              # Default processing size
BATCH_SIZE = 1                # Batch processing size
```

### Performance Tuning
```python
# For faster processing (lower accuracy)
config = {
    'imgsz': 320,             # Smaller images
    'conf': 0.30,             # Higher confidence
    'iou': 0.45               # NMS threshold
}

# For better accuracy (slower processing)
config = {
    'imgsz': 640,             # Larger images
    'conf': 0.15,             # Lower confidence
    'iou': 0.50               # Stricter NMS
}
```

### Dataset Configuration
```yaml
# data/fgvd/data.yaml
path: ./data/fgvd
train: train
val: val
test: test

names:
  0: auto-rickshaw
  1: bus
  2: car
  3: motorcycle
  4: scooter
  5: truck

nc: 6  # number of classes
```

## 🔌 API Reference

### Core Detection Function

```python
def detect_vehicles_with_yolo(model, image, conf_threshold=0.25):
    """
    Detect vehicles in an image using YOLO model trained on FGVD dataset
    
    Args:
        model: YOLO model instance
        image: PIL Image or numpy array
        conf_threshold: Detection confidence (0.0-1.0)
    
    Returns:
        dict: {
            'detections': List of detection objects,
            'vehicle_counts': Dict of vehicle type counts,
            'total_detections': Total number of vehicles,
            'annotated_image': Image with bounding boxes
        }
    """
```

### Load Models Function

```python
def load_trained_models():
    """
    Load available YOLO models from specified paths
    
    Returns:
        tuple: (models_dict, demo_mode_boolean)
    """
```

### Example Usage

```python
from ultralytics import YOLO
from PIL import Image

# Load model trained on FGVD dataset
model = YOLO('best.pt')

# Load image
image = Image.open('traffic_scene.jpg')

# Detect vehicles
results = detect_vehicles_with_yolo(model, image, conf_threshold=0.25)

# Print results
print(f"Total vehicles: {results['total_detections']}")
for vehicle_type, count in results['vehicle_counts'].items():
    if count > 0:
        print(f"{vehicle_type}: {count}")
```



## 🔍 Troubleshooting

### Common Issues & Solutions

**1. Dataset Download Issues**
```bash
# If FGVD dataset download fails
# Try using a different browser or download manager
# Ensure stable internet connection (2.6 GB download)
# Verify dataset integrity after download
```

**2. Camera not working**
```bash
# Check camera permissions in browser
# Try different camera indices: 0, 1, 2
# Ensure no other applications are using camera
# On Linux, check: ls /dev/video*
```

**3. Model not loading**
```bash
# Verify model file exists
ls -la best.pt

# Check model format
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); print('Model loaded successfully')"

# If missing, app runs in demo mode automatically
```

**4. Out of memory errors during training**
```bash
# Reduce batch size in configuration
# Use smaller image sizes (320px instead of 640px)
# Close other applications to free RAM
# Monitor with: htop or Task Manager
# Consider using a subset of FGVD dataset for initial training
```

**5. Slow performance**
```bash
# Check if using CPU vs GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Optimize settings for your hardware
# Reduce confidence threshold for fewer detections
# Use smaller image processing size
```

**6. Installation issues**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install PyTorch separately if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Dataset Statistics & Insights

### FGVD Dataset Analysis
The FGVD (Fine-Grained Vehicle Detection) dataset provides comprehensive coverage of Indian traffic scenarios:

**Image Distribution:**
- **Training Set**: 3,535 images (64.2%) - Primary learning data
- **Validation Set**: 884 images (16.1%) - Model tuning and evaluation
- **Test Set**: 1,083 images (19.7%) - Final performance assessment

**Vehicle Class Distribution** (approximate):
- **Cars**: ~35% of total detections (most common)
- **Motorcycles/Scooters**: ~30% combined (personal transport)
- **Auto-rickshaws**: ~15% (unique to Indian context)
- **Trucks**: ~12% (commercial vehicles)
- **Buses**: ~8% (public transport)

**Scenario Coverage:**
- Urban traffic intersections
- Highway scenarios
- Mixed traffic conditions
- Day/night variations
- Different weather conditions

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/indian-traffic-ai.git
cd indian-traffic-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Download FGVD dataset for development
# Place in ./data/fgvd/ directory
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to all functions
- Include unit tests for new features

### Pull Request Process
1. Create feature branch: `git checkout -b feature/amazing-feature`
2. Make changes and test thoroughly
3. Update documentation if needed
4. Submit pull request with clear description

## 📊 Project Statistics

- **Model Accuracy**: 68% mAP50 overall
- **Best Class Performance**: Auto Rickshaw (81.3%)
- **Training Time**: ~3 hours on modern CPU
- **Model Size**: 5.9MB (ultra-compact)
- **Inference Speed**: 26.7ms per image
- **Real-time Performance**: 30+ FPS
- **Supported Vehicles**: 6 types
- **Dataset Size**: 5,502 images (FGVD dataset)
- **Training Objects**: 24,450 labeled instances
- **Dataset Download**: 2.6 GB

## 📚 References & Citations

### Dataset Citation
```bibtex
@dataset{fgvd_dataset,
  title={Fine-Grained Vehicle Detection Dataset for Indian Traffic},
  author={IDD Research Team},
  year={2024},
  url={https://idd.insaan.iiit.ac.in/},
  note={Downloaded from IDD Dataset Portal}
}
```

### Framework Citations
```bibtex
@software{ultralytics_yolo,
  title={Ultralytics YOLOv8},
  author={Ultralytics},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Dataset License**: Please refer to the FGVD dataset license terms on the [IDD portal](https://idd.insaan.iiit.ac.in/).


**Made with ❤️ for Indian Traffic Analysis using the FGVD Dataset**

*If you find this project useful, please consider giving it a ⭐ on GitHub!*

![GitHub stars](https://img.shields.io/github/stars/yourusername/indian-traffic-ai?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/indian-traffic-ai?style=social)

## 📊 Additional Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/indian-traffic-ai)
![GitHub forks](https://img.shields.io/github/forks/yourusername/indian-traffic-ai)
![GitHub issues](https://img.shields.io/github/issues/yourusername/indian-traffic-ai)
![GitHub license](https://img.shields.io/github/license/yourusername/indian-traffic-ai)
![Dataset Size](https://img.shields.io/badge/Dataset%20Size-2.6%20GB-blue)
![Total Images](https://img.shields.io/badge/Total%20Images-5,502-green)
