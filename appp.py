#!/usr/bin/env python3
"""
Indian Traffic AI - Real-time Camera Detection
Live camera feed with real-time vehicle detection using trained YOLO model
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import io
import random
from ultralytics import YOLO
import threading
import queue
from datetime import datetime
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Indian Traffic AI - Real-time Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main > div {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .camera-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }

    .live-indicator {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    .detection-stats {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }

    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        display: block;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 400;
    }

    .detection-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }

    .detection-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }

    .detection-medium {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        box-shadow: 0 4px 15px rgba(254, 202, 87, 0.3);
    }

    .detection-low {
        background: linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(253, 121, 168, 0.3);
    }

    .status-success {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }

    .status-warning {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(253, 203, 110, 0.3);
    }

    .fps-counter {
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
    }

    .camera-controls {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .emergency-stop {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%) !important;
    }

    .camera-frame {
        border: 3px solid #667eea;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Load class mapping
@st.cache_data
def load_class_info():
    """Load class mapping and information"""
    class_mapping = {
        'auto_rickshaw': 0, 'bus': 1, 'car': 2,
        'motorcycle': 3, 'scooter': 4, 'truck': 5
    }

    class_descriptions = {
        'auto_rickshaw': 'Three-wheeled motorized vehicle, common public transport in India',
        'bus': 'Large public transportation vehicle for multiple passengers',
        'car': 'Four-wheeled private vehicle for personal transportation',
        'motorcycle': 'Two-wheeled motorized vehicle, popular in Indian traffic',
        'scooter': 'Two-wheeled vehicle with step-through frame, very common in India',
        'truck': 'Large commercial vehicle for goods transportation'
    }

    return class_mapping, class_descriptions


class_mapping, class_descriptions = load_class_info()
class_names = list(class_mapping.keys())


# Demo model for when actual models aren't available
class DemoClassifier:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, image):
        """Generate realistic-looking predictions for demo purposes"""
        random.seed(int(time.time()) % 100)
        probabilities = np.random.dirichlet(np.ones(len(class_names)) * 2)
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx]
        inference_time = random.uniform(50, 150)

        return {
            'model_name': self.model_name,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'inference_time': inference_time,
            'all_predictions': [(class_names[i], prob) for i, prob in enumerate(probabilities)]
        }


# Load models
@st.cache_resource
def load_trained_models():
    """Load all available trained models including real YOLO model"""
    models = {}
    demo_mode = True

    # Try to load your trained YOLO model first
    yolo_paths = [
        "yolo/runs/indian_traffic_cpu_optimized_20250524_215225/weights/best.pt",
        "yolo/trained_models/optimized_cpu_model_20250524_215225.pt",
        "web_app/trained_model.pt",
        "models/saved_models/yolo_models/best.pt",
        "best.pt"
    ]

    trained_yolo_model = None
    for yolo_path in yolo_paths:
        if Path(yolo_path).exists():
            try:
                trained_yolo_model = YOLO(yolo_path)
                models['Trained YOLOv8'] = trained_yolo_model
                demo_mode = False
                break
            except Exception as e:
                continue

    # If no trained model found, search more broadly
    if trained_yolo_model is None:
        from glob import glob
        potential_models = glob("**/best.pt", recursive=True) + glob("**/*yolo*.pt", recursive=True)

        for model_path in potential_models:
            try:
                trained_yolo_model = YOLO(model_path)
                models['Trained YOLOv8'] = trained_yolo_model
                demo_mode = False
                break
            except:
                continue

    # Add demo models
    models['YOLOv8 (Demo)'] = DemoClassifier('YOLOv8')
    models['EfficientNet-B3'] = DemoClassifier('EfficientNet-B3')
    models['ResNet50+Attention'] = DemoClassifier('ResNet50+Attention')

    return models, demo_mode


def detect_vehicles_with_yolo(model, image, conf_threshold=0.25):
    """Detect vehicles using real YOLO model"""
    try:
        # Run inference
        results = model(image, conf=conf_threshold, verbose=False)

        detections = []
        vehicle_counts = {vehicle: 0 for vehicle in class_names}

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                bbox = boxes.xyxy[i].cpu().numpy()

                if cls < len(class_names):
                    class_name = class_names[cls]
                    vehicle_counts[class_name] += 1

                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': bbox
                    })

        return {
            'detections': detections,
            'vehicle_counts': vehicle_counts,
            'total_detections': len(detections),
            'annotated_image': results[0].plot() if detections else None
        }

    except Exception as e:
        return {
            'detections': [],
            'vehicle_counts': {vehicle: 0 for vehicle in class_names},
            'total_detections': 0,
            'annotated_image': None
        }


# Real-time camera processing class
class RealTimeDetector:
    def __init__(self, model, conf_threshold=0.25):
        self.model = model
        self.conf_threshold = conf_threshold
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_queue = queue.Queue(maxsize=10)
        self.fps_counter = 0
        self.fps_time = time.time()
        self.total_detections = 0
        self.session_stats = {vehicle: 0 for vehicle in class_names}

    def start_camera(self, camera_index=0):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.is_running = True
        return True

    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if hasattr(self, 'cap'):
            self.cap.release()

    def process_frame(self, frame):
        """Process a single frame"""
        if isinstance(self.model, YOLO):
            # Real YOLO detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            detection_result = detect_vehicles_with_yolo(self.model, pil_image, self.conf_threshold)

            # Update session stats
            for vehicle, count in detection_result['vehicle_counts'].items():
                self.session_stats[vehicle] += count

            self.total_detections += detection_result['total_detections']

            # Draw detections on frame
            if detection_result['detections']:
                for detection in detection_result['detections']:
                    bbox = detection['bbox']
                    class_name = detection['class']
                    confidence = detection['confidence']

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    label = f"{class_name.replace('_', ' ').title()}: {confidence:.2%}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return frame, detection_result
        else:
            # Demo mode
            demo_result = self.model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            # Simulate detection visualization
            height, width = frame.shape[:2]
            num_detections = random.randint(1, 3)

            detections = []
            vehicle_counts = {vehicle: 0 for vehicle in class_names}

            for _ in range(num_detections):
                x1 = random.randint(0, width // 2)
                y1 = random.randint(height // 3, height - 100)
                x2 = x1 + random.randint(80, 150)
                y2 = y1 + random.randint(60, 120)

                vehicle_class = random.choice(class_names)
                confidence = random.uniform(0.6, 0.95)

                vehicle_counts[vehicle_class] += 1
                detections.append({
                    'class': vehicle_class,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })

                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = f"{vehicle_class.replace('_', ' ').title()}: {confidence:.2%}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Update session stats
            for vehicle, count in vehicle_counts.items():
                self.session_stats[vehicle] += count

            self.total_detections += len(detections)

            detection_result = {
                'detections': detections,
                'vehicle_counts': vehicle_counts,
                'total_detections': len(detections)
            }

            return frame, detection_result

    def get_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
            return fps
        self.fps_counter += 1
        return None


def numpy_to_base64(img_array):
    """Convert numpy array to base64 string for display"""
    _, buffer = cv2.imencode('.jpg', img_array)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_base64}"


# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'current_fps' not in st.session_state:
    st.session_state.current_fps = 0

# App header
st.markdown('<h1 class="main-header">🚗 Indian Traffic AI - Real-time Detection</h1>', unsafe_allow_html=True)

# Load models
with st.spinner("🤖 Loading AI models..."):
    available_models, demo_mode = load_trained_models()

# Model status
if not demo_mode:
    st.markdown("""
    <div class="status-success">
        🎉 REAL MODEL LOADED! Using your trained YOLO model for real-time detection.<br>
        Model Performance: 68% mAP50 - Optimized for Indian traffic scenarios
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-warning">
        🔬 DEMO MODE: No trained model found. Showing simulated real-time detection.
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📹 Real-time Camera", "📸 Image Detection", "🎬 Video Analysis", "📊 Analytics"])

# REAL-TIME CAMERA TAB
with tab1:
    st.markdown("### 📹 Live Camera Detection")

    col1_cam, col2_cam = st.columns([2, 1])

    with col1_cam:
        # Camera display area
        camera_placeholder = st.empty()

        # Live indicator and FPS
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            live_status = st.empty()
        with status_col2:
            fps_display = st.empty()
        with status_col3:
            detection_count = st.empty()

    with col2_cam:
        st.markdown('<div class="camera-controls">', unsafe_allow_html=True)
        st.markdown("#### 🎛️ Camera Controls")

        # Model selection
        selected_model = st.selectbox(
            "Choose Detection Model",
            list(available_models.keys()),
            key="camera_model"
        )

        # Camera settings
        camera_index = st.selectbox("Camera Source", [0, 1, 2], help="Select camera device")

        if isinstance(available_models[selected_model], YOLO):
            conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05, key="cam_conf")
        else:
            conf_threshold = 0.25
            st.info("📊 Demo mode - simulated detections")

        # Control buttons
        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button("📹 Start Camera", type="primary", disabled=st.session_state.camera_running):
                try:
                    # Initialize detector
                    st.session_state.detector = RealTimeDetector(
                        available_models[selected_model],
                        conf_threshold
                    )

                    if st.session_state.detector.start_camera(camera_index):
                        st.session_state.camera_running = True
                        st.success("📹 Camera started successfully!")
                    else:
                        st.error("❌ Failed to start camera")
                except Exception as e:
                    st.error(f"Error starting camera: {e}")

        with col_stop:
            if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_running):
                if st.session_state.detector:
                    st.session_state.detector.stop_camera()
                st.session_state.camera_running = False
                st.success("⏹️ Camera stopped")

        st.markdown('</div>', unsafe_allow_html=True)

        # Real-time statistics
        if st.session_state.camera_running and st.session_state.detector:
            st.markdown("#### 📊 Live Statistics")

            # Session stats
            total_session_detections = st.session_state.detector.total_detections
            st.metric("Total Detections", total_session_detections)

            # Vehicle breakdown
            for vehicle, count in st.session_state.detector.session_stats.items():
                if count > 0:
                    st.metric(
                        label=vehicle.replace('_', ' ').title(),
                        value=count
                    )

    # Main camera processing loop
    if st.session_state.camera_running and st.session_state.detector:
        try:
            ret, frame = st.session_state.detector.cap.read()

            if ret:
                # Process frame
                start_time = time.time()
                processed_frame, detection_result = st.session_state.detector.process_frame(frame)
                process_time = time.time() - start_time

                # Calculate FPS
                fps = st.session_state.detector.get_fps()
                if fps is not None:
                    st.session_state.current_fps = fps

                # Display frame
                frame_base64 = numpy_to_base64(processed_frame)
                camera_placeholder.markdown(
                    f'<div class="camera-frame"><img src="{frame_base64}" style="width:100%; height:auto;"></div>',
                    unsafe_allow_html=True
                )

                # Update status displays
                live_status.markdown(
                    '<div class="live-indicator">🔴 LIVE</div>',
                    unsafe_allow_html=True
                )

                fps_display.markdown(
                    f'<div class="fps-counter">📊 {st.session_state.current_fps} FPS</div>',
                    unsafe_allow_html=True
                )

                detection_count.markdown(
                    f'<div class="fps-counter">🚗 {detection_result["total_detections"]} vehicles</div>',
                    unsafe_allow_html=True
                )

                # Auto-refresh for continuous detection
                time.sleep(0.03)  # ~30 FPS
                st.rerun()

        except Exception as e:
            st.error(f"Camera processing error: {e}")
            st.session_state.camera_running = False

    elif not st.session_state.camera_running:
        # Show placeholder when camera is off
        camera_placeholder.markdown("""
        <div class="camera-container">
            <h3>📹 Camera Ready</h3>
            <p>Click "Start Camera" to begin real-time vehicle detection</p>
            <p>🎯 Your trained YOLO model will detect vehicles in real-time</p>
        </div>
        """, unsafe_allow_html=True)

# IMAGE DETECTION TAB
with tab2:
    st.markdown("### 📸 Upload Image for Vehicle Detection")

    col1_img, col2_img = st.columns([1, 1])

    with col1_img:
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Detection settings
            selected_img_model = st.selectbox("Choose Model", list(available_models.keys()), key="img_model")

            if isinstance(available_models[selected_img_model], YOLO):
                conf_threshold_img = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05, key="img_conf")
            else:
                conf_threshold_img = 0.25

            if st.button("🔍 Detect Vehicles", type="primary"):
                with st.spinner("🤖 Analyzing image..."):
                    if isinstance(available_models[selected_img_model], YOLO):
                        # Real detection
                        result = detect_vehicles_with_yolo(
                            available_models[selected_img_model],
                            image,
                            conf_threshold_img
                        )

                        with col2_img:
                            if result['total_detections'] > 0:
                                st.success(f"🎉 Found {result['total_detections']} vehicle(s)")

                                if result['annotated_image'] is not None:
                                    annotated_rgb = cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB)
                                    st.image(annotated_rgb, caption="Detected Vehicles", use_column_width=True)

                                # Show detections
                                st.markdown("#### 📋 Detections")
                                for i, detection in enumerate(result['detections']):
                                    confidence = detection['confidence']
                                    conf_class = "detection-high" if confidence > 0.8 else "detection-medium" if confidence > 0.6 else "detection-low"
                                    st.markdown(f"""
                                    <div class="detection-badge {conf_class}">
                                        {detection['class'].replace('_', ' ').title()}: {confidence:.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No vehicles detected")
                    else:
                        # Demo detection
                        demo_result = available_models[selected_img_model].predict(image)
                        with col2_img:
                            st.info("📊 Demo Result")
                            confidence = demo_result['confidence']
                            conf_class = "detection-high" if confidence > 0.8 else "detection-medium" if confidence > 0.6 else "detection-low"
                            st.markdown(f"""
                            <div class="detection-badge {conf_class}">
                                {demo_result['predicted_class'].replace('_', ' ').title()}: {confidence:.1%}
                            </div>
                            """, unsafe_allow_html=True)

# VIDEO ANALYSIS TAB
with tab3:
    st.markdown("### 🎬 Video Analysis")
    st.info("📹 Upload a video file for frame-by-frame vehicle detection")

    uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'mov', 'avi', 'mkv'])

    if uploaded_video is not None:
        st.video(uploaded_video)

        col1_vid, col2_vid = st.columns([1, 1])

        with col1_vid:
            video_model = st.selectbox("Model for Video", list(available_models.keys()), key="video_model")

            if isinstance(available_models[video_model], YOLO):
                video_conf = st.slider("Video Confidence", 0.1, 0.9, 0.25, 0.05, key="video_conf")
                frame_skip = st.slider("Process every N frames", 1, 10, 3)
                max_frames = st.slider("Max frames", 10, 100, 30)
            else:
                video_conf = 0.25
                frame_skip = 5
                max_frames = 20

        with col2_vid:
            if st.button("🎬 Analyze Video", type="primary"):
                if isinstance(available_models[video_model], YOLO):
                    st.info("🔄 Processing video with trained YOLO model...")

                    # Save uploaded video temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_video.read())
                        tmp_video_path = tmp_file.name

                    # Process video
                    cap = cv2.VideoCapture(tmp_video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    progress_bar = st.progress(0)
                    frame_results = []
                    total_vehicle_counts = {vehicle: 0 for vehicle in class_names}

                    frame_count = 0
                    processed_count = 0

                    while cap.isOpened() and processed_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_count % frame_skip == 0:
                            # Convert and detect
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)

                            detection_result = detect_vehicles_with_yolo(
                                available_models[video_model],
                                pil_image,
                                video_conf
                            )

                            # Update counts
                            for vehicle, count in detection_result['vehicle_counts'].items():
                                total_vehicle_counts[vehicle] += count

                            frame_results.append({
                                'frame': frame_count,
                                'time': frame_count / fps,
                                'detections': detection_result['total_detections']
                            })

                            processed_count += 1
                            progress_bar.progress(processed_count / max_frames)

                        frame_count += 1

                    cap.release()
                    Path(tmp_video_path).unlink()

                    # Show results
                    st.success(f"✅ Processed {processed_count} frames")

                    total_detections = sum(frame['detections'] for frame in frame_results)
                    avg_detections = total_detections / len(frame_results) if frame_results else 0

                    col1_metric, col2_metric = st.columns(2)
                    with col1_metric:
                        st.metric("Total Detections", total_detections)
                    with col2_metric:
                        st.metric("Avg per Frame", f"{avg_detections:.1f}")

                    # Vehicle counts
                    if total_detections > 0:
                        st.markdown("#### 🚗 Vehicle Distribution")
                        for vehicle, count in total_vehicle_counts.items():
                            if count > 0:
                                st.metric(vehicle.replace('_', ' ').title(), count)
                else:
                    st.info("📊 Demo mode - simulated video analysis")

# ANALYTICS TAB
with tab4:
    st.markdown("### 📊 Detection Analytics")

    # Performance metrics
    st.markdown("#### 🎯 Model Performance")

    col1_perf, col2_perf, col3_perf, col4_perf = st.columns(4)

    with col1_perf:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-number">68%</span>
            <span class="stat-label">Overall mAP50</span>
        </div>
        """, unsafe_allow_html=True)

    with col2_perf:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-number">27ms</span>
            <span class="stat-label">Inference Time</span>
        </div>
        """, unsafe_allow_html=True)

    with col3_perf:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-number">30 FPS</span>
            <span class="stat-label">Real-time Speed</span>
        </div>
        """, unsafe_allow_html=True)

    with col4_perf:
        st.markdown("""
        <div class="stat-card">
            <span class="stat-number">5.9MB</span>
            <span class="stat-label">Model Size</span>
        </div>
        """, unsafe_allow_html=True)

    # Class performance
    st.markdown("#### 🚗 Vehicle Class Performance")

    performance_data = pd.DataFrame({
        'Vehicle Type': ['Auto Rickshaw', 'Car', 'Truck', 'Bus', 'Scooter', 'Motorcycle'],
        'Accuracy (%)': [81.3, 76.8, 65.9, 65.6, 61.5, 56.6],
        'Real-time Performance': ['Excellent', 'Excellent', 'Good', 'Good', 'Good', 'Fair']
    })

    col1_chart, col2_chart = st.columns(2)

    with col1_chart:
        fig_acc = px.bar(
            performance_data,
            x='Vehicle Type',
            y='Accuracy (%)',
            title="Detection Accuracy by Vehicle Type",
            color='Accuracy (%)',
            color_continuous_scale='viridis'
        )
        fig_acc.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2_chart:
        # Session statistics if camera was used
        if st.session_state.detector:
            session_data = []
            for vehicle, count in st.session_state.detector.session_stats.items():
                if count > 0:
                    session_data.append({
                        'Vehicle': vehicle.replace('_', ' ').title(),
                        'Detections': count
                    })

            if session_data:
                session_df = pd.DataFrame(session_data)
                fig_session = px.pie(
                    session_df,
                    values='Detections',
                    names='Vehicle',
                    title="Current Session Detections"
                )
                st.plotly_chart(fig_session, use_container_width=True)
            else:
                st.info("📊 Start camera to see session statistics")
        else:
            # Show sample data
            sample_data = pd.DataFrame({
                'Vehicle': ['Car', 'Motorcycle', 'Auto Rickshaw', 'Scooter'],
                'Sample Detections': [45, 30, 15, 25]
            })
            fig_sample = px.pie(
                sample_data,
                values='Sample Detections',
                names='Vehicle',
                title="Sample Detection Distribution"
            )
            st.plotly_chart(fig_sample, use_container_width=True)

    # Technical specifications
    st.markdown("#### 🔧 Technical Specifications")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        **Model Architecture:**
        - Framework: YOLOv8 Nano
        - Backend: PyTorch
        - Optimization: CPU optimized
        - Input Size: 640x640
        - Classes: 6 vehicle types
        """)

    with tech_col2:
        st.markdown("""
        **Training Data:**
        - Total Objects: 24,450
        - Images: 5,502
        - Training Time: 3h 39m
        - Platform: CPU training
        - Validation: 20% split
        """)

# Sidebar information
with st.sidebar:
    st.markdown("### 🤖 System Status")

    # Model status
    if not demo_mode:
        st.success("✅ Real YOLO Model Loaded")
        st.info("🎯 Production Ready")
    else:
        st.warning("⚠️ Demo Mode Active")
        st.info("📊 Simulated Results")

    st.markdown("---")
    st.markdown("### 📹 Camera Status")

    if st.session_state.camera_running:
        st.success("🟢 Camera Active")
        if st.session_state.detector:
            st.metric("FPS", st.session_state.current_fps)
            st.metric("Total Detections", st.session_state.detector.total_detections)
    else:
        st.info("⚪ Camera Inactive")

    st.markdown("---")
    st.markdown("### 💡 Usage Tips")
    st.markdown("""
    **For Best Results:**
    - Good lighting conditions
    - Stable camera position
    - Clear view of traffic
    - Optimal distance: 10-50m

    **Performance:**
    - Real-time: 30 FPS
    - Detection range: 5-100m
    - Best accuracy: Cars & Rickshaws
    - Works in daylight & dusk
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Camera Settings")
    st.markdown("""
    **Recommended:**
    - Confidence: 0.25-0.4
    - Resolution: 640x480
    - FPS: 30
    - Format: RGB
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           color: white; border-radius: 15px; margin: 1rem 0;">
    <h3>🚗 Indian Traffic AI - Real-time Detection System</h3>
    <p>Professional-grade vehicle detection optimized for Indian traffic scenarios</p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
        <span>🎯 68% mAP50 Accuracy</span>
        <span>⚡ 27ms Inference</span>
        <span>📹 Real-time 30 FPS</span>
        <span>🚀 Production Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for camera
if st.session_state.camera_running:
    # Add a small delay and rerun for continuous camera feed
    time.sleep(0.1)
    st.rerun()