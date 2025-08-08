#!/usr/bin/env python3
"""
Real-time Video Depth Estimation with Object Detection using RT-MonoDepth
Provides human detection with accurate depth measurements (no depth map visualization)
Optimized for MacBook M1 Pro and Jetson Nano deployment
"""

import argparse
from PIL import Image
import cv2
import numpy as np
import torch
import platform
import time
import os
from tqdm import tqdm
import threading
from queue import Queue, Empty
import json
import signal
import sys
import csv
from datetime import datetime

# RT-MonoDepth imports
from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder
from networks.RTMonoDepth.RTMonoDepth_s import DepthDecoder as DepthDecoderS, DepthEncoder as DepthEncoderS
from layers import disp_to_depth
from torchvision import transforms

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO (ultralytics) is available for object detection.")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not found. To enable object detection: pip install ultralytics")

# MLX Configuration
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
MLX_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import mlx.core as mx
        import mlx.nn as mlx_nn
        MLX_AVAILABLE = True
        print("✅ MLX is available for Apple Silicon acceleration.")
    except ImportError:
        print("⚠️ MLX not found. To enable MLX acceleration: pip install mlx")

# Configuration
OUTPUT_FOLDER = "output_depth_video"
DEPTH_LOG_FOLDER = "depth_logs"

# YOLO Configuration - Custom YOLOv11n model
YOLO_MODEL_PATH = 'custom_yolo11n.pt'  # Custom YOLOv11n model in current directory
CLASSES_TO_DETECT = {"person"}  # Focus on person detection only
CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for better detection with custom model

DEFAULT_CAMERA_PARAMS = {
    "macbook_m1_pro": {
        "fx": 640.0,  # More realistic focal length for MacBook webcam
        "fy": 640.0,  # Assuming square pixels
        "cx": 320.0,  # Principal point at center
        "cy": 240.0,
        "width": 640,
        "height": 480
    },
    "jetson_nano": {
        "fx": 800.0,
        "fy": 800.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480
    }
}


class RTMonoDepthModel:
    """RT-MonoDepth model wrapper with MLX acceleration"""
    
    def __init__(self, weight_path, device='cpu', use_mlx=True):
        self.device = device
        self.use_mlx = use_mlx and MLX_AVAILABLE
        
        print(f"Loading RT-MonoDepth from: {weight_path}")
        
        # Determine model type
        self.is_small_model = "/s/" in weight_path or "_s" in weight_path
        
        # Load encoder
        encoder_path = os.path.join(weight_path, "encoder.pth")
        if self.is_small_model:
            self.encoder = DepthEncoderS()
            print("   Using small model architecture")
        else:
            self.encoder = DepthEncoder()
            print("   Using full model architecture")
            
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device).eval()
        
        # Load decoder
        decoder_path = os.path.join(weight_path, "depth.pth")
        if self.is_small_model:
            self.decoder = DepthDecoderS(num_ch_enc=self.encoder.num_ch_enc)
        else:
            self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
            
        loaded_dict = torch.load(decoder_path, map_location=device)
        self.decoder.load_state_dict(loaded_dict)
        self.decoder.to(device).eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.feed_height, self.feed_width)),
            transforms.ToTensor(),
        ])
        
        if self.use_mlx:
            print("   MLX acceleration enabled")
        
        print(f"   Model loaded: {self.feed_width}x{self.feed_height}")
    
    def predict_depth(self, rgb_image, camera_params=None, depth_scale_factor=1.0):
        """Predict raw monocular depth then apply external scale.
        The network (Monodepth2-style) produces metric depth up to an unknown global scale.
        We avoid arbitrary nonlinear scaling and leave scale resolution to calibration.
        """
        # Preprocess image
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.encoder(input_tensor)
            outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        _, depth = disp_to_depth(disp, 0.1, 100)  # Returns depth proportional to metric (unknown scale)
        # Apply (potentially combined user * auto) scale factor ONLY
        metric_depth = depth * depth_scale_factor
        return metric_depth.squeeze().cpu().numpy()


class YOLODetector:
    """YOLO object detection wrapper"""
    
    def __init__(self, model_path=YOLO_MODEL_PATH, device='cpu'):
        self.device = device
        self.model = None
        
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO not available - object detection disabled")
            return
        
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                model_name = os.path.basename(model_path)
                print(f"✅ YOLO model loaded from: {model_path}")
                
                # Check if it's a custom trained model
                if "custom" in model_path.lower() or "best.pt" in model_path:
                    print("🔧 Custom-trained YOLO model detected")
                    print("   Using custom weights for optimized person detection")
                    # Print model classes for debugging
                    if hasattr(self.model, 'names'):
                        print(f"   Available classes: {list(self.model.names.values())}")
                else:
                    print(f"   Model: {model_name}")
                    
            else:
                # Fallback to YOLOv11n
                print(f"⚠️  Specified model not found: {model_path}")
                print("   Using standard YOLOv11n for person detection")
                self.model = YOLO('yolo11n.pt')
                print("✅ Fallback: YOLO model loaded: yolo11n.pt")
                
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("   Object detection will be disabled")
            self.model = None
    
    def detect_objects(self, image, confidence=CONFIDENCE_THRESHOLD):
        """Detect objects in image and return bounding boxes"""
        if self.model is None:
            return []
        
        try:
            results = self.model(image, device=self.device, verbose=False, conf=confidence)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, cls_idx, conf in zip(boxes, classes, confidences):
                        class_name = self.model.names[int(cls_idx)]
                        
                        # Filter out unwanted classes and detect person only
                        unwanted_classes = {'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Hardhat', 'Safety Vest', 'Safety Cone', 'machinery', 'vehicle'}
                        if class_name in unwanted_classes:
                            continue
                            
                        # Detect person class (case-insensitive)
                        if class_name.lower() != "person":
                            continue
                            
                        # Debug: Print person detections only
                        print(f"✅ Person detected: {confidence:.2f}")
                        
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'class': class_name,
                            'confidence': float(conf)
                        })
            
            return detections
            
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            return []


class FPSCounter:
    """FPS counter utility"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        """Update FPS counter with new frame"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0


class DepthLogger:
    """Real-time depth logging for detected objects"""
    
    def __init__(self, log_interval=60, enabled=False, measure_distances=False):
        self.log_interval = log_interval  # seconds
        self.enabled = enabled
        self.measure_distances = measure_distances
        self.last_log_time = time.time()
        self.depth_data = []
        self.log_file = None
        self.csv_writer = None
        self.distance_log_file = None
        self.distance_csv_writer = None
        
        if self.enabled:
            self._setup_log_files()
    
    def _setup_log_files(self):
        """Setup CSV log files with headers"""
        os.makedirs(DEPTH_LOG_FOLDER, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main depth log file
        self.log_file_path = os.path.join(DEPTH_LOG_FOLDER, f"depth_log_{timestamp}.csv")
        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Write headers for depth log
        self.csv_writer.writerow([
            'timestamp', 'datetime', 'frame_count', 'object_class', 
            'confidence', 'depth_meters', 'bbox_x1', 'bbox_y1', 
            'bbox_x2', 'bbox_y2', 'center_x', 'center_y'
        ])
        self.log_file.flush()
        print(f"📊 Depth logging enabled: {self.log_file_path}")
        
        # Distance measurement log file (if enabled)
        if self.measure_distances:
            self.distance_log_file_path = os.path.join(DEPTH_LOG_FOLDER, f"distance_log_{timestamp}.csv")
            self.distance_log_file = open(self.distance_log_file_path, 'w', newline='')
            self.distance_csv_writer = csv.writer(self.distance_log_file)
            
            # Write headers for distance log
            self.distance_csv_writer.writerow([
                'timestamp', 'datetime', 'frame_count', 'obj1_class', 'obj2_class',
                'distance_3d_meters', 'depth_difference', 'obj1_depth', 'obj2_depth',
                'obj1_x', 'obj1_y', 'obj2_x', 'obj2_y'
            ])
            self.distance_log_file.flush()
            print(f"📐 Distance logging enabled: {self.distance_log_file_path}")
    
    def should_log(self, current_time):
        """Check if it's time to log data"""
        if not self.enabled:
            return False
        return (current_time - self.last_log_time) >= self.log_interval
    
    def log_detections(self, detections, depth_map, frame_count, camera_params=None):
        """Log detection data with depth measurements and distances"""
        if not self.enabled or not self.csv_writer:
            return
        
        current_time = time.time()
        if not self.should_log(current_time):
            return
        
        # Update last log time
        self.last_log_time = current_time
        
        # Create timestamp
        timestamp = current_time
        datetime_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        if not detections:
            # Log no detections
            self.csv_writer.writerow([
                timestamp, datetime_str, frame_count, 'none', 
                0.0, 0.0, 0, 0, 0, 0, 0, 0
            ])
            print(f"📊 Logged: No objects detected at {datetime_str}")
        else:
            # Log each detection
            for detection in detections:
                bbox = detection['bbox']
                center = detection['center']
                class_name = detection['class']
                confidence = detection['confidence']
                
                x1, y1, x2, y2 = bbox
                center_x, center_y = center
                
                # Calculate depth at detection center
                # Ensure coordinates are within bounds
                center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                
                # Sample multiple points for robust depth measurement
                sample_points = [
                    (center_x, center_y),  # Center
                    (center_x, min(center_y + 20, depth_map.shape[0] - 1)),  # Lower
                    (max(center_x - 10, 0), center_y),  # Left
                    (min(center_x + 10, depth_map.shape[1] - 1), center_y),  # Right
                ]
                
                depths = []
                for px, py in sample_points:
                    px, py = int(px), int(py)
                    if 0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]:
                        depth_at_point = depth_map[py, px]
                        depths.append(depth_at_point)
                
                # Use median depth for robustness
                depth_value = float(np.median(depths)) if depths else 0.0
                
                # Log to CSV
                self.csv_writer.writerow([
                    timestamp, datetime_str, frame_count, class_name,
                    float(confidence), depth_value, int(x1), int(y1),
                    int(x2), int(y2), int(center_x), int(center_y)
                ])
                
                print(f"📊 Logged: {class_name} at {depth_value:.2f}m ({datetime_str})")
            
            # Log distance measurements if enabled and we have multiple objects
            if self.measure_distances and self.distance_csv_writer and len(detections) >= 2 and camera_params:
                self._log_distances(detections, depth_map, frame_count, timestamp, datetime_str, camera_params)
        
        # Flush to ensure data is written
        self.log_file.flush()
        if self.distance_log_file:
            self.distance_log_file.flush()
    
    def _log_distances(self, detections, depth_map, frame_count, timestamp, datetime_str, camera_params):
        """Log distances between all pairs of detected objects"""
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                dist_info = calculate_object_distance(detections[i], detections[j], depth_map, camera_params)
                if dist_info:
                    obj1 = detections[i]
                    obj2 = detections[j]
                    
                    self.distance_csv_writer.writerow([
                        timestamp, datetime_str, frame_count,
                        obj1['class'], obj2['class'],
                        dist_info['distance_3d'], dist_info['depth_difference'],
                        dist_info['object1_depth'], dist_info['object2_depth'],
                        obj1['center'][0], obj1['center'][1],
                        obj2['center'][0], obj2['center'][1]
                    ])
                    
                    print(f"📐 Distance logged: {obj1['class']} ↔ {obj2['class']} = {dist_info['distance_3d']:.2f}m")
    
    def close(self):
        """Close log files"""
        if self.log_file:
            self.log_file.close()
            print(f"📊 Depth log saved: {self.log_file_path}")
        if self.distance_log_file:
            self.distance_log_file.close()
            print(f"📐 Distance log saved: {self.distance_log_file_path}")


class DepthFrameProcessor(threading.Thread):
    """Threaded frame processor for real-time depth estimation with object detection"""
    
    def __init__(self, frame_queue, result_queue, depth_model, yolo_detector=None, camera_params=None):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.depth_model = depth_model
        self.yolo_detector = yolo_detector
        self.camera_params = camera_params
        self.running = True
        self.user_scale = 1.0      # User interactive scale (+/-)
        self.auto_scale = 1.0      # Auto geometric calibration scale
        self.last_auto_update = 0
        self.auto_enabled = True

    def effective_scale(self):
        return self.user_scale * self.auto_scale

    def run(self):
        while self.running:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)  # Reduced timeout for faster response
                if frame_data is None:
                    break
                frame, frame_count, timestamp = frame_data
                start_time = time.time()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = []
                if self.yolo_detector and self.running:
                    detections = self.yolo_detector.detect_objects(rgb_frame)
                if self.running:
                    depth_map = self.depth_model.predict_depth(rgb_frame, self.camera_params, self.effective_scale())
                    processing_time = time.time() - start_time
                    try:
                        # Drop old results if queue is full for responsiveness
                        if self.result_queue.full():
                            try:
                                self.result_queue.get_nowait()
                            except:
                                pass
                        self.result_queue.put((frame, depth_map, detections, frame_count, timestamp, processing_time), timeout=0.05)
                    except:
                        pass
            except Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"⚠️ Processing error: {e}")
                break
    
    def stop(self):
        self.running = False
        # Clear queues to prevent blocking
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except:
            pass
        # Add sentinel to wake up thread
        try:
            self.frame_queue.put(None, timeout=0.1)
        except:
            pass


def load_camera_params(camera_name="macbook_m1_pro"):
    """Load camera parameters from file or use defaults"""
    # Try loading from individual calibration file first
    params_file = f"camera_params_{camera_name}.json"
    
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"✅ Loaded camera parameters from {params_file}")
        return params
    
    # Try loading from config.json
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        if camera_name in config:
            params = config[camera_name]
            print(f"✅ Loaded camera parameters from {config_file}")
            return params
    
    # Use defaults
    params = DEFAULT_CAMERA_PARAMS.get(camera_name, DEFAULT_CAMERA_PARAMS["macbook_m1_pro"])
    print(f"⚠️ Using default camera parameters for {camera_name}")
    print("   Run 'python camera_calibration.py' to calibrate your camera")
    return params


def add_depth_info_overlay(image, depth_map, camera_params, cursor_pos=None):
    """Add depth information overlay to image"""
    h, w = image.shape[:2]
    
    # Add cursor depth if available
    if cursor_pos:
        x, y = cursor_pos
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            depth_value = depth_map[y, x]
            cv2.circle(image, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(image, f"{depth_value:.2f}m", (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return image


def draw_yolo_detections(image, depth_map, detections):
    """Draw YOLO detection bounding boxes and depth measurements for humans"""
    for detection in detections:
        bbox = detection['bbox']
        center = detection['center']
        class_name = detection['class']
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = bbox
        center_x, center_y = center
        
        # Ensure coordinates are integers and within bounds
        center_x = max(0, min(center_x, depth_map.shape[1] - 1))
        center_y = max(0, min(center_y, depth_map.shape[0] - 1))
        x1 = max(0, min(x1, depth_map.shape[1] - 1))
        y1 = max(0, min(y1, depth_map.shape[0] - 1))
        x2 = max(0, min(x2, depth_map.shape[1] - 1))
        y2 = max(0, min(y2, depth_map.shape[0] - 1))
        
        # Get depth at multiple points for better accuracy
        depths = []
        # Sample from center area (more stable for humans) - simplified approach
        sample_points = [
            (center_x, center_y),  # Center
            (center_x, min(center_y + 20, depth_map.shape[0] - 1)),  # Lower
            (max(center_x - 10, 0), center_y),  # Left
            (min(center_x + 10, depth_map.shape[1] - 1), center_y),  # Right
        ]
        
        for px, py in sample_points:
            # Ensure coordinates are integers
            px, py = int(px), int(py)
            if 0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]:
                depth_at_point = depth_map[py, px]
                depths.append(depth_at_point)
        
        # Use median depth for robustness
        depth_value = np.median(depths) if depths else 0.0
        
        # Draw bounding box with thicker lines for humans
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw center point
        cv2.circle(image, center, 5, (0, 255, 0), -1)
        
        # Draw sample points
        for px, py in sample_points:
            if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                cv2.circle(image, (px, py), 2, (255, 255, 0), -1)
        
        # Draw label with depth - larger text for visibility
        label = f"PERSON: {confidence:.2f} | {depth_value:.1f}m"
        
        # Use larger font
        font_scale = 0.8
        thickness = 2
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Background for text - make it more visible
        padding = 10
        cv2.rectangle(image, (x1, y1 - label_size[1] - padding*2), 
                     (x1 + label_size[0] + padding, y1), (0, 255, 0), -1)
        
        # Text
        cv2.putText(image, label, (x1 + padding//2, y1 - padding), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Add distance warning if very close
        if depth_value > 0 and depth_value < 2.0:  # Adjusted threshold for new scaling
            warning = "VERY CLOSE!"
            warning_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y2 + 5), 
                         (x1 + warning_size[0] + 10, y2 + warning_size[1] + 15), 
                         (0, 0, 255), -1)
            cv2.putText(image, warning, (x1 + 5, y2 + warning_size[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def calculate_object_distance(detection1, detection2, depth_map, camera_params):
    """Calculate 3D distance between two detected objects using depth and camera parameters"""
    # Get centers and depths for both objects
    x1, y1 = detection1['center']
    x2, y2 = detection2['center']
    
    # Ensure coordinates are within bounds
    x1 = max(0, min(x1, depth_map.shape[1] - 1))
    y1 = max(0, min(y1, depth_map.shape[0] - 1))
    x2 = max(0, min(x2, depth_map.shape[1] - 1))
    y2 = max(0, min(y2, depth_map.shape[0] - 1))
    
    # Sample multiple points around each center for robust depth measurement
    def get_robust_depth(cx, cy):
        sample_points = [
            (cx, cy),  # Center
            (cx, min(cy + 20, depth_map.shape[0] - 1)),  # Lower
            (max(cx - 10, 0), cy),  # Left
            (min(cx + 10, depth_map.shape[1] - 1), cy),  # Right
        ]
        
        depths = []
        for px, py in sample_points:
            px, py = int(px), int(py)
            if 0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]:
                depth_at_point = depth_map[py, px]
                if depth_at_point > 0:  # Only use valid depths
                    depths.append(depth_at_point)
        
        return np.median(depths) if depths else 0.0
    
    # Get depths for both objects
    depth1 = get_robust_depth(x1, y1)
    depth2 = get_robust_depth(x2, y2)
    
    if depth1 <= 0 or depth2 <= 0:
        return None  # Invalid depth data
    
    # Get camera intrinsic parameters
    fx = camera_params.get('fx', 640.0)
    fy = camera_params.get('fy', 640.0)
    cx = camera_params.get('cx', 320.0)
    cy = camera_params.get('cy', 240.0)
    
    # Convert pixel coordinates to 3D world coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    
    # Object 1 in 3D space
    X1 = (x1 - cx) * depth1 / fx
    Y1 = (y1 - cy) * depth1 / fy
    Z1 = depth1
    
    # Object 2 in 3D space
    X2 = (x2 - cx) * depth2 / fx
    Y2 = (y2 - cy) * depth2 / fy
    Z2 = depth2
    
    # Calculate 3D Euclidean distance
    distance_3d = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
    
    return {
        'distance_3d': float(distance_3d),
        'object1_depth': float(depth1),
        'object2_depth': float(depth2),
        'object1_3d': (float(X1), float(Y1), float(Z1)),
        'object2_3d': (float(X2), float(Y2), float(Z2)),
        'depth_difference': float(abs(depth2 - depth1))
    }


def draw_distance_measurements(image, depth_map, detections, camera_params):
    """Draw distance measurements between detected objects"""
    if len(detections) < 2:
        return image
    
    # Calculate distances between all pairs of objects
    distance_data = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            dist_info = calculate_object_distance(detections[i], detections[j], depth_map, camera_params)
            if dist_info:
                distance_data.append({
                    'obj1_idx': i,
                    'obj2_idx': j,
                    'distance_info': dist_info
                })
    
    # Draw distance lines and measurements
    for data in distance_data:
        obj1 = detections[data['obj1_idx']]
        obj2 = detections[data['obj2_idx']]
        dist_info = data['distance_info']
        
        # Get center points
        center1 = obj1['center']
        center2 = obj2['center']
        
        # Draw line between objects
        cv2.line(image, center1, center2, (255, 0, 255), 2)  # Magenta line
        
        # Calculate midpoint for text placement
        mid_x = (center1[0] + center2[0]) // 2
        mid_y = (center1[1] + center2[1]) // 2
        
        # Create distance label
        distance_3d = dist_info['distance_3d']
        depth_diff = dist_info['depth_difference']
        
        # Main distance text
        dist_text = f"{distance_3d:.2f}m"
        
        # Additional info text
        info_text = f"Δz:{depth_diff:.2f}m"
        
        # Draw background for main text
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Background rectangle
        padding = 5
        cv2.rectangle(image, 
                     (mid_x - text_size[0]//2 - padding, mid_y - text_size[1] - padding*2),
                     (mid_x + text_size[0]//2 + padding, mid_y + padding),
                     (255, 0, 255), -1)
        
        # Main distance text (white)
        cv2.putText(image, dist_text, 
                   (mid_x - text_size[0]//2, mid_y - padding),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Additional info text (smaller, below main text)
        info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(image, info_text,
                   (mid_x - info_size[0]//2, mid_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw small circles at connection points
        cv2.circle(image, center1, 3, (255, 0, 255), -1)
        cv2.circle(image, center2, 3, (255, 0, 255), -1)
    
    return image


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for depth measurement"""
    if event == cv2.EVENT_MOUSEMOVE:
        param['cursor_pos'] = (x, y)


def main():
    # Global variables for cleanup
    global processor, cap, writer
    processor = None
    cap = None
    writer = None
    
    def signal_handler(sig, frame):
        print('\n🛑 Signal received, exiting quickly...')
        cleanup_and_exit()
        sys.exit(0)
    
    def cleanup_and_exit():
        if processor:
            processor.running = False
        if cap:
            try:
                cap.release()
            except:
                pass
        if writer:
            try:
                writer.release()
            except:
                pass
        # Close depth logger
        if 'depth_logger' in locals():
            depth_logger.close()
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="Real-time depth estimation with RT-MonoDepth")
    parser.add_argument("-i", "--input", type=str, help="Input video file (default: webcam)")
    parser.add_argument("-w", "--weights", type=str, 
                       default="./weights/RTMonoDepth/s/m_640_192/",
                       help="Path to RT-MonoDepth weights")
    parser.add_argument("-r", "--record", action='store_true', help="Record output video")
    parser.add_argument("--width", type=int, default=640, help="Processing width")
    parser.add_argument("--camera", type=str, default="macbook_m1_pro", 
                       choices=["macbook_m1_pro", "jetson_nano"], help="Camera type")
    parser.add_argument("--no-mlx", action='store_true', help="Disable MLX acceleration")
    parser.add_argument("--no-yolo", action='store_true', help="Disable YOLO object detection")
    parser.add_argument("--use-yolov8", action='store_true', help="Use standard YOLOv8n instead of custom YOLOv11n model")
    parser.add_argument("--fps-limit", type=int, default=30, help="FPS limit for processing")
    # Auto calibration now DISABLED by default; enable with --auto-calib
    parser.add_argument('--auto-calib', action='store_true', help='Enable automatic geometric scale calibration (disabled by default)')
    # (Deprecated) keep no-auto-calib for backward compatibility (ignored)
    parser.add_argument('--no-auto-calib', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--person-height', type=float, default=1.70, help='Assumed average person height in meters for auto calibration')
    parser.add_argument('--auto-calib-min-frames', type=int, default=15, help='Frames to wait before applying first auto scale update')
    parser.add_argument('--auto-calib-smoothing', type=float, default=0.9, help='EMA smoothing factor (0-1, higher = slower changes)')
    
    # Depth logging options
    parser.add_argument('--log-depth', action='store_true', help='Enable real-time depth logging to CSV file')
    parser.add_argument('--log-interval', type=int, default=60, help='Depth logging interval in seconds (default: 60)')
    
    # Distance measurement options
    parser.add_argument('--measure-distance', action='store_true', help='Enable distance measurement between detected objects')
    
    args = parser.parse_args()
    
    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
    elif IS_APPLE_SILICON and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"🖥️  Using device: {device.upper()}")
    
    # Load camera parameters
    camera_params = load_camera_params(args.camera)
    
    # Load depth model
    use_mlx = not args.no_mlx and MLX_AVAILABLE
    depth_model = RTMonoDepthModel(args.weights, device, use_mlx)
    
    # Initialize YOLO detector
    yolo_detector = None
    if YOLO_AVAILABLE and not args.no_yolo:
        model_path = 'yolov8n.pt' if args.use_yolov8 else YOLO_MODEL_PATH
        yolo_detector = YOLODetector(model_path=model_path, device=device)
        if yolo_detector.model is not None:
            print("✅ YOLO object detection enabled")
        else:
            print("⚠️ YOLO failed to initialize - object detection disabled")
            yolo_detector = None
    else:
        if args.no_yolo:
            print("⚠️ YOLO disabled by user argument")
        else:
            print("⚠️ YOLO not available - object detection disabled")
    
    # Setup video capture
    input_source = args.input if args.input else 0
    # If input is a string that represents a number, convert it to int
    if isinstance(input_source, str) and input_source.isdigit():
        input_source = int(input_source)
    cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{input_source}'")
        return
    
    # Set camera resolution if using webcam
    if not args.input:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_params['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_params['height'])
        cap.set(cv2.CAP_PROP_FPS, args.fps_limit)
    
    # Setup recording if requested
    writer = None
    if args.record:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) if args.input else args.fps_limit
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_FOLDER, f"depth_estimation_{timestamp}.mp4")
        
        # Get frame dimensions for video writer
        ret, test_frame = cap.read()
        if ret:
            if test_frame.shape[1] != args.width:
                aspect_ratio = test_frame.shape[0] / test_frame.shape[1]
                new_height = int(args.width * aspect_ratio)
                frame_height = new_height
            else:
                frame_height = test_frame.shape[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        else:
            frame_height = camera_params['height']
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (args.width, frame_height))
        
        print(f"📹 Recording to: {output_path}")
    
    # Setup threading with smaller queues for faster response
    frame_queue = Queue(maxsize=1)  # Reduced queue size for less lag
    result_queue = Queue(maxsize=1)
    
    # Initialize depth logger
    depth_logger = DepthLogger(log_interval=args.log_interval, enabled=args.log_depth, measure_distances=args.measure_distance)
    
    processor = DepthFrameProcessor(frame_queue, result_queue, depth_model, yolo_detector, camera_params)
    # Auto calibration default OFF unless --auto-calib supplied
    processor.auto_enabled = bool(getattr(args, 'auto_calib', False))
    if processor.auto_enabled:
        print("🔧 Auto calibration ENABLED (user requested)")
    else:
        print("🔧 Auto calibration DISABLED (default)")
    processor.start()
    
    # Setup mouse callback for depth measurement
    mouse_data = {'cursor_pos': None}
    cv2.namedWindow('RT-MonoDepth Real-time')
    cv2.setMouseCallback('RT-MonoDepth Real-time', mouse_callback, mouse_data)
    
    # Performance tracking
    fps_counter = FPSCounter(window_size=10)  # Smaller window for faster response
    processing_times = []
    
    print("🚀 Starting real-time depth estimation...")
    print("   📏 CALIBRATION INSTRUCTIONS:")
    print("   - Move mouse over image to measure depth")
    print("   - If depth is wrong, use these controls:")
    print("     • '+' or '=' : Increase depth scale (if reading too low)")
    print("     • '-' : Decrease depth scale (if reading too high)")
    print("     • 'c' : Quick calibrate assuming 1.5m distance")
    print("     • 'p' : Precise calibrate (enter actual distance)")
    print("     • 'r' : Reset depth scale to 1.0")
    print("   📹 OTHER CONTROLS:")
    print("   - Press 'q' or ESC to quit quickly")
    print("   - Press 's' to save current frame")
    if args.use_yolov8:
        print("   📸 Using YOLOv8n for better human detection")
    if depth_logger.enabled:
        print(f"   📊 Depth logging enabled: Every {args.log_interval} seconds")
    if args.measure_distance:
        print("   📐 Distance measurement enabled: Shows 3D distance between objects")
    print("   🎯 Stand 1-2m away and calibrate for best accuracy")
    
    # Helper for auto calibration (geometric) - placed before processing loop so it's in scope
    def auto_calibrate_scale(detections, depth_map_resized):
        if not processor.auto_enabled or not detections:
            return
        fx = camera_params.get('fx', 640.0)
        valid_scales = []
        for det in detections:
            if det['class'].lower() != 'person':
                continue
            (x1, y1, x2, y2) = det['bbox']
            bbox_h = max(1, y2 - y1)
            # Reject too small or too large boxes
            frame_h = depth_map_resized.shape[0]
            if bbox_h < 80 or bbox_h > frame_h * 0.9:
                continue
            # Sample median network depth inside upper body region
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            sample_pts = []
            for dy in [-10, 0, 10]:
                for dx in [-10, 0, 10]:
                    sx = np.clip(cx + dx, 0, depth_map_resized.shape[1]-1)
                    sy = np.clip(cy + dy, 0, depth_map_resized.shape[0]-1)
                    sample_pts.append(depth_map_resized[sy, sx])
            net_depth = float(np.median(sample_pts))
            if net_depth <= 0:
                continue
            geom_depth = (fx * args.person_height) / float(bbox_h)
            if geom_depth <= 0:
                continue
            scale_candidate = geom_depth / net_depth
            if 0.05 < scale_candidate < 100:
                valid_scales.append(scale_candidate)
        if not valid_scales:
            return
        median_scale = float(np.median(valid_scales))
        alpha = args.auto_calib_smoothing
        processor.auto_scale = alpha * processor.auto_scale + (1 - alpha) * median_scale
    
    try:
        frame_count = 0
        exit_requested = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Check for exit and process input every frame for faster response
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC key
                print("🛑 Exit requested...")
                exit_requested = True
                break
            elif key == ord('s'):
                # Save current frame
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    if 'combined' in locals():
                        cv2.imwrite(f"depth_frame_{timestamp}.jpg", combined)
                        print(f"💾 Saved frame: depth_frame_{timestamp}.jpg")
                except:
                    print("⚠️ Failed to save frame")
            elif key == ord('+') or key == ord('='):
                processor.user_scale *= 1.05
                print(f"🔧 User scale: {processor.user_scale:.3f} (auto:{processor.auto_scale:.3f} eff:{processor.effective_scale():.3f})")
            elif key == ord('-'):
                processor.user_scale *= 0.95
                print(f"🔧 User scale: {processor.user_scale:.3f} (auto:{processor.auto_scale:.3f} eff:{processor.effective_scale():.3f})")
            elif key == ord('c'):
                # Quick manual calibration: assume cursor distance entered
                if 'combined' in locals() and mouse_data['cursor_pos'] and 'depth_map_resized' in locals():
                    x, y = mouse_data['cursor_pos']
                    raw_depth = depth_map_resized[y, x] / max(processor.auto_scale, 1e-6)  # remove auto scale influence
                    assumed_dist = 1.5
                    if raw_depth > 0:
                        processor.user_scale = assumed_dist / raw_depth
                        print(f"🎯 Quick calib user_scale->{processor.user_scale:.3f}; eff:{processor.effective_scale():.3f}")
            elif key == ord('p'):
                if 'combined' in locals() and mouse_data['cursor_pos'] and 'depth_map_resized' in locals():
                    x, y = mouse_data['cursor_pos']
                    raw_depth = depth_map_resized[y, x] / max(processor.auto_scale, 1e-6)
                    try:
                        entered = float(input('Enter actual distance (m): '))
                        if entered > 0 and raw_depth > 0:
                            processor.user_scale = entered / raw_depth
                            print(f"🎯 Precise calib user_scale->{processor.user_scale:.3f}; eff:{processor.effective_scale():.3f}")
                    except Exception:
                        print('⚠️ Invalid input for precise calibration')
            elif key == ord('r'):
                processor.user_scale = 1.0
                processor.auto_scale = 1.0 if not processor.auto_enabled else processor.auto_scale
                print(f"🔄 Reset user scale. Auto:{processor.auto_scale:.3f} Eff:{processor.effective_scale():.3f}")
            
            # Resize frame for processing
            if frame.shape[1] != args.width:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(args.width * aspect_ratio)
                frame = cv2.resize(frame, (args.width, new_height))
            
            # Always try to add frame to processing queue, drop old frames for responsiveness
            try:
                frame_queue.put_nowait((frame, frame_count, current_time))
            except:
                # Queue full, drop the oldest frame and add new one
                try:
                    frame_queue.get_nowait()  # Remove old frame
                    frame_queue.put_nowait((frame, frame_count, current_time))  # Add new frame
                except:
                    pass
            
            # Get processed results
            try:
                result = result_queue.get_nowait()
                original_frame, depth_map, detections, proc_frame_count, timestamp, proc_time = result
                processing_times.append(proc_time)
                
                # Resize depth map to match original frame for accurate coordinate mapping
                depth_map_resized = cv2.resize(depth_map, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                # Add overlay information
                display_frame = original_frame.copy()
                
                # Draw YOLO detections if available (use resized depth map)
                if detections:
                    # Perform geometric auto calibration BEFORE drawing detections (so display uses updated scale next frame)
                    auto_calibrate_scale(detections, depth_map_resized)
                    display_frame = draw_yolo_detections(display_frame, depth_map_resized, detections)
                    
                    # Draw distance measurements between objects if enabled
                    if args.measure_distance and len(detections) >= 2:
                        display_frame = draw_distance_measurements(display_frame, depth_map_resized, detections, camera_params)
                
                # Log depth data if enabled
                depth_logger.log_detections(detections, depth_map_resized, frame_count, camera_params)
                
                # Add depth info overlay (use resized depth map)
                display_frame = add_depth_info_overlay(display_frame, depth_map_resized, camera_params, mouse_data['cursor_pos'])
                
                # Use only the main frame (no depth map visualization)
                combined = display_frame
                
                # Update FPS counter
                fps_counter.update()
                current_fps = fps_counter.get_fps()
                avg_proc_time = np.mean(processing_times[-30:]) if processing_times else 0
                
                # Performance overlay with more responsive updates
                perf_text = f"FPS: {current_fps:.1f} | Proc: {avg_proc_time*1000:.1f}ms | Frame: {frame_count}"
                cv2.putText(combined, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Depth scale overlay with calibration help
                scale_text = f"Scale User:{processor.user_scale:.2f} Auto:{processor.auto_scale:.2f} Eff:{processor.effective_scale():.2f}"
                cv2.putText(combined, scale_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if processor.auto_enabled:
                    cv2.putText(combined, 'AutoCalib ON', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                else:
                    cv2.putText(combined, 'AutoCalib OFF', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 1)
                
                # Display
                cv2.imshow('RT-MonoDepth Real-time', combined)
                
                # Record if enabled
                if writer:
                    writer.write(combined)
                
            except Empty:
                pass
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        print("🧹 Cleaning up...")
        # Stop processor thread immediately
        processor.running = False
        try:
            processor.join(timeout=1.0)  # Wait max 1 second
        except:
            pass
        
        # Release resources quickly
        try:
            cap.release()
        except:
            pass
        
        try:
            if writer:
                writer.release()
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Close depth logger
        depth_logger.close()
        
        # Print performance summary
        if processing_times:
            avg_time = np.mean(processing_times)
            print(f"\n📊 Performance Summary:")
            print(f"   Average processing time: {avg_time*1000:.1f}ms")
            print(f"   Average FPS: {1/avg_time:.1f}")
            print(f"   Processed {len(processing_times)} frames")
        
        print("✅ Done!")


if __name__ == '__main__':
    main()
