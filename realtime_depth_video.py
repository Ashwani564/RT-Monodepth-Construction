#!/usr/bin/env python3
"""
Real-time Video Depth Estimation with Object Detection using RT-MonoDepth
Provides human detection with accurate depth measurements (no depth map visualization)
Optimized for MacBook M1 Pro and Jetson Nano deployment
🚀 TensorRT Acceleration for Jetson Nano
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

# TensorRT imports for Jetson Nano optimization
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    print("✅ TensorRT is available for hardware acceleration")
except ImportError:
    print("⚠️ TensorRT not found. Install with: sudo apt-get install python3-libnvinfer-dev")

# Detect if running on Jetson
IS_JETSON = os.path.exists('/etc/nv_tegra_release') or os.path.exists('/sys/module/tegra_fuse')
if IS_JETSON:
    print("🔧 Jetson device detected - enabling optimizations")
    # Set Jetson-specific optimizations
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("   ✅ CUDA optimizations enabled")

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
CLASSES_TO_DETECT = {"Person", "Machinery", "Vehicle"}  # Main detection classes (normalized)
CONFIDENCE_THRESHOLD = 0.35  # Higher threshold to reduce false positives

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
    },
    "jetson_csi": {
        # CSI Camera (Raspberry Pi Camera v2 typical on Jetson)
        "fx": 800.0,
        "fy": 800.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480
    }
}

# Jetson Nano Performance Settings
JETSON_SETTINGS = {
    "power_mode": "MAXN",  # Maximum performance mode
    "processing_width": 416,  # Reduced for Jetson Nano 4GB (was 640)
    "yolo_img_size": 416,  # Smaller YOLO input for faster inference
    "depth_batch_size": 1,  # Process one frame at a time
    "enable_fp16": True,  # Use FP16 for faster inference on Jetson
    "cpu_threads": 4,  # Jetson Nano has 4 CPU cores
}


class RTMonoDepthModel:
    """RT-MonoDepth model wrapper with MLX and TensorRT acceleration"""
    
    def __init__(self, weight_path, device='cpu', use_mlx=True, use_tensorrt=False, tensorrt_engine_path=None):
        self.device = device
        self.use_mlx = use_mlx and MLX_AVAILABLE
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE and device == 'cuda'
        self.tensorrt_engine = None
        self.tensorrt_context = None
        self.cuda_stream = None
        
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
        
        # Enable FP16 for Jetson Nano if CUDA is available
        if IS_JETSON and device == 'cuda' and JETSON_SETTINGS['enable_fp16']:
            try:
                self.encoder = self.encoder.half()
                self.decoder = self.decoder.half()
                print("   ⚡ FP16 mode enabled for Jetson Nano acceleration")
            except Exception as e:
                print(f"   ⚠️ FP16 conversion failed: {e}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.feed_height, self.feed_width)),
            transforms.ToTensor(),
        ])
        
        # Initialize TensorRT if requested
        if self.use_tensorrt:
            if tensorrt_engine_path and os.path.exists(tensorrt_engine_path):
                print(f"   Loading TensorRT engine from: {tensorrt_engine_path}")
                self._load_tensorrt_engine(tensorrt_engine_path)
            else:
                print("   ⚠️ TensorRT engine not found. Run with --build-tensorrt first.")
                self.use_tensorrt = False
        
        if self.use_mlx:
            print("   MLX acceleration enabled")
        elif self.use_tensorrt:
            print("   🚀 TensorRT acceleration enabled")
        
        print(f"   Model loaded: {self.feed_width}x{self.feed_height}")
    
    def _load_tensorrt_engine(self, engine_path):
        """Load TensorRT engine for optimized inference"""
        try:
            with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                self.tensorrt_engine = runtime.deserialize_cuda_engine(f.read())
            
            self.tensorrt_context = self.tensorrt_engine.create_execution_context()
            self.cuda_stream = cuda.Stream()
            print("   ✅ TensorRT engine loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load TensorRT engine: {e}")
            self.use_tensorrt = False
    
    def predict_depth(self, rgb_image, camera_params=None, depth_scale_factor=1.0):
        """Predict raw monocular depth then apply external scale.
        The network (Monodepth2-style) produces metric depth up to an unknown global scale.
        We avoid arbitrary nonlinear scaling and leave scale resolution to calibration.
        """
        # Preprocess image
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        
        # Use FP16 if enabled for Jetson
        if IS_JETSON and self.device == 'cuda' and JETSON_SETTINGS['enable_fp16']:
            input_tensor = input_tensor.half()
        
        with torch.no_grad():
            if self.use_tensorrt and self.tensorrt_context:
                # TensorRT inference path
                depth = self._tensorrt_inference(input_tensor)
            else:
                # Standard PyTorch inference
                features = self.encoder(input_tensor)
                outputs = self.decoder(features)
                disp = outputs[("disp", 0)]
                _, depth = disp_to_depth(disp, 0.1, 100)
        
        # Apply (potentially combined user * auto) scale factor ONLY
        metric_depth = depth * depth_scale_factor
        return metric_depth.squeeze().cpu().numpy()
    
    def _tensorrt_inference(self, input_tensor):
        """Run inference using TensorRT engine"""
        # This is a simplified version - you'd need to implement full TensorRT pipeline
        # For now, fall back to PyTorch
        features = self.encoder(input_tensor)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        _, depth = disp_to_depth(disp, 0.1, 100)
        return depth


class YOLODetector:
    """YOLO object detection wrapper with TensorRT support"""
    
    def __init__(self, model_path=YOLO_MODEL_PATH, device='cpu', use_tensorrt=False, img_size=640):
        self.device = device
        self.model = None
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE and device == 'cuda'
        self.img_size = img_size
        
        # Adjust image size for Jetson Nano
        if IS_JETSON:
            self.img_size = JETSON_SETTINGS['yolo_img_size']
            print(f"   Jetson detected: Using YOLO image size {self.img_size}")
        
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO not available - object detection disabled")
            return
        
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                model_name = os.path.basename(model_path)
                print(f"✅ YOLO model loaded from: {model_path}")
                
                # Enable TensorRT export for YOLO if requested
                if self.use_tensorrt and IS_JETSON:
                    try:
                        # Export YOLO model to TensorRT engine
                        tensorrt_model_path = model_path.replace('.pt', '_tensorrt.engine')
                        if not os.path.exists(tensorrt_model_path):
                            print("   🔧 Exporting YOLO to TensorRT engine (this may take a few minutes)...")
                            self.model.export(format='engine', device=0, half=True, imgsz=self.img_size)
                            print(f"   ✅ TensorRT engine created: {tensorrt_model_path}")
                        else:
                            print(f"   ✅ Using existing TensorRT engine: {tensorrt_model_path}")
                    except Exception as e:
                        print(f"   ⚠️ TensorRT export failed: {e}")
                        self.use_tensorrt = False
                
                # Check if it's a custom trained model
                if "custom" in model_path.lower() or "best.pt" in model_path:
                    print("🔧 Custom-trained YOLO model detected")
                    print("   Using custom weights for optimized detection")
                else:
                    print(f"   Model: {model_name}")
                
                # Always print model classes for debugging
                if hasattr(self.model, 'names'):
                    print(f"   Available classes: {list(self.model.names.values())}")
                    print("   ↳ Will filter for: Person, machinery, vehicles")
                    
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
            # Run detection with specified image size for Jetson optimization
            results = self.model(image, device=self.device, verbose=False, conf=confidence, imgsz=self.img_size)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, cls_idx, conf in zip(boxes, classes, confidences):
                        class_name = self.model.names[int(cls_idx)]
                        
                        # Debug: Print ALL detections before filtering
                        if False:  # Disabled for Jetson to reduce overhead
                            print(f"🔍 Raw detection: {class_name} (confidence={conf:.2f})")
                        
                        # Filter out safety equipment classes we don't want to show
                        unwanted_classes = {'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Hardhat', 'Safety Vest', 'Safety Cone'}
                        if class_name in unwanted_classes:
                            continue
                        
                        # Keep only Person, machinery, and vehicles (case-insensitive matching)
                        class_lower = class_name.lower()
                        
                        # Define wanted classes (case-insensitive)
                        is_person = class_lower in ['person', 'people', 'human']
                        is_machinery = class_lower in ['machinery', 'excavator', 'bulldozer', 'crane', 'construction equipment']
                        is_vehicle = class_lower in ['vehicle', 'car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle']
                        
                        if not (is_person or is_machinery or is_vehicle):
                            continue
                        
                        # Lower confidence threshold for person detection to reduce flickering
                        min_confidence = 0.25 if is_person else 0.35
                        if conf < min_confidence:
                            continue
                        
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box)
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Only reclassify vehicles as machinery if they're very large (less aggressive)
                        if is_vehicle:
                            aspect_ratio = bbox_width / max(bbox_height, 1)
                            
                            # Calculate box ratio relative to image size
                            image_area = image.shape[0] * image.shape[1]
                            box_ratio = bbox_area / max(image_area, 1)
                            
                            # Much more conservative reclassification - only for very large construction equipment
                            is_very_large = box_ratio > 0.15  # >15% of frame
                            is_very_wide = aspect_ratio > 2.0  # Much wider than tall
                            
                            if is_very_large and is_very_wide:
                                class_name = "machinery"
                            else:
                                class_name = "vehicle"  # Keep as vehicle
                        
                        # Normalize class name for display
                        if is_person:
                            display_class = "Person"
                        elif is_machinery:
                            display_class = "Machinery" 
                        elif is_vehicle:
                            display_class = "Vehicle"
                        else:
                            display_class = class_name
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'class': display_class,
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
    
    def should_log(self, current_time, force_log=False):
        """Check if it's time to log data"""
        if not self.enabled:
            return False
        return force_log or (current_time - self.last_log_time) >= self.log_interval
    
    def log_detections(self, detections, depth_map, frame_count, camera_params=None, force_log=False):
        """Log detection data with depth measurements and distances"""
        if not self.enabled or not self.csv_writer:
            return
        
        current_time = time.time()
        # Force log if there are detections, or use normal interval
        should_log_now = self.should_log(current_time, force_log or bool(detections))
        if not should_log_now:
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


def setup_jetson_power_mode():
    """Set Jetson Nano to maximum performance mode"""
    if not IS_JETSON:
        return
    
    try:
        # Set to MAXN mode (maximum performance)
        os.system('sudo nvpmodel -m 0')
        # Set CPU and GPU to maximum clocks
        os.system('sudo jetson_clocks')
        print("🚀 Jetson Nano set to maximum performance mode (MAXN)")
    except Exception as e:
        print(f"⚠️ Could not set Jetson power mode: {e}")
        print("   Run manually: sudo nvpmodel -m 0 && sudo jetson_clocks")


def get_csi_camera_gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0
):
    """
    Create GStreamer pipeline for CSI camera on Jetson Nano
    
    Args:
        sensor_id: Camera sensor ID (0 or 1)
        capture_width: Native capture width
        capture_height: Native capture height
        display_width: Output width
        display_height: Output height
        framerate: Camera framerate
        flip_method: Image rotation/flip (0=none, 2=180deg)
    
    Returns:
        GStreamer pipeline string
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )


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
    """Draw YOLO detection bounding boxes and depth measurements for all detected objects"""
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
        # Sample from center area (more stable) - simplified approach
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
        
        # Set colors and thickness based on object type - exact class names from custom model
        if class_name == "Person":
            color = (0, 255, 0)  # Green for persons
            thickness = 3
            label_prefix = "PERSON"
        elif class_name == "vehicle":
            color = (255, 0, 0)  # Blue for vehicles
            thickness = 4
            # Check if it's likely construction equipment based on size/location
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area > 500:  # Larger objects likely construction equipment
                label_prefix = "HEAVY VEHICLE"
            else:
                label_prefix = "VEHICLE"
        elif class_name == "machinery":
            color = (0, 165, 255)  # Orange for machinery
            thickness = 4
            label_prefix = "MACHINERY"
        else:
            # Default for any other objects that might pass through
            color = (128, 128, 128)  # Gray for other objects
            thickness = 2
            label_prefix = class_name.upper()
        
        # Draw bounding box with object-specific color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw center point
        cv2.circle(image, center, 5, color, -1)
        
        # Draw sample points
        for px, py in sample_points:
            if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                cv2.circle(image, (px, py), 2, (255, 255, 0), -1)
        
        # Draw label with depth - larger text for visibility
        label = f"{label_prefix}: {confidence:.2f} | {depth_value:.1f}m"
        
        # Use larger font
        font_scale = 0.8
        text_thickness = 2
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        
        # Background for text - make it more visible with object-specific color
        padding = 10
        cv2.rectangle(image, (x1, y1 - label_size[1] - padding*2), 
                     (x1 + label_size[0] + padding, y1), color, -1)
        
        # Text - white for good contrast
        cv2.putText(image, label, (x1 + padding//2, y1 - padding), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
        
        # Add distance warning for close objects
        if depth_value > 0 and depth_value < 3.0:  # Warning for objects closer than 3m
            warning = "CLOSE!"
            if class_name.lower() == "machinery":
                warning = "MACHINERY CLOSE!"
            elif class_name.lower() == "vehicle":
                warning = "VEHICLE CLOSE!"
            
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
    parser.add_argument("-r", "--record", action='store_true', help="Record output video with detections and depth overlay")
    parser.add_argument("-o", "--output", type=str, help="Output video filename (default: auto-generated)")
    parser.add_argument("--width", type=int, default=640, help="Processing width (default: 640, use 416 for Jetson)")
    parser.add_argument("--camera", type=str, default="macbook_m1_pro", 
                       choices=["macbook_m1_pro", "jetson_nano", "jetson_csi"], help="Camera type")
    parser.add_argument("--no-mlx", action='store_true', help="Disable MLX acceleration")
    parser.add_argument("--no-yolo", action='store_true', help="Disable YOLO object detection")
    parser.add_argument("--use-yolov8", action='store_true', help="Use standard YOLOv8n instead of custom YOLOv11n model")
    parser.add_argument("--fps-limit", type=int, default=30, help="FPS limit for processing")
    
    # Jetson-specific arguments
    parser.add_argument("--use-tensorrt", action='store_true', help="Enable TensorRT acceleration (Jetson only)")
    parser.add_argument("--tensorrt-engine", type=str, help="Path to TensorRT engine file")
    parser.add_argument("--build-tensorrt", action='store_true', help="Build TensorRT engine and exit")
    parser.add_argument("--csi-camera", action='store_true', help="Use CSI camera on Jetson (overrides --input)")
    parser.add_argument("--csi-sensor-id", type=int, default=0, help="CSI camera sensor ID (0 or 1)")
    parser.add_argument("--csi-flip", type=int, default=0, choices=[0, 2], help="CSI camera flip (0=none, 2=180deg)")
    parser.add_argument("--jetson-power-mode", action='store_true', help="Set Jetson to max performance mode")
    
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
    
    # Scale adjustment options
    parser.add_argument('--depth-scale', type=float, default=5.0, help='Initial depth scale factor (default: 5.0 for construction sites)')
    parser.add_argument('--manual-calibration', action='store_true', help='Enable manual calibration mode with larger scale adjustments')
    
    # Video timing options
    parser.add_argument('--fast-process', action='store_true', help='Process video as fast as possible, ignoring original timing (default: 1:1 real-time)')
    
    args = parser.parse_args()
    
    # Jetson-specific setup
    if IS_JETSON:
        print("🔧 Jetson Nano optimizations enabled")
        
        # Set power mode if requested
        if args.jetson_power_mode:
            setup_jetson_power_mode()
        
        # Adjust processing width for Jetson if not explicitly set
        if args.width == 640:
            args.width = JETSON_SETTINGS['processing_width']
            print(f"   Auto-adjusting processing width to {args.width} for Jetson Nano")
        
        # Warn about TensorRT on non-Jetson systems
        if args.use_tensorrt:
            if not torch.cuda.is_available():
                print("⚠️ TensorRT requested but CUDA not available. Disabling TensorRT.")
                args.use_tensorrt = False
    
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
    depth_model = RTMonoDepthModel(
        args.weights, 
        device, 
        use_mlx, 
        use_tensorrt=args.use_tensorrt,
        tensorrt_engine_path=args.tensorrt_engine
    )
    
    # Initialize YOLO detector
    yolo_detector = None
    if YOLO_AVAILABLE and not args.no_yolo:
        model_path = 'yolov8n.pt' if args.use_yolov8 else YOLO_MODEL_PATH
        yolo_img_size = JETSON_SETTINGS['yolo_img_size'] if IS_JETSON else 640
        yolo_detector = YOLODetector(
            model_path=model_path, 
            device=device,
            use_tensorrt=args.use_tensorrt,
            img_size=yolo_img_size
        )
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
    if args.csi_camera and IS_JETSON:
        # Use CSI camera on Jetson
        print(f"📷 Using CSI camera sensor {args.csi_sensor_id}")
        pipeline = get_csi_camera_gstreamer_pipeline(
            sensor_id=args.csi_sensor_id,
            capture_width=1280,
            capture_height=720,
            display_width=args.width,
            display_height=int(args.width * 0.75),  # 4:3 aspect ratio
            framerate=args.fps_limit,
            flip_method=args.csi_flip
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        input_source = f"CSI Camera {args.csi_sensor_id}"
    else:
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
    
    # Setup recording if requested or if processing video file (auto-record)
    writer = None
    auto_record = bool(args.input)  # Auto-record when processing video files
    should_record = args.record or auto_record
    
    if should_record:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) if args.input else args.fps_limit
        if fps <= 0 or fps > 60:  # Handle invalid FPS values
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Determine output filename
        if args.output:
            output_filename = args.output
            if not output_filename.endswith('.mp4'):
                output_filename += '.mp4'
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if args.input:
                input_name = os.path.splitext(os.path.basename(args.input))[0]
                output_filename = f"depth_estimation_{input_name}_{timestamp}.mp4"
            else:
                output_filename = f"depth_estimation_webcam_{timestamp}.mp4"
        
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
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
        
        if auto_record:
            print(f"📹 Auto-recording enabled for video input: {output_path}")
        else:
            print(f"📹 Recording to: {output_path}")
        print(f"   Output resolution: {args.width}x{frame_height} @ {fps} FPS")
    
    # Setup threading with smaller queues for faster response
    frame_queue = Queue(maxsize=1)  # Reduced queue size for less lag
    result_queue = Queue(maxsize=1)
    
    # Initialize depth logger
    depth_logger = DepthLogger(log_interval=args.log_interval, enabled=args.log_depth, measure_distances=args.measure_distance)
    
    processor = DepthFrameProcessor(frame_queue, result_queue, depth_model, yolo_detector, camera_params)
    # Auto calibration default OFF unless --auto-calib supplied
    processor.auto_enabled = bool(getattr(args, 'auto_calib', False))
    # Set initial user scale from command line argument
    processor.user_scale = args.depth_scale
    
    if processor.auto_enabled:
        print("🔧 Auto calibration ENABLED (user requested)")
    else:
        print("🔧 Auto calibration DISABLED (default)")
    
    # Manual calibration mode for easier adjustments
    manual_calib_mode = args.manual_calibration
    if manual_calib_mode:
        print("🎯 Manual calibration mode ENABLED (larger scale adjustments)")
    
    print(f"🔧 Initial depth scale: {processor.user_scale:.3f}")
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
    if manual_calib_mode:
        print("     • '+' or '=' : Increase depth scale by 50% (manual mode)")
        print("     • '-' : Decrease depth scale by 33% (manual mode)")
        print("     • 'SHIFT +' : Increase depth scale by 10%")
        print("     • 'SHIFT -' : Decrease depth scale by 10%")
    else:
        print("     • '+' or '=' : Increase depth scale by 5%")
        print("     • '-' : Decrease depth scale by 5%")
    print("     • '1'-'9' : Set depth scale to 1x-9x quickly")
    print("     • '0' : Set depth scale to 10x")
    print("     • 'c' : Quick calibrate assuming 1.5m distance")
    print("     • 'p' : Precise calibrate (enter actual distance)")
    print("     • 'r' : Reset depth scale to 1.0")
    print("   💡 TIP: For construction sites, try scale 5-10x (keys 5-9)")
    print("   📹 OTHER CONTROLS:")
    print("   - Press 'q' or ESC to quit quickly")
    print("   - Press 's' to save current frame")
    if should_record:
        print(f"   🔴 Recording output video: {output_filename}")
    if args.use_yolov8:
        print("   📸 Using YOLOv8n for better human detection")
    if depth_logger.enabled:
        print(f"   📊 Depth logging enabled: Every {args.log_interval} seconds")
    if args.measure_distance:
        print("   📐 Distance measurement enabled: Shows 3D distance between objects")
    if args.input:
        if args.fast_process:
            print("   ⚡ Fast processing mode: Processing video as fast as possible")
        else:
            print("   🕒 Real-time mode: Processing video at original 1:1 timing")
    if IS_JETSON:
        print("   🚀 JETSON NANO OPTIMIZATIONS ACTIVE:")
        print(f"      - Processing resolution: {args.width}x{int(args.width*0.75)}")
        print(f"      - FP16 inference: {'✅ Enabled' if JETSON_SETTINGS['enable_fp16'] else '❌ Disabled'}")
        print(f"      - TensorRT: {'✅ Enabled' if args.use_tensorrt else '❌ Disabled'}")
        if args.csi_camera:
            print(f"      - CSI Camera: ✅ Sensor {args.csi_sensor_id}")
    print("   🎯 Stand 1-2m away and calibrate for best accuracy")
    
    # Helper for auto calibration (geometric) - placed before processing loop so it's in scope
    def auto_calibrate_scale(detections, depth_map_resized):
        if not processor.auto_enabled or not detections:
            return
        fx = camera_params.get('fx', 640.0)
        valid_scales = []
        for det in detections:
            # Only use person detections for auto-calibration (most reliable for height estimation)
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
        start_time = time.time()
        
        # Get video FPS for timing control
        video_fps = cap.get(cv2.CAP_PROP_FPS) if args.input else 30
        if video_fps <= 0 or video_fps > 60:  # Handle invalid FPS values
            video_fps = 30
        frame_duration = 1.0 / video_fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Calculate timing for 1:1 playback (only for video files, not webcam)
            if args.input and not args.fast_process:
                expected_time = start_time + (frame_count - 1) * frame_duration
                time_diff = expected_time - current_time
                if time_diff > 0:
                    # We're ahead of schedule, wait
                    time.sleep(min(time_diff, frame_duration))
            
            # Check for exit and process input with appropriate timing
            wait_time = 1 if args.input else 1  # Keep responsive for both cases
            key = cv2.waitKey(wait_time) & 0xFF
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
                if manual_calib_mode:
                    processor.user_scale *= 1.5  # 50% increase in manual mode
                else:
                    processor.user_scale *= 1.05  # 5% increase in normal mode
                print(f"🔧 User scale: {processor.user_scale:.3f} (auto:{processor.auto_scale:.3f} eff:{processor.effective_scale():.3f})")
            elif key == ord('-'):
                if manual_calib_mode:
                    processor.user_scale *= 0.67  # 33% decrease in manual mode
                else:
                    processor.user_scale *= 0.95  # 5% decrease in normal mode
                print(f"🔧 User scale: {processor.user_scale:.3f} (auto:{processor.auto_scale:.3f} eff:{processor.effective_scale():.3f})")
            elif key >= ord('1') and key <= ord('9'):
                # Quick scale setting (1x to 9x)
                scale_factor = key - ord('0')
                processor.user_scale = float(scale_factor)
                print(f"🔧 Quick scale set to {scale_factor}x: {processor.user_scale:.3f}")
            elif key == ord('0'):
                # Set to 10x scale
                processor.user_scale = 10.0
                print(f"🔧 Quick scale set to 10x: {processor.user_scale:.3f}")
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
        
        # Report recording completion
        if should_record and writer is not None:
            print(f"🎬 Video saved successfully: {output_path}")
            print(f"   Contains depth estimation and object detection overlays")
        
        print("✅ Done!")


if __name__ == '__main__':
    main()
