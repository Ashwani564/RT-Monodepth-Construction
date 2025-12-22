#!/usr/bin/env python3
"""
Real-time Video Depth Estimation with Object Detection using RT-MonoDepth
OPTIMIZED FOR JETSON NANO (2019) 4GB RAM

Key Optimizations:
1. Reduced input resolution (320x192 vs 640x480)
2. Skip frames for faster processing
3. ONNX/TensorRT inference for YOLO
4. Optimized memory usage with explicit garbage collection
5. Simplified depth processing pipeline
6. Reduced threading overhead
7. FP16 inference where possible

Target: 10-15+ FPS on Jetson Nano
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
import json
import signal
import sys
import gc

# RT-MonoDepth imports
from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder
from networks.RTMonoDepth.RTMonoDepth_s import DepthDecoder as DepthDecoderS, DepthEncoder as DepthEncoderS
from layers import disp_to_depth
from torchvision import transforms

# Check platform
IS_JETSON = os.path.exists('/etc/nv_tegra_release') or 'aarch64' in platform.machine()

# YOLO imports with ONNX Runtime support
YOLO_AVAILABLE = False
ONNX_RUNTIME_AVAILABLE = False
TENSORRT_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
    print("✅ ONNX Runtime available")
except ImportError:
    print("⚠️ ONNX Runtime not found. Install: pip install onnxruntime-gpu")

# TensorRT check (Jetson specific)
if IS_JETSON:
    try:
        import tensorrt as trt
        TENSORRT_AVAILABLE = True
        print("✅ TensorRT available")
    except ImportError:
        print("⚠️ TensorRT not found")

# Configuration optimized for Jetson Nano
OUTPUT_FOLDER = "output_depth_video"
DEPTH_LOG_FOLDER = "depth_logs"

# Reduced thresholds for faster processing
YOLO_MODEL_PATH = 'custom_yolo11n.pt'
ONNX_MODEL_PATH = 'custom_yolo11n_jetson_320.onnx'
CONFIDENCE_THRESHOLD = 0.4  # Slightly higher to reduce post-processing
CLASSES_TO_DETECT = {"Person", "Machinery", "Vehicle"}

# Jetson Nano optimized camera params
JETSON_CAMERA_PARAMS = {
    "fx": 400.0,  # Adjusted for 320x240
    "fy": 400.0,
    "cx": 160.0,
    "cy": 120.0,
    "width": 320,
    "height": 240
}


class JetsonOptimizedDepthModel:
    """RT-MonoDepth model optimized for Jetson Nano"""
    
    def __init__(self, weight_path, use_fp16=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        
        print(f"Loading RT-MonoDepth from: {weight_path}")
        print(f"   Device: {self.device}, FP16: {self.use_fp16}")
        
        # Use small model for Jetson
        self.is_small_model = True  # Force small model on Jetson
        
        # Load encoder
        encoder_path = os.path.join(weight_path, "encoder.pth")
        self.encoder = DepthEncoderS()
        
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        
        # Use smaller resolution for Jetson
        self.feed_height = 192  # Reduced from 192
        self.feed_width = 320   # Reduced from 640
        
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device).eval()
        
        # Load decoder
        decoder_path = os.path.join(weight_path, "depth.pth")
        self.decoder = DepthDecoderS(num_ch_enc=self.encoder.num_ch_enc)
        loaded_dict = torch.load(decoder_path, map_location=self.device)
        self.decoder.load_state_dict(loaded_dict)
        self.decoder.to(self.device).eval()
        
        # Convert to FP16 if available
        if self.use_fp16:
            self.encoder = self.encoder.half()
            self.decoder = self.decoder.half()
            print("   Using FP16 inference")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.feed_height, self.feed_width)),
            transforms.ToTensor(),
        ])
        
        # Pre-allocate tensors
        self._warmup()
        
        print(f"   Model loaded: {self.feed_width}x{self.feed_height}")
    
    def _warmup(self):
        """Warmup the model to initialize CUDA kernels"""
        print("   Warming up model...")
        dummy = torch.randn(1, 3, self.feed_height, self.feed_width).to(self.device)
        if self.use_fp16:
            dummy = dummy.half()
        
        with torch.no_grad():
            for _ in range(3):
                features = self.encoder(dummy)
                _ = self.decoder(features)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Clear cache
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    @torch.no_grad()
    def predict_depth(self, rgb_image, depth_scale_factor=1.0):
        """Predict depth with optimized inference"""
        # Preprocess
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        
        if self.use_fp16:
            input_tensor = input_tensor.half()
        
        # Inference
        features = self.encoder(input_tensor)
        outputs = self.decoder(features)
        
        disp = outputs[("disp", 0)]
        _, depth = disp_to_depth(disp, 0.1, 100)
        
        # Apply scale
        metric_depth = depth * depth_scale_factor
        
        return metric_depth.squeeze().cpu().float().numpy()


class ONNXYOLODetector:
    """YOLO detector using ONNX Runtime for Jetson Nano"""
    
    def __init__(self, model_path=ONNX_MODEL_PATH, confidence=CONFIDENCE_THRESHOLD):
        self.confidence = confidence
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.class_names = None
        
        # Try ONNX Runtime first, then fallback to ultralytics
        if ONNX_RUNTIME_AVAILABLE and os.path.exists(model_path):
            self._load_onnx(model_path)
        elif YOLO_AVAILABLE:
            self._load_ultralytics()
        else:
            print("❌ No YOLO backend available")
    
    def _load_onnx(self, model_path):
        """Load ONNX model with optimized settings for Jetson"""
        print(f"Loading ONNX model: {model_path}")
        
        # Configure session options for Jetson
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Jetson Nano has 4 cores
        
        # Try TensorRT execution provider first, then CUDA, then CPU
        providers = []
        if IS_JETSON:
            providers.append('TensorrtExecutionProvider')
        providers.extend(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            print(f"   ✅ ONNX model loaded")
            print(f"   Input shape: {self.input_shape}")
            print(f"   Provider: {self.session.get_providers()[0]}")
            
            # Default YOLO class names (will be overwritten if available)
            self.class_names = {0: 'Person', 1: 'Machinery', 2: 'Vehicle'}
            
        except Exception as e:
            print(f"   ❌ Failed to load ONNX: {e}")
            self._load_ultralytics()
    
    def _load_ultralytics(self):
        """Fallback to ultralytics YOLO"""
        print("Using ultralytics YOLO (fallback)")
        
        try:
            if os.path.exists(YOLO_MODEL_PATH):
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
            else:
                self.yolo_model = YOLO('yolov8n.pt')
            
            if hasattr(self.yolo_model, 'names'):
                self.class_names = self.yolo_model.names
            
            print("   ✅ Ultralytics YOLO loaded")
        except Exception as e:
            print(f"   ❌ Failed to load YOLO: {e}")
            self.yolo_model = None
    
    def _preprocess_onnx(self, image):
        """Preprocess image for ONNX inference"""
        # Get target size from model
        if self.input_shape:
            target_h = self.input_shape[2] if len(self.input_shape) > 2 else 320
            target_w = self.input_shape[3] if len(self.input_shape) > 3 else 320
        else:
            target_h, target_w = 320, 320
        
        # Resize
        img = cv2.resize(image, (target_w, target_h))
        
        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img, (image.shape[1], image.shape[0])  # Return original size
    
    def _postprocess_onnx(self, outputs, orig_size, img_size=(320, 320)):
        """Postprocess ONNX outputs to detections"""
        detections = []
        
        # YOLO output format: [batch, num_detections, 5 + num_classes]
        # or [batch, 5 + num_classes, num_detections]
        output = outputs[0]
        
        if len(output.shape) == 3:
            # Check orientation
            if output.shape[1] > output.shape[2]:
                output = np.transpose(output, (0, 2, 1))
            
            output = output[0]  # Remove batch dimension
        
        orig_w, orig_h = orig_size
        scale_x = orig_w / img_size[0]
        scale_y = orig_h / img_size[1]
        
        for detection in output:
            if len(detection) < 5:
                continue
            
            # Get class scores
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            
            if confidence < self.confidence:
                continue
            
            # Get bounding box (center format to corner format)
            cx, cy, w, h = detection[:4]
            x1 = int((cx - w/2) * scale_x)
            y1 = int((cy - h/2) * scale_y)
            x2 = int((cx + w/2) * scale_x)
            y2 = int((cy + h/2) * scale_y)
            
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w - 1, x2)
            y2 = min(orig_h - 1, y2)
            
            # Get class name
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # Filter classes
            class_lower = class_name.lower()
            if not any(c in class_lower for c in ['person', 'machinery', 'vehicle', 'car', 'truck']):
                continue
            
            # Normalize class name
            if 'person' in class_lower:
                display_class = 'Person'
            elif 'machinery' in class_lower or 'excavator' in class_lower:
                display_class = 'Machinery'
            else:
                display_class = 'Vehicle'
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'class': display_class,
                'confidence': confidence
            })
        
        return detections
    
    def detect_objects(self, image, confidence=None):
        """Detect objects in image"""
        if confidence is None:
            confidence = self.confidence
        
        # Use ONNX if available
        if self.session is not None:
            try:
                img_input, orig_size = self._preprocess_onnx(image)
                outputs = self.session.run(None, {self.input_name: img_input})
                return self._postprocess_onnx(outputs, orig_size)
            except Exception as e:
                print(f"⚠️ ONNX inference error: {e}")
                return []
        
        # Fallback to ultralytics
        if hasattr(self, 'yolo_model') and self.yolo_model is not None:
            return self._detect_ultralytics(image, confidence)
        
        return []
    
    def _detect_ultralytics(self, image, confidence):
        """Detect using ultralytics YOLO"""
        try:
            results = self.yolo_model(image, verbose=False, conf=confidence)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, cls_idx, conf in zip(boxes, classes, confidences):
                        class_name = self.yolo_model.names[int(cls_idx)]
                        class_lower = class_name.lower()
                        
                        # Filter classes
                        is_person = 'person' in class_lower
                        is_machinery = 'machinery' in class_lower
                        is_vehicle = any(v in class_lower for v in ['vehicle', 'car', 'truck'])
                        
                        if not (is_person or is_machinery or is_vehicle):
                            continue
                        
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        if is_person:
                            display_class = 'Person'
                        elif is_machinery:
                            display_class = 'Machinery'
                        else:
                            display_class = 'Vehicle'
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'class': display_class,
                            'confidence': float(conf)
                        })
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []


class SimpleFPSCounter:
    """Lightweight FPS counter"""
    
    def __init__(self, window_size=10):
        self.times = []
        self.window_size = window_size
        self.last_time = time.time()
    
    def update(self):
        now = time.time()
        self.times.append(now - self.last_time)
        self.last_time = now
        if len(self.times) > self.window_size:
            self.times.pop(0)
    
    def get_fps(self):
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


def draw_detections_simple(image, depth_map, detections):
    """Simplified detection drawing for better performance"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = det['center']
        class_name = det['class']
        conf = det['confidence']
        
        # Clamp coordinates
        cy = max(0, min(cy, depth_map.shape[0] - 1))
        cx = max(0, min(cx, depth_map.shape[1] - 1))
        
        # Get depth
        depth = depth_map[cy, cx]
        
        # Color by class
        if class_name == 'Person':
            color = (0, 255, 0)
        elif class_name == 'Machinery':
            color = (0, 165, 255)
        else:
            color = (255, 0, 0)
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {depth:.1f}m"
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Jetson Nano Optimized Depth Estimation")
    parser.add_argument("-i", "--input", type=str, help="Input video file (default: webcam)")
    parser.add_argument("-w", "--weights", type=str, 
                       default="./weights/RTMonoDepth/s/m_640_192/",
                       help="Path to RT-MonoDepth weights (will use small model)")
    parser.add_argument("-o", "--output", type=str, help="Output video file")
    parser.add_argument("--width", type=int, default=320, help="Processing width (default: 320)")
    parser.add_argument("--height", type=int, default=240, help="Processing height (default: 240)")
    parser.add_argument("--skip-frames", type=int, default=2, 
                       help="Skip N frames between processing (default: 2)")
    parser.add_argument("--yolo-onnx", type=str, default=ONNX_MODEL_PATH,
                       help="Path to YOLO ONNX model")
    parser.add_argument("--no-yolo", action='store_true', help="Disable object detection")
    parser.add_argument("--depth-scale", type=float, default=5.0, help="Depth scale factor")
    parser.add_argument("--no-display", action='store_true', help="Disable display (for headless)")
    parser.add_argument("--fp16", action='store_true', default=True, help="Use FP16 inference")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Jetson Nano Optimized Depth Estimation")
    print("=" * 60)
    
    # Memory optimization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        # Reduce memory fragmentation
        torch.cuda.set_per_process_memory_fraction(0.7)
    
    # Load models
    print("\n📦 Loading models...")
    
    # Depth model
    depth_model = JetsonOptimizedDepthModel(args.weights, use_fp16=args.fp16)
    depth_scale = args.depth_scale
    
    # YOLO detector
    yolo_detector = None
    if not args.no_yolo:
        yolo_detector = ONNXYOLODetector(args.yolo_onnx)
    
    # Camera setup
    print("\n📷 Setting up camera...")
    cap = cv2.VideoCapture(args.input if args.input else 0)
    
    if not cap.isOpened():
        print("❌ Failed to open video source")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # For GStreamer pipeline on Jetson (better performance)
    if IS_JETSON and not args.input:
        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={args.width}, height={args.height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        try:
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            print("   Using GStreamer pipeline for camera")
        except:
            print("   GStreamer not available, using default capture")
    
    # Video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15  # Target FPS for output
        writer = cv2.VideoWriter(args.output, fourcc, fps, (args.width, args.height))
        print(f"📹 Recording to: {args.output}")
    
    # Processing loop
    fps_counter = SimpleFPSCounter()
    frame_count = 0
    last_depth_map = None
    last_detections = []
    
    print("\n🎬 Starting processing...")
    print(f"   Resolution: {args.width}x{args.height}")
    print(f"   Skip frames: {args.skip_frames}")
    print(f"   Depth scale: {depth_scale}")
    print("   Press 'q' to quit, '+/-' to adjust scale")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.input:  # End of video file
                    break
                continue
            
            frame_count += 1
            
            # Resize frame
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            
            # Process every Nth frame
            should_process = (frame_count % (args.skip_frames + 1)) == 0
            
            if should_process:
                # Convert to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Depth estimation
                depth_map = depth_model.predict_depth(rgb, depth_scale)
                
                # Resize depth map to frame size
                depth_map = cv2.resize(depth_map, (args.width, args.height))
                
                # Object detection
                if yolo_detector:
                    detections = yolo_detector.detect_objects(frame)
                else:
                    detections = []
                
                last_depth_map = depth_map
                last_detections = detections
            
            # Use cached results for skipped frames
            display_frame = frame.copy()
            
            if last_depth_map is not None and last_detections:
                display_frame = draw_detections_simple(
                    display_frame, last_depth_map, last_detections
                )
            
            # FPS overlay
            fps_counter.update()
            fps = fps_counter.get_fps()
            cv2.putText(display_frame, f"FPS: {fps:.1f} | Scale: {depth_scale:.1f}", 
                       (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display
            if not args.no_display:
                cv2.imshow('Jetson Depth', display_frame)
            
            # Write output
            if writer:
                writer.write(display_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('+') or key == ord('='):
                depth_scale *= 1.1
                print(f"Scale: {depth_scale:.2f}")
            elif key == ord('-'):
                depth_scale *= 0.9
                print(f"Scale: {depth_scale:.2f}")
            
            # Periodic garbage collection
            if frame_count % 100 == 0:
                gc.collect()
    
    except KeyboardInterrupt:
        print("\n⏹ Interrupted")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processed {frame_count} frames")
        print(f"   Average FPS: {fps_counter.get_fps():.1f}")


if __name__ == "__main__":
    main()
