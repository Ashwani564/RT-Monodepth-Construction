# Jetson Nano Deployment Guide

This guide covers deploying RT-MonoDepth with YOLO object detection on Jetson Nano (2019) 4GB.

## Performance Targets
- **Original script**: ~1 FPS
- **Optimized script**: 10-15+ FPS

## Key Optimizations Made

1. **Reduced Resolution**: 320x240 instead of 640x480
2. **Frame Skipping**: Process every 2-3 frames, display interpolated
3. **ONNX/TensorRT**: Hardware-accelerated inference
4. **FP16 Precision**: Half-precision floating point
5. **Small Model Only**: Uses RT-MonoDepth-S variant
6. **Memory Management**: Explicit garbage collection, reduced queue sizes
7. **GStreamer Pipeline**: Direct camera access bypassing V4L2 overhead

## Files for Jetson

Copy these files to your Jetson Nano:
```
realtime_depth_jetson.py          # Optimized main script
custom_yolo11n_jetson_320.onnx    # YOLO model (ONNX format)
requirements_jetson.txt           # Dependencies
networks/                         # Model architecture
weights/RTMonoDepth/s/            # Small model weights
layers.py                         # Utility functions
```

## Installation on Jetson Nano

### 1. Install PyTorch (JetPack compatible)
```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Install PyTorch for Jetson (example for JetPack 4.6)
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install TorchVision
git clone --branch v0.11.0 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
```

### 2. Install ONNX Runtime (GPU)
```bash
# Download from Jetson Zoo: https://elinux.org/Jetson_Zoo#ONNX_Runtime
wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kgez9.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
```

### 3. Install Other Dependencies
```bash
pip3 install -r requirements_jetson.txt
```

### 4. (Optional) Convert to TensorRT for Maximum Performance
```bash
# On Jetson Nano, convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=custom_yolo11n_jetson_320.onnx \
    --saveEngine=custom_yolo11n_jetson_320.engine \
    --fp16 \
    --workspace=1024

# This creates a TensorRT engine optimized for your specific Jetson
```

## Running the Optimized Script

### Basic Usage (Webcam)
```bash
python3 realtime_depth_jetson.py
```

### With ONNX Model
```bash
python3 realtime_depth_jetson.py --yolo-onnx custom_yolo11n_jetson_320.onnx
```

### Process Video File
```bash
python3 realtime_depth_jetson.py -i input_video.mp4 -o output_video.mp4
```

### Maximum Performance Mode
```bash
python3 realtime_depth_jetson.py \
    --width 320 \
    --height 240 \
    --skip-frames 3 \
    --fp16 \
    --yolo-onnx custom_yolo11n_jetson_320.onnx
```

### Headless Mode (No Display)
```bash
python3 realtime_depth_jetson.py --no-display -o output.mp4
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | webcam | Input video file |
| `-o, --output` | None | Output video file |
| `--width` | 320 | Processing width |
| `--height` | 240 | Processing height |
| `--skip-frames` | 2 | Frames to skip between processing |
| `--yolo-onnx` | custom_yolo11n_jetson_320.onnx | YOLO ONNX model path |
| `--no-yolo` | False | Disable object detection |
| `--depth-scale` | 5.0 | Depth scale factor |
| `--no-display` | False | Headless mode |
| `--fp16` | True | Use FP16 inference |

## Controls

- `q` or `ESC`: Quit
- `+` or `=`: Increase depth scale
- `-`: Decrease depth scale

## Troubleshooting

### Low FPS
1. Increase `--skip-frames` to 3 or 4
2. Reduce resolution: `--width 256 --height 192`
3. Disable YOLO: `--no-yolo` (depth only)
4. Use TensorRT engine instead of ONNX

### Out of Memory
1. Close other applications
2. Reduce resolution
3. Increase skip frames
4. Run: `sudo nvpmodel -m 0` (MAXN power mode)
5. Run: `sudo jetson_clocks` (max clock speeds)

### Camera Not Working
```bash
# Test camera
nvgstcapture-1.0

# Check camera device
ls /dev/video*

# Use specific device
python3 realtime_depth_jetson.py -i /dev/video0
```

### CUDA/TensorRT Errors
```bash
# Check CUDA
nvcc --version

# Check TensorRT
dpkg -l | grep tensorrt

# Reinstall JetPack if needed
```

## Performance Comparison

| Configuration | Resolution | Skip | FPS |
|--------------|------------|------|-----|
| Original script | 640x480 | 0 | ~1 |
| Optimized (ONNX) | 320x240 | 2 | ~10-12 |
| Optimized (TensorRT) | 320x240 | 2 | ~12-15 |
| Depth only | 320x240 | 2 | ~15-20 |

## Power Modes

For best performance, use MAXN mode:
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Check current mode:
```bash
sudo nvpmodel -q
```

## Memory Usage

The optimized script is designed for 4GB RAM:
- PyTorch model: ~500MB
- ONNX Runtime: ~200MB
- Frame buffers: ~100MB
- System overhead: ~1GB
- Available for processing: ~2GB

## Notes

- The ONNX model uses 320x320 input for YOLO (vs 640x640 in original)
- Depth model uses 320x192 input (vs 640x192)
- FP16 reduces memory usage by ~50% with minimal accuracy loss
- Frame skipping maintains smooth display while reducing computation
