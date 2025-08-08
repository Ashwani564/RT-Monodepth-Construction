# RT-MonoDepth: Metric Monocular Depth Estimation

A real-time **metric monocular depth estimation** system using RT-MonoDepth neural network with YOLO object detection. Provides accurate metric depth measurements for human detection, optimized for MacBook M1/M2 Pro, Linux, and Jetson Nano deployment.
## Results
<img width="267" height="221" alt="image" src="https://github.com/user-attachments/assets/e3538e9f-1d4b-4c30-b626-7aeb03c3505e" />
<img width="1508" height="939" alt="image" src="https://github.com/user-attachments/assets/1f450ff9-d159-4deb-b3df-bd1198c9a9ea" />

## Features

- 🎯 **Metric monocular depth estimation** with RT-MonoDepth neural network
-  **Accurate metric depth measurements** at detection points  
- � **Person detection** using custom YOLOv11n weights
- �🔧 **Interactive calibration** with real-time keyboard controls
- 🚀 **Multi-platform support** - macOS (Apple Silicon), Linux, Jetson Nano
- ⚡ **Hardware acceleration** - MLX (macOS), CUDA (Linux/Jetson), CPU fallback
- 📹 **Video recording** capability with depth annotations

## System Requirements

### Supported Platforms
- **macOS** (optimized for Apple Silicon M1/M2 Pro)
- **Linux** (Ubuntu 18.04+, tested on Jetson Nano)
- **Jetson Nano** (NVIDIA L4T/JetPack 4.6+)

### Hardware Requirements
- **Python 3.8+**
- **Webcam** or video input source
- **8GB+ RAM** recommended for smooth performance
- **GPU** (optional but recommended for better performance)

## Installation

### 1. Clone or Download

If you have this as a repository:
```bash
git clone <repository-url>
cd RT-MonoDepth
```

Or if you have the files locally, navigate to the project directory:
```bash
cd /path/to/RT-MonoDepth
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

**On Jetson Nano:**
```bash
python3 -m venv env
source env/bin/activate
# Install system dependencies for OpenCV
sudo apt-get update
sudo apt-get install -y python3-opencv libopencv-dev
```

### 3. Install Dependencies

#### Option A: Using requirements.txt (Recommended)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Platform-specific Installation

**For macOS (Apple Silicon M1/M2):**
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio
pip install -r requirements.txt

# Optional: MLX for Apple Silicon acceleration
pip install mlx mlx-nn
```

**For Linux/Ubuntu:**
```bash
# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For Jetson Nano (NVIDIA L4T):**
```bash
# Install PyTorch for Jetson (ARM64)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision for Jetson
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python setup.py install --user
cd ..

# Install remaining dependencies
pip install opencv-python numpy pillow tqdm ultralytics
```

### 4. Download Model Weights

You need the RT-MonoDepth pre-trained weights. The script expects them in:
```
weights/RTMonoDepth/s/m_640_192/
├── encoder.pth
└── depth.pth
```

Create the directory structure and place your model weights:
```bash
mkdir -p weights/RTMonoDepth/s/m_640_192/
# Place encoder.pth and depth.pth in this directory
```

### 5. Custom YOLO Weights (Optional)

If you have custom YOLOv11n weights, place them in the project root:
```bash
# Place your custom_yolo11n.pt file in the project root directory
cp /path/to/your/custom_yolo11n.pt ./
```

If you don't have custom weights, the system will automatically download and use standard YOLOv11n weights.

## Usage

### Basic Usage

Start the real-time depth estimation system:

```bash
python realtime_depth_video.py
```

### Command Line Options

```bash
# Use webcam (default)
python realtime_depth_video.py

# Use video file as input
python realtime_depth_video.py -i /path/to/video.mp4

# Record output video
python realtime_depth_video.py -r

# Use different camera type
python realtime_depth_video.py --camera jetson_nano

# Enable auto-calibration
python realtime_depth_video.py --auto-calib

# Use standard YOLOv8 instead of custom YOLOv11n
python realtime_depth_video.py --use-yolov8

# Disable YOLO object detection
python realtime_depth_video.py --no-yolo

# Set processing width
python realtime_depth_video.py --width 800

# Set FPS limit
python realtime_depth_video.py --fps-limit 30
```

### Complete Command Reference

```bash
python realtime_depth_video.py [OPTIONS]

Options:
  -i, --input TEXT           Input video file (default: webcam)
  -w, --weights TEXT         Path to RT-MonoDepth weights [default: ./weights/RTMonoDepth/s/m_640_192/]
  -r, --record              Record output video
  --width INTEGER            Processing width [default: 640]
  --camera [macbook_m1_pro|jetson_nano]  Camera type [default: macbook_m1_pro]
  --no-mlx                   Disable MLX acceleration
  --no-yolo                  Disable YOLO object detection
  --use-yolov8              Use standard YOLOv8n instead of custom YOLOv11n
  --fps-limit INTEGER        FPS limit for processing [default: 30]
  --auto-calib              Enable automatic geometric scale calibration
  --person-height FLOAT      Assumed average person height in meters [default: 1.70]
  --auto-calib-min-frames INTEGER  Frames to wait before applying first auto scale update [default: 15]
  --auto-calib-smoothing FLOAT     EMA smoothing factor (0-1, higher = slower changes) [default: 0.9]
```

## Real-time Controls

Once the application is running, use these keyboard controls:

### Depth Calibration
- **`+` or `=`** : Increase depth scale (if readings are too low)
- **`-`** : Decrease depth scale (if readings are too high)
- **`c`** : Quick calibrate assuming 1.5m distance at cursor
- **`p`** : Precise calibrate (enter actual distance in terminal)
- **`r`** : Reset depth scale to 1.0

### General Controls
- **`q` or `ESC`** : Quit application
- **`s`** : Save current frame as image
- **Mouse movement** : Show depth measurement at cursor position

## Calibration Guide

### Initial Setup
1. **Position yourself** 1-2 meters away from the camera
2. **Point your mouse** at a person in the video feed
3. **Check the depth reading** displayed at the cursor

### If Depth is Incorrect
1. **Too low?** Press `+` to increase scale
2. **Too high?** Press `-` to decrease scale
3. **Quick fix:** Press `c` to assume 1.5m at cursor position
4. **Precise fix:** Press `p` and enter the actual distance

### Auto-Calibration (Optional)
Enable with `--auto-calib` flag. The system will automatically adjust depth scale based on detected person heights using geometric estimation.

## File Structure

```
RT-MonoDepth/
├── realtime_depth_video.py      # Main application
├── layers.py                    # RT-MonoDepth layers
├── requirements.txt             # Python dependencies
├── custom_yolo11n.pt           # Custom YOLO weights (optional)
├── README.md                   # This file
├── env/                        # Virtual environment
├── networks/                   # Neural network modules
│   └── RTMonoDepth/
│       ├── __init__.py
│       ├── RTMonoDepth.py
│       └── RTMonoDepth_s.py
└── weights/                    # Model weights
    └── RTMonoDepth/
        └── s/
            └── m_640_192/
                ├── encoder.pth
                └── depth.pth
```

## Performance Tips

### For Best Performance

**macOS (Apple Silicon):**
1. **Use MLX acceleration** (automatically detected)
2. **Close unnecessary applications** to free up CPU/GPU
3. **Use lower resolution** (--width 480) for faster processing
4. **Enable auto-calibration** for hands-free operation

**Linux/Ubuntu:**
1. **Use CUDA acceleration** if NVIDIA GPU available
2. **Install CUDA drivers** and PyTorch with CUDA support
3. **Use lower resolution** for CPU-only systems
4. **Monitor GPU memory** usage

**Jetson Nano:**
1. **Use GPU acceleration** (automatic with proper PyTorch installation)
2. **Set power mode** to maximum: `sudo nvpmodel -m 0`
3. **Increase swap space** for memory-intensive operations
4. **Use lower resolution** (--width 480) and FPS (--fps-limit 15)
5. **Disable auto-calibration** unless needed
6. **Close unnecessary services** to free up resources

### Platform-specific Optimizations

**Jetson Nano Setup:**
```bash
# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Increase swap space
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
echo '/mnt/4GB.swap swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

### Troubleshooting

**Low FPS / Lag:**
- Reduce processing width: `--width 480`
- Lower FPS limit: `--fps-limit 15`
- On macOS: Disable MLX: `--no-mlx`
- On Jetson: Set performance mode: `sudo nvpmodel -m 0`

**Inaccurate Depth:**
- Calibrate using `+/-` keys
- Enable auto-calibration: `--auto-calib`
- Check camera parameters in code

**YOLO Not Working:**
- Install ultralytics: `pip install ultralytics`
- Use standard YOLO: `--use-yolov8`
- Disable YOLO: `--no-yolo`

**No Model Weights:**
- Download RT-MonoDepth weights
- Check weights path: `--weights /path/to/weights`

**Platform-specific Issues:**

**macOS:**
- MLX not installing: Ensure you have Apple Silicon Mac
- Camera permission: Allow camera access in System Preferences

**Linux:**
- CUDA issues: Verify NVIDIA drivers and CUDA installation
- Camera not detected: Check `/dev/video*` devices
- Permission denied: Add user to video group: `sudo usermod -a -G video $USER`

**Jetson Nano:**
- Out of memory: Increase swap space (see Performance Tips)
- Slow performance: Set maximum performance mode
- Camera issues: Check CSI camera connection and enable in `/boot/extlinux/extlinux.conf`

## Output

### Display Window
- **Green bounding boxes** around detected persons
- **Depth measurements** at detection centers
- **Sample points** showing depth sampling locations
- **Performance metrics** (FPS, processing time)
- **Calibration status** and scale factors

### Recorded Video
When using `--record`, videos are saved to:
```
output_depth_video/depth_estimation_YYYYMMDD_HHMMSS.mp4
```

### Saved Frames
Press `s` to save current frame:
```
depth_frame_YYYYMMDD_HHMMSS.jpg
```

## Technical Details

- **Depth Model:** RT-MonoDepth (small architecture)
- **Object Detection:** YOLOv11n (custom) or YOLOv8n (standard)
- **Input Resolution:** 640x480 (default, configurable)
- **Depth Range:** 0.1m to 100m (theoretical)
- **Practical Range:** 0.5m to 10m (optimal accuracy)
- **Supported Platforms:** macOS (Apple Silicon), Linux, Jetson Nano
- **Acceleration:** MLX (macOS), CUDA (Linux/Jetson), CPU fallback

## Quick Start Examples

### macOS (Apple Silicon)
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install mlx mlx-nn  # For MLX acceleration
python realtime_depth_video.py --auto-calib
```

### Linux (with NVIDIA GPU)
```bash
python3 -m venv env
source env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python realtime_depth_video.py --auto-calib
```

### Jetson Nano
```bash
# After following Jetson-specific installation steps
sudo nvpmodel -m 0  # Set performance mode
python realtime_depth_video.py --width 480 --fps-limit 15
```

## License

See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model weights are in correct location
4. Test with `--no-yolo` flag to isolate issues
