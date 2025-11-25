#!/bin/bash
# Quick start script for fine-tuning RT-MonoDepth on Cityscapes

echo "=========================================="
echo "RT-MonoDepth Fine-tuning Setup"
echo "=========================================="
echo ""

# Check if running with GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda"
else
    echo "⚠️  No NVIDIA GPU detected. Training will be slow on CPU."
    echo "   Consider using:"
    echo "   - Google Colab Pro ($10/month)"
    echo "   - AWS EC2 g4dn.xlarge (~$0.50/hour)"
    echo "   - Lambda Labs (~$0.50/hour)"
    read -p "Continue with CPU? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    DEVICE="cpu"
fi

echo ""
echo "=========================================="
echo "Checking Dependencies"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

if ! python -c 'import torch' &> /dev/null; then
    echo "❌ PyTorch not found. Installing..."
    if [ "$DEVICE" = "cuda" ]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio
    fi
else
    echo "✅ PyTorch installed"
    python -c 'import torch; print(f"   Version: {torch.__version__}")'
fi

# Install other dependencies
echo ""
echo "Installing additional dependencies..."
pip install -q -r finetune/requirements.txt
echo "✅ All dependencies installed"

echo ""
echo "=========================================="
echo "Checking Data"
echo "=========================================="
echo ""

# Check Cityscapes data
if [ ! -d "datasets/cityscapes" ]; then
    echo "❌ Cityscapes dataset not found at datasets/cityscapes/"
    echo ""
    echo "Please download Cityscapes dataset:"
    echo "1. leftImg8bit_trainvaltest.zip"
    echo "2. disparity_trainvaltest.zip"
    echo ""
    echo "Visit: https://www.cityscapes-dataset.com/"
    exit 1
fi

# Count training images
TRAIN_IMAGES=$(find datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train -name "*.png" 2>/dev/null | wc -l)
VAL_IMAGES=$(find datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val -name "*.png" 2>/dev/null | wc -l)
TRAIN_DISP=$(find datasets/cityscapes/disparity_trainvaltest/disparity/train -name "*.png" 2>/dev/null | wc -l)
VAL_DISP=$(find datasets/cityscapes/disparity_trainvaltest/disparity/val -name "*.png" 2>/dev/null | wc -l)

echo "Training images: $TRAIN_IMAGES"
echo "Training disparity: $TRAIN_DISP"
echo "Validation images: $VAL_IMAGES"
echo "Validation disparity: $VAL_DISP"
echo ""

if [ "$TRAIN_IMAGES" -eq 0 ] || [ "$TRAIN_DISP" -eq 0 ]; then
    echo "❌ Cityscapes data incomplete"
    exit 1
fi

echo "✅ Cityscapes data ready"

echo ""
echo "=========================================="
echo "Checking Pre-trained Weights"
echo "=========================================="
echo ""

# Check if pre-trained weights exist
if [ ! -d "weights/RTMonoDepth/full/sh_640_192" ]; then
    echo "❌ Pre-trained weights not found"
    echo "   Expected: weights/RTMonoDepth/full/sh_640_192/"
    exit 1
fi

echo "✅ Pre-trained KITTI weights found"

echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo ""

# Get user input or use defaults
read -p "Model to fine-tune (default: full_sh_640_192): " MODEL_NAME
MODEL_NAME=${MODEL_NAME:-full_sh_640_192}

read -p "Number of epochs (default: 20): " EPOCHS
EPOCHS=${EPOCHS:-20}

read -p "Batch size (default: 12, reduce if OOM): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-12}

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "1. Check terminal output for loss and accuracy"
echo "2. Open tensorboard: tensorboard --logdir finetune/logs"
echo "3. Visit: http://localhost:6006"
echo ""
echo "Estimated time: 8-12 hours (RTX 3080/4090)"
echo ""

# Create directories
mkdir -p finetune/checkpoints
mkdir -p finetune/logs

# Start training
python finetune/train_cityscapes.py \
    --model_name "$MODEL_NAME" \
    --model_type full \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --encoder_lr 1e-5 \
    --decoder_lr 1e-4

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Evaluate fine-tuned model:"
echo "   python benchmark/evaluate_depth_multi_dataset.py \\"
echo "       --model_path finetune/checkpoints/$MODEL_NAME/final_weights \\"
echo "       --model_type full \\"
echo "       --datasets cityscapes"
echo ""
echo "2. Compare with pre-trained model:"
echo "   python benchmark/evaluate_depth_multi_dataset.py \\"
echo "       --model_path weights/RTMonoDepth/full/$MODEL_NAME \\"
echo "       --model_type full \\"
echo "       --datasets cityscapes"
echo ""
echo "3. Check tensorboard logs:"
echo "   tensorboard --logdir finetune/logs"
echo ""
