#!/usr/bin/env python3
"""
Train YOLOv12n on Merged Construction Safety Dataset
Person (class 17) + Construction Equipment (classes 0-16) = 18 classes total

Uses MLX for optimized training on Apple Silicon (M1/M2/M3)
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import platform
import os

def setup_mlx_environment():
    """Setup MLX environment for Apple Silicon training."""
    
    # Check if running on Apple Silicon
    is_apple_silicon = platform.processor() == 'arm' or platform.machine() == 'arm64'
    
    if is_apple_silicon:
        print("🍎 Apple Silicon detected - configuring for MLX acceleration")
        
        # Set environment variables for MLX optimization
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Try to import mlx to verify it's available
        try:
            import mlx
            import mlx.core as mx
            print(f"✓ MLX available")
            print(f"✓ MLX device: {mx.default_device()}")
            return 'mps'  # Use MPS backend with MLX optimizations
        except ImportError:
            print("⚠️  MLX not installed. Install with: pip install mlx")
            print("Falling back to MPS (Metal Performance Shaders)")
            return 'mps'
    else:
        print("ℹ️  Not on Apple Silicon")
        return 'cuda' if torch.cuda.is_available() else 'cpu'

def train_yolo12n():
    """Train YOLOv12n model on merged construction safety dataset with MLX."""
    
    # Paths
    base_dir = Path(__file__).parent
    merged_dataset = base_dir / 'merged_construction_safety'
    data_yaml = merged_dataset / 'data.yaml'
    output_dir = base_dir / 'runs'
    
    # Verify data.yaml exists
    if not data_yaml.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        print("Please run merge_datasets.py first!")
        return
    
    # Verify merged dataset exists
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = Path(data_config['path'])
    if not dataset_path.exists():
        print(f"❌ Error: Merged dataset not found at {dataset_path}")
        print("Please run merge_datasets.py first!")
        return
    
    # Setup MLX environment
    device = setup_mlx_environment()
    print(f"Using device: {device}")
    
    # Print dataset info
    print("="*70)
    print("YOLOv12n Training - Construction Safety Dataset")
    print("="*70)
    print(f"Dataset: {data_yaml}")
    print(f"Classes: {data_config['nc']}")
    print(f"Device: {device}")
    print("="*70)
    
    # Training parameters optimized for MLX/Apple Silicon
    params = {
        'data': str(data_yaml),
        'epochs': 100,                # Number of epochs
        'imgsz': 640,                 # Image size
        'batch': 32,                  # Larger batch size for Apple Silicon unified memory
        'device': device,
        'project': str(output_dir),
        'name': 'yolo12n_construction_safety_mlx',
        'patience': 50,               # Early stopping patience
        'save': True,                 # Save checkpoints
        'save_period': 10,            # Save checkpoint every N epochs
        'cache': True,                # Cache images in RAM (Apple Silicon has unified memory)
        'workers': 8,                 # Number of dataloader workers (optimize for M-series)
        'optimizer': 'AdamW',         # AdamW works well with MLX
        'verbose': True,
        'seed': 42,
        'deterministic': False,       # Set False for better MLX performance
        'single_cls': False,
        'rect': False,
        'cos_lr': True,               # Cosine LR scheduler
        'close_mosaic': 10,           # Disable mosaic augmentation last N epochs
        'resume': False,              # Resume from last checkpoint
        'amp': True,                  # Automatic Mixed Precision (MLX supports FP16)
        'fraction': 1.0,              # Train on fraction of data
        'profile': False,             # Profile ONNX and TensorRT
        'freeze': None,               # Freeze layers (None or list of layer indices)
        'lr0': 0.01,                  # Initial learning rate
        'lrf': 0.01,                  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,            # SGD momentum
        'weight_decay': 0.0005,       # Optimizer weight decay
        'warmup_epochs': 3.0,         # Warmup epochs
        'warmup_momentum': 0.8,       # Warmup momentum
        'warmup_bias_lr': 0.1,        # Warmup bias learning rate
        'box': 7.5,                   # Box loss gain
        'cls': 0.5,                   # Classification loss gain
        'dfl': 1.5,                   # DFL loss gain
        'pose': 12.0,                 # Pose loss gain (keypoint)
        'kobj': 1.0,                  # Keypoint obj loss gain
        'label_smoothing': 0.0,       # Label smoothing
        'nbs': 64,                    # Nominal batch size
        'overlap_mask': True,         # Masks should overlap during training
        'mask_ratio': 4,              # Mask downsample ratio
        'dropout': 0.0,               # Dropout (classification only)
        'val': True,                  # Validate/test during training
    }
    
    print("\nTraining Parameters (MLX Optimized):")
    print(f"  Epochs: {params['epochs']}")
    print(f"  Batch Size: {params['batch']} (optimized for Apple Silicon unified memory)")
    print(f"  Image Size: {params['imgsz']}")
    print(f"  Device: {params['device']} (MLX acceleration)")
    print(f"  Optimizer: {params['optimizer']}")
    print(f"  Learning Rate: {params['lr0']} -> {params['lr0'] * params['lrf']}")
    print(f"  Cache Images: {params['cache']} (using unified memory)")
    print("="*70)
    
    try:
        # Initialize YOLOv12n model
        print("\n🚀 Loading YOLOv12n model...")
        model = YOLO('yolo12n.pt')  # Will auto-download if not exists
        
        # Start training
        print("\n🏋️ Starting training...\n")
        results = model.train(**params)
        
        print("\n" + "="*70)
        print("✓ Training Complete!")
        print("="*70)
        print(f"\nBest model saved at: {output_dir / params['name'] / 'weights' / 'best.pt'}")
        print(f"Last model saved at: {output_dir / params['name'] / 'weights' / 'last.pt'}")
        print(f"\nTensorBoard logs: {output_dir / params['name']}")
        print("View with: tensorboard --logdir=" + str(output_dir / params['name']))
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\nFinal Metrics:")
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"  mAP@50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  mAP@50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
        
        print("\n" + "="*70)
        print("Next steps:")
        print("1. Evaluate: python evaluate_yolo12n.py")
        print("2. Export: yolo export model=runs/.../weights/best.pt format=onnx")
        print("3. Inference: yolo predict model=runs/.../weights/best.pt source=image.jpg")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == '__main__':
    train_yolo12n()
