#!/usr/bin/env python3
"""
Stage 1: Multi-Dataset Depth Evaluation for RT-MonoDepth
Evaluates RT-MonoDepth across NYU, KITTI, and Cityscapes datasets

Usage:
    python evaluate_depth_multi_dataset.py --model_path weights/RTMonoDepth/full/s_640_192 --datasets nyu kitti cityscapes
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import csv
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import RT-MonoDepth models
from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder
from networks.RTMonoDepth.RTMonoDepth_s import DepthDecoder as DepthDecoderS, DepthEncoder as DepthEncoderS
from layers import disp_to_depth

# Import our utilities
from compute_depth_metrics import batch_compute_depth_errors, aggregate_metrics, print_metrics
from dataset_loaders import get_dataset


class RTMonoDepthEvaluator:
    """
    RT-MonoDepth model evaluator
    """
    def __init__(self, model_path, model_type='full', device='cuda'):
        """
        Args:
            model_path: Path to model weights folder (contains encoder.pth and depth.pth)
            model_type: 'full' or 's' (small)
            device: 'cuda', 'mps', or 'cpu'
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = device
        
        # Load model
        self.encoder, self.depth_decoder = self._load_model()
        
        # Set to eval mode
        self.encoder.eval()
        self.depth_decoder.eval()
        
        print(f"✅ Loaded RT-MonoDepth model from {model_path}")
        print(f"   Device: {device}")
        print(f"   Model type: {model_type}")
    
    def _load_model(self):
        """Load RT-MonoDepth encoder and decoder"""
        encoder_path = self.model_path / 'encoder.pth'
        depth_path = self.model_path / 'depth.pth'
        
        if not encoder_path.exists() or not depth_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")
        
        # Load appropriate model architecture
        if self.model_type == 's':
            encoder = DepthEncoderS()
            depth_decoder = DepthDecoderS(encoder.num_ch_enc, scales=range(1))
        else:
            encoder = DepthEncoder()
            depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(1))
        
        # Load weights (strict=False to ignore extra keys like 'height', 'width', 'use_stereo')
        encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'), strict=False)
        depth_decoder.load_state_dict(torch.load(depth_path, map_location='cpu'), strict=False)
        
        # Move to device
        encoder.to(self.device)
        depth_decoder.to(self.device)
        
        return encoder, depth_decoder
    
    @torch.no_grad()
    def predict_depth(self, image):
        """
        Predict depth for a single image
        
        Args:
            image: PIL Image or tensor [C, H, W]
        
        Returns:
            depth: Depth prediction [H, W]
        """
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(self.device)
        
        # Forward pass
        features = self.encoder(image)
        outputs = self.depth_decoder(features)
        
        # Get disparity at scale 0
        disp = outputs[("disp", 0)]
        
        # Convert disparity to depth
        _, depth = disp_to_depth(disp, 0.1, 100)
        
        # Remove batch dimension and move to CPU
        depth = depth.squeeze(0).squeeze(0).cpu()
        
        return depth
    
    @torch.no_grad()
    def predict_batch(self, images):
        """
        Predict depth for a batch of images
        
        Args:
            images: Batch of images [B, C, H, W]
        
        Returns:
            depths: Batch of depth predictions [B, H, W]
        """
        images = images.to(self.device)
        
        # Forward pass
        features = self.encoder(images)
        outputs = self.depth_decoder(features)
        
        # Get disparity at scale 0
        disp = outputs[("disp", 0)]
        
        # Convert disparity to depth
        _, depth = disp_to_depth(disp, 0.1, 100)
        
        # Remove channel dimension
        depth = depth.squeeze(1).cpu()
        
        return depth


def evaluate_dataset(evaluator, dataset, batch_size=8, num_workers=4, median_scaling=True):
    """
    Evaluate RT-MonoDepth on a dataset
    
    Args:
        evaluator: RTMonoDepthEvaluator instance
        dataset: Dataset to evaluate on
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        median_scaling: Whether to use median scaling
    
    Returns:
        dict: Aggregated metrics
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if evaluator.device != 'cpu' else False
    )
    
    all_metrics = []
    
    print(f"Evaluating on {len(dataset)} images...")
    for batch in tqdm(dataloader, desc="Processing"):
        images = batch['image']
        gt_depths = batch['depth']
        
        # Predict depths
        pred_depths = evaluator.predict_batch(images)
        
        # Resize predictions to match GT size
        pred_depths_resized = []
        for pred, gt in zip(pred_depths, gt_depths):
            # Resize prediction to GT size
            pred_np = pred.numpy()
            gt_np = gt.numpy()
            
            if pred_np.shape != gt_np.shape:
                from PIL import Image
                pred_pil = Image.fromarray(pred_np)
                pred_pil = pred_pil.resize((gt_np.shape[1], gt_np.shape[0]), Image.BILINEAR)
                pred_np = np.array(pred_pil)
            
            pred_depths_resized.append(pred_np)
        
        # Compute metrics
        batch_metrics = batch_compute_depth_errors(
            gt_depths.numpy(),
            np.array(pred_depths_resized),
            median_scaling=median_scaling
        )
        
        all_metrics.extend(batch_metrics)
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset Depth Evaluation for RT-MonoDepth')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model weights folder (e.g., weights/RTMonoDepth/full/s_640_192)')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 's'],
                        help='Model type: full or s (small)')
    
    # Dataset arguments
    parser.add_argument('--datasets', nargs='+', default=['nyu', 'kitti', 'cityscapes'],
                        choices=['nyu', 'kitti', 'cityscapes'],
                        help='Datasets to evaluate on (NYU Depth V2, KITTI, Cityscapes)')
    parser.add_argument('--data_root', type=str, default='datasets',
                        help='Root path to datasets folder')
    parser.add_argument('--cityscapes_split', type=str, default='val', choices=['val', 'test'],
                        help='Cityscapes split to use (val or test)')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--no_median_scaling', action='store_true',
                        help='Disable median scaling (not recommended)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='benchmark/results',
                        help='Base directory to save results (model subdirs will be created automatically)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save depth predictions (warning: large files)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print(f"RT-MonoDepth Multi-Dataset Depth Evaluation - Stage 1")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Device: {device}")
    print(f"Median scaling: {not args.no_median_scaling}")
    print(f"{'='*60}\n")
    
    # Create output directory with model-specific subdirectory
    # Extract model name from path (e.g., "m_640_192" from "weights/RTMonoDepth/full/m_640_192")
    model_path_parts = Path(args.model_path).parts
    if len(model_path_parts) >= 2:
        # Format: {model_type}_{model_name} (e.g., "full_m_640_192")
        model_subdir = f"{model_path_parts[-2]}_{model_path_parts[-1]}"
    else:
        model_subdir = model_path_parts[-1]
    
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / model_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Results will be saved to: {output_dir}\n")
    
    # Load model
    evaluator = RTMonoDepthEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device
    )
    
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((192, 640)),  # RT-MonoDepth input size
        transforms.ToTensor(),
    ])
    
    # Evaluate on each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name.upper()} dataset")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            # Use appropriate split for each dataset
            split = 'test'  # Default for NYU and KITTI Eigen
            if dataset_name.lower() == 'cityscapes':
                split = args.cityscapes_split
            
            dataset = get_dataset(
                dataset_name=dataset_name,
                data_root=args.data_root,
                split=split,
                transform=transform
            )
            
            if len(dataset) == 0:
                print(f"⚠️  {dataset_name} dataset is empty, skipping...")
                continue
            
            # Evaluate
            metrics = evaluate_dataset(
                evaluator=evaluator,
                dataset=dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                median_scaling=not args.no_median_scaling
            )
            
            # Print metrics
            print_metrics(metrics, dataset_name.upper())
            
            # Store results
            all_results[dataset_name] = metrics
            
        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if len(all_results) > 0:
        # Save as JSON
        json_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            # Convert tuples to lists for JSON serialization
            json_results = {}
            for dataset_name, metrics in all_results.items():
                json_results[dataset_name] = {
                    k: [float(v[0]), float(v[1])] if isinstance(v, tuple) else v
                    for k, v in metrics.items()
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Results saved to {json_path}")
        
        # Save as CSV
        csv_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'Metric', 'Mean', 'Std'])
            
            for dataset_name, metrics in all_results.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, tuple):
                        writer.writerow([dataset_name, metric_name, f"{value[0]:.4f}", f"{value[1]:.4f}"])
                    else:
                        writer.writerow([dataset_name, metric_name, value, ''])
        
        print(f"✅ Results saved to {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Evaluated {len(all_results)} datasets")
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    for dataset_name, metrics in all_results.items():
        print(f"  {dataset_name.upper()}: AbsRel={metrics['abs_rel'][0]:.4f}, δ<1.25={metrics['a1'][0]:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
