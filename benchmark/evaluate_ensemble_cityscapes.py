#!/usr/bin/env python3
"""
Quick Win: Ensemble Model Evaluation for Cityscapes
Combines predictions from all 6 RT-MonoDepth models with weighted averaging
Expected improvement: +8-12% accuracy (from 38% to 46-50%)
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

from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder
from networks.RTMonoDepth.RTMonoDepth_s import DepthDecoder as DepthDecoderS, DepthEncoder as DepthEncoderS
from layers import disp_to_depth
from benchmark.compute_depth_metrics import batch_compute_depth_errors, aggregate_metrics, print_metrics
from benchmark.dataset_loaders import get_dataset


class EnsembleModel:
    """Ensemble of multiple RT-MonoDepth models"""
    
    def __init__(self, model_configs, device='mps'):
        """
        Args:
            model_configs: List of (model_path, model_type, weight) tuples
            device: 'cuda', 'mps', or 'cpu'
        """
        self.device = device
        self.models = []
        self.weights = []
        
        print("Loading ensemble models...")
        for model_path, model_type, weight in model_configs:
            encoder, decoder = self._load_model(model_path, model_type)
            self.models.append((encoder, decoder))
            self.weights.append(weight)
            print(f"  ✓ Loaded {Path(model_path).name} (weight: {weight})")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"\n✅ Ensemble ready with {len(self.models)} models")
        print(f"   Normalized weights: {[f'{w:.3f}' for w in self.weights]}")
    
    def _load_model(self, model_path, model_type):
        """Load a single model"""
        encoder_path = Path(model_path) / 'encoder.pth'
        depth_path = Path(model_path) / 'depth.pth'
        
        if not encoder_path.exists() or not depth_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Load appropriate architecture
        if model_type == 's':
            encoder = DepthEncoderS()
            decoder = DepthDecoderS(encoder.num_ch_enc, scales=range(1))
        else:
            encoder = DepthEncoder()
            decoder = DepthDecoder(encoder.num_ch_enc, scales=range(1))
        
        # Load weights
        encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'), strict=False)
        decoder.load_state_dict(torch.load(depth_path, map_location='cpu'), strict=False)
        
        # Move to device and set to eval mode
        encoder.to(self.device)
        decoder.to(self.device)
        encoder.eval()
        decoder.eval()
        
        return encoder, decoder
    
    @torch.no_grad()
    def predict(self, images, use_tta=True):
        """
        Predict depth using ensemble
        
        Args:
            images: Batch of images [B, C, H, W]
            use_tta: Use test-time augmentation (horizontal flip)
        
        Returns:
            depth: Ensemble depth prediction [B, H, W]
        """
        batch_size = images.shape[0]
        predictions = []
        
        # Get predictions from each model
        for (encoder, decoder), weight in zip(self.models, self.weights):
            # Original prediction
            features = encoder(images)
            outputs = decoder(features)
            disp = outputs[("disp", 0)]
            
            # Convert to depth
            _, depth = disp_to_depth(disp, 0.1, 100.0)
            
            if use_tta:
                # Horizontal flip augmentation
                images_flipped = torch.flip(images, dims=[3])
                features_flipped = encoder(images_flipped)
                outputs_flipped = decoder(features_flipped)
                disp_flipped = outputs_flipped[("disp", 0)]
                disp_flipped = torch.flip(disp_flipped, dims=[3])
                
                _, depth_flipped = disp_to_depth(disp_flipped, 0.1, 100.0)
                
                # Average original and flipped
                depth = (depth + depth_flipped) / 2.0
            
            predictions.append(depth * weight)
        
        # Weighted average
        ensemble_depth = torch.sum(torch.stack(predictions), dim=0)
        
        return ensemble_depth


def evaluate_ensemble(ensemble, dataset, batch_size=8, num_workers=4, median_scaling=True):
    """Evaluate ensemble model on dataset"""
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    all_metrics = []
    
    print(f"Evaluating ensemble on {len(dataset)} images...")
    
    for batch in tqdm(dataloader, desc="Processing"):
        images = batch['image'].to(ensemble.device)
        gt_depths = batch['depth']
        
        # Predict depths
        pred_depths = ensemble.predict(images, use_tta=True)
        
        # Move predictions to CPU
        pred_depths = pred_depths.cpu()
        
        # Resize predictions to match ground truth
        pred_depths_resized = []
        for i in range(len(pred_depths)):
            pred_depth = pred_depths[i].squeeze().numpy()  # Remove batch and channel dims
            gt_depth = gt_depths[i].numpy()
            
            if pred_depth.shape != gt_depth.shape:
                from PIL import Image
                import cv2
                # Use cv2 for float arrays
                pred_depth = cv2.resize(
                    pred_depth,
                    (gt_depth.shape[1], gt_depth.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            pred_depths_resized.append(pred_depth)
        
        # Compute metrics
        batch_metrics = batch_compute_depth_errors(
            np.array(gt_depths.numpy()),
            np.array(pred_depths_resized),
            median_scaling=median_scaling
        )
        
        all_metrics.extend(batch_metrics)
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Ensemble Model Evaluation for Cityscapes')
    
    parser.add_argument('--data_root', type=str, default='datasets',
                        help='Root path to datasets folder')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (smaller for ensemble)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use test-time augmentation')
    parser.add_argument('--output_dir', type=str, default='benchmark/results/ensemble_cityscapes',
                        help='Directory to save results')
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
    print(f"RT-MonoDepth Ensemble Evaluation on Cityscapes")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Test-time augmentation: {args.use_tta}")
    print(f"{'='*60}\n")
    
    # Define model configurations (path, type, weight)
    # Weights based on KITTI performance (δ<1.25)
    model_configs = [
        ('weights/RTMonoDepth/full/sh_640_192', 'full', 0.9609),  # Best performer
        ('weights/RTMonoDepth/full/s_640_192', 'full', 0.9538),
        ('weights/RTMonoDepth/full/m_640_192', 'full', 0.9538),
        ('weights/RTMonoDepth/full/ms_640_192', 'full', 0.9559),
        ('weights/RTMonoDepth/s/m_640_192', 's', 0.9353),
        ('weights/RTMonoDepth/s/ms_640_192', 's', 0.9369),
    ]
    
    # Create ensemble
    ensemble = EnsembleModel(model_configs, device=device)
    
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
    ])
    
    # Load Cityscapes validation dataset
    print("\nLoading Cityscapes validation dataset...")
    dataset = get_dataset(
        dataset_name='cityscapes',
        data_root=args.data_root,
        split='val',
        transform=transform
    )
    
    if len(dataset) == 0:
        print("❌ Cityscapes dataset is empty!")
        return
    
    # Evaluate
    metrics = evaluate_ensemble(
        ensemble=ensemble,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        median_scaling=True
    )
    
    # Print metrics
    print_metrics(metrics, "CITYSCAPES ENSEMBLE")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        json_results = {
            k: [float(v[0]), float(v[1])] if isinstance(v, tuple) else v
            for k, v in metrics.items()
        }
        json_results['config'] = {
            'models': [str(Path(cfg[0]).name) for cfg in model_configs],
            'weights': [cfg[2] for cfg in model_configs],
            'use_tta': args.use_tta,
            'device': device
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to {json_path}")
    
    # Save as CSV
    csv_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std'])
        
        for metric_name, value in metrics.items():
            if isinstance(value, tuple):
                writer.writerow([metric_name, f"{value[0]:.4f}", f"{value[1]:.4f}"])
            else:
                writer.writerow([metric_name, value, ''])
    
    print(f"✅ Results saved to {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Ensemble Performance:")
    print(f"  AbsRel: {metrics['abs_rel'][0]:.4f}")
    print(f"  δ<1.25: {metrics['a1'][0]:.4f} ({metrics['a1'][0]*100:.2f}%)")
    print(f"\nComparison to single best model:")
    print(f"  Single model (full_sh): δ<1.25 = 38.25%")
    print(f"  Ensemble: δ<1.25 = {metrics['a1'][0]*100:.2f}%")
    print(f"  Improvement: +{(metrics['a1'][0] - 0.3825)*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
