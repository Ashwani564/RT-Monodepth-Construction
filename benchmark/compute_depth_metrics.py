#!/usr/bin/env python3
"""
Depth Metrics Computation for RT-MonoDepth Evaluation
Implements standard metrics from Eigen et al. and Monodepth2
"""

import numpy as np
import torch


def compute_depth_errors(gt, pred):
    """
    Compute depth estimation errors
    
    Args:
        gt: Ground truth depth (numpy array or tensor)
        pred: Predicted depth (numpy array or tensor)
    
    Returns:
        dict: Dictionary containing all depth metrics
    """
    # Convert to numpy if needed
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    # Flatten arrays
    gt = gt.flatten()
    pred = pred.flatten()
    
    # Create valid mask (remove zeros and invalid values)
    valid_mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    if valid_mask.sum() < 10:
        # Not enough valid pixels
        return {
            'abs_rel': np.nan,
            'sq_rel': np.nan,
            'rmse': np.nan,
            'rmse_log': np.nan,
            'a1': np.nan,
            'a2': np.nan,
            'a3': np.nan,
        }
    
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    
    # Absolute relative error
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    
    # Squared relative error
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    # Root mean squared error
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    
    # RMSE in log space
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    
    # Threshold accuracy (δ < 1.25, 1.25², 1.25³)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
    }


def compute_depth_errors_with_median_scaling(gt, pred):
    """
    Compute depth errors with median scaling (standard practice for monocular depth)
    
    Args:
        gt: Ground truth depth
        pred: Predicted depth
    
    Returns:
        dict: Depth metrics with median scaling applied
    """
    # Convert to numpy if needed
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    # Flatten
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    # Valid mask
    valid_mask = (gt_flat > 0) & (pred_flat > 0) & np.isfinite(gt_flat) & np.isfinite(pred_flat)
    
    if valid_mask.sum() < 10:
        return compute_depth_errors(gt, pred)
    
    # Compute median scaling ratio
    ratio = np.median(gt_flat[valid_mask]) / np.median(pred_flat[valid_mask])
    
    # Scale predictions
    pred_scaled = pred * ratio
    
    # Compute metrics on scaled predictions
    return compute_depth_errors(gt, pred_scaled)


def batch_compute_depth_errors(gt_batch, pred_batch, median_scaling=True):
    """
    Compute depth errors for a batch of images
    
    Args:
        gt_batch: Batch of ground truth depth maps [B, H, W] or [B, 1, H, W]
        pred_batch: Batch of predicted depth maps [B, H, W] or [B, 1, H, W]
        median_scaling: Whether to apply median scaling (default: True)
    
    Returns:
        list: List of metric dictionaries, one per image
    """
    # Convert to numpy
    if isinstance(gt_batch, torch.Tensor):
        gt_batch = gt_batch.detach().cpu().numpy()
    if isinstance(pred_batch, torch.Tensor):
        pred_batch = pred_batch.detach().cpu().numpy()
    
    # Handle [B, 1, H, W] format
    if gt_batch.ndim == 4 and gt_batch.shape[1] == 1:
        gt_batch = gt_batch.squeeze(1)
    if pred_batch.ndim == 4 and pred_batch.shape[1] == 1:
        pred_batch = pred_batch.squeeze(1)
    
    results = []
    for gt, pred in zip(gt_batch, pred_batch):
        if median_scaling:
            metrics = compute_depth_errors_with_median_scaling(gt, pred)
        else:
            metrics = compute_depth_errors(gt, pred)
        results.append(metrics)
    
    return results


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple images
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        dict: Aggregated metrics (mean and std)
    """
    # Filter out NaN entries
    valid_metrics = [m for m in metrics_list if not np.isnan(m['abs_rel'])]
    
    if len(valid_metrics) == 0:
        return {
            'abs_rel': (np.nan, np.nan),
            'sq_rel': (np.nan, np.nan),
            'rmse': (np.nan, np.nan),
            'rmse_log': (np.nan, np.nan),
            'a1': (np.nan, np.nan),
            'a2': (np.nan, np.nan),
            'a3': (np.nan, np.nan),
            'num_valid': 0,
        }
    
    # Compute mean and std for each metric
    result = {}
    for key in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']:
        values = [m[key] for m in valid_metrics]
        result[key] = (np.mean(values), np.std(values))
    
    result['num_valid'] = len(valid_metrics)
    
    return result


def print_metrics(metrics, dataset_name=""):
    """
    Pretty print depth metrics
    
    Args:
        metrics: Dictionary of aggregated metrics
        dataset_name: Name of the dataset (for display)
    """
    if dataset_name:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
    
    print(f"Valid samples: {metrics['num_valid']}")
    print(f"\nScale-Invariant Metrics:")
    print(f"  AbsRel:    {metrics['abs_rel'][0]:.4f} ± {metrics['abs_rel'][1]:.4f}")
    print(f"  SqRel:     {metrics['sq_rel'][0]:.4f} ± {metrics['sq_rel'][1]:.4f}")
    print(f"  RMSE:      {metrics['rmse'][0]:.4f} ± {metrics['rmse'][1]:.4f}")
    print(f"  RMSElog:   {metrics['rmse_log'][0]:.4f} ± {metrics['rmse_log'][1]:.4f}")
    print(f"\nThreshold Accuracy:")
    print(f"  δ < 1.25:  {metrics['a1'][0]:.4f} ± {metrics['a1'][1]:.4f}")
    print(f"  δ < 1.25²: {metrics['a2'][0]:.4f} ± {metrics['a2'][1]:.4f}")
    print(f"  δ < 1.25³: {metrics['a3'][0]:.4f} ± {metrics['a3'][1]:.4f}")


if __name__ == "__main__":
    # Test the metrics computation
    print("Testing depth metrics computation...")
    
    # Create dummy data
    gt = np.random.rand(480, 640) * 10 + 1  # GT depth 1-11m
    pred = gt + np.random.randn(480, 640) * 0.5  # Add noise
    pred = np.clip(pred, 0.1, 20)  # Clip to valid range
    
    # Compute metrics without scaling
    print("\n1. Without median scaling:")
    metrics_no_scale = compute_depth_errors(gt, pred)
    for k, v in metrics_no_scale.items():
        print(f"  {k}: {v:.4f}")
    
    # Compute metrics with median scaling
    print("\n2. With median scaling:")
    metrics_with_scale = compute_depth_errors_with_median_scaling(gt, pred)
    for k, v in metrics_with_scale.items():
        print(f"  {k}: {v:.4f}")
    
    # Test batch computation
    print("\n3. Batch computation (3 images):")
    gt_batch = np.random.rand(3, 480, 640) * 10 + 1
    pred_batch = gt_batch + np.random.randn(3, 480, 640) * 0.5
    pred_batch = np.clip(pred_batch, 0.1, 20)
    
    batch_metrics = batch_compute_depth_errors(gt_batch, pred_batch)
    aggregated = aggregate_metrics(batch_metrics)
    print_metrics(aggregated, "Test Dataset")
    
    print("\n✅ Metrics computation test complete!")
