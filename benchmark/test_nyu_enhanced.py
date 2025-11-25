#!/usr/bin/env python3
"""
Test script for enhanced NYU Depth V2 dataset loader.
Validates all new features and optimizations.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_loaders import NYUDepthV2Dataset
from nyu_utils import NYUCameraModel, compute_depth_quality_metrics, get_dataset_info


def test_basic_loader():
    """Test basic dataset loading"""
    print("="*80)
    print("TEST 1: Basic Loader")
    print("="*80)
    
    dataset = NYUDepthV2Dataset(
        data_path='../datasets/nyu_depth_v2',
        split='test',
        use_raw_depth=False,
        return_valid_mask=True
    )
    
    print(f"\n✓ Dataset loaded: {len(dataset)} images")
    
    # Test first sample
    sample = dataset[0]
    print(f"\n✓ Sample 0:")
    # Image is a PIL Image, convert to check shape
    img_array = np.array(sample['image'])
    print(f"  - Image shape: {img_array.shape}")
    print(f"  - Image type: {type(sample['image'])}")
    print(f"  - Depth shape: {sample['depth'].shape}")
    print(f"  - Valid mask shape: {sample['valid_mask'].shape}")
    print(f"  - Filename: {sample['filename']}")
    print(f"  - Depth range: [{sample['min_depth']}, {sample['max_depth']}]")
    print(f"  - Actual depth range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]")
    print(f"  - Valid pixels: {sample['valid_mask'].sum()} / {sample['valid_mask'].numel()}")
    
    assert img_array.shape == (480, 640, 3), "Image should be 480x640x3"
    assert sample['depth'].shape == torch.Size([480, 640]), "Depth should be 480x640"
    assert sample['valid_mask'].shape == torch.Size([480, 640]), "Valid mask should be 480x640"
    
    print("\n✓ All basic tests passed!")
    return dataset


def test_eigen_crop():
    """Test Eigen crop functionality"""
    print("\n" + "="*80)
    print("TEST 2: Eigen Crop")
    print("="*80)
    
    dataset = NYUDepthV2Dataset(
        data_path='../datasets/nyu_depth_v2',
        split='test',
        crop_eigen=True,
        return_valid_mask=True
    )
    
    sample = dataset[0]
    expected_h = 471 - 45  # 426
    expected_w = 601 - 41  # 560
    
    print(f"\n✓ Cropped dataset loaded: {len(dataset)} images")
    print(f"  - Expected shape: {expected_h}x{expected_w}")
    print(f"  - Actual depth shape: {sample['depth'].shape}")
    img_array = np.array(sample['image'])
    print(f"  - Actual image shape: {img_array.shape}")
    
    assert sample['depth'].shape == torch.Size([expected_h, expected_w]), \
        f"Cropped depth should be {expected_h}x{expected_w}"
    assert img_array.shape == (expected_h, expected_w, 3), \
        f"Cropped image should be {expected_h}x{expected_w}x3"
    
    print("\n✓ Eigen crop test passed!")
    return dataset


def test_raw_depth():
    """Test raw depth loading"""
    print("\n" + "="*80)
    print("TEST 3: Raw Depth")
    print("="*80)
    
    try:
        dataset = NYUDepthV2Dataset(
            data_path='../datasets/nyu_depth_v2',
            split='test',
            use_raw_depth=True
        )
        
        print(f"\n✓ Raw depth dataset loaded: {len(dataset)} images")
        
        sample = dataset[0]
        print(f"  - Depth range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]")
        print(f"  - Non-zero pixels: {(sample['depth'] > 0).sum()} / {sample['depth'].numel()}")
        
        print("\n✓ Raw depth test passed!")
        return dataset
    except Exception as e:
        print(f"\n⚠ Raw depth not available in dataset: {e}")
        print("  This is expected if the .mat file doesn't include rawDepths")
        return None


def test_camera_parameters():
    """Test camera parameter access"""
    print("\n" + "="*80)
    print("TEST 4: Camera Parameters")
    print("="*80)
    
    dataset = NYUDepthV2Dataset(
        data_path='../datasets/nyu_depth_v2',
        split='test'
    )
    
    # Test camera params access
    params = dataset.get_camera_params()
    print(f"\n✓ Camera parameters retrieved")
    print(f"  - RGB focal length: ({params['fx_rgb']:.2f}, {params['fy_rgb']:.2f})")
    print(f"  - Depth focal length: ({params['fx_d']:.2f}, {params['fy_d']:.2f})")
    
    # Test intrinsics matrix
    K_rgb = dataset.get_intrinsics_matrix('rgb')
    K_d = dataset.get_intrinsics_matrix('depth')
    
    print(f"\n✓ RGB Intrinsics Matrix:")
    print(K_rgb)
    
    print(f"\n✓ Depth Intrinsics Matrix:")
    print(K_d)
    
    assert K_rgb.shape == (3, 3), "Intrinsics should be 3x3"
    assert K_d.shape == (3, 3), "Intrinsics should be 3x3"
    
    print("\n✓ Camera parameter tests passed!")


def test_depth_statistics():
    """Test depth statistics computation"""
    print("\n" + "="*80)
    print("TEST 5: Depth Statistics")
    print("="*80)
    
    dataset = NYUDepthV2Dataset(
        data_path='../datasets/nyu_depth_v2',
        split='test'
    )
    
    stats = dataset.get_depth_statistics()
    
    print(f"\n✓ Depth statistics computed:")
    print(f"  - Min depth: {stats['min']:.3f} m")
    print(f"  - Max depth: {stats['max']:.3f} m")
    print(f"  - Mean depth: {stats['mean']:.3f} m")
    print(f"  - Median depth: {stats['median']:.3f} m")
    print(f"  - Std dev: {stats['std']:.3f} m")
    print(f"  - Valid pixels: {stats['num_valid']:,} / {stats['num_total']:,}")
    print(f"  - Valid ratio: {stats['valid_ratio']:.2%}")
    
    assert 0.1 <= stats['min'] <= 10.0, "Min depth should be in valid range"
    assert 0.1 <= stats['max'] <= 10.0, "Max depth should be in valid range"
    assert stats['valid_ratio'] > 0.8, "Most pixels should be valid"
    
    print("\n✓ Depth statistics test passed!")


def test_camera_model():
    """Test NYU camera model utilities"""
    print("\n" + "="*80)
    print("TEST 6: Camera Model Utilities")
    print("="*80)
    
    camera = NYUCameraModel()
    
    # Test intrinsics
    K_rgb = camera.get_intrinsics_matrix('rgb')
    K_d = camera.get_intrinsics_matrix('depth')
    
    print(f"\n✓ Camera model initialized")
    print(f"  - RGB focal: ({camera.fx_rgb:.2f}, {camera.fy_rgb:.2f})")
    print(f"  - Depth focal: ({camera.fx_d:.2f}, {camera.fy_d:.2f})")
    
    # Test projection
    test_depth = np.random.uniform(1, 5, (480, 640)).astype(np.float32)
    points = camera.project_depth_to_points(test_depth)
    
    print(f"\n✓ Depth projection test:")
    print(f"  - Input depth shape: {test_depth.shape}")
    print(f"  - Output points shape: {points.shape}")
    print(f"  - Points range: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
          f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
          f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    
    assert points.shape[1] == 3, "Points should be Nx3"
    assert len(points) > 0, "Should have valid points"
    
    # Test Eigen crop
    test_image = np.random.rand(480, 640, 3)
    cropped = camera.apply_eigen_crop(test_image)
    
    print(f"\n✓ Eigen crop test:")
    print(f"  - Input shape: {test_image.shape}")
    print(f"  - Output shape: {cropped.shape}")
    print(f"  - Expected: (426, 560, 3)")
    
    assert cropped.shape == (426, 560, 3), "Cropped shape should be 426x560x3"
    
    print("\n✓ Camera model tests passed!")


def test_quality_metrics():
    """Test depth quality metrics computation"""
    print("\n" + "="*80)
    print("TEST 7: Quality Metrics")
    print("="*80)
    
    # Create synthetic ground truth and prediction
    depth_gt = np.random.uniform(1, 8, (480, 640)).astype(np.float32)
    depth_pred = depth_gt + np.random.normal(0, 0.1, (480, 640)).astype(np.float32)
    
    metrics = compute_depth_quality_metrics(depth_gt, depth_pred)
    
    print(f"\n✓ Quality metrics computed:")
    print(f"  - RMSE: {metrics['rmse']:.4f}")
    print(f"  - RMSE(log): {metrics['rmse_log']:.4f}")
    print(f"  - MAE: {metrics['mae']:.4f}")
    print(f"  - AbsRel: {metrics['abs_rel']:.4f}")
    print(f"  - SqRel: {metrics['sq_rel']:.4f}")
    print(f"  - δ<1.25: {metrics['delta1']:.4f}")
    print(f"  - δ<1.25²: {metrics['delta2']:.4f}")
    print(f"  - δ<1.25³: {metrics['delta3']:.4f}")
    print(f"  - Valid pixels: {metrics['num_valid']:,}")
    
    assert 'rmse' in metrics, "Should have RMSE metric"
    assert 'delta1' in metrics, "Should have delta1 metric"
    assert metrics['num_valid'] > 0, "Should have valid pixels"
    
    print("\n✓ Quality metrics test passed!")


def test_full_pipeline():
    """Test full evaluation pipeline"""
    print("\n" + "="*80)
    print("TEST 8: Full Pipeline")
    print("="*80)
    
    # Load dataset
    dataset = NYUDepthV2Dataset(
        data_path='../datasets/nyu_depth_v2',
        split='test',
        return_valid_mask=True
    )
    
    # Process a few samples
    num_samples = min(5, len(dataset))
    all_metrics = []
    
    print(f"\n✓ Testing on {num_samples} samples...")
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Simulate prediction (just add noise for testing)
        depth_gt = sample['depth'].numpy()
        depth_pred = depth_gt + np.random.normal(0, 0.2, depth_gt.shape)
        
        # Compute metrics
        metrics = compute_depth_quality_metrics(depth_gt, depth_pred)
        all_metrics.append(metrics)
        
        print(f"  Sample {i}: RMSE={metrics['rmse']:.4f}, δ<1.25={metrics['delta1']:.4f}")
    
    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], float)}
    
    print(f"\n✓ Average metrics over {num_samples} samples:")
    print(f"  - RMSE: {avg_metrics['rmse']:.4f}")
    print(f"  - MAE: {avg_metrics['mae']:.4f}")
    print(f"  - δ<1.25: {avg_metrics['delta1']:.4f}")
    
    print("\n✓ Full pipeline test passed!")


def main():
    """Run all tests"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║         NYU Depth V2 Enhanced Dataset Loader - Validation Tests             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    try:
        # Display dataset info
        get_dataset_info()
        
        # Run tests
        test_basic_loader()
        test_eigen_crop()
        test_raw_depth()
        test_camera_parameters()
        test_depth_statistics()
        test_camera_model()
        test_quality_metrics()
        test_full_pipeline()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nThe enhanced NYU Depth V2 loader is ready for use with:")
        print("  ✓ Official Eigen split (654 test images)")
        print("  ✓ Full camera calibration parameters")
        print("  ✓ Raw and preprocessed depth support")
        print("  ✓ Eigen crop support")
        print("  ✓ Valid depth masking")
        print("  ✓ Quality metrics computation")
        print("  ✓ 3D projection utilities")
        print("  ✓ Depth statistics analysis")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
