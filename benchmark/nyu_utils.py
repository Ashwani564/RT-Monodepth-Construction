#!/usr/bin/env python3
"""
NYU Depth V2 Dataset Utilities

Advanced utilities for NYU Depth V2 dataset processing based on the official toolbox.
Includes projection, alignment, quality metrics, and visualization tools.

Reference: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
Toolbox: datasets/nyu_depth_v2/toolbox_nyu_depth_v2/
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


class NYUCameraModel:
    """
    NYU Depth V2 camera model with full calibration parameters.
    Implements projection, distortion, and alignment functions.
    """
    
    def __init__(self):
        """Initialize with official camera parameters from camera_params.m"""
        # RGB Camera Intrinsics
        self.fx_rgb = 5.1885790117450188e+02
        self.fy_rgb = 5.1946961112127485e+02
        self.cx_rgb = 3.2558244941119034e+02
        self.cy_rgb = 2.5373616633400465e+02
        
        # RGB Distortion (k1, k2, p1, p2, k3)
        self.dist_rgb = np.array([
            2.0796615318809061e-01,   # k1
            -5.8613825163911781e-01,  # k2
            7.2231363135888329e-04,   # p1
            1.0479627195765181e-03,   # p2
            4.9856986684705107e-01    # k3
        ])
        
        # Depth Camera Intrinsics
        self.fx_d = 5.8262448167737955e+02
        self.fy_d = 5.8269103270988637e+02
        self.cx_d = 3.1304475870804731e+02
        self.cy_d = 2.3844389626620386e+02
        
        # Depth Distortion (k1, k2, p1, p2, k3)
        self.dist_d = np.array([
            -9.9897236553084481e-02,  # k1
            3.9065324602765344e-01,   # k2
            1.9290592870229277e-03,   # p1
            -1.9422022475975055e-03,  # p2
            -5.1031725053400578e-01   # k3
        ])
        
        # Extrinsic parameters (Depth to RGB transformation)
        self.R = np.array([
            [9.9997798940829263e-01, -5.0359919480810989e-03, -4.3196624923060242e-03],
            [5.0518419386157446e-03,  9.9998051861143999e-01,  3.6662365748484798e-03],
            [4.3011152014118693e-03, -3.6879781309514218e-03,  9.9998394948385538e-01]
        ])
        self.t = np.array([2.5031875059141302e-02, 6.6238747008330102e-04, -2.9342312935846411e-04])
        
        # Depth range
        self.max_depth = 10.0
        self.min_depth = 0.1
        
        # Depth to absolute conversion parameter
        self.depth_param1 = 351.3
        
        # Eigen crop parameters
        self.eigen_crop = {'top': 45, 'bottom': 471, 'left': 41, 'right': 601}
    
    def get_intrinsics_matrix(self, camera='rgb'):
        """Get intrinsics as 3x3 matrix"""
        if camera == 'rgb':
            return np.array([
                [self.fx_rgb, 0, self.cx_rgb],
                [0, self.fy_rgb, self.cy_rgb],
                [0, 0, 1]
            ])
        else:  # depth
            return np.array([
                [self.fx_d, 0, self.cx_d],
                [0, self.fy_d, self.cy_d],
                [0, 0, 1]
            ])
    
    def project_depth_to_points(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Project depth map to 3D point cloud in depth camera coordinates.
        
        Args:
            depth_map: [H, W] depth values in meters
            
        Returns:
            points: [N, 3] array of (x, y, z) coordinates for valid depth pixels
        """
        h, w = depth_map.shape
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Valid depth mask
        valid = (depth_map >= self.min_depth) & (depth_map <= self.max_depth)
        
        # Get valid pixels
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth_map[valid]
        
        # Back-project to 3D (using depth camera intrinsics)
        x = (u_valid - self.cx_d) * z_valid / self.fx_d
        y = (v_valid - self.cy_d) * z_valid / self.fy_d
        
        points = np.stack([x, y, z_valid], axis=1)
        return points
    
    def align_depth_to_rgb(self, depth_map: np.ndarray, rgb_shape: Tuple[int, int]) -> np.ndarray:
        """
        Align depth map from depth camera frame to RGB camera frame.
        
        This is a simplified version. Full implementation would use the toolbox's
        project_depth_map.m function with proper distortion correction.
        
        Args:
            depth_map: [H, W] depth in depth camera frame
            rgb_shape: (H, W) target RGB image shape
            
        Returns:
            aligned_depth: [H, W] depth aligned to RGB camera
        """
        # Project to 3D points
        points_d = self.project_depth_to_points(depth_map)
        
        # Transform from depth to RGB camera frame
        points_rgb = (self.R @ points_d.T).T + self.t
        
        # Project to RGB image plane
        x_rgb = points_rgb[:, 0]
        y_rgb = points_rgb[:, 1]
        z_rgb = points_rgb[:, 2]
        
        u_rgb = (x_rgb * self.fx_rgb / z_rgb) + self.cx_rgb
        v_rgb = (y_rgb * self.fy_rgb / z_rgb) + self.cy_rgb
        
        # Create aligned depth map
        aligned_depth = np.zeros(rgb_shape, dtype=np.float32)
        
        # Round to nearest pixel
        u_rgb = np.round(u_rgb).astype(int)
        v_rgb = np.round(v_rgb).astype(int)
        
        # Check bounds
        valid_proj = (u_rgb >= 0) & (u_rgb < rgb_shape[1]) & (v_rgb >= 0) & (v_rgb < rgb_shape[0])
        
        u_rgb = u_rgb[valid_proj]
        v_rgb = v_rgb[valid_proj]
        z_rgb = z_rgb[valid_proj]
        
        # Fill aligned depth (handle overlaps by keeping closest depth)
        for i in range(len(u_rgb)):
            if aligned_depth[v_rgb[i], u_rgb[i]] == 0 or z_rgb[i] < aligned_depth[v_rgb[i], u_rgb[i]]:
                aligned_depth[v_rgb[i], u_rgb[i]] = z_rgb[i]
        
        return aligned_depth
    
    def undistort_image(self, image: np.ndarray, camera='rgb') -> np.ndarray:
        """
        Remove lens distortion from RGB or depth image.
        
        Args:
            image: [H, W, C] or [H, W] image
            camera: 'rgb' or 'depth'
            
        Returns:
            undistorted: Undistorted image
        """
        if camera == 'rgb':
            K = self.get_intrinsics_matrix('rgb')
            dist = self.dist_rgb
        else:
            K = self.get_intrinsics_matrix('depth')
            dist = self.dist_d
        
        h, w = image.shape[:2]
        
        # OpenCV expects distortion in format: (k1, k2, p1, p2, k3)
        undistorted = cv2.undistort(image, K, dist)
        
        return undistorted
    
    def apply_eigen_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply standard Eigen crop to remove invalid border regions"""
        return image[self.eigen_crop['top']:self.eigen_crop['bottom'],
                    self.eigen_crop['left']:self.eigen_crop['right']]


def compute_depth_quality_metrics(depth_gt: np.ndarray, depth_pred: np.ndarray,
                                  min_depth: float = 0.1, max_depth: float = 10.0) -> Dict[str, float]:
    """
    Compute comprehensive depth quality metrics for NYU Depth V2.
    
    Args:
        depth_gt: Ground truth depth [H, W]
        depth_pred: Predicted depth [H, W]
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        
    Returns:
        Dictionary of metrics including RMSE, MAE, accuracy thresholds, etc.
    """
    # Valid mask
    valid = (depth_gt >= min_depth) & (depth_gt <= max_depth)
    
    if not valid.any():
        return {
            'rmse': float('inf'),
            'rmse_log': float('inf'),
            'mae': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'delta1': 0.0,
            'delta2': 0.0,
            'delta3': 0.0,
            'num_valid': 0
        }
    
    gt_valid = depth_gt[valid]
    pred_valid = depth_pred[valid]
    
    # Clamp predictions to valid range
    pred_valid = np.clip(pred_valid, min_depth, max_depth)
    
    # Absolute errors
    abs_diff = np.abs(gt_valid - pred_valid)
    
    # RMSE
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    
    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    
    # MAE
    mae = np.mean(abs_diff)
    
    # Relative errors
    abs_rel = np.mean(abs_diff / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    
    # Accuracy under threshold (δ < threshold for threshold in [1.25, 1.25^2, 1.25^3])
    ratio = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)
    
    return {
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'mae': float(mae),
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
        'num_valid': int(valid.sum())
    }


def visualize_depth_comparison(rgb: np.ndarray, depth_gt: np.ndarray, depth_pred: np.ndarray,
                               save_path: Optional[str] = None, title: str = "Depth Comparison"):
    """
    Create a visualization comparing ground truth and predicted depth.
    
    Args:
        rgb: [H, W, 3] RGB image
        depth_gt: [H, W] ground truth depth
        depth_pred: [H, W] predicted depth
        save_path: Optional path to save the figure
        title: Figure title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RGB image
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # Ground truth depth
    im1 = axes[0, 1].imshow(depth_gt, cmap='viridis', vmin=0, vmax=10)
    axes[0, 1].set_title('Ground Truth Depth')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Predicted depth
    im2 = axes[1, 0].imshow(depth_pred, cmap='viridis', vmin=0, vmax=10)
    axes[1, 0].set_title('Predicted Depth')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Error map
    error = np.abs(depth_gt - depth_pred)
    im3 = axes[1, 1].imshow(error, cmap='hot', vmin=0, vmax=2)
    axes[1, 1].set_title('Absolute Error')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_depth_point_cloud(rgb: np.ndarray, depth: np.ndarray, 
                             camera_model: NYUCameraModel) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a colored point cloud from RGB and depth.
    
    Args:
        rgb: [H, W, 3] RGB image
        depth: [H, W] depth map
        camera_model: NYUCameraModel instance
        
    Returns:
        points: [N, 3] 3D points
        colors: [N, 3] RGB colors (0-255)
    """
    h, w = depth.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Valid depth mask
    valid = (depth >= camera_model.min_depth) & (depth <= camera_model.max_depth)
    
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = depth[valid]
    
    # Back-project to 3D
    x = (u_valid - camera_model.cx_d) * z_valid / camera_model.fx_d
    y = (v_valid - camera_model.cy_d) * z_valid / camera_model.fy_d
    
    points = np.stack([x, y, z_valid], axis=1)
    
    # Get corresponding colors
    colors = rgb[valid]
    
    return points, colors


def get_dataset_info():
    """
    Print comprehensive NYU Depth V2 dataset information.
    """
    info = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      NYU Depth V2 Dataset Information                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dataset: NYU Depth V2 (Indoor RGBD Dataset)
Source: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
Paper: "Indoor Segmentation and Support Inference from RGBD Images" (ECCV 2012)

DATASET STRUCTURE:
- Total images: 1,449 densely labeled pairs
- Resolution: 640x480 pixels
- Sensor: Microsoft Kinect v1
- Scenes: 464 diverse indoor scenes
- Room types: Living rooms, bedrooms, offices, bathrooms, etc.

STANDARD SPLITS:
- Eigen Test Split: 654 images (every 2nd image from 0-1308)
- Training: Remaining images + augmentation from raw data

CAMERA PARAMETERS:
RGB Camera:
  - Focal length: fx=518.86, fy=519.47
  - Principal point: cx=325.58, cy=253.74
  - Distortion: k1=0.208, k2=-0.586, p1=0.00072, p2=0.00105, k3=0.499

Depth Camera:
  - Focal length: fx=582.62, fy=582.69
  - Principal point: cx=313.04, cy=238.44
  - Distortion: k1=-0.099, k2=0.391, p1=0.00193, p2=-0.00194, k3=-0.510
  - Max depth: 10.0 meters
  - Min depth: 0.1 meters

EXTRINSICS (Depth to RGB):
  - Rotation: 3x3 matrix (near identity, small misalignment)
  - Translation: [25.0mm, 0.66mm, -0.29mm]

EVALUATION SETTINGS:
- Depth range: 0.1m to 10.0m
- Eigen crop: [45:471, 41:601] removes invalid border regions
- Metrics: RMSE, RMSE(log), AbsRel, SqRel, δ<1.25, δ<1.25², δ<1.25³

TOOLBOX FEATURES:
✓ Camera calibration parameters
✓ Depth projection and alignment
✓ Distortion correction (apply_distortion.m, undistort.m)
✓ Depth inpainting (fill_depth_colorization.m, fill_depth_cross_bf.m)
✓ Scene labels and instance segmentation
✓ Raw sensor data access

PYTHON SUPPORT:
✓ Official Eigen split (654 test images)
✓ Full camera parameter support
✓ Raw and preprocessed depth maps
✓ Eigen crop support
✓ Valid depth masking
✓ Quality metrics computation
✓ Visualization utilities
"""
    print(info)


if __name__ == '__main__':
    # Demo: Display dataset information and camera model
    get_dataset_info()
    
    print("\n" + "="*80)
    print("Camera Model Test")
    print("="*80)
    
    camera = NYUCameraModel()
    
    print("\nRGB Camera Intrinsics:")
    print(camera.get_intrinsics_matrix('rgb'))
    
    print("\nDepth Camera Intrinsics:")
    print(camera.get_intrinsics_matrix('depth'))
    
    print("\nExtrinsics (Depth to RGB):")
    print("Rotation:")
    print(camera.R)
    print("Translation:", camera.t)
    
    print("\nEigen Crop Region:")
    print(f"  Top: {camera.eigen_crop['top']}")
    print(f"  Bottom: {camera.eigen_crop['bottom']}")
    print(f"  Left: {camera.eigen_crop['left']}")
    print(f"  Right: {camera.eigen_crop['right']}")
    print(f"  Output size: {camera.eigen_crop['bottom'] - camera.eigen_crop['top']}x"
          f"{camera.eigen_crop['right'] - camera.eigen_crop['left']}")
    
    # Test projection
    print("\n" + "="*80)
    print("Testing Depth Projection")
    print("="*80)
    
    # Create a test depth map
    test_depth = np.ones((480, 640)) * 3.0  # 3 meters
    test_depth[100:200, 150:250] = 1.5  # Closer object
    
    points = camera.project_depth_to_points(test_depth)
    print(f"\nProjected {len(points)} valid 3D points")
    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
