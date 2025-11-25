#!/usr/bin/env python3
"""
NYU Depth V2 Camera Parameters and Dataset Information
Extract camera parameters from the MATLAB toolbox for reference.
"""

import numpy as np
from pathlib import Path

def print_nyu_camera_params():
    """
    Display NYU Depth V2 camera parameters from the toolbox.
    Reference: datasets/nyu_depth_v2/toolbox_nyu_depth_v2/camera_params.m
    """
    
    print("=" * 70)
    print("NYU Depth V2 Camera Parameters")
    print("=" * 70)
    print("Source: http://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")
    print("Toolbox: datasets/nyu_depth_v2/toolbox_nyu_depth_v2/camera_params.m")
    print()
    
    # RGB Camera Intrinsics
    print("RGB Camera Intrinsic Parameters:")
    print("-" * 70)
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    print(f"  Focal Length X (fx_rgb): {fx_rgb:.4f} pixels")
    print(f"  Focal Length Y (fy_rgb): {fy_rgb:.4f} pixels")
    print(f"  Principal Point X (cx_rgb): {cx_rgb:.4f} pixels")
    print(f"  Principal Point Y (cy_rgb): {cy_rgb:.4f} pixels")
    print()
    
    # RGB Distortion
    print("RGB Camera Distortion Parameters:")
    print("-" * 70)
    k1_rgb = 2.0796615318809061e-01
    k2_rgb = -5.8613825163911781e-01
    k3_rgb = 4.9856986684705107e-01
    p1_rgb = 7.2231363135888329e-04
    p2_rgb = 1.0479627195765181e-03
    print(f"  Radial k1: {k1_rgb:.6f}")
    print(f"  Radial k2: {k2_rgb:.6f}")
    print(f"  Radial k3: {k3_rgb:.6f}")
    print(f"  Tangential p1: {p1_rgb:.6f}")
    print(f"  Tangential p2: {p2_rgb:.6f}")
    print()
    
    # Depth Camera Intrinsics
    print("Depth Camera Intrinsic Parameters:")
    print("-" * 70)
    fx_d = 5.8262448167737955e+02
    fy_d = 5.8269103270988637e+02
    cx_d = 3.1304475870804731e+02
    cy_d = 2.3844389626620386e+02
    print(f"  Focal Length X (fx_d): {fx_d:.4f} pixels")
    print(f"  Focal Length Y (fy_d): {fy_d:.4f} pixels")
    print(f"  Principal Point X (cx_d): {cx_d:.4f} pixels")
    print(f"  Principal Point Y (cy_d): {cy_d:.4f} pixels")
    print()
    
    # Depth Distortion
    print("Depth Camera Distortion Parameters:")
    print("-" * 70)
    k1_d = -9.9897236553084481e-02
    k2_d = 3.9065324602765344e-01
    k3_d = -5.1031725053400578e-01
    p1_d = 1.9290592870229277e-03
    p2_d = -1.9422022475975055e-03
    print(f"  Radial k1: {k1_d:.6f}")
    print(f"  Radial k2: {k2_d:.6f}")
    print(f"  Radial k3: {k3_d:.6f}")
    print(f"  Tangential p1: {p1_d:.6f}")
    print(f"  Tangential p2: {p2_d:.6f}")
    print()
    
    # Translation and Rotation
    print("RGB-Depth Extrinsic Parameters:")
    print("-" * 70)
    t_x = 2.5031875059141302e-02
    t_y = 6.6238747008330102e-04
    t_z = -2.9342312935846411e-04
    print(f"  Translation X: {t_x:.6f} meters")
    print(f"  Translation Y: {t_y:.6f} meters")
    print(f"  Translation Z: {t_z:.6f} meters")
    print()
    
    # Depth Range
    print("Depth Range:")
    print("-" * 70)
    max_depth = 10.0
    min_depth = 0.1  # Practical minimum
    print(f"  Maximum Depth: {max_depth:.1f} meters")
    print(f"  Minimum Depth: {min_depth:.1f} meters (practical)")
    print()
    
    # Image Resolution
    print("Image Resolution:")
    print("-" * 70)
    print("  RGB Image: 640 x 480 pixels")
    print("  Depth Map: 640 x 480 pixels (aligned to RGB)")
    print()
    
    # Dataset Statistics
    print("Dataset Statistics:")
    print("-" * 70)
    print("  Total Images: 1449 RGB-D pairs")
    print("  Eigen Test Split: 654 images")
    print("  Split Method: Every 2nd image (indices 0, 2, 4, ..., 1308)")
    print()
    
    # Depth Map Types
    print("Depth Map Types:")
    print("-" * 70)
    print("  1. rawDepths: Raw projected depth (with holes)")
    print("  2. depths: Preprocessed depth (inpainted using colorization)")
    print("     -> Recommended: Use 'depths' for evaluation")
    print()
    
    print("=" * 70)
    print("Python Dictionary Format (for code):")
    print("=" * 70)
    print("""
camera_params = {
    'fx_rgb': 518.8579,
    'fy_rgb': 519.4696,
    'cx_rgb': 325.5824,
    'cy_rgb': 253.7362,
    'fx_d': 582.6245,
    'fy_d': 582.6910,
    'cx_d': 313.0448,
    'cy_d': 238.4439,
    'max_depth': 10.0,
    'min_depth': 0.1,
    'image_width': 640,
    'image_height': 480,
}
    """)
    print("=" * 70)


if __name__ == "__main__":
    print_nyu_camera_params()
