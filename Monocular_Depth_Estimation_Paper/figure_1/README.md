# Figure 1: Qualitative Depth Estimation Results

## Directory Contents

### Generated Images (5 samples × 4 types = 20 images)

**Image Dimensions: 1242 × 375 pixels (aspect ratio 3.31:1)**

| Image | RGB Input | Ground Truth | GT Overlay | Prediction |
|-------|-----------|--------------|------------|------------|
| 1 | `img1_rgb.png` | `img1_gt_depth.png` | `img1_gt_overlay.png` | `img1_pred_depth.png` |
| 2 | `img2_rgb.png` | `img2_gt_depth.png` | `img2_gt_overlay.png` | `img2_pred_depth.png` |
| 3 | `img3_rgb.png` | `img3_gt_depth.png` | `img3_gt_overlay.png` | `img3_pred_depth.png` |
| 4 | `img4_rgb.png` | `img4_gt_depth.png` | `img4_gt_overlay.png` | `img4_pred_depth.png` |
| 5 | `img5_rgb.png` | `img5_gt_depth.png` | `img5_gt_overlay.png` | `img5_pred_depth.png` |

### Template File
- `figure1_template.drawio` - Draw.io template for creating the collage
- Placeholders sized to **310 × 57 pixels** (scaled maintaining aspect ratio)
- Page size: **1800 × 700 pixels**

---

## Image Sources (KITTI Dataset)

| # | Drive | Frame | Scene Description |
|---|-------|-------|-------------------|
| 1 | 2011_09_26_drive_0002_sync | 0000000010 | Urban street |
| 2 | 2011_09_26_drive_0005_sync | 0000000020 | Residential area |
| 3 | 2011_09_26_drive_0013_sync | 0000000030 | Highway/road |
| 4 | 2011_09_26_drive_0020_sync | 0000000015 | City driving |
| 5 | 2011_09_26_drive_0023_sync | 0000000010 | Urban scene |

---

## How to Create the Collage in Draw.io

### Option 1: Use the Template
1. Open `figure1_template.drawio` in draw.io (app.diagrams.net)
2. For each placeholder box, double-click → Edit → Image
3. Import the corresponding PNG file
4. Delete the instruction box
5. Export as PNG or PDF

### Option 2: Manual Layout
1. Go to [draw.io](https://app.diagrams.net)
2. Create a new diagram (1600×900 pixels recommended)
3. Arrange in 3 rows × 5 columns:
   - **Row 1:** RGB images (img1_rgb → img5_rgb)
   - **Row 2:** Ground truth depth (img1_gt_depth → img5_gt_depth)
   - **Row 3:** Predicted depth (img1_pred_depth → img5_pred_depth)
4. Add labels and caption

---

## Suggested Caption

> **Figure 1:** Qualitative results on the KITTI Eigen split. The proposed RT-MonoDepth variant (full_sh_640_192) captures fine structural details of vehicles and road geometry. **Top row:** Original RGB input images from KITTI. **Middle row:** Sparse LiDAR ground truth depth maps. **Bottom row:** Predicted dense depth maps from our model. Depth is colorized using the magma colormap (near=purple, far=yellow). The model achieves 96.09% accuracy (δ<1.25) on this benchmark.

---

## Technical Details

- **Model:** RT-MonoDepth full_sh_640_192 (best variant)
- **Input Resolution:** 640×192 pixels
- **Original Image Resolution:** 1242×375 pixels (KITTI)
- **Depth Range:** 0-80 meters
- **Colormap:** Magma (matplotlib)
- **Median Scaling:** Applied for quantitative alignment

---

## Citation

If using these figures, please cite:

```bibtex
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR},
  year = {2012}
}

@inproceedings{Feng2024RTMonoDepth,
  author = {Cheng Feng and Congxuan Zhang and Zhen Chen and Weiming Hu and Liyue Ge},
  title = {Real-Time Monocular Depth Estimation on Embedded Systems},
  booktitle = {IEEE ICME},
  year = {2024}
}
```
