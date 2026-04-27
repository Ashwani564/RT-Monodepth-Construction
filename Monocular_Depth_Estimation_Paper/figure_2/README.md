# Figure 2: Construction Site Safety Dataset Samples

This directory contains sample images from the Construction Site Safety dataset used for training YOLOv11n.

## Dataset Information

- **Source**: Roboflow Universe Construction Safety datasets
- **Classes** (10 total):
  - Hardhat
  - Mask
  - NO-Hardhat
  - NO-Mask
  - NO-Safety Vest
  - Person
  - Safety Cone
  - Safety Vest
  - machinery
  - vehicle

## Available Images

### Individual Samples (480×480 px each)
Extracted from YOLO training batch grids:

| File | Description |
|------|-------------|
| `train_sample01.jpg` - `train_sample16.jpg` | 16 samples from train_batch0 |
| `train_b1_sample01.jpg` - `train_b1_sample16.jpg` | 16 samples from train_batch1 |

### Overview Images (1920×1920 px)
Full 4×4 grid images showing training progress:

| File | Description |
|------|-------------|
| `train_batch0_overview.jpg` | First training batch with GT annotations |
| `train_batch1_overview.jpg` | Second training batch with GT annotations |
| `val_batch0_labels.jpg` | Validation batch with ground truth labels |
| `labels_distribution.jpg` | Class distribution visualization |

## Suggested Samples for Figure 2

For a diverse 2×3 grid showing workers, PPE, and machinery, consider:

1. **train_sample01.jpg** - Workers with hardhats and vests
2. **train_sample02.jpg** - Multiple workers, varied poses
3. **train_sample05.jpg** - Machinery and workers
4. **train_sample09.jpg** - Different lighting conditions
5. **train_b1_sample03.jpg** - Occlusion scenarios
6. **train_b1_sample07.jpg** - Diverse scene composition

## How to Create Figure 2

1. Open `figure2_template.drawio` in VS Code (Draw.io extension) or [draw.io](https://app.diagrams.net)
2. For each placeholder:
   - Right-click → **Edit** → **Image**
   - Select the corresponding JPG from this folder
3. Delete the instruction box
4. Export as PNG or PDF for your paper

## Caption for Paper

> **Figure 2**: Representative samples from the Construction Site Safety dataset used for training YOLOv11n. The dataset features diverse lighting conditions, partial occlusions, and varied worker poses. Ground truth bounding boxes show annotated classes including: Hardhat (green), Safety Vest (blue), Person (cyan), NO-Hardhat (red), machinery (orange), and vehicle (yellow). Training data sourced from Roboflow Universe construction safety datasets.
