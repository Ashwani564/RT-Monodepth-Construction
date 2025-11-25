# Stage 2: YOLO Object Detection Evaluation

This directory contains scripts for evaluating the custom YOLOv11n model (`custom_yolo11n.pt`) on the PPE Detection dataset:

1. **PPE Detection** - Personal Protective Equipment detection (10 classes)

## Quick Start

### Run All Evaluations

```bash
# From project root
python stage2_benchmark/run_all_evaluations.py
```

### Run PPE Evaluation

**PPE Detection:**
```bash
python stage2_benchmark/evaluate_yolo_ppe.py
```

Or use custom model:
```bash
python stage2_benchmark/run_all_evaluations.py --model path/to/model.pt
```

## Datasets

### PPE Detection Dataset

**Location:** `datasets/yolo/ppe-detection/`

**Classes (10):**
- 0: Hardhat
- 1: Mask
- 2: NO-Hardhat
- 3: NO-Mask
- 4: NO-Safety Vest
- 5: Person
- 6: Safety Cone
- 7: Safety Vest
- 8: machinery
- 9: vehicle

**Use Case:** Construction site safety monitoring, PPE compliance detection

## Output Structure

All results are saved in `stage2_benchmark/results/`:

```
stage2_benchmark/results/
└── ppe/
    ├── ppe_results_TIMESTAMP.json          # Full metrics in JSON
    ├── ppe_results_TIMESTAMP.csv           # Overall metrics
    ├── ppe_per_class_TIMESTAMP.csv         # Per-class breakdown
    └── ppe_evaluation/                     # Ultralytics outputs (plots, etc.)
```

## Metrics Explained

### mAP@50 (Mean Average Precision at IoU=0.50)
- Most commonly reported metric
- Measures detection accuracy at 50% IoU threshold
- Range: 0.0 to 1.0 (higher is better)

### mAP@50-95 (Standard)
- Average mAP across IoU thresholds from 0.50 to 0.95
- More stringent metric, requires precise localization
- Standard metric for object detection benchmarks
- Range: 0.0 to 1.0 (higher is better)

### Precision
- Percentage of correct detections among all detections
- `Precision = TP / (TP + FP)`
- High precision = few false positives

### Recall
- Percentage of ground truth objects that were detected
- `Recall = TP / (TP + FN)`
- High recall = few missed detections

### Speed Metrics
- **Preprocess:** Image loading and preprocessing time
- **Inference:** Model forward pass time
- **Postprocess:** NMS and result formatting time
- **Total:** End-to-end time per image
- **FPS:** Frames per second (1000 / total_ms)

## Evaluation Parameters

### PPE Detection
- **Confidence Threshold:** 0.25 (standard for custom datasets)
- **IoU Threshold:** 0.45 (standard for NMS)
- **Image Size:** 640×640

## Requirements

All dependencies are in the main `requirements.txt`:
- `ultralytics` - YOLO implementation
- `torch` - PyTorch framework
- `torchvision` - Vision utilities
- `numpy`, `opencv-python`, `Pillow` - Image processing

## Hardware Acceleration

The scripts automatically detect and use:
- **CUDA** (NVIDIA GPU) - if available
- **MPS** (Apple Silicon) - if available
- **CPU** - fallback

## Expected Runtime

| Dataset | Images | Time (GPU) | Time (MPS) | Time (CPU) |
|---------|--------|------------|------------|------------|
| PPE Detection | ~varies | ~1-2 min | ~2-5 min | ~5-15 min |

*Times are approximate and depend on hardware*

## Troubleshooting

### Model not found
```bash
# Make sure custom_yolo11n.pt is in the project root
ls custom_yolo11n.pt
```

### Dataset not found
```bash
# Check dataset structure
ls datasets/yolo/ppe-detection/
```

### Out of memory
- Reduce batch size in the evaluation scripts
- Use smaller image size (e.g., 320 instead of 640)
- Close other applications

## Output Files

### JSON Files
Complete evaluation results including:
- Overall metrics (mAP, precision, recall)
- Per-class metrics
- Speed benchmarks
- Configuration parameters

### CSV Files
Tabular format for easy analysis:
- `*_results_*.csv`: Overall metrics
- `*_per_class_*.csv`: Per-class breakdown

### Plots (in evaluation directories)
- Confusion matrix
- Precision-Recall curves
- F1-Confidence curves

## Citation

When using these evaluation scripts in research:

```bibtex
@misc{rtmonodepth_yolo_eval_2025,
  author = {Ashwani},
  title = {YOLO Object Detection Evaluation - Stage 2},
  year = {2025},
  month = {November},
  howpublished = {\url{https://github.com/Ashwani564/RT-Monodepth-Construction}}
}
```

## Contact

For issues or questions:
- Create an issue on GitHub
- Check the main project README.md

---

**Last Updated:** November 24, 2025
