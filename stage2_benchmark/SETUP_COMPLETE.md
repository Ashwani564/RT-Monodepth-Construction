# Stage 2: YOLO Object Detection Evaluation - Complete Setup

**Date:** November 24, 2025  
**Project:** RT-MonoDepth-Construction  
**Branch:** benchmark  
**Status:** ✓ Ready for Execution

---

## Overview

Complete evaluation infrastructure for YOLO object detection using `custom_yolo11n.pt` model:

1. **PPE Detection** - Personal Protective Equipment (10 classes)

---

## Quick Start

### Option 1: Run All Evaluations (Recommended)

```bash
# From project root
./stage2_benchmark/run_evaluations.sh
```

or

```bash
python stage2_benchmark/run_all_evaluations.py
```

### Option 2: Run PPE Evaluation

**PPE Detection:**
```bash
python stage2_benchmark/evaluate_yolo_ppe.py
```

### Option 3: Custom Model

```bash
# Use custom model
python stage2_benchmark/run_all_evaluations.py --model path/to/model.pt
```

---

## File Structure

```
stage2_benchmark/
├── __init__.py                    # Module initialization
├── README.md                      # User documentation
├── evaluate_yolo_ppe.py          # PPE dataset evaluator
├── run_all_evaluations.py        # Combined runner (Python)
├── run_evaluations.sh            # Combined runner (Shell)
└── results/                      # Output directory (auto-created)
    └── ppe/                      # PPE evaluation results
        ├── ppe_results_TIMESTAMP.json
        ├── ppe_results_TIMESTAMP.csv
        ├── ppe_per_class_TIMESTAMP.csv
        └── ppe_evaluation/       # Ultralytics outputs
```

---

## Dataset Information

### PPE Detection Dataset

**Location:** `datasets/yolo/ppe-detection/css-data/`

**Structure:**
```
ppe-detection/
└── css-data/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

**Classes (10):**
| ID | Class Name | Description |
|----|------------|-------------|
| 0 | Hardhat | Person wearing hardhat |
| 1 | Mask | Person wearing mask |
| 2 | NO-Hardhat | Person without hardhat |
| 3 | NO-Mask | Person without mask |
| 4 | NO-Safety Vest | Person without safety vest |
| 5 | Person | General person detection |
| 6 | Safety Cone | Traffic/safety cone |
| 7 | Safety Vest | Person wearing safety vest |
| 8 | machinery | Construction machinery |
| 9 | vehicle | Vehicles on site |

**Evaluation Split:** `valid/` directory (validation set)

**Configuration:** Auto-generated `data.yaml` during evaluation

---

## Evaluation Metrics

### Primary Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **mAP@50** | Mean Average Precision at IoU=0.50 | 0-1 | Higher |
| **mAP@50-95** | Mean Average Precision (IoU 0.50-0.95) | 0-1 | Higher |
| **Precision** | Correct detections / All detections | 0-1 | Higher |
| **Recall** | Detected objects / All ground truth | 0-1 | Higher |

### Additional Metrics

- **mAP@75**: Stricter localization requirement
- **Per-class metrics**: Individual performance for each class
- **Speed metrics**: Inference time, FPS

### Interpretation

**Excellent Performance:**
- mAP@50 > 0.80 (80%)
- mAP@50-95 > 0.60 (60%)
- Precision > 0.85
- Recall > 0.75

**Good Performance:**
- mAP@50 > 0.65 (65%)
- mAP@50-95 > 0.45 (45%)
- Precision > 0.75
- Recall > 0.65

---

## Evaluation Parameters

### PPE Detection

| Parameter | Value | Reason |
|-----------|-------|--------|
| Confidence Threshold | 0.25 | Standard for custom datasets |
| IoU Threshold (NMS) | 0.45 | Standard for YOLO models |
| Image Size | 640×640 | YOLOv11 default |
| Device | Auto-detect | CUDA/MPS/CPU |

---

## Output Files

### JSON Files (*.json)

Complete evaluation results including:
- Model information
- Dataset details
- Overall metrics (mAP, precision, recall)
- Per-class metrics (all classes)
- Speed benchmarks (preprocess, inference, postprocess)
- Timestamp and configuration

**Example:**
```json
{
  "timestamp": "20251124_153045",
  "model": "custom_yolo11n.pt",
  "dataset": "PPE Detection",
  "overall": {
    "mAP50": 0.8532,
    "mAP50-95": 0.6421,
    "precision": 0.8721,
    "recall": 0.7893
  },
  "per_class": {...},
  "speed": {...}
}
```

### CSV Files (*.csv)

Tabular format for spreadsheet analysis:

**Overall Results CSV:**
- Model name, dataset, timestamp
- All overall metrics
- Speed benchmarks

**Per-Class CSV:**
- Class ID, name
- mAP@50-95, precision, recall for each class
- Easy sorting and filtering

### Ultralytics Outputs

Generated in `results/*/evaluation/` directories:
- **Confusion Matrix:** Visual representation of predictions vs ground truth
- **PR Curves:** Precision-Recall curves
- **F1 Curves:** F1 score vs confidence threshold
- **predictions.json:** Prediction results for analysis

---

## Expected Performance

### Runtime

| Dataset | Images | GPU (CUDA) | Apple Silicon (MPS) | CPU |
|---------|--------|------------|---------------------|-----|
| PPE Detection | ~varies | 1-2 min | 2-5 min | 5-15 min |

*Times are approximate and hardware-dependent*

### Model Performance (Expected Range)

**Custom Model on PPE:**
- Performance depends on training quality
- Should exceed 70% mAP@50 for good training
- Speed: ~100-150 FPS (GPU), ~40-80 FPS (MPS)

---

## Hardware Acceleration

Scripts automatically detect and use the best available hardware:

1. **CUDA (NVIDIA GPU)** - Fastest
   - Requires: NVIDIA GPU + CUDA toolkit
   - Expected speed: 100-200 FPS

2. **MPS (Apple Silicon)** - Fast
   - Requires: Mac with M1/M2/M3 chip
   - Expected speed: 40-80 FPS

3. **CPU** - Slowest (fallback)
   - Works on any system
   - Expected speed: 10-30 FPS

---

## Troubleshooting

### Common Issues

**1. Model not found**
```bash
Error: Model not found: custom_yolo11n.pt
```
**Solution:** Ensure model is in project root
```bash
ls custom_yolo11n.pt  # Should show the file
```

**2. Dataset not found**
```bash
Error: Dataset not found: datasets/yolo/ppe-detection
```
**Solution:** Check dataset directory structure
```bash
ls -la datasets/yolo/
ls -la datasets/yolo/ppe-detection/css-data/
```

**3. Out of memory**
```bash
RuntimeError: CUDA out of memory
```
**Solution:** 
- Close other applications
- Reduce batch size (edit scripts)
- Use smaller image size (320 instead of 640)
- Switch to CPU (slower but uses less memory)

**4. Import errors**
```bash
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

**5. Dataset structure issues**
```bash
Warning: annotations not found
```
**Solution:** Ensure dataset structure is correct:
```bash
ls datasets/yolo/ppe-detection/css-data/valid/
```

---

## Validation Checklist

Before running evaluations:

- [ ] Model file exists: `custom_yolo11n.pt`
- [ ] PPE dataset structure correct: `datasets/yolo/ppe-detection/css-data/`
- [ ] Python requirements installed: `pip install -r requirements.txt`
- [ ] Scripts are executable: `chmod +x stage2_benchmark/run_evaluations.sh`
- [ ] Output directory writable: `stage2_benchmark/results/`

---

## Next Steps

### After Evaluation

1. **Review Results**
   - Check CSV files for metrics
   - View confusion matrices
   - Analyze per-class performance

2. **Compare Datasets**
   - PPE performance analysis
   - Identify strengths/weaknesses
   - Note class-specific issues

3. **Documentation** (After all evaluations complete)
   - Results will be summarized in `documentations/` directory
   - Full metrics comparison
   - Performance analysis
   - Recommendations

---

## Technical Details

### Scripts Overview

**evaluate_yolo_ppe.py:**
- PPEEvaluator class
- Auto-generates data.yaml for PPE dataset
- Handles Roboflow format (css-data structure)
- Saves results in JSON/CSV format
- Includes 10-class PPE evaluation

**run_all_evaluations.py:**
- Evaluation runner
- Command-line argument support
- Error handling and reporting
- Final summary generation

**run_evaluations.sh:**
- Bash wrapper script
- Pre-flight checks (model, datasets)
- User-friendly output
- Exit status handling

---

## Research & Citation

When using these evaluations in research or reports:

```bibtex
@misc{rtmonodepth_stage2_2025,
  author = {Ashwani},
  title = {YOLO Object Detection Evaluation - Stage 2},
  subtitle = {PPE Detection Benchmark},
  year = {2025},
  month = {November},
  howpublished = {\url{https://github.com/Ashwani564/RT-Monodepth-Construction}},
  note = {Evaluation of custom YOLOv11n on construction safety detection}
}
```

### Dataset Citations

**PPE Detection Dataset:**
- Construction Site Safety Dataset (Roboflow)
- 10 classes for PPE compliance monitoring

---

## Support

For issues, questions, or contributions:
- GitHub: https://github.com/Ashwani564/RT-Monodepth-Construction
- Create an issue with detailed error messages
- Check README.md for project overview

---

**Document Version:** 1.0  
**Last Updated:** November 24, 2025  
**Status:** Ready for Execution  
**Author:** Ashwani

---

## Summary

✓ **Evaluation scripts created**
✓ **PPE dataset configured**
✓ **Auto-detection** of dataset structures
✓ **Comprehensive output** (JSON, CSV, plots)
✓ **Hardware acceleration** (CUDA/MPS/CPU)
✓ **Error handling** and validation
✓ **Easy execution** (shell + Python)

**Ready to run evaluations!** 🚀
